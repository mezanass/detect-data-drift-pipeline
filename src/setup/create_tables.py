import hydra
import pandas as pd
from omegaconf import DictConfig
from sqlalchemy import create_engine


def get_reference_df(filename: str):
    return pd.read_csv(filename, header=0, sep=",", parse_dates=["dteday"])


def get_empty_current_df(columns: list):
    return pd.DataFrame([], columns=columns)


def save_to_db(df: pd.DataFrame, db: DictConfig, table_name: str):
    engine = create_engine(
        f"postgresql://{db.username}:{db.password}@{db.host}:{db.port}/{db.database}"
    )
    df.to_sql(table_name, engine, if_exists="replace", index=False)


@hydra.main(config_path="../..", config_name="config", version_base=None)
def create_table(config: DictConfig):
    reference_df = get_reference_df(config.data.reference)
    current_df = get_empty_current_df(reference_df.columns)

    save_to_db(current_df, config.db, table_name="current")
    save_to_db(reference_df, config.db, table_name="reference")


if __name__ == "__main__":
    create_table()
