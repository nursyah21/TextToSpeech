"""
Legacy entry point. Use fairseq_cli/train.py or fairseq-train instead.
"""

from fairseq_cli.train import cli_main


if __name__ == "__main__":
    cli_main()