# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Library functions for setting up common parser arguments"""

from argparse import ArgumentParser

def add_verbosity_arguments(parser: ArgumentParser) -> None:
    """Adds the arguments for verbosity to the ArgumentParser

    Arguments:
        parser -- the argument parser being modified with verbosity arguments
    """
    parser.add_argument(
        "--verbosity",
        type=str,
        help="The verbosity level to use for logging",
        default="INFO",
        nargs="?",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )