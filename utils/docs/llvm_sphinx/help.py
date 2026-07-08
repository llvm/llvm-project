# -*- coding: utf-8 -*-

import textwrap


def venv_help(err) -> str:
    return textwrap.dedent(
        f"""
        Missing LLVM documentation build dependencies.

        Import failed with:
          {err}

        The standard requirements file is:
          llvm-project/llvm/docs/requirements.txt

        From an llvm-project checkout, a typical pip setup is:
          python3 -m venv .venv
          . .venv/bin/activate
          python3 -m pip install -r llvm/docs/requirements.txt
          python3 utils/docs --test

        With uv, a typical one-shot command is:
          uv run --with-requirements llvm/docs/requirements.txt \\
            python utils/docs --test
        """
    ).strip()
