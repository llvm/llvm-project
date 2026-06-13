# utils/docs

Common configuration and extensions for all of the sphinx projects in the
llvm-project monorepo (e.g. `llvm/docs/`, `clang/docs/`, `bolt/docs`, ...).

This directory is injected into the `PYTHONPATH` of `sphinx-build` processes as
part of the llvm-project CMake build. On the command-line, either use the
`utils/docs/__main__.py` entrypoint (e.g. `python3 utils/docs --test` to
run smoke tests for your environment) or manually ensure this directory is part
of your `PYTHONPATH`.
