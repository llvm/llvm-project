# utils/docs

Common configuration and extensions for all of the sphinx projects in the
llvm-project monorepo (e.g. `llvm/docs/`, `clang/docs/`, `bolt/docs`, ...).

This directory is injected into the `PYTHONPATH` of `sphinx-build` processes as
part of the llvm-project CMake build.

**Note:** On the command-line, you must manually ensure this directory is part
of your `PYTHONPATH`. For example, to use `sphinx-autobuild` for projects which
support it, you might use:

```
$ PYTHONPATH=$PWD/utils/docs sphinx-autobuild llvm/docs/ /tmp/sphinx-build
```

## Testing your Environment

To smoke-test your sphinx environment outside of a CMake build, you can use the
`utils/docs/__main__.py` entrypoint (e.g. `python3 utils/docs --test`).
