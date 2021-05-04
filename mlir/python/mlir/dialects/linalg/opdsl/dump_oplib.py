#!/usr/bin/which python
# Command line tool to load an oplib module and dump all of the operations
# it contains in some format.
"""Loads one or more modules containing op definitions and dumps them.

The dump format can be:

* `--dump_format=yaml` (default)
* `--dump_format=repr`

Positional arguments are interpreted as module names (optionally, relative to
this module). Loose module files can be specified via `--file <filepath>`.

Sample usage:
  # Dump the YAML op definitions for the core named ops (as in the dialect
  # source tree).
  python -m mlir.dialects.linalg.opdsl.dump_oplib .ops.core_named_ops

Note: YAML output is emitted in "document list" format with each operation
as its own "document". Practically, this means that each operation (or group
of composite ops) is emitted with a "---" preceding it, which can be useful
for testing.
"""

import argparse
import importlib

from .lang import *
from .lang.config import *
from .lang.yaml_helper import *


def create_arg_parser() -> argparse.ArgumentParser:
  p = argparse.ArgumentParser(description="Dump an oplib in various formats")
  p.add_argument("modules",
                 metavar="M",
                 type=str,
                 nargs="*",
                 help="Op module to dump")
  p.add_argument("--file",
                 metavar="F",
                 type=str,
                 nargs="*",
                 help="Python op file to dump")
  p.add_argument("--format",
                 type=str,
                 dest="format",
                 default="yaml",
                 choices=("yaml", "repr"),
                 help="Format in which to dump")
  return p


def load_module_from_file(module_name, file_path):
  spec = importlib.util.spec_from_file_location(module_name, file_path)
  m = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(m)
  return m


def main(args):
  # Load all configs.
  configs = []
  modules = []
  for module_name in args.modules:
    modules.append(
        importlib.import_module(module_name,
                                package="mlir.dialects.linalg.opdsl"))
  for i, file_path in enumerate(args.file or []):
    modules.append(load_module_from_file(f"_mlir_eval_oplib{i}", file_path))
  for m in modules:
    for attr_name, value in m.__dict__.items():
      # TODO: This class layering is awkward.
      if isinstance(value, DefinedOpCallable):
        try:
          linalg_config = LinalgOpConfig.from_linalg_op_def(value.model)
        except Exception as e:
          raise ValueError(
              f"Could not create LinalgOpConfig from {value.model}") from e
        configs.extend(linalg_config)

  # Print.
  if args.format == "yaml":
    print(yaml_dump_all(configs))
  elif args.format == "repr":
    for config in configs:
      print(repr(config))


if __name__ == "__main__":
  main(create_arg_parser().parse_args())
