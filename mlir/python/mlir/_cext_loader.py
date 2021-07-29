#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Common module for looking up and manipulating C-Extensions."""

# The normal layout is to have a nested _mlir_libs package that contains
# all native libraries and extensions. If that exists, use it, but also fallback
# to old behavior where extensions were at the top level as loose libraries.
# TODO: Remove the fallback once downstreams adapt.
try:
  from ._mlir_libs import *
  # TODO: Remove these aliases once everything migrates
  _preload_dependency = preload_dependency
  _load_extension = load_extension
except ModuleNotFoundError:
  # Assume that we are in-tree.
  # The _dlloader takes care of platform specific setup before we try to
  # load a shared library.
  # TODO: Remove _dlloader once all consolidated on the _mlir_libs approach.
  from ._dlloader import preload_dependency

  def load_extension(name):
    import importlib
    return importlib.import_module(name)  # i.e. '_mlir' at the top level

preload_dependency("MLIRPythonCAPI")

# Expose the corresponding C-Extension module with a well-known name at this
# top-level module. This allows relative imports like the following to
# function:
#   from .._cext_loader import _cext
# This reduces coupling, allowing embedding of the python sources into another
# project that can just vary based on this top-level loader module.
_cext = load_extension("_mlir")


def _reexport_cext(cext_module_name, target_module_name):
  """Re-exports a named sub-module of the C-Extension into another module.

  Typically:
    from ._cext_loader import _reexport_cext
    _reexport_cext("ir", __name__)
    del _reexport_cext
  """
  import sys
  target_module = sys.modules[target_module_name]
  submodule_names = cext_module_name.split(".")
  source_module = _cext
  for submodule_name in submodule_names:
    source_module = getattr(source_module, submodule_name)
  for attr_name in dir(source_module):
    if not attr_name.startswith("__"):
      setattr(target_module, attr_name, getattr(source_module, attr_name))


# Add our 'dialects' parent module to the search path for implementations.
_cext.globals.append_dialect_search_prefix("mlir.dialects")
