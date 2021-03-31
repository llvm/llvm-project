#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Common module for looking up and manipulating C-Extensions."""

# Packaged installs have a top-level _mlir_libs package with symbols:
#   load_extension(name): Loads a named extension module
#   preload_dependency(public_name): Loads a shared-library/DLL into the
#     namespace. TODO: Remove this in favor of a more robust mechanism.
# Conditionally switch based on whether we are in a package context.
try:
  import _mlir_libs
except ModuleNotFoundError:
  # Assume that we are in-tree.
  # The _dlloader takes care of platform specific setup before we try to
  # load a shared library.
  from ._dlloader import preload_dependency as _preload_dependency

  def _load_extension(name):
    import importlib
    return importlib.import_module(name)  # i.e. '_mlir' at the top level
else:
  # Packaged distribution.
  _load_extension = _mlir_libs.load_extension
  _preload_dependency = _mlir_libs.preload_dependency

_preload_dependency("MLIRPublicAPI")

# Expose the corresponding C-Extension module with a well-known name at this
# top-level module. This allows relative imports like the following to
# function:
#   from .._cext_loader import _cext
# This reduces coupling, allowing embedding of the python sources into another
# project that can just vary based on this top-level loader module.
_cext = _load_extension("_mlir")


def _reexport_cext(cext_module_name, target_module_name):
  """Re-exports a named sub-module of the C-Extension into another module.

  Typically:
    from ._cext_loader import _reexport_cext
    _reexport_cext("ir", __name__)
    del _reexport_cext
  """
  import sys
  target_module = sys.modules[target_module_name]
  source_module = getattr(_cext, cext_module_name)
  for attr_name in dir(source_module):
    if not attr_name.startswith("__"):
      setattr(target_module, attr_name, getattr(source_module, attr_name))


# Add our 'dialects' parent module to the search path for implementations.
_cext.globals.append_dialect_search_prefix("mlir.dialects")
