#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Note that the only function of this module is currently to load the
# native module and re-export its symbols. In the future, this file is
# reserved as a trampoline to handle environment specific loading needs
# and arbitrate any one-time initialization needed in various shared-library
# scenarios.

__all__ = [
  "ir",
  "passmanager",
]

# The _dlloader takes care of platform specific setup before we try to
# load a shared library.
from . import _dlloader
_dlloader.preload_dependency("MLIRPublicAPI")

# Expose the corresponding C-Extension module with a well-known name at this
# top-level module. This allows relative imports like the following to
# function:
#   from .. import _cext
# This reduces coupling, allowing embedding of the python sources into another
# project that can just vary based on this top-level loader module.
import _mlir as _cext

def _reexport_cext(cext_module_name, target_module_name):
  """Re-exports a named sub-module of the C-Extension into another module.

  Typically:
    from . import _reexport_cext
    _reexport_cext("ir", __name__)
    del _reexport_cext
  """
  import sys
  target_module = sys.modules[target_module_name]
  source_module = getattr(_cext, cext_module_name)
  for attr_name in dir(source_module):
    if not attr_name.startswith("__"):
      setattr(target_module, attr_name, getattr(source_module, attr_name))


# Import sub-modules. Since these may import from here, this must come after
# any exported definitions.
from . import ir, passmanager

# Add our 'dialects' parent module to the search path for implementations.
_cext.globals.append_dialect_search_prefix("mlir.dialects")
