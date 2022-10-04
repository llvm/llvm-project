# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Any, Sequence

import os

_this_dir = os.path.dirname(__file__)


def get_lib_dirs() -> Sequence[str]:
  """Gets the lib directory for linking to shared libraries.

  On some platforms, the package may need to be built specially to export
  development libraries.
  """
  return [_this_dir]


def get_include_dirs() -> Sequence[str]:
  """Gets the include directory for compiling against exported C libraries.

  Depending on how the package was build, development C libraries may or may
  not be present.
  """
  return [os.path.join(_this_dir, "include")]


# Perform Python level site initialization. This involves:
#   1. Attempting to load initializer modules, specific to the distribution.
#   2. Defining the concrete mlir.ir.Context that does site specific 
#      initialization.
#
# Aside from just being far more convenient to do this at the Python level,
# it is actually quite hard/impossible to have such __init__ hooks, given
# the pybind memory model (i.e. there is not a Python reference to the object
# in the scope of the base class __init__).
#
# For #1, we:
#   a. Probe for modules named '_mlirRegisterEverything' and 
#     '_site_initialize_{i}', where 'i' is a number starting at zero and 
#     proceeding so long as a module with the name is found.
#   b. If the module has a 'register_dialects' attribute, it will be called
#     immediately with a DialectRegistry to populate.
#   c. If the module has a 'context_init_hook', it will be added to a list
#     of callbacks that are invoked as the last step of Context 
#     initialization (and passed the Context under construction).
#
# This facility allows downstreams to customize Context creation to their
# needs.
def _site_initialize():
  import importlib
  import itertools
  import logging
  from ._mlir import ir
  registry = ir.DialectRegistry()
  post_init_hooks = []

  def process_initializer_module(module_name):
    try:
      m = importlib.import_module(f".{module_name}", __name__)
    except ModuleNotFoundError:
      return False
    except ImportError:
      message = (f"Error importing mlir initializer {module_name}. This may "
      "happen in unclean incremental builds but is likely a real bug if "
      "encountered otherwise and the MLIR Python API may not function.")
      logging.warning(message, exc_info=True)

    logging.debug("Initializing MLIR with module: %s", module_name)
    if hasattr(m, "register_dialects"):
      logging.debug("Registering dialects from initializer %r", m)
      m.register_dialects(registry)
    if hasattr(m, "context_init_hook"):
      logging.debug("Adding context init hook from %r", m)
      post_init_hooks.append(m.context_init_hook)
    return True


  # If _mlirRegisterEverything is built, then include it as an initializer
  # module.
  process_initializer_module("_mlirRegisterEverything")

  # Load all _site_initialize_{i} modules, where 'i' is a number starting
  # at 0.
  for i in itertools.count():
    module_name = f"_site_initialize_{i}"
    if not process_initializer_module(module_name):
      break

  class Context(ir._BaseContext):
    def __init__(self, *args, **kwargs):
      super().__init__(*args, **kwargs)
      self.append_dialect_registry(registry)
      for hook in post_init_hooks:
        hook(self)
      # TODO: There is some debate about whether we should eagerly load
      # all dialects. It is being done here in order to preserve existing
      # behavior. See: https://github.com/llvm/llvm-project/issues/56037
      self.load_all_available_dialects()

  ir.Context = Context


_site_initialize()
