# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import importlib
import itertools
import logging
import os
import sys
from typing import Sequence

from ._mlir import ir


_this_dir = os.path.dirname(__file__)

logger = logging.getLogger(__name__)

_path = __path__
_spec = __spec__
_name = __name__


class _M:
    __path__ = _path
    __spec__ = _spec
    __name__ = _name

    @staticmethod
    def get_lib_dirs() -> Sequence[str]:
        """Gets the lib directory for linking to shared libraries.

        On some platforms, the package may need to be built specially to export
        development libraries.
        """
        return [_this_dir]

    @staticmethod
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
    #   d. If the module has a 'disable_multithreading' attribute, it will be
    #     taken as a boolean. If it is True for any initializer, then the
    #     default behavior of enabling multithreading on the context
    #     will be suppressed. This complies with the original behavior of all
    #     contexts being created with multithreading enabled while allowing
    #     this behavior to be changed if needed (i.e. if a context_init_hook
    #     explicitly sets up multithreading).
    #
    # This facility allows downstreams to customize Context creation to their
    # needs.

    __registry = ir.DialectRegistry()
    __post_init_hooks = []
    __disable_multithreading = False
    from . import _mlir as _mlir

    def __get_registry(self):
        return self.__registry

    def process_initializer_module(self, module_name):
        try:
            m = importlib.import_module(f".{module_name}", __name__)
        except ModuleNotFoundError:
            return False
        except ImportError:
            message = (
                f"Error importing mlir initializer {module_name}. This may "
                "happen in unclean incremental builds but is likely a real bug if "
                "encountered otherwise and the MLIR Python API may not function."
            )
            logger.warning(message, exc_info=True)

        logger.debug("Initializing MLIR with module: %s", module_name)
        if hasattr(m, "register_dialects"):
            logger.debug("Registering dialects from initializer %r", m)
            m.register_dialects(self.__get_registry())
        if hasattr(m, "context_init_hook"):
            logger.debug("Adding context init hook from %r", m)
            self.__post_init_hooks.append(m.context_init_hook)
        if hasattr(m, "disable_multithreading"):
            if bool(m.disable_multithreading):
                logger.debug("Disabling multi-threading for context")
                self.__disable_multithreading = True
        return True

    def __init__(self):
        # If _mlirRegisterEverything is built, then include it as an initializer
        # module.
        init_module = None
        if self.process_initializer_module("_mlirRegisterEverything"):
            init_module = importlib.import_module(f"._mlirRegisterEverything", __name__)

        # Load all _site_initialize_{i} modules, where 'i' is a number starting
        # at 0.
        for i in itertools.count():
            module_name = f"_site_initialize_{i}"
            if not self.process_initializer_module(module_name):
                break

        that = self

        class Context(ir._BaseContext):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.append_dialect_registry(that._M__get_registry())
                for hook in that._M__post_init_hooks:
                    hook(self)
                if not that._M__disable_multithreading:
                    self.enable_multithreading(True)
                # TODO: There is some debate about whether we should eagerly load
                # all dialects. It is being done here in order to preserve existing
                # behavior. See: https://github.com/llvm/llvm-project/issues/56037
                self.load_all_available_dialects()
                if init_module:
                    logger.debug(
                        "Registering translations from initializer %r", init_module
                    )
                    init_module.register_llvm_translations(self)

        ir.Context = Context

    class MLIRError(Exception):
        """
        An exception with diagnostic information. Has the following fields:
          message: str
          error_diagnostics: List[ir.DiagnosticInfo]
        """

        def __init__(self, message, error_diagnostics):
            self.message = message
            self.error_diagnostics = error_diagnostics
            super().__init__(message, error_diagnostics)

        def __str__(self):
            s = self.message
            if self.error_diagnostics:
                s += ":"
            for diag in self.error_diagnostics:
                s += (
                    "\nerror: "
                    + str(diag.location)[4:-1]
                    + ": "
                    + diag.message.replace("\n", "\n  ")
                )
                for note in diag.notes:
                    s += (
                        "\n note: "
                        + str(note.location)[4:-1]
                        + ": "
                        + note.message.replace("\n", "\n  ")
                    )
            return s

    ir.MLIRError = MLIRError


sys.modules[__name__] = _M()
