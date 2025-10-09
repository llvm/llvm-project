# RUN: echo "do nothing"
# just so lit doesn't complain about a missing RUN line
# noinspection PyUnusedImports
import contextlib
import ctypes
import sys


@contextlib.contextmanager
def dl_open_guard():
    old_flags = sys.getdlopenflags()
    sys.setdlopenflags(old_flags | ctypes.RTLD_GLOBAL)
    yield
    sys.setdlopenflags(old_flags)


with dl_open_guard():
    # noinspection PyUnresolvedReferences
    from mlir._mlir_libs import _mlir
    from mlir import ir

from mlir_standreallyalone.dialects import standalonereallyalone as standalone_d

with ir.Context() as ctx:
    standalone_d.register_dialects()
    module = ir.Module.parse(
        """
    %0 = arith.constant 2 : i32
    %1 = standalone.foo %0 : i32
    """
    )
    # CHECK: %[[C:.*]] = arith.constant 2 : i32
    # CHECK: standalone.foo %[[C]] : i32
    print(str(module))

# just so lit doesn't complain about this file
# UNSUPPORTED: target={{.*}}
