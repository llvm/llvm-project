# RUN: %PYTHON %s | FileCheck %s
# RUN: %PYTHON -m mypy %s --config-file %mlir_src_root/test/python/mypy.ini
# This is just a smoke test that the dialect is functional.
from array import array

from mlir.ir import *
from mlir.dialects import ub
from mlir.extras import types as T


def constructAndPrintInModule(f):
    print("\nTEST:", f.__name__)
    with Context(), Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            f()
        print(module)
    return f


# CHECK-LABEL: testSmoke
@constructAndPrintInModule
def testSmoke():
    # CHECK: Value(%{{.*}} = ub.poison : f32
    f32 = F32Type.get()
    poison = ub.poison(f32)
    print(poison)
    assert isinstance(poison, Value)
