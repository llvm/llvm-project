# RUN: %PYTHON %s | FileCheck %s

from mlir.dialects import ptr
from mlir.ir import Context, Location, Module, InsertionPoint, Attribute


def run(f):
    print("\nTEST:", f.__name__)
    with Context(), Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            f(module)
        print(module)
        assert module.operation.verify()


# CHECK-LABEL: TEST: test_smoke
@run
def test_smoke(_module):
    null_ptr = Attribute.parse("#ptr.null : !ptr.ptr<#llvm.address_space<1>>")
    null = ptr.constant(null_ptr)
    # CHECK: %0 = ptr.constant #ptr.null : !ptr.ptr<#llvm.address_space<1>>
    print(null)
