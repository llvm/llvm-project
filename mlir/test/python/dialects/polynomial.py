# RUN: %PYTHON %s | FileCheck %s

from mlir.ir import *
from mlir.dialects import polynomial


def constructAndPrintInModule(f):
    print("\nTEST:", f.__name__)
    with Context(), Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            f()
        print(module)
    return f


# CHECK-LABEL: TEST: test_smoke
@constructAndPrintInModule
def test_smoke():
    value = Attribute.parse("#polynomial.float_polynomial<0.5 + 1.3e06 x**2>")
    output = Type.parse("!polynomial.polynomial<ring=<coefficientType=f32>>")
    res = polynomial.constant(output, value)
    # CHECK: polynomial.constant {value = #polynomial.float_polynomial<0.5 + 1.3E+6x**2>} : <ring = <coefficientType = f32>>
    print(res)
