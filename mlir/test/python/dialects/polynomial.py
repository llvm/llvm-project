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
    res = polynomial.constant(value)
    # CHECK: polynomial.constant float<0.5 + 1.3E+6x**2> : <ring = <coefficientType = f32>>
    print(res)

    int_poly = polynomial.IntMonomial(1, 10)
    # CHECK: <1, 10>
    print(int_poly)
