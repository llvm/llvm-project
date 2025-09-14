# RUN: %PYTHON %s 2>&1 | FileCheck %s

from mlir.ir import *
from mlir.dialects import irdl
import sys


def run(f):
    print("\nTEST:", f.__name__, file=sys.stderr)
    f()


# CHECK: TEST: testIRDL
@run
def testIRDL():
    with Context() as ctx, Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            dialect = irdl.DialectOp("irdl_test")
            with InsertionPoint(dialect.body):
                op = irdl.OperationOp("test_op")
                with InsertionPoint(op.body):
                    f32 = irdl.is_(TypeAttr.get(F32Type.get()))
                    irdl.operands_([f32], ["input"], [irdl.Variadicity.single])

        # CHECK: module {
        # CHECK:   irdl.dialect @irdl_test {
        # CHECK:     irdl.operation @test_op {
        # CHECK:       %0 = irdl.is f32
        # CHECK:       irdl.operands(input: %0)
        # CHECK:     }
        # CHECK:   }
        # CHECK: }
        module.dump()

        irdl.load_dialects(module)

        m = Module.parse("""
          module {
            %a = arith.constant 1.0 : f32
            "irdl_test.test_op"(%a) : (f32) -> ()
          }
        """)
        # CHECK: "irdl_test.test_op"(%cst) : (f32) -> ()
        m.dump()
