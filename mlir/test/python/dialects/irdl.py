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
                type1 = irdl.TypeOp("type1")
                with InsertionPoint(type1.body):
                    f32 = irdl.is_(TypeAttr.get(F32Type.get()))
                    irdl.parameters([f32], ["val"])
                attr1 = irdl.AttributeOp("attr1")
                with InsertionPoint(attr1.body):
                    test = irdl.is_(StringAttr.get("test"))
                    irdl.parameters([test], ["val"])

        # CHECK: module {
        # CHECK:   irdl.dialect @irdl_test {
        # CHECK:     irdl.operation @test_op {
        # CHECK:       %0 = irdl.is f32
        # CHECK:       irdl.operands(input: %0)
        # CHECK:     }
        # CHECK:     irdl.type @type1 {
        # CHECK:       %0 = irdl.is f32
        # CHECK:       irdl.parameters(val: %0)
        # CHECK:     }
        # CHECK:     irdl.attribute @attr1 {
        # CHECK:       %0 = irdl.is "test"
        # CHECK:       irdl.parameters(val: %0)
        # CHECK:     }
        # CHECK:   }
        # CHECK: }
        module.operation.verify()
        module.dump()

        irdl.load_dialects(module)

        m = Module.parse(
            """
          module {
            %a = arith.constant 1.0 : f32
            "irdl_test.test_op"(%a) : (f32) -> ()
          }
        """
        )
        # CHECK: module {
        # CHECK:   "irdl_test.test_op"(%cst) : (f32) -> ()
        # CHECK: }
        m.dump()
