# RUN: %PYTHON %s 2>&1 | FileCheck %s

from mlir.ir import *
from mlir.dialects.irdl import *
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
            irdl_test = dialect("irdl_test")
            with InsertionPoint(irdl_test.body):
                op = operation_("test_op")
                with InsertionPoint(op.body):
                    f32 = is_(TypeAttr.get(F32Type.get()))
                    operands_([f32], ["input"], [Variadicity.single])
                type1 = type_("type1")
                with InsertionPoint(type1.body):
                    f32 = is_(TypeAttr.get(F32Type.get()))
                    parameters([f32], ["val"])
                attr1 = attribute("attr1")
                with InsertionPoint(attr1.body):
                    test = is_(StringAttr.get("test"))
                    parameters([test], ["val"])

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

        load_dialects(module)

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
