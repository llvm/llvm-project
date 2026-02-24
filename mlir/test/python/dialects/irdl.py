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


@run
def testIRDLTypes():
    with Context() as ctx, Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            irdl_test = dialect("irdl_type_test")
            with InsertionPoint(irdl_test.body):
                type1 = type_("type1")
                with InsertionPoint(type1.body):
                    iattr = base(base_name="#builtin.integer")
                    parameters([iattr], ["val"])
                type2 = type_("type2")
                with InsertionPoint(type2.body):
                    iattr = base(base_name="#builtin.integer")
                    unit = is_(UnitAttr.get())
                    parameters([iattr, unit], ["val1", "val2"])
                op1 = operation_("op1")
                with InsertionPoint(op1.body):
                    t1 = base(base_ref=["irdl_type_test", "type1"])
                    results_([t1], ["res"], [Variadicity.single])

        # CHECK: module {
        # CHECK:   irdl.dialect @irdl_type_test {
        # CHECK:     irdl.type @type1 {
        # CHECK:       %0 = irdl.base "#builtin.integer"
        # CHECK:       irdl.parameters(val: %0)
        # CHECK:     }
        # CHECK:     irdl.type @type2 {
        # CHECK:       %0 = irdl.base "#builtin.integer"
        # CHECK:       %1 = irdl.is unit
        # CHECK:       irdl.parameters(val1: %0, val2: %1)
        # CHECK:     }
        # CHECK:     irdl.operation @op1 {
        # CHECK:       %0 = irdl.base @irdl_type_test::@type1
        # CHECK:       irdl.results(res: %0)
        # CHECK:     }
        # CHECK:   }
        # CHECK: }
        module.operation.verify()
        module.dump()

        load_dialects(module)

        i32 = IntegerType.get(32)
        t1 = DynamicType.get("irdl_type_test.type1", [IntegerAttr.get(i32, 42)])
        # CHECK: !irdl_type_test.type1<42 : i32>
        t1.dump()
        # CHECK: irdl_type_test.type1
        print(t1.type_name, file=sys.stderr)
        # CHECK: 1
        print(len(t1.params), file=sys.stderr)
        # CHECK: 42 : i32
        t1.params[0].dump()
        t2 = DynamicType.get(
            "irdl_type_test.type2", [IntegerAttr.get(i32, 33), UnitAttr.get()]
        )
        # CHECK: !irdl_type_test.type2<33 : i32, unit>
        t2.dump()
        # CHECK: irdl_type_test.type2
        print(t2.type_name, file=sys.stderr)
        # CHECK: 2
        print(len(t2.params), file=sys.stderr)
        # CHECK: 33 : i32
        t2.params[0].dump()
        # CHECK: unit
        t2.params[1].dump()

        m = Module.create()
        with InsertionPoint(m.body):
            Operation.create("irdl_type_test.op1", results=[t1])

        # CHECK: %0 = "irdl_type_test.op1"() : () -> !irdl_type_test.type1<42 : i32>
        m.dump()
