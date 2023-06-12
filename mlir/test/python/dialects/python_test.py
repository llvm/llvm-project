# RUN: %PYTHON %s | FileCheck %s

from mlir.ir import *
import mlir.dialects.func as func
import mlir.dialects.python_test as test
import mlir.dialects.tensor as tensor
import mlir.dialects.arith as arith


def run(f):
    print("\nTEST:", f.__name__)
    f()
    return f


# CHECK-LABEL: TEST: testAttributes
@run
def testAttributes():
    with Context() as ctx, Location.unknown():
        ctx.allow_unregistered_dialects = True

        #
        # Check op construction with attributes.
        #

        i32 = IntegerType.get_signless(32)
        one = IntegerAttr.get(i32, 1)
        two = IntegerAttr.get(i32, 2)
        unit = UnitAttr.get()

        # CHECK: "python_test.attributed_op"() {
        # CHECK-DAG: mandatory_i32 = 1 : i32
        # CHECK-DAG: optional_i32 = 2 : i32
        # CHECK-DAG: unit
        # CHECK: }
        op = test.AttributedOp(one, optional_i32=two, unit=unit)
        print(f"{op}")

        # CHECK: "python_test.attributed_op"() {
        # CHECK: mandatory_i32 = 2 : i32
        # CHECK: }
        op2 = test.AttributedOp(two)
        print(f"{op2}")

        #
        # Check generic "attributes" access and mutation.
        #

        assert "additional" not in op.attributes

        # CHECK: "python_test.attributed_op"() {
        # CHECK-DAG: additional = 1 : i32
        # CHECK-DAG: mandatory_i32 = 2 : i32
        # CHECK: }
        op2.attributes["additional"] = one
        print(f"{op2}")

        # CHECK: "python_test.attributed_op"() {
        # CHECK-DAG: additional = 2 : i32
        # CHECK-DAG: mandatory_i32 = 2 : i32
        # CHECK: }
        op2.attributes["additional"] = two
        print(f"{op2}")

        # CHECK: "python_test.attributed_op"() {
        # CHECK-NOT: additional = 2 : i32
        # CHECK:     mandatory_i32 = 2 : i32
        # CHECK: }
        del op2.attributes["additional"]
        print(f"{op2}")

        try:
            print(op.attributes["additional"])
        except KeyError:
            pass
        else:
            assert False, "expected KeyError on unknown attribute key"

        #
        # Check accessors to defined attributes.
        #

        # CHECK: Mandatory: 1
        # CHECK: Optional: 2
        # CHECK: Unit: True
        print(f"Mandatory: {op.mandatory_i32.value}")
        print(f"Optional: {op.optional_i32.value}")
        print(f"Unit: {op.unit}")

        # CHECK: Mandatory: 2
        # CHECK: Optional: None
        # CHECK: Unit: False
        print(f"Mandatory: {op2.mandatory_i32.value}")
        print(f"Optional: {op2.optional_i32}")
        print(f"Unit: {op2.unit}")

        # CHECK: Mandatory: 2
        # CHECK: Optional: None
        # CHECK: Unit: False
        op.mandatory_i32 = two
        op.optional_i32 = None
        op.unit = False
        print(f"Mandatory: {op.mandatory_i32.value}")
        print(f"Optional: {op.optional_i32}")
        print(f"Unit: {op.unit}")
        assert "optional_i32" not in op.attributes
        assert "unit" not in op.attributes

        try:
            op.mandatory_i32 = None
        except ValueError:
            pass
        else:
            assert False, "expected ValueError on setting a mandatory attribute to None"

        # CHECK: Optional: 2
        op.optional_i32 = two
        print(f"Optional: {op.optional_i32.value}")

        # CHECK: Optional: None
        del op.optional_i32
        print(f"Optional: {op.optional_i32}")

        # CHECK: Unit: False
        op.unit = None
        print(f"Unit: {op.unit}")
        assert "unit" not in op.attributes

        # CHECK: Unit: True
        op.unit = True
        print(f"Unit: {op.unit}")

        # CHECK: Unit: False
        del op.unit
        print(f"Unit: {op.unit}")


# CHECK-LABEL: TEST: attrBuilder
@run
def attrBuilder():
    with Context() as ctx, Location.unknown():
        ctx.allow_unregistered_dialects = True
        op = test.AttributesOp(
            x_bool=True,
            x_i16=1,
            x_i32=2,
            x_i64=3,
            x_si16=-1,
            x_si32=-2,
            x_f32=1.5,
            x_f64=2.5,
            x_str="x_str",
            x_i32_array=[1, 2, 3],
            x_i64_array=[4, 5, 6],
            x_f32_array=[1.5, -2.5, 3.5],
            x_f64_array=[4.5, 5.5, -6.5],
            x_i64_dense=[1, 2, 3, 4, 5, 6],
        )
        print(op)


# CHECK-LABEL: TEST: inferReturnTypes
@run
def inferReturnTypes():
    with Context() as ctx, Location.unknown(ctx):
        test.register_python_test_dialect(ctx)
        module = Module.create()
        with InsertionPoint(module.body):
            op = test.InferResultsOp()
            dummy = test.DummyOp()

        # CHECK: [Type(i32), Type(i64)]
        iface = InferTypeOpInterface(op)
        print(iface.inferReturnTypes())

        # CHECK: [Type(i32), Type(i64)]
        iface_static = InferTypeOpInterface(test.InferResultsOp)
        print(iface.inferReturnTypes())

        assert isinstance(iface.opview, test.InferResultsOp)
        assert iface.opview == iface.operation.opview

        try:
            iface_static.opview
        except TypeError:
            pass
        else:
            assert False, (
                "not expected to be able to obtain an opview from a static" " interface"
            )

        try:
            InferTypeOpInterface(dummy)
        except ValueError:
            pass
        else:
            assert False, "not expected dummy op to implement the interface"

        try:
            InferTypeOpInterface(test.DummyOp)
        except ValueError:
            pass
        else:
            assert False, "not expected dummy op class to implement the interface"


# CHECK-LABEL: TEST: resultTypesDefinedByTraits
@run
def resultTypesDefinedByTraits():
    with Context() as ctx, Location.unknown(ctx):
        test.register_python_test_dialect(ctx)
        module = Module.create()
        with InsertionPoint(module.body):
            inferred = test.InferResultsOp()
            same = test.SameOperandAndResultTypeOp([inferred.results[0]])
            # CHECK-COUNT-2: i32
            print(same.one.type)
            print(same.two.type)

            first_type_attr = test.FirstAttrDeriveTypeAttrOp(
                inferred.results[1], TypeAttr.get(IndexType.get())
            )
            # CHECK-COUNT-2: index
            print(first_type_attr.one.type)
            print(first_type_attr.two.type)

            first_attr = test.FirstAttrDeriveAttrOp(FloatAttr.get(F32Type.get(), 3.14))
            # CHECK-COUNT-3: f32
            print(first_attr.one.type)
            print(first_attr.two.type)
            print(first_attr.three.type)

            implied = test.InferResultsImpliedOp()
            # CHECK: i32
            print(implied.integer.type)
            # CHECK: f64
            print(implied.flt.type)
            # CHECK: index
            print(implied.index.type)


# CHECK-LABEL: TEST: testOptionalOperandOp
@run
def testOptionalOperandOp():
    with Context() as ctx, Location.unknown():
        test.register_python_test_dialect(ctx)

        module = Module.create()
        with InsertionPoint(module.body):

            op1 = test.OptionalOperandOp()
            # CHECK: op1.input is None: True
            print(f"op1.input is None: {op1.input is None}")

            op2 = test.OptionalOperandOp(input=op1)
            # CHECK: op2.input is None: False
            print(f"op2.input is None: {op2.input is None}")


# CHECK-LABEL: TEST: testCustomAttribute
@run
def testCustomAttribute():
    with Context() as ctx:
        test.register_python_test_dialect(ctx)
        a = test.TestAttr.get()
        # CHECK: #python_test.test_attr
        print(a)

        # The following cast must not assert.
        b = test.TestAttr(a)

        unit = UnitAttr.get()
        try:
            test.TestAttr(unit)
        except ValueError as e:
            assert "Cannot cast attribute to TestAttr" in str(e)
        else:
            raise

        # The following must trigger a TypeError from our adaptors and must not
        # crash.
        try:
            test.TestAttr(42)
        except TypeError as e:
            assert "Expected an MLIR object" in str(e)
        else:
            raise

        # The following must trigger a TypeError from pybind (therefore, not
        # checking its message) and must not crash.
        try:
            test.TestAttr(42, 56)
        except TypeError:
            pass
        else:
            raise


@run
def testCustomType():
    with Context() as ctx:
        test.register_python_test_dialect(ctx)
        a = test.TestType.get()
        # CHECK: !python_test.test_type
        print(a)

        # The following cast must not assert.
        b = test.TestType(a)
        # Instance custom types should have typeids
        assert isinstance(b.typeid, TypeID)
        # Subclasses of ir.Type should not have a static_typeid
        # CHECK: 'TestType' object has no attribute 'static_typeid'
        try:
            b.static_typeid
        except AttributeError as e:
            print(e)

        i8 = IntegerType.get_signless(8)
        try:
            test.TestType(i8)
        except ValueError as e:
            assert "Cannot cast type to TestType" in str(e)
        else:
            raise

        # The following must trigger a TypeError from our adaptors and must not
        # crash.
        try:
            test.TestType(42)
        except TypeError as e:
            assert "Expected an MLIR object" in str(e)
        else:
            raise

        # The following must trigger a TypeError from pybind (therefore, not
        # checking its message) and must not crash.
        try:
            test.TestType(42, 56)
        except TypeError:
            pass
        else:
            raise


@run
# CHECK-LABEL: TEST: testTensorValue
def testTensorValue():
    with Context() as ctx, Location.unknown():
        test.register_python_test_dialect(ctx)

        i8 = IntegerType.get_signless(8)

        class Tensor(test.TestTensorValue):
            def __str__(self):
                return super().__str__().replace("Value", "Tensor")

        module = Module.create()
        with InsertionPoint(module.body):
            t = tensor.EmptyOp([10, 10], i8).result

            # CHECK: Value(%{{.*}} = tensor.empty() : tensor<10x10xi8>)
            print(Value(t))

            tt = Tensor(t)
            # CHECK: Tensor(%{{.*}} = tensor.empty() : tensor<10x10xi8>)
            print(tt)

            # CHECK: False
            print(tt.is_null())

            # Classes of custom types that inherit from concrete types should have
            # static_typeid
            assert isinstance(test.TestIntegerRankedTensorType.static_typeid, TypeID)
            # And it should be equal to the in-tree concrete type
            assert test.TestIntegerRankedTensorType.static_typeid == t.type.typeid


# CHECK-LABEL: TEST: inferReturnTypeComponents
@run
def inferReturnTypeComponents():
    with Context() as ctx, Location.unknown(ctx):
        test.register_python_test_dialect(ctx)
        module = Module.create()
        i32 = IntegerType.get_signless(32)
        with InsertionPoint(module.body):
            resultType = UnrankedTensorType.get(i32)
            operandTypes = [
                RankedTensorType.get([1, 3, 10, 10], i32),
                UnrankedTensorType.get(i32),
            ]
            f = func.FuncOp(
                "test_inferReturnTypeComponents", (operandTypes, [resultType])
            )
            entry_block = Block.create_at_start(f.operation.regions[0], operandTypes)
            with InsertionPoint(entry_block):
                ranked_op = test.InferShapedTypeComponentsOp(
                    resultType, entry_block.arguments[0]
                )
                unranked_op = test.InferShapedTypeComponentsOp(
                    resultType, entry_block.arguments[1]
                )

        # CHECK: has rank: True
        # CHECK: rank: 4
        # CHECK: element type: i32
        # CHECK: shape: [1, 3, 10, 10]
        iface = InferShapedTypeOpInterface(ranked_op)
        shaped_type_components = iface.inferReturnTypeComponents(
            operands=[ranked_op.operand]
        )[0]
        print("has rank:", shaped_type_components.has_rank)
        print("rank:", shaped_type_components.rank)
        print("element type:", shaped_type_components.element_type)
        print("shape:", shaped_type_components.shape)

        # CHECK: has rank: False
        # CHECK: rank: None
        # CHECK: element type: i32
        # CHECK: shape: None
        iface = InferShapedTypeOpInterface(unranked_op)
        shaped_type_components = iface.inferReturnTypeComponents(
            operands=[unranked_op.operand]
        )[0]
        print("has rank:", shaped_type_components.has_rank)
        print("rank:", shaped_type_components.rank)
        print("element type:", shaped_type_components.element_type)
        print("shape:", shaped_type_components.shape)


# CHECK-LABEL: TEST: testCustomTypeTypeCaster
@run
def testCustomTypeTypeCaster():
    with Context() as ctx, Location.unknown():
        test.register_python_test_dialect(ctx)

        a = test.TestType.get()
        assert a.typeid is not None

        b = Type.parse("!python_test.test_type")
        # CHECK: !python_test.test_type
        print(b)
        # CHECK: TestType(!python_test.test_type)
        print(repr(b))

        c = test.TestIntegerRankedTensorType.get([10, 10], 5)
        # CHECK: tensor<10x10xi5>
        print(c)
        # CHECK: TestIntegerRankedTensorType(tensor<10x10xi5>)
        print(repr(c))

        # CHECK: Type caster is already registered
        try:

            def type_caster(pytype):
                return test.TestIntegerRankedTensorType(pytype)

            register_type_caster(c.typeid, type_caster)
        except RuntimeError as e:
            print(e)

        def type_caster(pytype):
            return test.TestIntegerRankedTensorType(pytype)

        register_type_caster(c.typeid, type_caster, replace=True)

        d = tensor.EmptyOp([10, 10], IntegerType.get_signless(5)).result
        # CHECK: tensor<10x10xi5>
        print(d.type)
        # CHECK: TestIntegerRankedTensorType(tensor<10x10xi5>)
        print(repr(d.type))


# CHECK-LABEL: TEST: testInferTypeOpInterface
@run
def testInferTypeOpInterface():
    with Context() as ctx, Location.unknown(ctx):
        test.register_python_test_dialect(ctx)
        module = Module.create()
        with InsertionPoint(module.body):
            i64 = IntegerType.get_signless(64)
            zero = arith.ConstantOp(i64, 0)

            one_operand = test.InferResultsVariadicInputsOp(single=zero, doubled=None)
            # CHECK: i32
            print(one_operand.result.type)

            two_operands = test.InferResultsVariadicInputsOp(single=zero, doubled=zero)
            # CHECK: f32
            print(two_operands.result.type)
