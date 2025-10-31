# RUN: %PYTHON %s 2>&1 | FileCheck %s

from mlir.dialects import arith, func, pdl
from mlir.dialects.builtin import module
from mlir.ir import *
from mlir.rewrite import *


def construct_and_print_in_module(f):
    print("\nTEST:", f.__name__)
    with Context(), Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            module = f(module)
        if module is not None:
            print(module)
    return f


def get_pdl_patterns():
    # Create a rewrite from add to mul. This will match
    # - operation name is arith.addi
    # - operands are index types.
    # - there are two operands.
    with Location.unknown():
        m = Module.create()
        with InsertionPoint(m.body):
            # Change all arith.addi with index types to arith.muli.
            @pdl.pattern(benefit=1, sym_name="addi_to_mul")
            def pat():
                # Match arith.addi with index types.
                index_type = pdl.TypeOp(IndexType.get())
                operand0 = pdl.OperandOp(index_type)
                operand1 = pdl.OperandOp(index_type)
                op0 = pdl.OperationOp(
                    name="arith.addi", args=[operand0, operand1], types=[index_type]
                )

                # Replace the matched op with arith.muli.
                @pdl.rewrite()
                def rew():
                    newOp = pdl.OperationOp(
                        name="arith.muli", args=[operand0, operand1], types=[index_type]
                    )
                    pdl.ReplaceOp(op0, with_op=newOp)

    # Create a PDL module from module and freeze it. At this point the ownership
    # of the module is transferred to the PDL module. This ownership transfer is
    # not yet captured Python side/has sharp edges. So best to construct the
    # module and PDL module in same scope.
    # FIXME: This should be made more robust.
    return PDLModule(m).freeze()


# CHECK-LABEL: TEST: test_add_to_mul
# CHECK: arith.muli
@construct_and_print_in_module
def test_add_to_mul(module_):
    index_type = IndexType.get()

    # Create a test case.
    @module(sym_name="ir")
    def ir():
        @func.func(index_type, index_type)
        def add_func(a, b):
            return arith.addi(a, b)

    frozen = get_pdl_patterns()
    # Could apply frozen pattern set multiple times.
    apply_patterns_and_fold_greedily(module_, frozen)
    return module_


# CHECK-LABEL: TEST: test_add_to_mul_with_op
# CHECK: arith.muli
@construct_and_print_in_module
def test_add_to_mul_with_op(module_):
    index_type = IndexType.get()

    # Create a test case.
    @module(sym_name="ir")
    def ir():
        @func.func(index_type, index_type)
        def add_func(a, b):
            return arith.addi(a, b)

    frozen = get_pdl_patterns()
    apply_patterns_and_fold_greedily(module_.operation, frozen)
    return module_


# If we use arith.constant and arith.addi here,
# these C++-defined folding/canonicalization will be applied
# implicitly in the greedy pattern rewrite driver to
# make our Python-defined folding useless,
# so here we define a new dialect to workaround this.
def load_myint_dialect():
    from mlir.dialects import irdl

    m = Module.create()
    with InsertionPoint(m.body):
        myint = irdl.dialect("myint")
        with InsertionPoint(myint.body):
            constant = irdl.operation_("constant")
            with InsertionPoint(constant.body):
                iattr = irdl.base(base_name="#builtin.integer")
                i32 = irdl.is_(TypeAttr.get(IntegerType.get_signless(32)))
                irdl.attributes_([iattr], ["value"])
                irdl.results_([i32], ["cst"], [irdl.Variadicity.single])
            add = irdl.operation_("add")
            with InsertionPoint(add.body):
                i32 = irdl.is_(TypeAttr.get(IntegerType.get_signless(32)))
                irdl.operands_(
                    [i32, i32],
                    ["lhs", "rhs"],
                    [irdl.Variadicity.single, irdl.Variadicity.single],
                )
                irdl.results_([i32], ["res"], [irdl.Variadicity.single])

    m.operation.verify()
    irdl.load_dialects(m)


# This PDL pattern is to fold constant additions,
# including two patterns:
# 1. add(constant0, constant1) -> constant2
#    where constant2 = constant0 + constant1;
# 2. add(x, 0) or add(0, x) -> x.
def get_pdl_pattern_fold():
    m = Module.create()
    i32 = IntegerType.get_signless(32)
    with InsertionPoint(m.body):

        @pdl.pattern(benefit=1, sym_name="myint_add_fold")
        def pat():
            t = pdl.TypeOp(i32)
            a0 = pdl.AttributeOp()
            a1 = pdl.AttributeOp()
            c0 = pdl.OperationOp(
                name="myint.constant", attributes={"value": a0}, types=[t]
            )
            c1 = pdl.OperationOp(
                name="myint.constant", attributes={"value": a1}, types=[t]
            )
            v0 = pdl.ResultOp(c0, 0)
            v1 = pdl.ResultOp(c1, 0)
            op0 = pdl.OperationOp(name="myint.add", args=[v0, v1], types=[t])

            @pdl.rewrite()
            def rew():
                sum = pdl.apply_native_rewrite(
                    [pdl.AttributeType.get()], "add_fold", [a0, a1]
                )
                newOp = pdl.OperationOp(
                    name="myint.constant", attributes={"value": sum}, types=[t]
                )
                pdl.ReplaceOp(op0, with_op=newOp)

        @pdl.pattern(benefit=1, sym_name="myint_add_zero_fold")
        def pat():
            t = pdl.TypeOp(i32)
            v0 = pdl.OperandOp()
            v1 = pdl.OperandOp()
            v = pdl.apply_native_constraint([pdl.ValueType.get()], "has_zero", [v0, v1])
            op0 = pdl.OperationOp(name="myint.add", args=[v0, v1], types=[t])

            @pdl.rewrite()
            def rew():
                pdl.ReplaceOp(op0, with_values=[v])

    def add_fold(rewriter, results, values):
        a0, a1 = values
        results.append(IntegerAttr.get(i32, a0.value + a1.value))

    def is_zero(value):
        op = value.owner
        if isinstance(op, Operation):
            return op.name == "myint.constant" and op.attributes["value"].value == 0
        return False

    # Check if either operand is a constant zero,
    # and append the other operand to the results if so.
    def has_zero(rewriter, results, values):
        v0, v1 = values
        if is_zero(v0):
            results.append(v1)
            return False
        if is_zero(v1):
            results.append(v0)
            return False
        return True

    pdl_module = PDLModule(m)
    pdl_module.register_rewrite_function("add_fold", add_fold)
    pdl_module.register_constraint_function("has_zero", has_zero)
    return pdl_module.freeze()


# CHECK-LABEL: TEST: test_pdl_register_function
# CHECK: "myint.constant"() {value = 8 : i32} : () -> i32
@construct_and_print_in_module
def test_pdl_register_function(module_):
    load_myint_dialect()

    module_ = Module.parse(
        """
        %c0 = "myint.constant"() { value = 2 }: () -> (i32)
        %c1 = "myint.constant"() { value = 3 }: () -> (i32)
        %x = "myint.add"(%c0, %c1): (i32, i32) -> (i32)
        "myint.add"(%x, %c1): (i32, i32) -> (i32)
        """
    )

    frozen = get_pdl_pattern_fold()
    apply_patterns_and_fold_greedily(module_, frozen)

    return module_


# CHECK-LABEL: TEST: test_pdl_register_function_constraint
# CHECK: return %arg0 : i32
@construct_and_print_in_module
def test_pdl_register_function_constraint(module_):
    load_myint_dialect()

    module_ = Module.parse(
        """
        func.func @f(%x : i32) -> i32 {
            %c0 = "myint.constant"() { value = 1 }: () -> (i32)
            %c1 = "myint.constant"() { value = -1 }: () -> (i32)
            %a = "myint.add"(%c0, %c1): (i32, i32) -> (i32)
            %b = "myint.add"(%a, %x): (i32, i32) -> (i32)
            %c = "myint.add"(%b, %a): (i32, i32) -> (i32)
            func.return %c : i32
        }
        """
    )

    frozen = get_pdl_pattern_fold()
    apply_patterns_and_fold_greedily(module_, frozen)

    return module_


# This pattern is to expand constant to additions
# unless the constant is no more than 1,
# e.g. 3 -> 1 + 2 -> 1 + (1 + 1).
def get_pdl_pattern_expand():
    m = Module.create()
    i32 = IntegerType.get_signless(32)
    with InsertionPoint(m.body):

        @pdl.pattern(benefit=1, sym_name="myint_constant_expand")
        def pat():
            t = pdl.TypeOp(i32)
            cst = pdl.AttributeOp()
            pdl.apply_native_constraint([], "is_one", [cst])
            op0 = pdl.OperationOp(
                name="myint.constant", attributes={"value": cst}, types=[t]
            )

            @pdl.rewrite()
            def rew():
                expanded = pdl.apply_native_rewrite(
                    [pdl.OperationType.get()], "expand", [cst]
                )
                pdl.ReplaceOp(op0, with_op=expanded)

    def is_one(rewriter, results, values):
        cst = values[0].value
        return cst <= 1

    def expand(rewriter, results, values):
        cst = values[0].value
        c1 = cst // 2
        c2 = cst - c1
        with rewriter.ip:
            op1 = Operation.create(
                "myint.constant",
                results=[i32],
                attributes={"value": IntegerAttr.get(i32, c1)},
            )
            op2 = Operation.create(
                "myint.constant",
                results=[i32],
                attributes={"value": IntegerAttr.get(i32, c2)},
            )
            res = Operation.create(
                "myint.add", results=[i32], operands=[op1.result, op2.result]
            )
        results.append(res)

    pdl_module = PDLModule(m)
    pdl_module.register_constraint_function("is_one", is_one)
    pdl_module.register_rewrite_function("expand", expand)
    return pdl_module.freeze()


# CHECK-LABEL: TEST: test_pdl_register_function_expand
# CHECK: %0 = "myint.constant"() {value = 1 : i32} : () -> i32
# CHECK: %1 = "myint.constant"() {value = 1 : i32} : () -> i32
# CHECK: %2 = "myint.add"(%0, %1) : (i32, i32) -> i32
# CHECK: %3 = "myint.constant"() {value = 1 : i32} : () -> i32
# CHECK: %4 = "myint.constant"() {value = 1 : i32} : () -> i32
# CHECK: %5 = "myint.constant"() {value = 1 : i32} : () -> i32
# CHECK: %6 = "myint.add"(%4, %5) : (i32, i32) -> i32
# CHECK: %7 = "myint.add"(%3, %6) : (i32, i32) -> i32
# CHECK: %8 = "myint.add"(%2, %7) : (i32, i32) -> i32
# CHECK: return %8 : i32
@construct_and_print_in_module
def test_pdl_register_function_expand(module_):
    load_myint_dialect()

    module_ = Module.parse(
        """
        func.func @f() -> i32 {
          %0 = "myint.constant"() { value = 5 }: () -> (i32)
          return %0 : i32
        }
        """
    )

    frozen = get_pdl_pattern_expand()
    apply_patterns_and_fold_greedily(module_, frozen)

    return module_
