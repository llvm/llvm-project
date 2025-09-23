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
# i.e. add(constant0, constant1) -> constant2
# where constant2 = constant0 + constant1.
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

    def add_fold(rewriter, results, values):
        a0, a1 = values
        results.append(IntegerAttr.get(i32, a0.value + a1.value))

    pdl_module = PDLModule(m)
    pdl_module.register_rewrite_function("add_fold", add_fold)
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
