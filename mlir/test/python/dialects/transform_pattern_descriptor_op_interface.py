# RUN: env PYTHONUNBUFFERED=1 %PYTHON %s 2>&1 | FileCheck %s

from contextlib import contextmanager

from mlir import ir, rewrite
from mlir.dialects import transform, func, arith, ext
from mlir.dialects.transform import AnyOpType, structured


@ext.register_dialect
class MyPatternDescriptors(ext.Dialect, name="my_pattern_descriptors"):
    pass


def run(emit_schedule):
    print(f"Test: {emit_schedule.__name__}")
    with ir.Context(), ir.Location.unknown():
        payload = emit_payload()

        MyPatternDescriptors.load(register=False, reload=True)

        # NB: Pattern descriptor ops have their interfaces attached
        #     in their respective test functions.
        schedule = emit_schedule()

        (_named_seq := schedule.body.operations[0]).apply(payload)

        print(payload)


# Payload used by all tests.
def emit_payload():
    payload_module = ir.Module.create()
    with ir.InsertionPoint(payload_module.body):
        i32 = ir.IntegerType.get_signless(32)

        @func.FuncOp.from_py_func(i32, i32)
        def test_func(a, b):
            c = arith.addi(a, b)
            d = arith.subi(c, b)
            return d

    return payload_module


@contextmanager
def schedule_boilerplate():
    schedule = ir.Module.create()
    schedule.operation.attributes["transform.with_named_sequence"] = ir.UnitAttr.get()
    with ir.InsertionPoint(schedule.body):
        named_sequence = transform.NamedSequenceOp(
            "__transform_main",
            [AnyOpType.get()],
            [AnyOpType.get()],
            arg_attrs=[{"transform.consumed": ir.UnitAttr.get()}],
        )
        with ir.InsertionPoint(named_sequence.body):
            yield schedule, named_sequence


@ext.register_operation(MyPatternDescriptors)
class SubiAddiRewritePatternOp(MyPatternDescriptors.Operation, name="add_pattern"):
    @classmethod
    def attach_interface_impls(cls, ctx=None):
        cls.PatternDescriptorOpInterfaceFallbackModel.attach(
            cls.OPERATION_NAME, context=ctx
        )

    class PatternDescriptorOpInterfaceFallbackModel(
        transform.PatternDescriptorOpInterface
    ):
        @staticmethod
        def populate_patterns(
            op: "SubiAddiRewritePatternOp",
            patterns: rewrite.RewritePatternSet,
        ) -> None:
            # Define a pattern that rewrites subi(addi(a, b), b) -> a
            def match_and_rewrite(subi, rewriter):
                if not isinstance(addi := subi.lhs.owner, arith.AddiOp):
                    return True  # Failed match, return truthy value
                if subi.rhs != addi.rhs:
                    return True
                # Replace subi's result with addi's lhs
                rewriter.replace_op(subi, [addi.lhs])
                return None  # Success

            # Add the pattern to the pattern set.
            patterns.add("arith.subi", match_and_rewrite, benefit=1)


# CHECK-LABEL: Test: test_pattern_descriptor_add_pattern
@run
def test_pattern_descriptor_add_pattern():
    """Tests python-defined rewrite pattern via PatternDescriptorOpInterface on AddPatternOp"""

    SubiAddiRewritePatternOp.attach_interface_impls()

    with schedule_boilerplate() as (schedule, named_seq):
        func_handle = structured.MatchOp.match_op_names(
            named_seq.bodyTarget, ["func.func"]
        ).result

        # After pattern application, check that subi is removed and func returns
        # the first argument directly:
        # CHECK: func.func @test_func(%[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32)
        # CHECK: return %[[ARG0]] : i32
        apply_patterns_op = transform.ApplyPatternsOp(func_handle)
        with ir.InsertionPoint(apply_patterns_op.patterns):
            SubiAddiRewritePatternOp()

        transform.yield_([func_handle])
        named_seq.verify()

    return schedule
