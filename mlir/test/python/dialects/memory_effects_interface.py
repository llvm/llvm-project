# RUN: %PYTHON %s | FileCheck %s

from mlir import ir
from mlir.dialects import ext


class TestMemEffectsDialect(ext.Dialect, name="test_mem_effects"):
    pass


class OptionalEffectsOp(TestMemEffectsDialect.Operation, name="optional_effects"):
    pass


def run(f):
    print("\nTEST:", f.__name__)
    f()
    return f


class OptionalMemoryEffectsModel(ir.MemoryEffectsOpInterface):
    """MemoryEffectsOpInterface that is only active when 'effects' attr is present."""

    @staticmethod
    def has_known_memory_effects(op):
        return "effects" in op.attributes

    @staticmethod
    def get_effects(op, effects):
        pass


def implements_memory_effects(op):
    try:
        ir.MemoryEffectsOpInterface(op)
        return True
    except ValueError:
        return False


# CHECK-LABEL: TEST: test_known_no_effects
@run
def test_known_no_effects():
    """When 'effects' attribute is present, hasKnownMemoryEffects returns true
    and the interface is active."""
    with ir.Context() as ctx, ir.Location.unknown():
        TestMemEffectsDialect.load(reload=True)
        OptionalMemoryEffectsModel.attach(
            OptionalEffectsOp.OPERATION_NAME, context=ctx
        )

        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            op = ir.Operation.create("test_mem_effects.optional_effects")
            op.attributes["effects"] = ir.UnitAttr.get()

        # CHECK: implements MemoryEffectsOpInterface: True
        print(f"implements MemoryEffectsOpInterface: {implements_memory_effects(op)}")


# CHECK-LABEL: TEST: test_unknown_effects
@run
def test_unknown_effects():
    """When 'effects' attribute is absent, hasKnownMemoryEffects returns false
    and the interface is not active (the op pretends not to implement it)."""
    with ir.Context() as ctx, ir.Location.unknown():
        TestMemEffectsDialect.load(reload=True)
        OptionalMemoryEffectsModel.attach(
            OptionalEffectsOp.OPERATION_NAME, context=ctx
        )

        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            op = ir.Operation.create("test_mem_effects.optional_effects")

        # CHECK: implements MemoryEffectsOpInterface: False
        print(f"implements MemoryEffectsOpInterface: {implements_memory_effects(op)}")
