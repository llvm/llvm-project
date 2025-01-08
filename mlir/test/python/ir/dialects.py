# RUN: %PYTHON %s | FileCheck %s

import gc
import sys
from mlir.ir import *
from mlir.dialects._ods_common import _cext


def run(f):
    print("\nTEST:", f.__name__)
    f()
    gc.collect()
    assert Context._get_live_count() == 0
    return f


# CHECK-LABEL: TEST: testDialectDescriptor
@run
def testDialectDescriptor():
    ctx = Context()
    d = ctx.get_dialect_descriptor("func")
    # CHECK: <DialectDescriptor func>
    print(d)
    # CHECK: func
    print(d.namespace)
    try:
        _ = ctx.get_dialect_descriptor("not_existing")
    except ValueError:
        pass
    else:
        assert False, "Expected exception"


# CHECK-LABEL: TEST: testUserDialectClass
@run
def testUserDialectClass():
    ctx = Context()
    # Access using attribute.
    d = ctx.dialects.func
    # CHECK: <Dialect func (class mlir.dialects._func_ops_gen._Dialect)>
    print(d)
    try:
        _ = ctx.dialects.not_existing
    except AttributeError:
        pass
    else:
        assert False, "Expected exception"

    # Access using index.
    d = ctx.dialects["func"]
    # CHECK: <Dialect func (class mlir.dialects._func_ops_gen._Dialect)>
    print(d)
    try:
        _ = ctx.dialects["not_existing"]
    except IndexError:
        pass
    else:
        assert False, "Expected exception"

    # Using the 'd' alias.
    d = ctx.d["func"]
    # CHECK: <Dialect func (class mlir.dialects._func_ops_gen._Dialect)>
    print(d)


# CHECK-LABEL: TEST: testCustomOpView
# This test uses the standard dialect AddFOp as an example of a user op.
# TODO: Op creation and access is still quite verbose: simplify this test as
# additional capabilities come online.
@run
def testCustomOpView():
    def createInput():
        op = Operation.create("pytest_dummy.intinput", results=[f32])
        # TODO: Auto result cast from operation
        return op.results[0]

    with Context() as ctx, Location.unknown():
        ctx.allow_unregistered_dialects = True
        m = Module.create()

        with InsertionPoint(m.body):
            f32 = F32Type.get()
            # Create via dialects context collection.
            input1 = createInput()
            input2 = createInput()
            op1 = ctx.dialects.arith.AddFOp(input1, input2)

            # Create via an import
            from mlir.dialects.arith import AddFOp

            AddFOp(input1, op1.result)

    # CHECK: %[[INPUT0:.*]] = "pytest_dummy.intinput"
    # CHECK: %[[INPUT1:.*]] = "pytest_dummy.intinput"
    # CHECK: %[[R0:.*]] = arith.addf %[[INPUT0]], %[[INPUT1]] : f32
    # CHECK: %[[R1:.*]] = arith.addf %[[INPUT0]], %[[R0]] : f32
    m.operation.print()


# CHECK-LABEL: TEST: testIsRegisteredOperation
@run
def testIsRegisteredOperation():
    ctx = Context()

    # CHECK: cf.cond_br: True
    print(f"cf.cond_br: {ctx.is_registered_operation('cf.cond_br')}")
    # CHECK: func.not_existing: False
    print(f"func.not_existing: {ctx.is_registered_operation('func.not_existing')}")


# CHECK-LABEL: TEST: testAppendPrefixSearchPath
@run
def testAppendPrefixSearchPath():
    ctx = Context()
    ctx.allow_unregistered_dialects = True
    with Location.unknown(ctx):
        assert not _cext.globals._check_dialect_module_loaded("custom")
        Operation.create("custom.op")
        assert not _cext.globals._check_dialect_module_loaded("custom")

        sys.path.append(".")
        _cext.globals.append_dialect_search_prefix("custom_dialect")
        assert _cext.globals._check_dialect_module_loaded("custom")


# CHECK-LABEL: TEST: testDialectLoadOnCreate
@run
def testDialectLoadOnCreate():
    with Context(load_on_create_dialects=[]) as ctx:
        ctx.emit_error_diagnostics = True
        ctx.allow_unregistered_dialects = True

        def callback(d):
            # CHECK: DIAGNOSTIC
            # CHECK-SAME: op created with unregistered dialect
            print(f"DIAGNOSTIC={d.message}")
            return True

        handler = ctx.attach_diagnostic_handler(callback)
        loc = Location.unknown(ctx)
        try:
            op = Operation.create("arith.addi", loc=loc)
            ctx.allow_unregistered_dialects = False
            op.verify()
        except MLIRError as e:
            pass

    with Context(load_on_create_dialects=["func"]) as ctx:
        loc = Location.unknown(ctx)
        fn = Operation.create("func.func", loc=loc)

    # TODO: This may require an update if a site wide policy is set.
    # CHECK: Load on create: []
    print(f"Load on create: {get_load_on_create_dialects()}")
    append_load_on_create_dialect("func")
    # CHECK: Load on create:
    # CHECK-SAME: func
    print(f"Load on create: {get_load_on_create_dialects()}")
    print(get_load_on_create_dialects())
