# RUN: %PYTHON %s | FileCheck %s
# This is just a smoke test that the dialect is functional.

from mlir.ir import *
from mlir.dialects import ml_program, arith, builtin


def constructAndPrintInModule(f):
    print("\nTEST:", f.__name__)
    with Context(), Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            f()
        print(module)
    return f


# CHECK-LABEL: testFuncOp
@constructAndPrintInModule
def testFuncOp():
    # CHECK: ml_program.func @foobar(%arg0: si32) -> si32
    f = ml_program.FuncOp(
        name="foobar", type=([IntegerType.get_signed(32)], [IntegerType.get_signed(32)])
    )
    block = f.add_entry_block()
    with InsertionPoint(block):
        # CHECK: ml_program.return
        ml_program.ReturnOp([block.arguments[0]])


# CHECK-LABEL: testGlobalStoreOp
@constructAndPrintInModule
def testGlobalStoreOp():
    # CHECK: %cst = arith.constant 4.242000e+01 : f32
    cst = arith.ConstantOp(value=42.42, result=F32Type.get())

    m = builtin.ModuleOp()
    m.sym_name = StringAttr.get("symbol1")
    m.sym_visibility = StringAttr.get("public")
    # CHECK: module @symbol1 attributes {sym_visibility = "public"} {
    # CHECK:   ml_program.global public mutable @symbol2 : f32
    # CHECK: }
    with InsertionPoint(m.body):
        ml_program.GlobalOp("symbol2", F32Type.get(), is_mutable=True)
    # CHECK: ml_program.global_store @symbol1::@symbol2 = %cst : f32
    ml_program.GlobalStoreOp(["symbol1", "symbol2"], cst)
