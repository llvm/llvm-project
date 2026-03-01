# RUN: %PYTHON %s 2>&1 | FileCheck %s
# REQUIRES: host-supports-jit

from mlir.ir import *
from mlir.dialects.ext import *
from mlir.rewrite import *
from mlir.passmanager import *
from mlir.execution_engine import *
from mlir.dialects import llvm, scf, func
from functools import partial


class BfDialect(Dialect, name="bf"):
    pass


class PtrType(BfDialect.Type, name="ptr"):
    pass


class NextOp(BfDialect.Operation, name="next"):
    in_: Operand[PtrType]
    out: Result[PtrType[()]]


class PrevOp(BfDialect.Operation, name="prev"):
    in_: Operand[PtrType]
    out: Result[PtrType[()]]


class IncOp(BfDialect.Operation, name="inc"):
    in_: Operand[PtrType]


class DecOp(BfDialect.Operation, name="dec"):
    in_: Operand[PtrType]


class InputOp(BfDialect.Operation, name="input"):
    in_: Operand[PtrType]


class OutputOp(BfDialect.Operation, name="output"):
    in_: Operand[PtrType]


class WhileOp(BfDialect.Operation, name="while"):
    in_: Operand[PtrType]
    out: Result[PtrType[()]]
    body: Region


class YieldOp(BfDialect.Operation, name="yield", traits=[IsTerminatorTrait]):
    in_: Operand[PtrType]


class MainOp(BfDialect.Operation, name="main"):
    body: Region


def parse(code: str):
    module = Module.create()

    with InsertionPoint(module.body):
        main = MainOp()
        main.body.blocks.append()
        current_val = main.body.blocks[0].add_argument(
            PtrType.get(), Location.unknown()
        )

        ip = InsertionPoint(main.body.blocks[0])
        for c in code:
            with ip:
                if c == ">":
                    current_val = NextOp(current_val).out
                elif c == "<":
                    current_val = PrevOp(current_val).out
                elif c == "+":
                    IncOp(current_val)
                elif c == "-":
                    DecOp(current_val)
                elif c == ".":
                    OutputOp(current_val)
                elif c == ",":
                    InputOp(current_val)
                elif c == "[":
                    loop = WhileOp(current_val)
                    loop.body.blocks.append()
                    current_val = loop.body.blocks[0].add_argument(
                        PtrType.get(), Location.unknown()
                    )
                    ip = InsertionPoint(loop.body.blocks[0])
                elif c == "]":
                    YieldOp(current_val)
                    current_val = ip.block.owner.opview.out
                    ip = InsertionPoint.after(current_val.owner)

        with ip:
            YieldOp(current_val)

    return module


def convert_bf_to_llvm(op, pass_):
    patterns = RewritePatternSet()
    ptr = llvm.PointerType.get()
    i8 = IntegerType.get_signless(8)
    i32 = IntegerType.get_signless(32)

    type_converter = TypeConverter()

    def convert_ptr(t):
        return ptr if isinstance(t, PtrType) else None

    type_converter.add_conversion(convert_ptr)

    def convert_next(op, adaptor, converter, rewriter, offset=1):
        with rewriter.ip:
            gep = llvm.GEPOp(ptr, adaptor.in_, [], [offset], i8, [])
        rewriter.replace_op(op, gep)

    def convert_inc(op, adaptor, converter, rewriter, cst=1):
        with rewriter.ip:
            load = llvm.load(i8, adaptor.in_)
            one = llvm.mlir_constant(IntegerAttr.get(i8, cst))
            added = llvm.add(load, one, [])
            store = llvm.StoreOp(added, adaptor.in_)
        rewriter.replace_op(op, store)

    def convert_main(op, adaptor, converter, rewriter):
        with rewriter.ip:
            fn = func.FuncOp("bf_main", FunctionType.get([ptr], [ptr]))
            op.body.blocks[0].append_to(fn.body)
            rewriter.convert_region_types(fn.body, converter)
        rewriter.replace_op(op, fn)

    def convert_yield(op, adaptor, converter, rewriter):
        with rewriter.ip:
            if isinstance(op.parent.opview, WhileOp | scf.WhileOp):
                yield_ = scf.YieldOp([adaptor.in_])
            else:
                yield_ = func.ReturnOp([adaptor.in_])
        rewriter.replace_op(op, yield_)

    def convert_while(op, adaptor, converter, rewriter):
        with rewriter.ip:
            loop = scf.WhileOp([ptr], [adaptor.in_])
            loop.before.blocks.append()
            arg = loop.before.blocks[0].add_argument(ptr, Location.unknown())
            with InsertionPoint(loop.before.blocks[0]):
                c = llvm.load(i8, arg)
                zero = llvm.mlir_constant(IntegerAttr.get(i8, 0))
                cond = llvm.icmp(llvm.ICmpPredicate.ne, c, zero)
                scf.ConditionOp(cond, [arg])
            op.body.blocks[0].append_to(loop.after)
            rewriter.convert_region_types(loop.after, converter)
        rewriter.replace_op(op, loop)

    def convert_output(op, adaptor, converter, rewriter):
        with rewriter.ip:
            val = llvm.load(i8, adaptor.in_)
            call = func.CallOp([], "bf_output", [val])
        rewriter.replace_op(op, call)

    def convert_input(op, adaptor, converter, rewriter):
        with rewriter.ip:
            call = func.call([i8], "bf_input", [])
            store = llvm.StoreOp(call, adaptor.in_)
        rewriter.replace_op(op, store)

    patterns.add_conversion(NextOp, convert_next, type_converter)
    patterns.add_conversion(PrevOp, partial(convert_next, offset=-1), type_converter)
    patterns.add_conversion(IncOp, convert_inc, type_converter)
    patterns.add_conversion(DecOp, partial(convert_inc, cst=-1), type_converter)
    patterns.add_conversion(MainOp, convert_main, type_converter)
    patterns.add_conversion(YieldOp, convert_yield, type_converter)
    patterns.add_conversion(WhileOp, convert_while, type_converter)
    patterns.add_conversion(OutputOp, convert_output, type_converter)
    patterns.add_conversion(InputOp, convert_input, type_converter)

    target = ConversionTarget()
    target.add_illegal_dialect(BfDialect)

    apply_partial_conversion(op, target, patterns.freeze())

    with InsertionPoint(op.opview.body):
        func.FuncOp("putchar", FunctionType.get([i32], [i32]), visibility="private")
        func.FuncOp("getchar", FunctionType.get([], [i32]), visibility="private")

        output = func.FuncOp("bf_output", FunctionType.get([i8], []))
        output.body.blocks.append()
        arg = output.body.blocks[0].add_argument(i8, Location.unknown())
        with InsertionPoint(output.body.blocks[0]):
            sext = llvm.sext(i32, arg)
            func.call([i32], "putchar", [sext])
            func.ReturnOp([])

        input = func.FuncOp("bf_input", FunctionType.get([], [i8]))
        input.body.blocks.append()
        with InsertionPoint(input.body.blocks[0]):
            call = func.call([i32], "getchar", [])
            trunc = llvm.trunc(i8, call, [])
            func.ReturnOp([trunc])

        init = func.FuncOp("bf_init", FunctionType.get([], []))
        init.attributes["llvm.emit_c_interface"] = UnitAttr.get()
        init.body.blocks.append()
        with InsertionPoint(init.body.blocks[0]):
            c1024 = llvm.mlir_constant(IntegerAttr.get(i32, 1024))
            zero = llvm.mlir_constant(IntegerAttr.get(i8, 0))
            p = llvm.alloca(ptr, c1024, i8)
            llvm.intr_memset(p, zero, c1024, False)
            func.call([ptr], "bf_main", [p])
            func.ReturnOp([])


def execute(code):
    module = parse(code)
    assert module.operation.verify()

    pm = PassManager()
    pm.add(convert_bf_to_llvm)
    pm.add("convert-scf-to-cf, convert-to-llvm")

    pm.run(module.operation)

    ee = ExecutionEngine(module)
    ee.lookup("bf_init")(0)


def run(f):
    print("TEST:", f.__name__)
    f()


with Context(), Location.unknown():
    BfDialect.load()

    # CHECK: TEST: test_convert_bf_to_llvm
    @run
    def test_convert_bf_to_llvm():
        module = parse("[-]")
        assert module.operation.verify()

        # CHECK: "bf.main"() ({
        # CHECK: ^bb0(%arg0: !bf.ptr):
        # CHECK:   %0 = "bf.while"(%arg0) ({
        # CHECK:   ^bb0(%arg1: !bf.ptr):
        # CHECK:     "bf.dec"(%arg1) : (!bf.ptr) -> ()
        # CHECK:     "bf.yield"(%arg1) : (!bf.ptr) -> ()
        # CHECK:   }) : (!bf.ptr) -> !bf.ptr
        # CHECK:   "bf.yield"(%0) : (!bf.ptr) -> ()
        # CHECK: }) : () -> ()
        print(module)

        pm = PassManager()
        pm.add(convert_bf_to_llvm)
        pm.run(module.operation)

        # CHECK: func.func @bf_main(%arg0: !llvm.ptr) -> !llvm.ptr {
        # CHECK:   %0 = scf.while (%arg1 = %arg0) : (!llvm.ptr) -> !llvm.ptr {
        # CHECK:     %1 = llvm.load %arg1 : !llvm.ptr -> i8
        # CHECK:     %2 = llvm.mlir.constant(0 : i8) : i8
        # CHECK:     %3 = llvm.icmp "ne" %1, %2 : i8
        # CHECK:     scf.condition(%3) %arg1 : !llvm.ptr
        # CHECK:   } do {
        # CHECK:   ^bb0(%arg1: !llvm.ptr):
        # CHECK:     %1 = llvm.load %arg1 : !llvm.ptr -> i8
        # CHECK:     %2 = llvm.mlir.constant(-1 : i8) : i8
        # CHECK:     %3 = llvm.add %1, %2 : i8
        # CHECK:     llvm.store %3, %arg1 : i8, !llvm.ptr
        # CHECK:     scf.yield %arg1 : !llvm.ptr
        # CHECK:   }
        # CHECK:   return %0 : !llvm.ptr
        # CHECK: }
        print(module)

    # CHECK: TEST: test_bf_e2e
    @run
    def test_bf_e2e():
        # CHECK: Hello World!
        execute(
            "++++++++[>++++[>++>+++>+++>+<<<<-]>+>+>->>+[<]<-]>>.>---.+++++++..+++.>>.<-.<.+++.------.--------.>>+.>++."
        )
