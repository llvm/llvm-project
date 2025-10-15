# RUN: python %s | FileCheck %s
from mlir.ir import (
    Context,
    FunctionType,
    Location,
    Module,
    InsertionPoint,
    IntegerType,
    IndexType,
    MemRefType,
    F32Type,
    Block,
    ArrayAttr,
    Attribute,
    UnitAttr,
    StringAttr,
    DenseI32ArrayAttr,
    ShapedType,
)
from mlir.dialects import openacc, func, arith, memref


def run(f):
    print("\n// TEST:", f.__name__)
    with Context(), Location.unknown():
        f()
    return f


@run
def testManualReconstructedKernel():
    module = Module.create()

    i32 = IntegerType.get_signless(32)
    i64 = IntegerType.get_signless(64)
    f32 = F32Type.get()
    dynamic = ShapedType.get_dynamic_size()
    memref_f32_1d_any = MemRefType.get([dynamic], f32)

    with InsertionPoint(module.body):
        function_type = FunctionType.get(
            [memref_f32_1d_any, memref_f32_1d_any, i64], []
        )
        f = func.FuncOp(
            type=function_type,
            name="memcpy_idiom",
        )
        f.attributes["sym_visibility"] = StringAttr.get("public")

    with InsertionPoint(f.add_entry_block()):
        c1024 = arith.ConstantOp(i32, 1024)
        c128 = arith.ConstantOp(i32, 128)

        parallel_op = openacc.ParallelOp(
            asyncOperands=[],
            waitOperands=[],
            numGangs=[c1024],
            numWorkers=[],
            vectorLength=[c128],
            reductionOperands=[],
            privateOperands=[],
            firstprivateOperands=[],
            dataClauseOperands=[],
        )

        # Set required device_type and segment attributes to satisfy verifier
        acc_device_none = ArrayAttr.get([Attribute.parse("#acc.device_type<none>")])
        parallel_op.numGangsDeviceType = acc_device_none
        parallel_op.numGangsSegments = DenseI32ArrayAttr.get([1])
        parallel_op.vectorLengthDeviceType = acc_device_none

        parallel_block = Block.create_at_start(parent=parallel_op.region, arg_types=[])

        with InsertionPoint(parallel_block):
            c0 = arith.ConstantOp(i64, 0)
            c1 = arith.ConstantOp(i64, 1)

            loop_op = openacc.LoopOp(
                results_=[],
                lowerbound=[c0],
                upperbound=[f.arguments[2]],
                step=[c1],
                gangOperands=[],
                workerNumOperands=[],
                vectorOperands=[],
                tileOperands=[],
                cacheOperands=[],
                privateOperands=[],
                reductionOperands=[],
                firstprivateOperands=[],
            )

            # Set loop attributes: gang and independent on device_type<none>
            acc_device_none = ArrayAttr.get([Attribute.parse("#acc.device_type<none>")])
            loop_op.gang = acc_device_none
            loop_op.independent = acc_device_none

            loop_block = Block.create_at_start(parent=loop_op.region, arg_types=[i64])

            with InsertionPoint(loop_block):
                idx = arith.index_cast(out=IndexType.get(), in_=loop_block.arguments[0])
                val = memref.load(memref=f.arguments[1], indices=[idx])
                memref.store(value=val, memref=f.arguments[0], indices=[idx])
                openacc.YieldOp([])

            openacc.YieldOp([])

        func.ReturnOp([])

    print(module)

    # CHECK-LABEL:   func.func public @memcpy_idiom(
    # CHECK-SAME:      %[[ARG0:.*]]: memref<?xf32>, %[[ARG1:.*]]: memref<?xf32>, %[[ARG2:.*]]: i64) {
    # CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1024 : i32
    # CHECK:           %[[CONSTANT_1:.*]] = arith.constant 128 : i32
    # CHECK:           acc.parallel num_gangs({%[[CONSTANT_0]] : i32}) vector_length(%[[CONSTANT_1]] : i32) {
    # CHECK:             %[[CONSTANT_2:.*]] = arith.constant 0 : i64
    # CHECK:             %[[CONSTANT_3:.*]] = arith.constant 1 : i64
    # CHECK:             acc.loop gang control(%[[VAL_0:.*]] : i64) = (%[[CONSTANT_2]] : i64) to (%[[ARG2]] : i64)  step (%[[CONSTANT_3]] : i64) {
    # CHECK:               %[[INDEX_CAST_0:.*]] = arith.index_cast %[[VAL_0]] : i64 to index
    # CHECK:               %[[LOAD_0:.*]] = memref.load %[[ARG1]]{{\[}}%[[INDEX_CAST_0]]] : memref<?xf32>
    # CHECK:               memref.store %[[LOAD_0]], %[[ARG0]]{{\[}}%[[INDEX_CAST_0]]] : memref<?xf32>
    # CHECK:               acc.yield
    # CHECK:             } attributes {independent = [#acc.device_type<none>]}
    # CHECK:             acc.yield
    # CHECK:           }
    # CHECK:           return
    # CHECK:         }
