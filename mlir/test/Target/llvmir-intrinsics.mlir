// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// CHECK-LABEL: @intrinsics
llvm.func @intrinsics(%arg0: !llvm.float, %arg1: !llvm.float, %arg2: !llvm<"<8 x float>">, %arg3: !llvm<"i8*">) {
  %c3 = llvm.mlir.constant(3 : i32) : !llvm.i32
  %c1 = llvm.mlir.constant(1 : i32) : !llvm.i32
  %c0 = llvm.mlir.constant(0 : i32) : !llvm.i32
  // CHECK: call float @llvm.fmuladd.f32
  "llvm.intr.fmuladd"(%arg0, %arg1, %arg0) : (!llvm.float, !llvm.float, !llvm.float) -> !llvm.float
  // CHECK: call <8 x float> @llvm.fmuladd.v8f32
  "llvm.intr.fmuladd"(%arg2, %arg2, %arg2) : (!llvm<"<8 x float>">, !llvm<"<8 x float>">, !llvm<"<8 x float>">) -> !llvm<"<8 x float>">
  // CHECK: call float @llvm.fma.f32
  "llvm.intr.fma"(%arg0, %arg1, %arg0) : (!llvm.float, !llvm.float, !llvm.float) -> !llvm.float
  // CHECK: call <8 x float> @llvm.fma.v8f32
  "llvm.intr.fma"(%arg2, %arg2, %arg2) : (!llvm<"<8 x float>">, !llvm<"<8 x float>">, !llvm<"<8 x float>">) -> !llvm<"<8 x float>">
  // CHECK: call void @llvm.prefetch.p0i8(i8* %3, i32 0, i32 3, i32 1)
  "llvm.intr.prefetch"(%arg3, %c0, %c3, %c1) : (!llvm<"i8*">, !llvm.i32, !llvm.i32, !llvm.i32) -> ()
  llvm.return
}

// CHECK-LABEL: @exp_test
llvm.func @exp_test(%arg0: !llvm.float, %arg1: !llvm<"<8 x float>">) {
  // CHECK: call float @llvm.exp.f32
  "llvm.intr.exp"(%arg0) : (!llvm.float) -> !llvm.float
  // CHECK: call <8 x float> @llvm.exp.v8f32
  "llvm.intr.exp"(%arg1) : (!llvm<"<8 x float>">) -> !llvm<"<8 x float>">
  llvm.return
}

// CHECK-LABEL: @exp2_test
llvm.func @exp2_test(%arg0: !llvm.float, %arg1: !llvm<"<8 x float>">) {
  // CHECK: call float @llvm.exp2.f32
  "llvm.intr.exp2"(%arg0) : (!llvm.float) -> !llvm.float
  // CHECK: call <8 x float> @llvm.exp2.v8f32
  "llvm.intr.exp2"(%arg1) : (!llvm<"<8 x float>">) -> !llvm<"<8 x float>">
  llvm.return
}

// CHECK-LABEL: @log_test
llvm.func @log_test(%arg0: !llvm.float, %arg1: !llvm<"<8 x float>">) {
  // CHECK: call float @llvm.log.f32
  "llvm.intr.log"(%arg0) : (!llvm.float) -> !llvm.float
  // CHECK: call <8 x float> @llvm.log.v8f32
  "llvm.intr.log"(%arg1) : (!llvm<"<8 x float>">) -> !llvm<"<8 x float>">
  llvm.return
}

// CHECK-LABEL: @log10_test
llvm.func @log10_test(%arg0: !llvm.float, %arg1: !llvm<"<8 x float>">) {
  // CHECK: call float @llvm.log10.f32
  "llvm.intr.log10"(%arg0) : (!llvm.float) -> !llvm.float
  // CHECK: call <8 x float> @llvm.log10.v8f32
  "llvm.intr.log10"(%arg1) : (!llvm<"<8 x float>">) -> !llvm<"<8 x float>">
  llvm.return
}

// CHECK-LABEL: @log2_test
llvm.func @log2_test(%arg0: !llvm.float, %arg1: !llvm<"<8 x float>">) {
  // CHECK: call float @llvm.log2.f32
  "llvm.intr.log2"(%arg0) : (!llvm.float) -> !llvm.float
  // CHECK: call <8 x float> @llvm.log2.v8f32
  "llvm.intr.log2"(%arg1) : (!llvm<"<8 x float>">) -> !llvm<"<8 x float>">
  llvm.return
}

// CHECK-LABEL: @fabs_test
llvm.func @fabs_test(%arg0: !llvm.float, %arg1: !llvm<"<8 x float>">) {
  // CHECK: call float @llvm.fabs.f32
  "llvm.intr.fabs"(%arg0) : (!llvm.float) -> !llvm.float
  // CHECK: call <8 x float> @llvm.fabs.v8f32
  "llvm.intr.fabs"(%arg1) : (!llvm<"<8 x float>">) -> !llvm<"<8 x float>">
  llvm.return
}

// CHECK-LABEL: @sqrt_test
llvm.func @sqrt_test(%arg0: !llvm.float, %arg1: !llvm<"<8 x float>">) {
  // CHECK: call float @llvm.sqrt.f32
  "llvm.intr.sqrt"(%arg0) : (!llvm.float) -> !llvm.float
  // CHECK: call <8 x float> @llvm.sqrt.v8f32
  "llvm.intr.sqrt"(%arg1) : (!llvm<"<8 x float>">) -> !llvm<"<8 x float>">
  llvm.return
}

// CHECK-LABEL: @ceil_test
llvm.func @ceil_test(%arg0: !llvm.float, %arg1: !llvm<"<8 x float>">) {
  // CHECK: call float @llvm.ceil.f32
  "llvm.intr.ceil"(%arg0) : (!llvm.float) -> !llvm.float
  // CHECK: call <8 x float> @llvm.ceil.v8f32
  "llvm.intr.ceil"(%arg1) : (!llvm<"<8 x float>">) -> !llvm<"<8 x float>">
  llvm.return
}

// CHECK-LABEL: @cos_test
llvm.func @cos_test(%arg0: !llvm.float, %arg1: !llvm<"<8 x float>">) {
  // CHECK: call float @llvm.cos.f32
  "llvm.intr.cos"(%arg0) : (!llvm.float) -> !llvm.float
  // CHECK: call <8 x float> @llvm.cos.v8f32
  "llvm.intr.cos"(%arg1) : (!llvm<"<8 x float>">) -> !llvm<"<8 x float>">
  llvm.return
}

// CHECK-LABEL: @copysign_test
llvm.func @copysign_test(%arg0: !llvm.float, %arg1: !llvm.float, %arg2: !llvm<"<8 x float>">, %arg3: !llvm<"<8 x float>">) {
  // CHECK: call float @llvm.copysign.f32
  "llvm.intr.copysign"(%arg0, %arg1) : (!llvm.float, !llvm.float) -> !llvm.float
  // CHECK: call <8 x float> @llvm.copysign.v8f32
  "llvm.intr.copysign"(%arg2, %arg3) : (!llvm<"<8 x float>">, !llvm<"<8 x float>">) -> !llvm<"<8 x float>">
  llvm.return
}

// CHECK-LABEL: @pow_test
llvm.func @pow_test(%arg0: !llvm.float, %arg1: !llvm.float, %arg2: !llvm<"<8 x float>">, %arg3: !llvm<"<8 x float>">) {
  // CHECK: call float @llvm.pow.f32
  "llvm.intr.pow"(%arg0, %arg1) : (!llvm.float, !llvm.float) -> !llvm.float
  // CHECK: call <8 x float> @llvm.pow.v8f32
  "llvm.intr.pow"(%arg2, %arg3) : (!llvm<"<8 x float>">, !llvm<"<8 x float>">) -> !llvm<"<8 x float>">
  llvm.return
}

// CHECK-LABEL: @vector_reductions
llvm.func @vector_reductions(%arg0: !llvm.float, %arg1: !llvm<"<8 x float>">, %arg2: !llvm<"<8 x i32>">) {
  // CHECK: call i32 @llvm.experimental.vector.reduce.add.v8i32
  "llvm.intr.experimental.vector.reduce.add"(%arg2) : (!llvm<"<8 x i32>">) -> !llvm.i32
  // CHECK: call i32 @llvm.experimental.vector.reduce.and.v8i32
  "llvm.intr.experimental.vector.reduce.and"(%arg2) : (!llvm<"<8 x i32>">) -> !llvm.i32
  // CHECK: call float @llvm.experimental.vector.reduce.fmax.v8f32
  "llvm.intr.experimental.vector.reduce.fmax"(%arg1) : (!llvm<"<8 x float>">) -> !llvm.float
  // CHECK: call float @llvm.experimental.vector.reduce.fmin.v8f32
  "llvm.intr.experimental.vector.reduce.fmin"(%arg1) : (!llvm<"<8 x float>">) -> !llvm.float
  // CHECK: call i32 @llvm.experimental.vector.reduce.mul.v8i32
  "llvm.intr.experimental.vector.reduce.mul"(%arg2) : (!llvm<"<8 x i32>">) -> !llvm.i32
  // CHECK: call i32 @llvm.experimental.vector.reduce.or.v8i32
  "llvm.intr.experimental.vector.reduce.or"(%arg2) : (!llvm<"<8 x i32>">) -> !llvm.i32
  // CHECK: call i32 @llvm.experimental.vector.reduce.smax.v8i32
  "llvm.intr.experimental.vector.reduce.smax"(%arg2) : (!llvm<"<8 x i32>">) -> !llvm.i32
  // CHECK: call i32 @llvm.experimental.vector.reduce.smin.v8i32
  "llvm.intr.experimental.vector.reduce.smin"(%arg2) : (!llvm<"<8 x i32>">) -> !llvm.i32
  // CHECK: call i32 @llvm.experimental.vector.reduce.umax.v8i32
  "llvm.intr.experimental.vector.reduce.umax"(%arg2) : (!llvm<"<8 x i32>">) -> !llvm.i32
  // CHECK: call i32 @llvm.experimental.vector.reduce.umin.v8i32
  "llvm.intr.experimental.vector.reduce.umin"(%arg2) : (!llvm<"<8 x i32>">) -> !llvm.i32
  // CHECK: call float @llvm.experimental.vector.reduce.v2.fadd.f32.v8f32
  "llvm.intr.experimental.vector.reduce.v2.fadd"(%arg0, %arg1) : (!llvm.float, !llvm<"<8 x float>">) -> !llvm.float
  // CHECK: call float @llvm.experimental.vector.reduce.v2.fmul.f32.v8f32
  "llvm.intr.experimental.vector.reduce.v2.fmul"(%arg0, %arg1) : (!llvm.float, !llvm<"<8 x float>">) -> !llvm.float
  // CHECK: call i32 @llvm.experimental.vector.reduce.xor.v8i32
  "llvm.intr.experimental.vector.reduce.xor"(%arg2) : (!llvm<"<8 x i32>">) -> !llvm.i32
  llvm.return
}

// CHECK-LABEL: @matrix_intrinsics
//                                       4x16                       16x3
llvm.func @matrix_intrinsics(%A: !llvm<"<64 x float>">, %B: !llvm<"<48 x float>">,
                             %ptr: !llvm<"float*">, %stride: !llvm.i32) {
  // CHECK: call <12 x float> @llvm.matrix.multiply.v12f32.v64f32.v48f32(<64 x float> %0, <48 x float> %1, i32 4, i32 16, i32 3)
  %C = llvm.intr.matrix.multiply %A, %B
    { lhs_rows = 4: i32, lhs_columns = 16: i32 , rhs_columns = 3: i32} :
    (!llvm<"<64 x float>">, !llvm<"<48 x float>">) -> !llvm<"<12 x float>">
  // CHECK: call <48 x float> @llvm.matrix.transpose.v48f32(<48 x float> %1, i32 3, i32 16)
  %D = llvm.intr.matrix.transpose %B { rows = 3: i32, columns = 16: i32} :
    !llvm<"<48 x float>"> into !llvm<"<48 x float>">
  // CHECK: call <48 x float> @llvm.matrix.columnwise.load.v48f32.p0f32(float* %2, i32 %3, i32 3, i32 16)
  %E = llvm.intr.matrix.columnwise.load %ptr, <stride=%stride>
    { rows = 3: i32, columns = 16: i32} :
    !llvm<"<48 x float>"> from !llvm<"float*"> stride !llvm.i32
  // CHECK: call void @llvm.matrix.columnwise.store.v48f32.p0f32(<48 x float> %7, float* %2, i32 %3, i32 3, i32 16)
  llvm.intr.matrix.columnwise.store %E, %ptr, <stride=%stride>
    { rows = 3: i32, columns = 16: i32} :
    !llvm<"<48 x float>"> to !llvm<"float*"> stride !llvm.i32
  llvm.return
}

// CHECK-LABEL: @masked_intrinsics
llvm.func @masked_intrinsics(%A: !llvm<"<7 x float>*">, %mask: !llvm<"<7 x i1>">) {
  // CHECK: call <7 x float> @llvm.masked.load.v7f32.p0v7f32(<7 x float>* %{{.*}}, i32 1, <7 x i1> %{{.*}}, <7 x float> undef)
  %a = llvm.intr.masked.load %A, %mask { alignment = 1: i32} :
    (!llvm<"<7 x float>*">, !llvm<"<7 x i1>">) -> !llvm<"<7 x float>">
  // CHECK: call <7 x float> @llvm.masked.load.v7f32.p0v7f32(<7 x float>* %{{.*}}, i32 1, <7 x i1> %{{.*}}, <7 x float> %{{.*}})
  %b = llvm.intr.masked.load %A, %mask, %a { alignment = 1: i32} :
    (!llvm<"<7 x float>*">, !llvm<"<7 x i1>">, !llvm<"<7 x float>">) -> !llvm<"<7 x float>">
  // CHECK: call void @llvm.masked.store.v7f32.p0v7f32(<7 x float> %{{.*}}, <7 x float>* %0, i32 {{.*}}, <7 x i1> %{{.*}})
  llvm.intr.masked.store %b, %A, %mask { alignment = 1: i32} :
    !llvm<"<7 x float>">, !llvm<"<7 x i1>"> into !llvm<"<7 x float>*">
  llvm.return
}

// Check that intrinsics are declared with appropriate types.
// CHECK-DAG: declare float @llvm.fma.f32(float, float, float)
// CHECK-DAG: declare <8 x float> @llvm.fma.v8f32(<8 x float>, <8 x float>, <8 x float>) #0
// CHECK-DAG: declare float @llvm.fmuladd.f32(float, float, float)
// CHECK-DAG: declare <8 x float> @llvm.fmuladd.v8f32(<8 x float>, <8 x float>, <8 x float>) #0
// CHECK-DAG: declare void @llvm.prefetch.p0i8(i8* nocapture readonly, i32 immarg, i32 immarg, i32)
// CHECK-DAG: declare float @llvm.exp.f32(float)
// CHECK-DAG: declare <8 x float> @llvm.exp.v8f32(<8 x float>) #0
// CHECK-DAG: declare float @llvm.log.f32(float)
// CHECK-DAG: declare <8 x float> @llvm.log.v8f32(<8 x float>) #0
// CHECK-DAG: declare float @llvm.log10.f32(float)
// CHECK-DAG: declare <8 x float> @llvm.log10.v8f32(<8 x float>) #0
// CHECK-DAG: declare float @llvm.log2.f32(float)
// CHECK-DAG: declare <8 x float> @llvm.log2.v8f32(<8 x float>) #0
// CHECK-DAG: declare float @llvm.fabs.f32(float)
// CHECK-DAG: declare <8 x float> @llvm.fabs.v8f32(<8 x float>) #0
// CHECK-DAG: declare float @llvm.sqrt.f32(float)
// CHECK-DAG: declare <8 x float> @llvm.sqrt.v8f32(<8 x float>) #0
// CHECK-DAG: declare float @llvm.ceil.f32(float)
// CHECK-DAG: declare <8 x float> @llvm.ceil.v8f32(<8 x float>) #0
// CHECK-DAG: declare float @llvm.cos.f32(float)
// CHECK-DAG: declare <8 x float> @llvm.cos.v8f32(<8 x float>) #0
// CHECK-DAG: declare float @llvm.copysign.f32(float, float)
// CHECK-DAG: declare <12 x float> @llvm.matrix.multiply.v12f32.v64f32.v48f32(<64 x float>, <48 x float>, i32 immarg, i32 immarg, i32 immarg)
// CHECK-DAG: declare <48 x float> @llvm.matrix.transpose.v48f32(<48 x float>, i32 immarg, i32 immarg)
// CHECK-DAG: declare <48 x float> @llvm.matrix.columnwise.load.v48f32.p0f32(float*, i32, i32 immarg, i32 immarg)
// CHECK-DAG: declare void @llvm.matrix.columnwise.store.v48f32.p0f32(<48 x float>, float* writeonly, i32, i32 immarg, i32 immarg)
// CHECK-DAG: declare <7 x float> @llvm.masked.load.v7f32.p0v7f32(<7 x float>*, i32 immarg, <7 x i1>, <7 x float>)
// CHECK-DAG: declare void @llvm.masked.store.v7f32.p0v7f32(<7 x float>, <7 x float>*, i32 immarg, <7 x i1>)
