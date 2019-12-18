// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// CHECK-LABEL: @intrinsics
llvm.func @intrinsics(%arg0: !llvm.float, %arg1: !llvm.float, %arg2: !llvm<"<8 x float>">, %arg3: !llvm<"i8*">) {
  %c3 = llvm.mlir.constant(3 : i32) : !llvm.i32
  %c1 = llvm.mlir.constant(1 : i32) : !llvm.i32
  %c0 = llvm.mlir.constant(0 : i32) : !llvm.i32
  // CHECK: call float @llvm.fmuladd.f32.f32.f32
  "llvm.intr.fmuladd"(%arg0, %arg1, %arg0) : (!llvm.float, !llvm.float, !llvm.float) -> !llvm.float
  // CHECK: call <8 x float> @llvm.fmuladd.v8f32.v8f32.v8f32
  "llvm.intr.fmuladd"(%arg2, %arg2, %arg2) : (!llvm<"<8 x float>">, !llvm<"<8 x float>">, !llvm<"<8 x float>">) -> !llvm<"<8 x float>">
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

// Check that intrinsics are declared with appropriate types.
// CHECK: declare float @llvm.fmuladd.f32.f32.f32(float, float, float)
// CHECK: declare <8 x float> @llvm.fmuladd.v8f32.v8f32.v8f32(<8 x float>, <8 x float>, <8 x float>) #0
// CHECK: declare void @llvm.prefetch.p0i8(i8* nocapture readonly, i32 immarg, i32 immarg, i32)
// CHECK: declare float @llvm.exp.f32(float)
// CHECK: declare <8 x float> @llvm.exp.v8f32(<8 x float>) #0
// CHECK: declare float @llvm.log.f32(float)
// CHECK: declare <8 x float> @llvm.log.v8f32(<8 x float>) #0
// CHECK: declare float @llvm.log10.f32(float)
// CHECK: declare <8 x float> @llvm.log10.v8f32(<8 x float>) #0
// CHECK: declare float @llvm.log2.f32(float)
// CHECK: declare <8 x float> @llvm.log2.v8f32(<8 x float>) #0
// CHECK: declare float @llvm.fabs.f32(float)
// CHECK: declare <8 x float> @llvm.fabs.v8f32(<8 x float>) #0
// CHECK: declare float @llvm.ceil.f32(float)
// CHECK: declare <8 x float> @llvm.ceil.v8f32(<8 x float>) #0
// CHECK: declare float @llvm.cos.f32(float)
// CHECK: declare <8 x float> @llvm.cos.v8f32(<8 x float>) #0
