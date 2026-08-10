// RUN: mlir-opt %s -llvm-optimize-for-nvvm-target | FileCheck %s

// CHECK-LABEL: llvm.func @fdiv_fp16
llvm.func @fdiv_fp16(%arg0 : f16, %arg1 : f16) -> f16 {
  // CHECK-DAG: %[[c0:.*]]      = llvm.mlir.constant(0 : ui32) : i32
  // CHECK-DAG: %[[mask:.*]]    = llvm.mlir.constant(2139095040 : ui32) : i32
  // CHECK-DAG: %[[lhs:.*]]     = llvm.fpext %arg0 : f16 to f32
  // CHECK-DAG: %[[rhs:.*]]     = llvm.fpext %arg1 : f16 to f32
  // CHECK-DAG: %[[rcp:.*]]     = nvvm.rcp.approx.ftz.f %[[rhs]] : f32
  // CHECK-DAG: %[[approx:.*]]  = llvm.fmul %[[lhs]], %[[rcp]] : f32
  // CHECK-DAG: %[[neg:.*]]     = llvm.fneg %[[rhs]] : f32
  // CHECK-DAG: %[[err:.*]]     = llvm.intr.fma(%[[approx]], %[[neg]], %[[lhs]]) : (f32, f32, f32) -> f32
  // CHECK-DAG: %[[refined:.*]] = llvm.intr.fma(%[[err]], %[[rcp]], %[[approx]]) : (f32, f32, f32) -> f32
  // CHECK-DAG: %[[cast:.*]]    = llvm.bitcast %[[approx]] : f32 to i32
  // CHECK-DAG: %[[exp:.*]]     = llvm.and %[[cast]], %[[mask]] : i32
  // CHECK-DAG: %[[is_zero:.*]] = llvm.icmp "eq" %[[exp]], %[[c0]] : i32
  // CHECK-DAG: %[[is_mask:.*]] = llvm.icmp "eq" %[[exp]], %[[mask]] : i32
  // CHECK-DAG: %[[pred:.*]]    = llvm.or %[[is_zero]], %[[is_mask]] : i1
  // CHECK-DAG: %[[select:.*]]  = llvm.select %[[pred]], %[[approx]], %[[refined]] : i1, f32
  // CHECK-DAG: %[[result:.*]]  = llvm.fptrunc %[[select]] : f32 to f16
  %result = llvm.fdiv %arg0, %arg1 : f16
  // CHECK: llvm.return %[[result]] : f16
  llvm.return %result : f16
}
