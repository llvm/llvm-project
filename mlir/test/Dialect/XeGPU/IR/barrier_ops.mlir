// RUN: mlir-opt %s | FileCheck %s
// Verify the printed output can be parsed.
// RUN: mlir-opt %s | mlir-opt | FileCheck %s
// Verify the generic form can be parsed.
// RUN: mlir-opt -mlir-print-op-generic %s | mlir-opt | FileCheck %s

// CHECK-LABEL: func @alloc_nbarrier({{.*}}) {
func.func @alloc_nbarrier() {
  // CHECK: xegpu.alloc_nbarrier
  xegpu.alloc_nbarrier 8
  return
}

// CHECK-LABEL: func @create_nbarrier({{.*}}) {
func.func @create_nbarrier() {
  %nbarrier_id = arith.constant 1 : i8
  %nbarrier_role = arith.constant 0 : i8
  // CHECK: xegpu.create_nbarrier
  // CHECK-SAME: {num_consumers = 32 : i8, num_producers = 32 : i8}
  // CHECK-SAME: (i8, i8) -> !xegpu.nbarrier
  %nbarrier = xegpu.create_nbarrier %nbarrier_id, %nbarrier_role {num_producers = 32 :i8 , num_consumers = 32 : i8}
    : (i8, i8) -> !xegpu.nbarrier
  return
}

// CHECK-LABEL: func @nbarrier_arrive({{.*}}) {
func.func @nbarrier_arrive(%nbarrier : !xegpu.nbarrier) {
  // CHECK:  xegpu.nbarrier_arrive
  // CHECK-SAME: !xegpu.nbarrier
  xegpu.nbarrier_arrive %nbarrier : !xegpu.nbarrier
  return
}

// CHECK-LABEL: func @nbarrier_wait({{.*}}) {
func.func @nbarrier_wait(%nbarrier : !xegpu.nbarrier) {
  // CHECK: xegpu.nbarrier_wait
  // CHECK-SAME: !xegpu.nbarrier
  xegpu.nbarrier_wait %nbarrier : !xegpu.nbarrier
  return
}

// CHECK-LABEL: func @compile_hint({{.*}}) {
func.func @compile_hint() {
  // CHECK: xegpu.compile_hint
  xegpu.compile_hint
  return
}

// CHECK-LABEL: func @mfence({{.*}}) {
func.func @mfence() {
  // CHECK: xegpu.mfence {fence_op = "none", fence_scope = "local", memory_kind = "ugm"}
  xegpu.mfence {memory_kind = "ugm" , fence_op = "none", fence_scope = "local"}
  return
}
