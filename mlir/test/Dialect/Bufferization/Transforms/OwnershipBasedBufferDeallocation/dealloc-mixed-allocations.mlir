// RUN: mlir-opt %s --test-ownership-based-buffer-deallocation -split-input-file | FileCheck %s

func.func @mixed_allocations(%cond: i1) -> (memref<f32>, !gpu.async.token) {
  %a1 = memref.alloc() : memref<f32>
  %a2 = gpu.alloc() : memref<f32>
  %0 = arith.select %cond, %a1, %a2 : memref<f32>
  %token = gpu.dealloc async [] %a2 : memref<f32>
  memref.dealloc %a1 : memref<f32>
  return %0, %token : memref<f32>, !gpu.async.token
}

// CHECK: [[A1:%.+]] = memref.alloc(
// CHECK: [[A2:%.+]] = gpu.alloc
// CHECK: [[SELECT:%.+]] = arith.select {{.*}}, [[A1]], [[A2]]
// CHECK: [[TOKEN:%.+]] = gpu.wait async
// CHECK: [[A1_BASE:%.+]],{{.*}} = memref.extract_strided_metadata [[A1]]
// CHECK: [[A1_PTR:%.+]] = memref.extract_aligned_pointer_as_index [[A1_BASE]]
// CHECK: [[SELECT_PTR:%.+]] = memref.extract_aligned_pointer_as_index [[SELECT]]
// CHECK: [[ALIAS0:%.+]] = arith.cmpi ne, [[A1_PTR]], [[SELECT_PTR]]
// CHECK: [[COND0:%.+]] = arith.andi [[ALIAS0]], %true
// CHECK: scf.if [[COND0]] {
// CHECK:   memref.dealloc [[A1_BASE]]
// CHECK: }
// CHECK: [[A2_BASE:%.+]],{{.*}} = memref.extract_strided_metadata [[A2]]
// CHECK: [[A2_PTR:%.+]] = memref.extract_aligned_pointer_as_index [[A2_BASE]]
// CHECK: [[SELECT_PTR:%.+]] = memref.extract_aligned_pointer_as_index [[SELECT]]
// CHECK: [[ALIAS1:%.+]] = arith.cmpi ne, [[A2_PTR]], [[SELECT_PTR]]
// CHECK: [[COND1:%.+]] = arith.andi [[ALIAS1]], %true
// CHECK: scf.if [[COND1]] {
// CHECK:   [[T:%.+]] = gpu.dealloc async [[A2_BASE]]
// CHECK:   gpu.wait [[[T]]]
// CHECK: }
// CHECK: return [[SELECT]], [[TOKEN]]
