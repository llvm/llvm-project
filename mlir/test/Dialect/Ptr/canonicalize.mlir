// RUN: mlir-opt --canonicalize %s | FileCheck %s

/// Check `ptr_add` canonicalizer patterns.

// CHECK-LABEL: @zero_offset
// CHECK-SAME: (%[[PTR_0:.*]]: !ptr.ptr<#ptr.generic_space>)
func.func @zero_offset(%ptr: !ptr.ptr<#ptr.generic_space>) -> !ptr.ptr<#ptr.generic_space> {
  // CHECK-NOT: index.constant
  // CHECK-NOT: ptr.ptr_add
  // CHECK: return %[[PTR_0]] : !ptr.ptr<#ptr.generic_space>
  // CHECK: }
  %off = index.constant 0
  %res0 = ptr.ptr_add %ptr, %off : !ptr.ptr<#ptr.generic_space>, index
  return %res0 : !ptr.ptr<#ptr.generic_space>
}

/// Tests the the `from_ptr` folder.
// CHECK-LABEL: @test_from_ptr_0
// CHECK-SAME: (%[[MEM_REF:.*]]: memref<f32, #ptr.generic_space>)
func.func @test_from_ptr_0(%mr: memref<f32, #ptr.generic_space>) -> memref<f32, #ptr.generic_space> {
  // CHECK-NOT: ptr.to_ptr
  // CHECK-NOT: ptr.get_metadata
  // CHECK-NOT: ptr.from_ptr
  // CHECK: return %[[MEM_REF]]
  %ptr = ptr.to_ptr %mr : memref<f32, #ptr.generic_space> -> !ptr.ptr<#ptr.generic_space>
  %mda = ptr.get_metadata %mr : memref<f32, #ptr.generic_space>
  %res = ptr.from_ptr %ptr metadata %mda : !ptr.ptr<#ptr.generic_space> -> memref<f32, #ptr.generic_space>
  return %res : memref<f32, #ptr.generic_space>
}

// CHECK-LABEL: @test_from_ptr_1
// CHECK-SAME: (%[[MEM_REF:.*]]: memref<f32, #ptr.generic_space>)
func.func @test_from_ptr_1(%mr: memref<f32, #ptr.generic_space>) -> memref<f32, #ptr.generic_space> {
  // CHECK-NOT: ptr.to_ptr
  // CHECK-NOT: ptr.from_ptr
  // CHECK: return %[[MEM_REF]]
  %ptr = ptr.to_ptr %mr : memref<f32, #ptr.generic_space> -> !ptr.ptr<#ptr.generic_space>
  %res = ptr.from_ptr %ptr trivial_metadata : !ptr.ptr<#ptr.generic_space> -> memref<f32, #ptr.generic_space>
  return %res : memref<f32, #ptr.generic_space>
}

/// Check that the ops cannot be folded because the metadata cannot be guaranteed to be the same.
// CHECK-LABEL: @test_from_ptr_2
func.func @test_from_ptr_2(%mr: memref<f32, #ptr.generic_space>, %md: !ptr.ptr_metadata<memref<f32, #ptr.generic_space>>) -> memref<f32, #ptr.generic_space> {
  // CHECK: ptr.to_ptr
  // CHECK: ptr.from_ptr
  %ptr = ptr.to_ptr %mr : memref<f32, #ptr.generic_space> -> !ptr.ptr<#ptr.generic_space>
  %res = ptr.from_ptr %ptr metadata %md : !ptr.ptr<#ptr.generic_space> -> memref<f32, #ptr.generic_space>
  return %res : memref<f32, #ptr.generic_space>
}

/// Tests the the `to_ptr` folder.
// CHECK-LABEL: @test_to_ptr_0
// CHECK-SAME: (%[[PTR:.*]]: !ptr.ptr<#ptr.generic_space>
func.func @test_to_ptr_0(%ptr: !ptr.ptr<#ptr.generic_space>, %md: !ptr.ptr_metadata<memref<f32, #ptr.generic_space>>) -> !ptr.ptr<#ptr.generic_space> {
  // CHECK: return %[[PTR]]
  // CHECK-NOT: ptr.from_ptr
  // CHECK-NOT: ptr.to_ptr
  %mrf = ptr.from_ptr %ptr metadata %md : !ptr.ptr<#ptr.generic_space> -> memref<f32, #ptr.generic_space>
  %res = ptr.to_ptr %mrf : memref<f32, #ptr.generic_space> -> !ptr.ptr<#ptr.generic_space>
  return %res : !ptr.ptr<#ptr.generic_space>
}

// CHECK-LABEL: @test_to_ptr_1
// CHECK-SAME: (%[[PTR:.*]]: !ptr.ptr<#ptr.generic_space>)
func.func @test_to_ptr_1(%ptr: !ptr.ptr<#ptr.generic_space>) -> !ptr.ptr<#ptr.generic_space> {
  // CHECK-NOT: ptr.from_ptr
  // CHECK-NOT: ptr.to_ptr
  // CHECK: return %[[PTR]]
  %mrf = ptr.from_ptr %ptr trivial_metadata : !ptr.ptr<#ptr.generic_space> -> memref<f32, #ptr.generic_space>
  %res = ptr.to_ptr %mrf : memref<f32, #ptr.generic_space> -> !ptr.ptr<#ptr.generic_space>
  return %res : !ptr.ptr<#ptr.generic_space>
}
