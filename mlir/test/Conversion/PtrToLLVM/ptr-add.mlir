// RUN: mlir-opt %s -convert-to-llvm | FileCheck %s

// Broadcast a scalar base to the shape of the offsets.
// CHECK-LABEL: llvm.func @scalar_base_vector_offset(
// CHECK-SAME: %[[BASE:.*]]: !llvm.ptr, %[[OFFSET:.*]]: vector<4xi64>) -> vector<4x!llvm.ptr>
// CHECK-NEXT: %[[RESULT:.*]] = llvm.getelementptr %[[BASE]][%[[OFFSET]]] : (!llvm.ptr, vector<4xi64>) -> vector<4x!llvm.ptr>, i8
// CHECK-NEXT: llvm.return %[[RESULT]] : vector<4x!llvm.ptr>
func.func @scalar_base_vector_offset(%base: !ptr.ptr<#ptr.generic_space>, %offset: vector<4xi64>) -> vector<4x!ptr.ptr<#ptr.generic_space>> {
  %result = ptr.ptr_add %base, %offset : !ptr.ptr<#ptr.generic_space>, vector<4xi64>
  return %result : vector<4x!ptr.ptr<#ptr.generic_space>>
}

// Broadcast a scalar offset to the shape of the base; preserve the flags.
// CHECK-LABEL: llvm.func @vector_base_scalar_offset(
// CHECK-SAME: %[[BASE:.*]]: vector<4x!llvm.ptr>, %[[OFFSET:.*]]: i32) -> vector<4x!llvm.ptr>
// CHECK-NEXT: %[[RESULT:.*]] = llvm.getelementptr nusw %[[BASE]][%[[OFFSET]]] : (vector<4x!llvm.ptr>, i32) -> vector<4x!llvm.ptr>, i8
// CHECK-NEXT: llvm.return %[[RESULT]] : vector<4x!llvm.ptr>
func.func @vector_base_scalar_offset(%base: vector<4x!ptr.ptr<#ptr.generic_space>>, %offset: i32) -> vector<4x!ptr.ptr<#ptr.generic_space>> {
  %result = ptr.ptr_add nusw %base, %offset : vector<4x!ptr.ptr<#ptr.generic_space>>, i32
  return %result : vector<4x!ptr.ptr<#ptr.generic_space>>
}

// Convert index elements while preserving element-wise pointer addition.
// CHECK-LABEL: llvm.func @vector_base_vector_offset(
// CHECK-SAME: %[[BASE:.*]]: vector<4x!llvm.ptr>, %[[OFFSET:.*]]: vector<4xi64>) -> vector<4x!llvm.ptr>
// CHECK-NEXT: %[[RESULT:.*]] = llvm.getelementptr nuw %[[BASE]][%[[OFFSET]]] : (vector<4x!llvm.ptr>, vector<4xi64>) -> vector<4x!llvm.ptr>, i8
// CHECK-NEXT: llvm.return %[[RESULT]] : vector<4x!llvm.ptr>
func.func @vector_base_vector_offset(%base: vector<4x!ptr.ptr<#ptr.generic_space>>, %offset: vector<4xindex>) -> vector<4x!ptr.ptr<#ptr.generic_space>> {
  %result = ptr.ptr_add nuw %base, %offset : vector<4x!ptr.ptr<#ptr.generic_space>>, vector<4xindex>
  return %result : vector<4x!ptr.ptr<#ptr.generic_space>>
}

// The same combinations work for scalable vectors.
// CHECK-LABEL: llvm.func @scalar_base_scalable_offset(
// CHECK-SAME: %[[BASE:.*]]: !llvm.ptr, %[[OFFSET:.*]]: vector<[4]xi64>) -> vector<[4]x!llvm.ptr>
// CHECK-NEXT: %[[RESULT:.*]] = llvm.getelementptr inbounds %[[BASE]][%[[OFFSET]]] : (!llvm.ptr, vector<[4]xi64>) -> vector<[4]x!llvm.ptr>, i8
// CHECK-NEXT: llvm.return %[[RESULT]] : vector<[4]x!llvm.ptr>
func.func @scalar_base_scalable_offset(%base: !ptr.ptr<#ptr.generic_space>, %offset: vector<[4]xindex>) -> vector<[4]x!ptr.ptr<#ptr.generic_space>> {
  %result = ptr.ptr_add inbounds %base, %offset : !ptr.ptr<#ptr.generic_space>, vector<[4]xindex>
  return %result : vector<[4]x!ptr.ptr<#ptr.generic_space>>
}

// CHECK-LABEL: llvm.func @scalable_base_scalar_offset(
// CHECK-SAME: %[[BASE:.*]]: vector<[4]x!llvm.ptr>, %[[OFFSET:.*]]: i64) -> vector<[4]x!llvm.ptr>
// CHECK-NEXT: %[[RESULT:.*]] = llvm.getelementptr %[[BASE]][%[[OFFSET]]] : (vector<[4]x!llvm.ptr>, i64) -> vector<[4]x!llvm.ptr>, i8
// CHECK-NEXT: llvm.return %[[RESULT]] : vector<[4]x!llvm.ptr>
func.func @scalable_base_scalar_offset(%base: vector<[4]x!ptr.ptr<#ptr.generic_space>>, %offset: i64) -> vector<[4]x!ptr.ptr<#ptr.generic_space>> {
  %result = ptr.ptr_add %base, %offset : vector<[4]x!ptr.ptr<#ptr.generic_space>>, i64
  return %result : vector<[4]x!ptr.ptr<#ptr.generic_space>>
}

// CHECK-LABEL: llvm.func @scalable_base_scalable_offset(
// CHECK-SAME: %[[BASE:.*]]: vector<[4]x!llvm.ptr>, %[[OFFSET:.*]]: vector<[4]xi32>) -> vector<[4]x!llvm.ptr>
// CHECK-NEXT: %[[RESULT:.*]] = llvm.getelementptr %[[BASE]][%[[OFFSET]]] : (vector<[4]x!llvm.ptr>, vector<[4]xi32>) -> vector<[4]x!llvm.ptr>, i8
// CHECK-NEXT: llvm.return %[[RESULT]] : vector<[4]x!llvm.ptr>
func.func @scalable_base_scalable_offset(%base: vector<[4]x!ptr.ptr<#ptr.generic_space>>, %offset: vector<[4]xi32>) -> vector<[4]x!ptr.ptr<#ptr.generic_space>> {
  %result = ptr.ptr_add %base, %offset : vector<[4]x!ptr.ptr<#ptr.generic_space>>, vector<[4]xi32>
  return %result : vector<[4]x!ptr.ptr<#ptr.generic_space>>
}
