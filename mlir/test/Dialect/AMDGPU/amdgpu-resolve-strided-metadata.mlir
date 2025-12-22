// RUN: mlir-opt -amdgpu-resolve-strided-metadata -split-input-file %s | FileCheck %s

!tSrc = memref<?x?xi32, strided<[?, ?], offset: ?>>
!tDst = memref<?x?xi32, strided<[?, ?], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>
!tRes = memref<i32, #amdgpu.address_space<fat_raw_buffer>>
// CHECK-LABEL: @resolve_metadata_no_offset_reset
// CHECK-SAME: (%[[arg0:.*]]: memref<?x?xi32, strided<[?, ?], offset: ?>>)
// CHECK-NEXT: %[[cast:.+]] = amdgpu.fat_raw_buffer_cast %[[arg0]]
// CHECK-NEXT: %{{.+}}, %[[offset:.+]], %[[size:.+]]:2, %[[stride:.+]]:2 = memref.extract_strided_metadata %[[arg0]]
// CHECK-NEXT: %[[reinterp:.+]] = memref.reinterpret_cast %[[cast]]
// CHECK-NEXT: return %[[reinterp]], %[[offset]], %[[size]]#0, %[[size]]#1, %[[stride]]#0, %[[stride]]#1
func.func @resolve_metadata_no_offset_reset(%arg0: !tSrc) -> (!tRes, index, index, index, index, index) {
  %cast = amdgpu.fat_raw_buffer_cast %arg0 : !tSrc to !tDst
  %base, %offset, %size:2, %stride:2 = memref.extract_strided_metadata %cast : !tDst -> !tRes, index, index, index, index, index
  func.return %base, %offset, %size#0, %size#1, %stride#0, %stride#1 : !tRes, index, index, index, index, index
}

// -----

!tSrc = memref<?x?xi32, strided<[?, ?], offset: ?>>
!tDst = memref<?x?xi32, strided<[?, ?]>, #amdgpu.address_space<fat_raw_buffer>>
!tRes = memref<i32, #amdgpu.address_space<fat_raw_buffer>>
// CHECK-LABEL: @resolve_metadata_offset_reset
// CHECK-SAME: (%[[arg0:.*]]: memref<?x?xi32, strided<[?, ?], offset: ?>>)
// CHECK-NEXT: %[[offset:.+]] = arith.constant 0 : index
// CHECK-NEXT: %[[cast:.+]] = amdgpu.fat_raw_buffer_cast %[[arg0]]
// CHECK-NEXT: %{{.+}}, %{{.+}}, %[[size:.+]]:2, %[[stride:.+]]:2 = memref.extract_strided_metadata %[[arg0]]
// CHECK-NEXT: %[[reinterp:.+]] = memref.reinterpret_cast %[[cast]]
// CHECK-NEXT: return %[[reinterp]], %[[offset]], %[[size]]#0, %[[size]]#1, %[[stride]]#0, %[[stride]]#1
func.func @resolve_metadata_offset_reset(%arg0: !tSrc) -> (!tRes, index, index, index, index, index) {
  %cast = amdgpu.fat_raw_buffer_cast %arg0 resetOffset : !tSrc to !tDst
  %base, %offset, %size:2, %stride:2 = memref.extract_strided_metadata %cast : !tDst -> !tRes, index, index, index, index, index
  func.return %base, %offset, %size#0, %size#1, %stride#0, %stride#1 : !tRes, index, index, index, index, index
}

// -----

!tSrc = memref<?x?xi32, strided<[?, ?], offset: ?>>
!tDst = memref<?x?xi32, strided<[?, ?]>, #amdgpu.address_space<fat_raw_buffer>>
!tRes = memref<i32, #amdgpu.address_space<fat_raw_buffer>>
// CHECK-LABEL: @resolve_metadata_no_base_ptr
// CHECK-SAME: (%[[arg0:.*]]: memref<?x?xi32, strided<[?, ?], offset: ?>>)
// CHECK-NEXT: %[[offset:.+]] = arith.constant 0 : index
// CHECK-NEXT: %[[cast:.+]] = amdgpu.fat_raw_buffer_cast %[[arg0]]
// CHECK-NEXT: %{{.+}}, %{{.+}}, %[[size:.+]]:2, %[[stride:.+]]:2 = memref.extract_strided_metadata %[[arg0]]
// CHECK-NEXT: return %[[cast]], %[[offset]], %[[size]]#0, %[[size]]#1, %[[stride]]#0, %[[stride]]#1
func.func @resolve_metadata_no_base_ptr(%arg0: !tSrc) -> (!tDst, index, index, index, index, index) {
  %cast = amdgpu.fat_raw_buffer_cast %arg0 resetOffset : !tSrc to !tDst
  %base, %offset, %size:2, %stride:2 = memref.extract_strided_metadata %cast : !tDst -> !tRes, index, index, index, index, index
  func.return %cast, %offset, %size#0, %size#1, %stride#0, %stride#1 : !tDst, index, index, index, index, index
}
