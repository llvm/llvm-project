// RUN: mlir-opt %s -convert-to-llvm | FileCheck %s

// Tests different variants of ptr_add operation with various attributes
// (regular, nusw, nuw, inbounds)
// CHECK-LABEL:   llvm.func @test_ptr_add(
// CHECK-SAME:      %[[ARG0:.*]]: !llvm.ptr, %[[ARG1:.*]]: i64) -> !llvm.struct<(ptr, ptr, ptr, ptr)> {
// CHECK:           %[[VAL_0:.*]] = llvm.getelementptr %[[ARG0]]{{\[}}%[[ARG1]]] : (!llvm.ptr, i64) -> !llvm.ptr, i8
// CHECK:           %[[VAL_1:.*]] = llvm.getelementptr nusw %[[ARG0]]{{\[}}%[[ARG1]]] : (!llvm.ptr, i64) -> !llvm.ptr, i8
// CHECK:           %[[VAL_2:.*]] = llvm.getelementptr nuw %[[ARG0]]{{\[}}%[[ARG1]]] : (!llvm.ptr, i64) -> !llvm.ptr, i8
// CHECK:           %[[VAL_3:.*]] = llvm.getelementptr inbounds %[[ARG0]]{{\[}}%[[ARG1]]] : (!llvm.ptr, i64) -> !llvm.ptr, i8
// CHECK:           %[[VAL_4:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, ptr, ptr)>
// CHECK:           %[[VAL_5:.*]] = llvm.insertvalue %[[VAL_0]], %[[VAL_4]][0] : !llvm.struct<(ptr, ptr, ptr, ptr)>
// CHECK:           %[[VAL_6:.*]] = llvm.insertvalue %[[VAL_1]], %[[VAL_5]][1] : !llvm.struct<(ptr, ptr, ptr, ptr)>
// CHECK:           %[[VAL_7:.*]] = llvm.insertvalue %[[VAL_2]], %[[VAL_6]][2] : !llvm.struct<(ptr, ptr, ptr, ptr)>
// CHECK:           %[[VAL_8:.*]] = llvm.insertvalue %[[VAL_3]], %[[VAL_7]][3] : !llvm.struct<(ptr, ptr, ptr, ptr)>
// CHECK:           llvm.return %[[VAL_8]] : !llvm.struct<(ptr, ptr, ptr, ptr)>
// CHECK:         }
func.func @test_ptr_add(%arg0: !ptr.ptr<#ptr.generic_space>, %arg1: index) -> (!ptr.ptr<#ptr.generic_space>, !ptr.ptr<#ptr.generic_space>, !ptr.ptr<#ptr.generic_space>, !ptr.ptr<#ptr.generic_space>) {
  %0 = ptr.ptr_add %arg0, %arg1 : !ptr.ptr<#ptr.generic_space>, index
  %1 = ptr.ptr_add nusw %arg0, %arg1 : !ptr.ptr<#ptr.generic_space>, index
  %2 = ptr.ptr_add nuw %arg0, %arg1 : !ptr.ptr<#ptr.generic_space>, index
  %3 = ptr.ptr_add inbounds %arg0, %arg1 : !ptr.ptr<#ptr.generic_space>, index
  return %0, %1, %2, %3 : !ptr.ptr<#ptr.generic_space>, !ptr.ptr<#ptr.generic_space>, !ptr.ptr<#ptr.generic_space>, !ptr.ptr<#ptr.generic_space>
}

// Tests type_offset operation which returns the size of different types
// CHECK-LABEL:   llvm.func @test_type_offset() -> !llvm.struct<(i64, i64, i64)> {
// CHECK:           %[[VAL_0:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           %[[VAL_1:.*]] = llvm.getelementptr %[[VAL_0]][1] : (!llvm.ptr) -> !llvm.ptr, f32
// CHECK:           %[[VAL_2:.*]] = llvm.ptrtoint %[[VAL_1]] : !llvm.ptr to i64
// CHECK:           %[[VAL_3:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           %[[VAL_4:.*]] = llvm.getelementptr %[[VAL_3]][1] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK:           %[[VAL_5:.*]] = llvm.ptrtoint %[[VAL_4]] : !llvm.ptr to i64
// CHECK:           %[[VAL_6:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           %[[VAL_7:.*]] = llvm.getelementptr %[[VAL_6]][1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, f64)>
// CHECK:           %[[VAL_8:.*]] = llvm.ptrtoint %[[VAL_7]] : !llvm.ptr to i64
// CHECK:           %[[VAL_9:.*]] = llvm.mlir.poison : !llvm.struct<(i64, i64, i64)>
// CHECK:           %[[VAL_10:.*]] = llvm.insertvalue %[[VAL_2]], %[[VAL_9]][0] : !llvm.struct<(i64, i64, i64)>
// CHECK:           %[[VAL_11:.*]] = llvm.insertvalue %[[VAL_5]], %[[VAL_10]][1] : !llvm.struct<(i64, i64, i64)>
// CHECK:           %[[VAL_12:.*]] = llvm.insertvalue %[[VAL_8]], %[[VAL_11]][2] : !llvm.struct<(i64, i64, i64)>
// CHECK:           llvm.return %[[VAL_12]] : !llvm.struct<(i64, i64, i64)>
// CHECK:         }
func.func @test_type_offset() -> (index, index, index) {
  %0 = ptr.type_offset f32 : index
  %1 = ptr.type_offset i64 : index
  %2 = ptr.type_offset !llvm.struct<(i32, f64)> : index
  return %0, %1, %2 : index, index, index
}

// Tests converting a memref to a pointer using to_ptr
// CHECK-LABEL:   llvm.func @test_to_ptr(
// CHECK-SAME:      %[[ARG0:.*]]: !llvm.ptr, %[[ARG1:.*]]: !llvm.ptr, %[[ARG2:.*]]: i64, %[[ARG3:.*]]: i64, %[[ARG4:.*]]: i64) -> !llvm.ptr {
// CHECK:           %[[VAL_0:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[VAL_1:.*]] = llvm.insertvalue %[[ARG0]], %[[VAL_0]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[VAL_2:.*]] = llvm.insertvalue %[[ARG1]], %[[VAL_1]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[VAL_3:.*]] = llvm.insertvalue %[[ARG2]], %[[VAL_2]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[VAL_4:.*]] = llvm.insertvalue %[[ARG3]], %[[VAL_3]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[VAL_5:.*]] = llvm.insertvalue %[[ARG4]], %[[VAL_4]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[VAL_6:.*]] = llvm.extractvalue %[[VAL_5]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           llvm.return %[[VAL_6]] : !llvm.ptr
// CHECK:         }
func.func @test_to_ptr(%arg0: memref<10xf32, #ptr.generic_space>) -> !ptr.ptr<#ptr.generic_space> {
  %0 = ptr.to_ptr %arg0 : memref<10xf32, #ptr.generic_space> -> <#ptr.generic_space>
  return %0 : !ptr.ptr<#ptr.generic_space>
}

// Tests extracting metadata from a static-sized memref
// CHECK-LABEL:   llvm.func @test_get_metadata_static(
// CHECK-SAME:      %[[ARG0:.*]]: !llvm.ptr, %[[ARG1:.*]]: !llvm.ptr, %[[ARG2:.*]]: i64, %[[ARG3:.*]]: i64, %[[ARG4:.*]]: i64, %[[ARG5:.*]]: i64, %[[ARG6:.*]]: i64) -> !llvm.struct<(ptr)> {
// CHECK:           %[[VAL_0:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_1:.*]] = llvm.insertvalue %[[ARG0]], %[[VAL_0]][0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_2:.*]] = llvm.insertvalue %[[ARG1]], %[[VAL_1]][1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_3:.*]] = llvm.insertvalue %[[ARG2]], %[[VAL_2]][2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_4:.*]] = llvm.insertvalue %[[ARG3]], %[[VAL_3]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_5:.*]] = llvm.insertvalue %[[ARG5]], %[[VAL_4]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_6:.*]] = llvm.insertvalue %[[ARG4]], %[[VAL_5]][3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_7:.*]] = llvm.insertvalue %[[ARG6]], %[[VAL_6]][4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_8:.*]] = llvm.mlir.undef : !llvm.struct<(ptr)>
// CHECK:           %[[VAL_9:.*]] = llvm.extractvalue %[[VAL_7]][0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_10:.*]] = llvm.insertvalue %[[VAL_9]], %[[VAL_8]][0] : !llvm.struct<(ptr)>
// CHECK:           llvm.return %[[VAL_10]] : !llvm.struct<(ptr)>
// CHECK:         }
func.func @test_get_metadata_static(%arg0: memref<10x20xf32, #ptr.generic_space>) -> !ptr.ptr_metadata<memref<10x20xf32, #ptr.generic_space>> {
  %0 = ptr.get_metadata %arg0 : memref<10x20xf32, #ptr.generic_space>
  return %0 : !ptr.ptr_metadata<memref<10x20xf32, #ptr.generic_space>>
}

// Tests extracting metadata from a dynamically-sized memref
// CHECK-LABEL:   llvm.func @test_get_metadata_dynamic(
// CHECK-SAME:      %[[ARG0:.*]]: !llvm.ptr, %[[ARG1:.*]]: !llvm.ptr, %[[ARG2:.*]]: i64, %[[ARG3:.*]]: i64, %[[ARG4:.*]]: i64, %[[ARG5:.*]]: i64, %[[ARG6:.*]]: i64) -> !llvm.struct<(ptr, i64, i64, i64)> {
// CHECK:           %[[VAL_0:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_1:.*]] = llvm.insertvalue %[[ARG0]], %[[VAL_0]][0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_2:.*]] = llvm.insertvalue %[[ARG1]], %[[VAL_1]][1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_3:.*]] = llvm.insertvalue %[[ARG2]], %[[VAL_2]][2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_4:.*]] = llvm.insertvalue %[[ARG3]], %[[VAL_3]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_5:.*]] = llvm.insertvalue %[[ARG5]], %[[VAL_4]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_6:.*]] = llvm.insertvalue %[[ARG4]], %[[VAL_5]][3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_7:.*]] = llvm.insertvalue %[[ARG6]], %[[VAL_6]][4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_8:.*]] = llvm.mlir.undef : !llvm.struct<(ptr, i64, i64, i64)>
// CHECK:           %[[VAL_9:.*]] = llvm.extractvalue %[[VAL_7]][0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_10:.*]] = llvm.insertvalue %[[VAL_9]], %[[VAL_8]][0] : !llvm.struct<(ptr, i64, i64, i64)>
// CHECK:           %[[VAL_11:.*]] = llvm.extractvalue %[[VAL_7]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_12:.*]] = llvm.insertvalue %[[VAL_11]], %[[VAL_10]][1] : !llvm.struct<(ptr, i64, i64, i64)>
// CHECK:           %[[VAL_13:.*]] = llvm.extractvalue %[[VAL_7]][3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_14:.*]] = llvm.insertvalue %[[VAL_13]], %[[VAL_12]][2] : !llvm.struct<(ptr, i64, i64, i64)>
// CHECK:           %[[VAL_15:.*]] = llvm.extractvalue %[[VAL_7]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_16:.*]] = llvm.insertvalue %[[VAL_15]], %[[VAL_14]][3] : !llvm.struct<(ptr, i64, i64, i64)>
// CHECK:           llvm.return %[[VAL_16]] : !llvm.struct<(ptr, i64, i64, i64)>
// CHECK:         }
func.func @test_get_metadata_dynamic(%arg0: memref<?x?xf32, #ptr.generic_space>) -> !ptr.ptr_metadata<memref<?x?xf32, #ptr.generic_space>> {
  %0 = ptr.get_metadata %arg0 : memref<?x?xf32, #ptr.generic_space>
  return %0 : !ptr.ptr_metadata<memref<?x?xf32, #ptr.generic_space>>
}

// Tests reconstructing a static-sized memref from a pointer and metadata
// CHECK-LABEL:   llvm.func @test_from_ptr_static(
// CHECK-SAME:      %[[ARG0:.*]]: !llvm.ptr, %[[ARG1:.*]]: !llvm.struct<(ptr)>) -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> {
// CHECK:           %[[VAL_0:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_1:.*]] = llvm.extractvalue %[[ARG1]][0] : !llvm.struct<(ptr)>
// CHECK:           %[[VAL_2:.*]] = llvm.insertvalue %[[VAL_1]], %[[VAL_0]][0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_3:.*]] = llvm.insertvalue %[[ARG0]], %[[VAL_2]][1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_4:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           %[[VAL_5:.*]] = llvm.insertvalue %[[VAL_4]], %[[VAL_3]][2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_6:.*]] = llvm.mlir.constant(10 : index) : i64
// CHECK:           %[[VAL_7:.*]] = llvm.insertvalue %[[VAL_6]], %[[VAL_5]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_8:.*]] = llvm.mlir.constant(20 : index) : i64
// CHECK:           %[[VAL_9:.*]] = llvm.insertvalue %[[VAL_8]], %[[VAL_7]][3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_10:.*]] = llvm.mlir.constant(20 : index) : i64
// CHECK:           %[[VAL_11:.*]] = llvm.insertvalue %[[VAL_10]], %[[VAL_9]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_12:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK:           %[[VAL_13:.*]] = llvm.insertvalue %[[VAL_12]], %[[VAL_11]][4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           llvm.return %[[VAL_13]] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:         }
func.func @test_from_ptr_static(%arg0: !ptr.ptr<#ptr.generic_space>, %arg1: !ptr.ptr_metadata<memref<10x20xf32, #ptr.generic_space>>) -> memref<10x20xf32, #ptr.generic_space> {
  %0 = ptr.from_ptr %arg0 metadata %arg1 : <#ptr.generic_space> -> memref<10x20xf32, #ptr.generic_space>
  return %0 : memref<10x20xf32, #ptr.generic_space>
}

// Tests reconstructing a dynamically-sized memref from a pointer and metadata
// CHECK-LABEL:   llvm.func @test_from_ptr_dynamic(
// CHECK-SAME:      %[[ARG0:.*]]: !llvm.ptr, %[[ARG1:.*]]: !llvm.struct<(ptr, i64, i64, i64)>) -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> {
// CHECK:           %[[VAL_0:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_1:.*]] = llvm.extractvalue %[[ARG1]][0] : !llvm.struct<(ptr, i64, i64, i64)>
// CHECK:           %[[VAL_2:.*]] = llvm.insertvalue %[[VAL_1]], %[[VAL_0]][0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_3:.*]] = llvm.insertvalue %[[ARG0]], %[[VAL_2]][1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_4:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           %[[VAL_5:.*]] = llvm.insertvalue %[[VAL_4]], %[[VAL_3]][2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_6:.*]] = llvm.extractvalue %[[ARG1]][1] : !llvm.struct<(ptr, i64, i64, i64)>
// CHECK:           %[[VAL_7:.*]] = llvm.insertvalue %[[VAL_6]], %[[VAL_5]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_8:.*]] = llvm.extractvalue %[[ARG1]][2] : !llvm.struct<(ptr, i64, i64, i64)>
// CHECK:           %[[VAL_9:.*]] = llvm.insertvalue %[[VAL_8]], %[[VAL_7]][3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_10:.*]] = llvm.extractvalue %[[ARG1]][3] : !llvm.struct<(ptr, i64, i64, i64)>
// CHECK:           %[[VAL_11:.*]] = llvm.insertvalue %[[VAL_10]], %[[VAL_9]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_12:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK:           %[[VAL_13:.*]] = llvm.insertvalue %[[VAL_12]], %[[VAL_11]][4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           llvm.return %[[VAL_13]] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:         }
func.func @test_from_ptr_dynamic(%arg0: !ptr.ptr<#ptr.generic_space>, %arg1: !ptr.ptr_metadata<memref<?x?xf32, #ptr.generic_space>>) -> memref<?x?xf32, #ptr.generic_space> {
  %0 = ptr.from_ptr %arg0 metadata %arg1 : <#ptr.generic_space> -> memref<?x?xf32, #ptr.generic_space>
  return %0 : memref<?x?xf32, #ptr.generic_space>
}

// Tests a round-trip conversion of a memref with mixed static/dynamic dimensions
// CHECK-LABEL:   llvm.func @test_memref_mixed(
// CHECK-SAME:      %[[ARG0:.*]]: !llvm.ptr, %[[ARG1:.*]]: !llvm.ptr, %[[ARG2:.*]]: i64, %[[ARG3:.*]]: i64, %[[ARG4:.*]]: i64, %[[ARG5:.*]]: i64, %[[ARG6:.*]]: i64, %[[ARG7:.*]]: i64, %[[ARG8:.*]]: i64) -> !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> {
// CHECK:           %[[VAL_0:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           %[[VAL_1:.*]] = llvm.insertvalue %[[ARG0]], %[[VAL_0]][0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           %[[VAL_2:.*]] = llvm.insertvalue %[[ARG1]], %[[VAL_1]][1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           %[[VAL_3:.*]] = llvm.insertvalue %[[ARG2]], %[[VAL_2]][2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           %[[VAL_4:.*]] = llvm.insertvalue %[[ARG3]], %[[VAL_3]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           %[[VAL_5:.*]] = llvm.insertvalue %[[ARG6]], %[[VAL_4]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           %[[VAL_6:.*]] = llvm.insertvalue %[[ARG4]], %[[VAL_5]][3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           %[[VAL_7:.*]] = llvm.insertvalue %[[ARG7]], %[[VAL_6]][4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           %[[VAL_8:.*]] = llvm.insertvalue %[[ARG5]], %[[VAL_7]][3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           %[[VAL_9:.*]] = llvm.insertvalue %[[ARG8]], %[[VAL_8]][4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           %[[VAL_10:.*]] = llvm.extractvalue %[[VAL_9]][1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           %[[VAL_11:.*]] = llvm.mlir.undef : !llvm.struct<(ptr, i64, i64)>
// CHECK:           %[[VAL_12:.*]] = llvm.extractvalue %[[VAL_9]][0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           %[[VAL_13:.*]] = llvm.insertvalue %[[VAL_12]], %[[VAL_11]][0] : !llvm.struct<(ptr, i64, i64)>
// CHECK:           %[[VAL_14:.*]] = llvm.extractvalue %[[VAL_9]][3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           %[[VAL_15:.*]] = llvm.insertvalue %[[VAL_14]], %[[VAL_13]][1] : !llvm.struct<(ptr, i64, i64)>
// CHECK:           %[[VAL_16:.*]] = llvm.extractvalue %[[VAL_9]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           %[[VAL_17:.*]] = llvm.insertvalue %[[VAL_16]], %[[VAL_15]][2] : !llvm.struct<(ptr, i64, i64)>
// CHECK:           llvm.return %[[VAL_9]] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:         }
func.func @test_memref_mixed(%arg0: memref<10x?x30xf32, #ptr.generic_space>) -> memref<10x?x30xf32, #ptr.generic_space> {
  %0 = ptr.to_ptr %arg0 : memref<10x?x30xf32, #ptr.generic_space> -> <#ptr.generic_space>
  %1 = ptr.get_metadata %arg0 : memref<10x?x30xf32, #ptr.generic_space>
  %2 = ptr.from_ptr %0 metadata %1 : <#ptr.generic_space> -> memref<10x?x30xf32, #ptr.generic_space>
  return %2 : memref<10x?x30xf32, #ptr.generic_space>
}

// Tests a round-trip conversion of a strided memref with explicit offset
// CHECK-LABEL:   llvm.func @test_memref_strided(
// CHECK-SAME:      %[[ARG0:.*]]: !llvm.ptr, %[[ARG1:.*]]: !llvm.ptr, %[[ARG2:.*]]: i64, %[[ARG3:.*]]: i64, %[[ARG4:.*]]: i64, %[[ARG5:.*]]: i64, %[[ARG6:.*]]: i64) -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> {
// CHECK:           %[[VAL_0:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_1:.*]] = llvm.insertvalue %[[ARG0]], %[[VAL_0]][0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_2:.*]] = llvm.insertvalue %[[ARG1]], %[[VAL_1]][1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_3:.*]] = llvm.insertvalue %[[ARG2]], %[[VAL_2]][2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_4:.*]] = llvm.insertvalue %[[ARG3]], %[[VAL_3]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_5:.*]] = llvm.insertvalue %[[ARG5]], %[[VAL_4]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_6:.*]] = llvm.insertvalue %[[ARG4]], %[[VAL_5]][3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_7:.*]] = llvm.insertvalue %[[ARG6]], %[[VAL_6]][4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_8:.*]] = llvm.extractvalue %[[VAL_7]][1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_9:.*]] = llvm.mlir.undef : !llvm.struct<(ptr)>
// CHECK:           %[[VAL_10:.*]] = llvm.extractvalue %[[VAL_7]][0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_11:.*]] = llvm.insertvalue %[[VAL_10]], %[[VAL_9]][0] : !llvm.struct<(ptr)>
// CHECK:           llvm.return %[[VAL_7]] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:         }
func.func @test_memref_strided(%arg0: memref<10x20xf32, strided<[40, 2], offset: 5>, #ptr.generic_space>) -> memref<10x20xf32, strided<[40, 2], offset: 5>, #ptr.generic_space> {
  %0 = ptr.to_ptr %arg0 : memref<10x20xf32, strided<[40, 2], offset: 5>, #ptr.generic_space> -> <#ptr.generic_space>
  %1 = ptr.get_metadata %arg0 : memref<10x20xf32, strided<[40, 2], offset: 5>, #ptr.generic_space>
  %2 = ptr.from_ptr %0 metadata %1 : <#ptr.generic_space> -> memref<10x20xf32, strided<[40, 2], offset: 5>, #ptr.generic_space>
  return %2 : memref<10x20xf32, strided<[40, 2], offset: 5>, #ptr.generic_space>
}

// Tests a comprehensive scenario with fully dynamic memref, including pointer arithmetic
// CHECK-LABEL:   llvm.func @test_comprehensive_dynamic(
// CHECK-SAME:      %[[ARG0:.*]]: !llvm.ptr, %[[ARG1:.*]]: !llvm.ptr, %[[ARG2:.*]]: i64, %[[ARG3:.*]]: i64, %[[ARG4:.*]]: i64, %[[ARG5:.*]]: i64, %[[ARG6:.*]]: i64) -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> {
// CHECK:           %[[VAL_0:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_1:.*]] = llvm.insertvalue %[[ARG0]], %[[VAL_0]][0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_2:.*]] = llvm.insertvalue %[[ARG1]], %[[VAL_1]][1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_3:.*]] = llvm.insertvalue %[[ARG2]], %[[VAL_2]][2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_4:.*]] = llvm.insertvalue %[[ARG3]], %[[VAL_3]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_5:.*]] = llvm.insertvalue %[[ARG5]], %[[VAL_4]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_6:.*]] = llvm.insertvalue %[[ARG4]], %[[VAL_5]][3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_7:.*]] = llvm.insertvalue %[[ARG6]], %[[VAL_6]][4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_8:.*]] = llvm.extractvalue %[[VAL_7]][1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_9:.*]] = llvm.mlir.undef : !llvm.struct<(ptr, i64, i64, i64, i64, i64)>
// CHECK:           %[[VAL_10:.*]] = llvm.extractvalue %[[VAL_7]][0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_11:.*]] = llvm.insertvalue %[[VAL_10]], %[[VAL_9]][0] : !llvm.struct<(ptr, i64, i64, i64, i64, i64)>
// CHECK:           %[[VAL_12:.*]] = llvm.extractvalue %[[VAL_7]][2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_13:.*]] = llvm.insertvalue %[[VAL_12]], %[[VAL_11]][1] : !llvm.struct<(ptr, i64, i64, i64, i64, i64)>
// CHECK:           %[[VAL_14:.*]] = llvm.extractvalue %[[VAL_7]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_15:.*]] = llvm.insertvalue %[[VAL_14]], %[[VAL_13]][2] : !llvm.struct<(ptr, i64, i64, i64, i64, i64)>
// CHECK:           %[[VAL_16:.*]] = llvm.extractvalue %[[VAL_7]][3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_17:.*]] = llvm.insertvalue %[[VAL_16]], %[[VAL_15]][3] : !llvm.struct<(ptr, i64, i64, i64, i64, i64)>
// CHECK:           %[[VAL_18:.*]] = llvm.extractvalue %[[VAL_7]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_19:.*]] = llvm.insertvalue %[[VAL_18]], %[[VAL_17]][4] : !llvm.struct<(ptr, i64, i64, i64, i64, i64)>
// CHECK:           %[[VAL_20:.*]] = llvm.extractvalue %[[VAL_7]][4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_21:.*]] = llvm.insertvalue %[[VAL_20]], %[[VAL_19]][5] : !llvm.struct<(ptr, i64, i64, i64, i64, i64)>
// CHECK:           %[[VAL_22:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           %[[VAL_23:.*]] = llvm.getelementptr %[[VAL_22]][1] : (!llvm.ptr) -> !llvm.ptr, f32
// CHECK:           %[[VAL_24:.*]] = llvm.ptrtoint %[[VAL_23]] : !llvm.ptr to i64
// CHECK:           %[[VAL_25:.*]] = llvm.getelementptr inbounds %[[VAL_8]]{{\[}}%[[VAL_24]]] : (!llvm.ptr, i64) -> !llvm.ptr, i8
// CHECK:           %[[VAL_26:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_27:.*]] = llvm.extractvalue %[[VAL_21]][0] : !llvm.struct<(ptr, i64, i64, i64, i64, i64)>
// CHECK:           %[[VAL_28:.*]] = llvm.insertvalue %[[VAL_27]], %[[VAL_26]][0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_29:.*]] = llvm.insertvalue %[[VAL_25]], %[[VAL_28]][1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_30:.*]] = llvm.extractvalue %[[VAL_21]][1] : !llvm.struct<(ptr, i64, i64, i64, i64, i64)>
// CHECK:           %[[VAL_31:.*]] = llvm.insertvalue %[[VAL_30]], %[[VAL_29]][2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_32:.*]] = llvm.extractvalue %[[VAL_21]][2] : !llvm.struct<(ptr, i64, i64, i64, i64, i64)>
// CHECK:           %[[VAL_33:.*]] = llvm.insertvalue %[[VAL_32]], %[[VAL_31]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_34:.*]] = llvm.extractvalue %[[VAL_21]][3] : !llvm.struct<(ptr, i64, i64, i64, i64, i64)>
// CHECK:           %[[VAL_35:.*]] = llvm.insertvalue %[[VAL_34]], %[[VAL_33]][3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_36:.*]] = llvm.extractvalue %[[VAL_21]][4] : !llvm.struct<(ptr, i64, i64, i64, i64, i64)>
// CHECK:           %[[VAL_37:.*]] = llvm.insertvalue %[[VAL_36]], %[[VAL_35]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_38:.*]] = llvm.extractvalue %[[VAL_21]][5] : !llvm.struct<(ptr, i64, i64, i64, i64, i64)>
// CHECK:           %[[VAL_39:.*]] = llvm.insertvalue %[[VAL_38]], %[[VAL_37]][4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           llvm.return %[[VAL_39]] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:         }
func.func @test_comprehensive_dynamic(%arg0: memref<?x?xf32, strided<[?, ?], offset: ?>, #ptr.generic_space>) -> memref<?x?xf32, strided<[?, ?], offset: ?>, #ptr.generic_space> {
  %0 = ptr.to_ptr %arg0 : memref<?x?xf32, strided<[?, ?], offset: ?>, #ptr.generic_space> -> <#ptr.generic_space>
  %1 = ptr.get_metadata %arg0 : memref<?x?xf32, strided<[?, ?], offset: ?>, #ptr.generic_space>
  %2 = ptr.type_offset f32 : index
  %3 = ptr.ptr_add inbounds %0, %2 : !ptr.ptr<#ptr.generic_space>, index
  %4 = ptr.from_ptr %3 metadata %1 : <#ptr.generic_space> -> memref<?x?xf32, strided<[?, ?], offset: ?>, #ptr.generic_space>
  return %4 : memref<?x?xf32, strided<[?, ?], offset: ?>, #ptr.generic_space>
}

// Tests a round-trip conversion of a 0D (scalar) memref
// CHECK-LABEL:   llvm.func @test_memref_0d(
// CHECK-SAME:      %[[ARG0:.*]]: !llvm.ptr, %[[ARG1:.*]]: !llvm.ptr, %[[ARG2:.*]]: i64) -> !llvm.struct<(ptr, ptr, i64)> {
// CHECK:           %[[VAL_0:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64)>
// CHECK:           %[[VAL_1:.*]] = llvm.insertvalue %[[ARG0]], %[[VAL_0]][0] : !llvm.struct<(ptr, ptr, i64)>
// CHECK:           %[[VAL_2:.*]] = llvm.insertvalue %[[ARG1]], %[[VAL_1]][1] : !llvm.struct<(ptr, ptr, i64)>
// CHECK:           %[[VAL_3:.*]] = llvm.insertvalue %[[ARG2]], %[[VAL_2]][2] : !llvm.struct<(ptr, ptr, i64)>
// CHECK:           %[[VAL_4:.*]] = llvm.extractvalue %[[VAL_3]][1] : !llvm.struct<(ptr, ptr, i64)>
// CHECK:           %[[VAL_5:.*]] = llvm.mlir.undef : !llvm.struct<(ptr)>
// CHECK:           %[[VAL_6:.*]] = llvm.extractvalue %[[VAL_3]][0] : !llvm.struct<(ptr, ptr, i64)>
// CHECK:           %[[VAL_7:.*]] = llvm.insertvalue %[[VAL_6]], %[[VAL_5]][0] : !llvm.struct<(ptr)>
// CHECK:           llvm.return %[[VAL_3]] : !llvm.struct<(ptr, ptr, i64)>
// CHECK:         }
func.func @test_memref_0d(%arg0: memref<f32, #ptr.generic_space>) -> memref<f32, #ptr.generic_space> {
  %0 = ptr.to_ptr %arg0 : memref<f32, #ptr.generic_space> -> <#ptr.generic_space>
  %1 = ptr.get_metadata %arg0 : memref<f32, #ptr.generic_space>
  %2 = ptr.from_ptr %0 metadata %1 : <#ptr.generic_space> -> memref<f32, #ptr.generic_space>
  return %2 : memref<f32, #ptr.generic_space>
}

// Tests ptr indexing with a pointer coming from a memref.
// CHECK-LABEL:   llvm.func @test_memref_ptradd_indexing(
// CHECK-SAME:      %[[ARG0:.*]]: !llvm.ptr, %[[ARG1:.*]]: !llvm.ptr, %[[ARG2:.*]]: i64, %[[ARG3:.*]]: i64, %[[ARG4:.*]]: i64, %[[ARG5:.*]]: i64, %[[ARG6:.*]]: i64, %[[ARG7:.*]]: i64, %[[ARG8:.*]]: i64, %[[ARG9:.*]]: i64) -> !llvm.ptr {
// CHECK:           %[[VAL_0:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           %[[VAL_1:.*]] = llvm.insertvalue %[[ARG0]], %[[VAL_0]][0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           %[[VAL_2:.*]] = llvm.insertvalue %[[ARG1]], %[[VAL_1]][1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           %[[VAL_3:.*]] = llvm.insertvalue %[[ARG2]], %[[VAL_2]][2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           %[[VAL_4:.*]] = llvm.insertvalue %[[ARG3]], %[[VAL_3]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           %[[VAL_5:.*]] = llvm.insertvalue %[[ARG6]], %[[VAL_4]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           %[[VAL_6:.*]] = llvm.insertvalue %[[ARG4]], %[[VAL_5]][3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           %[[VAL_7:.*]] = llvm.insertvalue %[[ARG7]], %[[VAL_6]][4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           %[[VAL_8:.*]] = llvm.insertvalue %[[ARG5]], %[[VAL_7]][3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           %[[VAL_9:.*]] = llvm.insertvalue %[[ARG8]], %[[VAL_8]][4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           %[[VAL_10:.*]] = llvm.extractvalue %[[VAL_9]][1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           %[[VAL_11:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           %[[VAL_12:.*]] = llvm.getelementptr %[[VAL_11]][1] : (!llvm.ptr) -> !llvm.ptr, f32
// CHECK:           %[[VAL_13:.*]] = llvm.ptrtoint %[[VAL_12]] : !llvm.ptr to i64
// CHECK:           %[[VAL_14:.*]] = llvm.mul %[[VAL_13]], %[[ARG9]] : i64
// CHECK:           %[[VAL_15:.*]] = llvm.getelementptr %[[VAL_10]]{{\[}}%[[VAL_14]]] : (!llvm.ptr, i64) -> !llvm.ptr, i8
// CHECK:           llvm.return %[[VAL_15]] : !llvm.ptr
// CHECK:         }
func.func @test_memref_ptradd_indexing(%arg0: memref<10x?x30xf32, #ptr.generic_space>, %arg1: index) -> !ptr.ptr<#ptr.generic_space> {
  %0 = ptr.to_ptr %arg0 : memref<10x?x30xf32, #ptr.generic_space> -> <#ptr.generic_space>
  %1 = ptr.type_offset f32 : index
  %2 = arith.muli %1, %arg1 : index
  %3 = ptr.ptr_add %0, %2 : !ptr.ptr<#ptr.generic_space>, index
  return %3 : !ptr.ptr<#ptr.generic_space>
}
