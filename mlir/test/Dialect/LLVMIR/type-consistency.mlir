// RUN: mlir-opt %s --pass-pipeline="builtin.module(llvm.func(llvm-type-consistency))" --split-input-file | FileCheck %s

// CHECK-LABEL: llvm.func @same_address
llvm.func @same_address(%arg: i32) {
  %0 = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %[[ALLOCA:.*]] = llvm.alloca %{{.*}} x !llvm.struct<"foo", (i32, i32, i32)>
  %1 = llvm.alloca %0 x !llvm.struct<"foo", (i32, i32, i32)> : (i32) -> !llvm.ptr
  // CHECK: = llvm.getelementptr %[[ALLOCA]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"foo", (i32, i32, i32)>
  %7 = llvm.getelementptr %1[8] : (!llvm.ptr) -> !llvm.ptr, i8
  llvm.store %arg, %7 : i32, !llvm.ptr
  llvm.return
}

// -----

// CHECK-LABEL: llvm.func @same_address_keep_inbounds
llvm.func @same_address_keep_inbounds(%arg: i32) {
  %0 = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %[[ALLOCA:.*]] = llvm.alloca %{{.*}} x !llvm.struct<"foo", (i32, i32, i32)>
  %1 = llvm.alloca %0 x !llvm.struct<"foo", (i32, i32, i32)> : (i32) -> !llvm.ptr
  // CHECK: = llvm.getelementptr inbounds %[[ALLOCA]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"foo", (i32, i32, i32)>
  %7 = llvm.getelementptr inbounds %1[8] : (!llvm.ptr) -> !llvm.ptr, i8
  llvm.store %arg, %7 : i32, !llvm.ptr
  llvm.return
}

// -----

// CHECK-LABEL: llvm.func @struct_store_instead_of_first_field
llvm.func @struct_store_instead_of_first_field(%arg: i32) {
  %0 = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %[[ALLOCA:.*]] = llvm.alloca %{{.*}} x !llvm.struct<"foo", (i32, i32, i32)>
  %1 = llvm.alloca %0 x !llvm.struct<"foo", (i32, i32, i32)> : (i32) -> !llvm.ptr
  // CHECK: %[[GEP:.*]] = llvm.getelementptr %[[ALLOCA]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"foo", (i32, i32, i32)>
  // CHECK: llvm.store %{{.*}}, %[[GEP]] : i32
  llvm.store %arg, %1 : i32, !llvm.ptr
  llvm.return
}

// -----

// CHECK-LABEL: llvm.func @struct_store_instead_of_first_field_same_size
// CHECK-SAME: (%[[ARG:.*]]: f32)
llvm.func @struct_store_instead_of_first_field_same_size(%arg: f32) {
  %0 = llvm.mlir.constant(1 : i32) : i32
  // CHECK-DAG: %[[ALLOCA:.*]] = llvm.alloca %{{.*}} x !llvm.struct<"foo", (i32, i32, i32)>
  %1 = llvm.alloca %0 x !llvm.struct<"foo", (i32, i32, i32)> : (i32) -> !llvm.ptr
  // CHECK-DAG: %[[GEP:.*]] = llvm.getelementptr %[[ALLOCA]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"foo", (i32, i32, i32)>
  // CHECK-DAG: %[[BITCAST:.*]] = llvm.bitcast %[[ARG]] : f32 to i32
  // CHECK: llvm.store %[[BITCAST]], %[[GEP]] : i32
  llvm.store %arg, %1 : f32, !llvm.ptr
  llvm.return
}

// -----

// CHECK-LABEL: llvm.func @struct_load_instead_of_first_field
llvm.func @struct_load_instead_of_first_field() -> i32 {
  %0 = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %[[ALLOCA:.*]] = llvm.alloca %{{.*}} x !llvm.struct<"foo", (i32, i32, i32)>
  %1 = llvm.alloca %0 x !llvm.struct<"foo", (i32, i32, i32)> : (i32) -> !llvm.ptr
  // CHECK: %[[GEP:.*]] = llvm.getelementptr %[[ALLOCA]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"foo", (i32, i32, i32)>
  // CHECK: %[[RES:.*]] = llvm.load %[[GEP]] : !llvm.ptr -> i32
  %2 = llvm.load %1 : !llvm.ptr -> i32
  // CHECK: llvm.return %[[RES]] : i32
  llvm.return %2 : i32
}

// -----

// CHECK-LABEL: llvm.func @struct_load_instead_of_first_field_same_size
llvm.func @struct_load_instead_of_first_field_same_size() -> f32 {
  %0 = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %[[ALLOCA:.*]] = llvm.alloca %{{.*}} x !llvm.struct<"foo", (i32, i32, i32)>
  %1 = llvm.alloca %0 x !llvm.struct<"foo", (i32, i32, i32)> : (i32) -> !llvm.ptr
  // CHECK: %[[GEP:.*]] = llvm.getelementptr %[[ALLOCA]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"foo", (i32, i32, i32)>
  // CHECK: %[[LOADED:.*]] = llvm.load %[[GEP]] : !llvm.ptr -> i32
  // CHECK: %[[RES:.*]] = llvm.bitcast %[[LOADED]] : i32 to f32
  %2 = llvm.load %1 : !llvm.ptr -> f32
  // CHECK: llvm.return %[[RES]] : f32
  llvm.return %2 : f32
}

// -----

// CHECK-LABEL: llvm.func @index_in_final_padding
llvm.func @index_in_final_padding(%arg: i32) {
  %0 = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %[[ALLOCA:.*]] = llvm.alloca %{{.*}} x !llvm.struct<"foo", (i32, i8)>
  %1 = llvm.alloca %0 x !llvm.struct<"foo", (i32, i8)> : (i32) -> !llvm.ptr
  // CHECK: = llvm.getelementptr %[[ALLOCA]][7] : (!llvm.ptr) -> !llvm.ptr, i8
  %7 = llvm.getelementptr %1[7] : (!llvm.ptr) -> !llvm.ptr, i8
  llvm.store %arg, %7 : i32, !llvm.ptr
  llvm.return
}

// -----

// CHECK-LABEL: llvm.func @index_out_of_bounds
llvm.func @index_out_of_bounds(%arg: i32) {
  %0 = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %[[ALLOCA:.*]] = llvm.alloca %{{.*}} x !llvm.struct<"foo", (i32, i32)>
  %1 = llvm.alloca %0 x !llvm.struct<"foo", (i32, i32)> : (i32) -> !llvm.ptr
  // CHECK: = llvm.getelementptr %[[ALLOCA]][9] : (!llvm.ptr) -> !llvm.ptr, i8
  %7 = llvm.getelementptr %1[9] : (!llvm.ptr) -> !llvm.ptr, i8
  llvm.store %arg, %7 : i32, !llvm.ptr
  llvm.return
}

// -----

// CHECK-LABEL: llvm.func @index_in_padding
llvm.func @index_in_padding(%arg: i16) {
  %0 = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %[[ALLOCA:.*]] = llvm.alloca %{{.*}} x !llvm.struct<"foo", (i16, i32)>
  %1 = llvm.alloca %0 x !llvm.struct<"foo", (i16, i32)> : (i32) -> !llvm.ptr
  // CHECK: = llvm.getelementptr %[[ALLOCA]][2] : (!llvm.ptr) -> !llvm.ptr, i8
  %7 = llvm.getelementptr %1[2] : (!llvm.ptr) -> !llvm.ptr, i8
  llvm.store %arg, %7 : i16, !llvm.ptr
  llvm.return
}

// -----

// CHECK-LABEL: llvm.func @index_not_in_padding_because_packed
llvm.func @index_not_in_padding_because_packed(%arg: i16) {
  %0 = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %[[ALLOCA:.*]] = llvm.alloca %{{.*}} x !llvm.struct<"foo", packed (i16, i32)>
  %1 = llvm.alloca %0 x !llvm.struct<"foo", packed (i16, i32)> : (i32) -> !llvm.ptr
  // CHECK: = llvm.getelementptr %[[ALLOCA]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"foo", packed (i16, i32)>
  %7 = llvm.getelementptr %1[2] : (!llvm.ptr) -> !llvm.ptr, i8
  llvm.store %arg, %7 : i16, !llvm.ptr
  llvm.return
}

// -----

// CHECK-LABEL: llvm.func @index_to_struct
// CHECK-SAME: (%[[ARG:.*]]: i32)
llvm.func @index_to_struct(%arg: i32) {
  %0 = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %[[ALLOCA:.*]] = llvm.alloca %{{.*}} x !llvm.struct<"foo", (i32, struct<"bar", (i32, i32)>)>
  %1 = llvm.alloca %0 x !llvm.struct<"foo", (i32, struct<"bar", (i32, i32)>)> : (i32) -> !llvm.ptr
  // CHECK: %[[GEP0:.*]] = llvm.getelementptr %[[ALLOCA]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"foo", (i32, struct<"bar", (i32, i32)>)>
  // CHECK: %[[GEP1:.*]] = llvm.getelementptr %[[GEP0]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"bar", (i32, i32)>
  %7 = llvm.getelementptr %1[4] : (!llvm.ptr) -> !llvm.ptr, i8
  // CHECK: llvm.store %[[ARG]], %[[GEP1]]
  llvm.store %arg, %7 : i32, !llvm.ptr
  llvm.return
}

// -----

// CHECK-LABEL: llvm.func @no_crash_on_negative_gep_index
llvm.func @no_crash_on_negative_gep_index() {
  %0 = llvm.mlir.constant(1.000000e+00 : f16) : f16
  %1 = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %[[ALLOCA:.*]] = llvm.alloca %{{.*}} x !llvm.struct<"foo", (i32, i32, i32)>
  %2 = llvm.alloca %1 x !llvm.struct<"foo", (i32, i32, i32)> : (i32) -> !llvm.ptr
  // CHECK: llvm.getelementptr %[[ALLOCA]][-1] : (!llvm.ptr) -> !llvm.ptr, f32
  %3 = llvm.getelementptr %2[-1] : (!llvm.ptr) -> !llvm.ptr, f32
  llvm.store %0, %3 : f16, !llvm.ptr
  llvm.return
}

// -----

// CHECK-LABEL: llvm.func @coalesced_store_ints
// CHECK-SAME: %[[ARG:.*]]: i64
llvm.func @coalesced_store_ints(%arg: i64) {
  // CHECK-DAG: %[[CST0:.*]] = llvm.mlir.constant(0 : i64) : i64
  // CHECK-DAG: %[[CST32:.*]] = llvm.mlir.constant(32 : i64) : i64

  %0 = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %[[ALLOCA:.*]] = llvm.alloca %{{.*}} x !llvm.struct<"foo", (i32, i32)>
  %1 = llvm.alloca %0 x !llvm.struct<"foo", (i32, i32)> : (i32) -> !llvm.ptr

  // CHECK: %[[GEP:.*]] = llvm.getelementptr %[[ALLOCA]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"foo", (i32, i32)>
  // CHECK: %[[SHR:.*]] = llvm.lshr %[[ARG]], %[[CST0]]
  // CHECK: %[[TRUNC:.*]] = llvm.trunc %[[SHR]] : i64 to i32
  // CHECK: llvm.store %[[TRUNC]], %[[GEP]]
  // CHECK: %[[SHR:.*]] = llvm.lshr %[[ARG]], %[[CST32]] : i64
  // CHECK: %[[TRUNC:.*]] = llvm.trunc %[[SHR]] : i64 to i32
  // CHECK: %[[GEP:.*]] = llvm.getelementptr %[[ALLOCA]][0, 1] : (!llvm.ptr)  -> !llvm.ptr, !llvm.struct<"foo", (i32, i32)>
  // CHECK: llvm.store %[[TRUNC]], %[[GEP]]
  llvm.store %arg, %1 : i64, !llvm.ptr
  // CHECK-NOT: llvm.store %[[ARG]], %[[ALLOCA]]
  llvm.return
}

// -----

// CHECK-LABEL: llvm.func @coalesced_store_ints_offset
// CHECK-SAME: %[[ARG:.*]]: i64
llvm.func @coalesced_store_ints_offset(%arg: i64) {
  // CHECK-DAG: %[[CST0:.*]] = llvm.mlir.constant(0 : i64) : i64
  // CHECK-DAG: %[[CST32:.*]] = llvm.mlir.constant(32 : i64) : i64
  %0 = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %[[ALLOCA:.*]] = llvm.alloca %{{.*}} x !llvm.struct<"foo", (i64, i32, i32)>
  %1 = llvm.alloca %0 x !llvm.struct<"foo", (i64, i32, i32)> : (i32) -> !llvm.ptr
  %3 = llvm.getelementptr %1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"foo", (i64, i32, i32)>

  // CHECK: %[[SHR:.*]] = llvm.lshr %[[ARG]], %[[CST0]]
  // CHECK: %[[TRUNC:.*]] = llvm.trunc %[[SHR]] : i64 to i32
  // CHECK: %[[GEP:.*]] = llvm.getelementptr %[[ALLOCA]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"foo", (i64, i32, i32)>
  // CHECK: llvm.store %[[TRUNC]], %[[GEP]]
  // CHECK: %[[SHR:.*]] = llvm.lshr %[[ARG]], %[[CST32]] : i64
  // CHECK: %[[TRUNC:.*]] = llvm.trunc %[[SHR]] : i64 to i32
  // CHECK: %[[GEP:.*]] = llvm.getelementptr %[[ALLOCA]][0, 2] : (!llvm.ptr)  -> !llvm.ptr, !llvm.struct<"foo", (i64, i32, i32)>
  // CHECK: llvm.store %[[TRUNC]], %[[GEP]]
  llvm.store %arg, %3 : i64, !llvm.ptr
  // CHECK-NOT: llvm.store %[[ARG]], %[[ALLOCA]]
  llvm.return
}

// -----

// CHECK-LABEL: llvm.func @coalesced_store_floats
// CHECK-SAME: %[[ARG:.*]]: i64
llvm.func @coalesced_store_floats(%arg: i64) {
  // CHECK-DAG: %[[CST0:.*]] = llvm.mlir.constant(0 : i64) : i64
  // CHECK-DAG: %[[CST32:.*]] = llvm.mlir.constant(32 : i64) : i64
  %0 = llvm.mlir.constant(1 : i32) : i32

  // CHECK: %[[ALLOCA:.*]] = llvm.alloca %{{.*}} x !llvm.struct<"foo", (f32, f32)>
  %1 = llvm.alloca %0 x !llvm.struct<"foo", (f32, f32)> : (i32) -> !llvm.ptr

  // CHECK: %[[GEP:.*]] = llvm.getelementptr %[[ALLOCA]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"foo", (f32, f32)>
  // CHECK: %[[SHR:.*]] = llvm.lshr %[[ARG]], %[[CST0]]
  // CHECK: %[[TRUNC:.*]] = llvm.trunc %[[SHR]] : i64 to i32
  // CHECK: %[[BIT_CAST:.*]] = llvm.bitcast %[[TRUNC]] : i32 to f32
  // CHECK: llvm.store %[[BIT_CAST]], %[[GEP]]
  // CHECK: %[[SHR:.*]] = llvm.lshr %[[ARG]], %[[CST32]] : i64
  // CHECK: %[[TRUNC:.*]] = llvm.trunc %[[SHR]] : i64 to i32
  // CHECK: %[[GEP:.*]] = llvm.getelementptr %[[ALLOCA]][0, 1] : (!llvm.ptr)  -> !llvm.ptr, !llvm.struct<"foo", (f32, f32)>
  // CHECK: %[[BIT_CAST:.*]] = llvm.bitcast %[[TRUNC]] : i32 to f32
  // CHECK: llvm.store %[[BIT_CAST]], %[[GEP]]
  llvm.store %arg, %1 : i64, !llvm.ptr
  // CHECK-NOT: llvm.store %[[ARG]], %[[ALLOCA]]
  llvm.return
}

// -----

// Padding test purposefully not modified.

// CHECK-LABEL: llvm.func @coalesced_store_padding_inbetween
// CHECK-SAME: %[[ARG:.*]]: i64
llvm.func @coalesced_store_padding_inbetween(%arg: i64) {
  %0 = llvm.mlir.constant(1 : i32) : i32

  // CHECK: %[[ALLOCA:.*]] = llvm.alloca %{{.*}} x !llvm.struct<"foo", (i16, i32)>
  %1 = llvm.alloca %0 x !llvm.struct<"foo", (i16, i32)> : (i32) -> !llvm.ptr
  // CHECK: llvm.store %[[ARG]], %[[ALLOCA]]
  llvm.store %arg, %1 : i64, !llvm.ptr
  llvm.return
}

// -----

// Padding test purposefully not modified.

// CHECK-LABEL: llvm.func @coalesced_store_padding_end
// CHECK-SAME: %[[ARG:.*]]: i64
llvm.func @coalesced_store_padding_end(%arg: i64) {
  %0 = llvm.mlir.constant(1 : i32) : i32

  // CHECK: %[[ALLOCA:.*]] = llvm.alloca %{{.*}} x !llvm.struct<"foo", (i32, i16)>
  %1 = llvm.alloca %0 x !llvm.struct<"foo", (i32, i16)> : (i32) -> !llvm.ptr
  // CHECK: llvm.store %[[ARG]], %[[ALLOCA]]
  llvm.store %arg, %1 : i64, !llvm.ptr
  llvm.return
}

// -----

// CHECK-LABEL: llvm.func @coalesced_store_past_end
// CHECK-SAME: %[[ARG:.*]]: i64
llvm.func @coalesced_store_past_end(%arg: i64) {
  %0 = llvm.mlir.constant(1 : i32) : i32

  // CHECK: %[[ALLOCA:.*]] = llvm.alloca %{{.*}} x !llvm.struct<"foo", (i32)>
  %1 = llvm.alloca %0 x !llvm.struct<"foo", (i32)> : (i32) -> !llvm.ptr
  // CHECK: llvm.store %[[ARG]], %[[ALLOCA]]
  llvm.store %arg, %1 : i64, !llvm.ptr
  llvm.return
}

// -----

// CHECK-LABEL: llvm.func @coalesced_store_packed_struct
// CHECK-SAME: %[[ARG:.*]]: i64
llvm.func @coalesced_store_packed_struct(%arg: i64) {
  %0 = llvm.mlir.constant(1 : i32) : i32
  // CHECK-DAG: %[[CST0:.*]] = llvm.mlir.constant(0 : i64) : i64
  // CHECK-DAG: %[[CST16:.*]] = llvm.mlir.constant(16 : i64) : i64
  // CHECK-DAG: %[[CST48:.*]] = llvm.mlir.constant(48 : i64) : i64

  // CHECK: %[[ALLOCA:.*]] = llvm.alloca %{{.*}} x !llvm.struct<"foo", packed (i16, i32, i16)>
  %1 = llvm.alloca %0 x !llvm.struct<"foo", packed (i16, i32, i16)> : (i32) -> !llvm.ptr
  // CHECK: %[[GEP:.*]] = llvm.getelementptr %[[ALLOCA]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"foo", packed (i16, i32, i16)>
  // CHECK: %[[SHR:.*]] = llvm.lshr %[[ARG]], %[[CST0]]
  // CHECK: %[[TRUNC:.*]] = llvm.trunc %[[SHR]] : i64 to i16
  // CHECK: llvm.store %[[TRUNC]], %[[GEP]]
  // CHECK: %[[SHR:.*]] = llvm.lshr %[[ARG]], %[[CST16]]
  // CHECK: %[[TRUNC:.*]] = llvm.trunc %[[SHR]] : i64 to i32
  // CHECK: %[[GEP:.*]] = llvm.getelementptr %[[ALLOCA]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"foo", packed (i16, i32, i16)>
  // CHECK: llvm.store %[[TRUNC]], %[[GEP]]
  // CHECK: %[[SHR:.*]] = llvm.lshr %[[ARG]], %[[CST48]]
  // CHECK: %[[TRUNC:.*]] = llvm.trunc %[[SHR]] : i64 to i16
  // CHECK: %[[GEP:.*]] = llvm.getelementptr %[[ALLOCA]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"foo", packed (i16, i32, i16)>
  // CHECK: llvm.store %[[TRUNC]], %[[GEP]]
  llvm.store %arg, %1 : i64, !llvm.ptr
  // CHECK-NOT: llvm.store %[[ARG]], %[[ALLOCA]]
  llvm.return
}

// -----

// CHECK-LABEL: llvm.func @vector_write_split
// CHECK-SAME: %[[ARG:.*]]: vector<4xi32>
llvm.func @vector_write_split(%arg: vector<4xi32>) {
  // CHECK-DAG: %[[CST0:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK-DAG: %[[CST1:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-DAG: %[[CST2:.*]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK-DAG: %[[CST3:.*]] = llvm.mlir.constant(3 : i32) : i32
  %0 = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %[[ALLOCA:.*]] = llvm.alloca %{{.*}} x !llvm.struct<"foo", (i32, i32, i32, i32)>
  %1 = llvm.alloca %0 x !llvm.struct<"foo", (i32, i32, i32, i32)> : (i32) -> !llvm.ptr

  // CHECK: %[[GEP:.*]] = llvm.getelementptr %[[ALLOCA]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"foo", (i32, i32, i32, i32)>
  // CHECK: %[[EXTRACT:.*]] = llvm.extractelement %[[ARG]][%[[CST0]] : i32] : vector<4xi32>
  // CHECK: llvm.store %[[EXTRACT]], %[[GEP]] : i32, !llvm.ptr

  // CHECK: %[[EXTRACT:.*]] = llvm.extractelement %[[ARG]][%[[CST1]] : i32] : vector<4xi32>
  // CHECK: %[[GEP:.*]] = llvm.getelementptr %[[ALLOCA]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"foo", (i32, i32, i32, i32)>
  // CHECK: llvm.store %[[EXTRACT]], %[[GEP]] : i32, !llvm.ptr

  // CHECK: %[[EXTRACT:.*]] = llvm.extractelement %[[ARG]][%[[CST2]] : i32] : vector<4xi32>
  // CHECK: %[[GEP:.*]] = llvm.getelementptr %[[ALLOCA]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"foo", (i32, i32, i32, i32)>
  // CHECK: llvm.store %[[EXTRACT]], %[[GEP]] : i32, !llvm.ptr

  // CHECK: %[[EXTRACT:.*]] = llvm.extractelement %[[ARG]][%[[CST3]] : i32] : vector<4xi32>
  // CHECK: %[[GEP:.*]] = llvm.getelementptr %[[ALLOCA]][0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"foo", (i32, i32, i32, i32)>
  // CHECK: llvm.store %[[EXTRACT]], %[[GEP]] : i32, !llvm.ptr

  llvm.store %arg, %1 : vector<4xi32>, !llvm.ptr
  // CHECK-NOT: llvm.store %[[ARG]], %[[ALLOCA]]
  llvm.return
}

// -----

// CHECK-LABEL: llvm.func @vector_write_split_offset
// CHECK-SAME: %[[ARG:.*]]: vector<4xi32>
llvm.func @vector_write_split_offset(%arg: vector<4xi32>) {
  // CHECK-DAG: %[[CST0:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK-DAG: %[[CST1:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-DAG: %[[CST2:.*]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK-DAG: %[[CST3:.*]] = llvm.mlir.constant(3 : i32) : i32
  %0 = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %[[ALLOCA:.*]] = llvm.alloca %{{.*}} x !llvm.struct<"foo", (i64, i32, i32, i32, i32)>
  %1 = llvm.alloca %0 x !llvm.struct<"foo", (i64, i32, i32, i32, i32)> : (i32) -> !llvm.ptr
  %2 = llvm.getelementptr %1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"foo", (i64, i32, i32, i32, i32)>

  // CHECK: %[[EXTRACT:.*]] = llvm.extractelement %[[ARG]][%[[CST0]] : i32] : vector<4xi32>
  // CHECK: %[[GEP:.*]] = llvm.getelementptr %[[ALLOCA]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"foo", (i64, i32, i32, i32, i32)>
  // CHECK: llvm.store %[[EXTRACT]], %[[GEP]] : i32, !llvm.ptr

  // CHECK: %[[EXTRACT:.*]] = llvm.extractelement %[[ARG]][%[[CST1]] : i32] : vector<4xi32>
  // CHECK: %[[GEP:.*]] = llvm.getelementptr %[[ALLOCA]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"foo", (i64, i32, i32, i32, i32)>
  // CHECK: llvm.store %[[EXTRACT]], %[[GEP]] : i32, !llvm.ptr

  // CHECK: %[[EXTRACT:.*]] = llvm.extractelement %[[ARG]][%[[CST2]] : i32] : vector<4xi32>
  // CHECK: %[[GEP:.*]] = llvm.getelementptr %[[ALLOCA]][0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"foo", (i64, i32, i32, i32, i32)>
  // CHECK: llvm.store %[[EXTRACT]], %[[GEP]] : i32, !llvm.ptr

  // CHECK: %[[EXTRACT:.*]] = llvm.extractelement %[[ARG]][%[[CST3]] : i32] : vector<4xi32>
  // CHECK: %[[GEP:.*]] = llvm.getelementptr %[[ALLOCA]][0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"foo", (i64, i32, i32, i32, i32)>
  // CHECK: llvm.store %[[EXTRACT]], %[[GEP]] : i32, !llvm.ptr

  llvm.store %arg, %2 : vector<4xi32>, !llvm.ptr
  // CHECK-NOT: llvm.store %[[ARG]], %[[ALLOCA]]
  llvm.return
}

// -----

// Small test that a split vector store will be further optimized (to than e.g.
// split integer loads to structs as shown here)

// CHECK-LABEL: llvm.func @vector_write_split_struct
// CHECK-SAME: %[[ARG:.*]]: vector<2xi64>
llvm.func @vector_write_split_struct(%arg: vector<2xi64>) {
  %0 = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %[[ALLOCA:.*]] = llvm.alloca %{{.*}} x !llvm.struct<"foo", (i32, i32, i32, i32)>
  %1 = llvm.alloca %0 x !llvm.struct<"foo", (i32, i32, i32, i32)> : (i32) -> !llvm.ptr

  // CHECK-COUNT-4: llvm.store %{{.*}}, %{{.*}} : i32, !llvm.ptr

  llvm.store %arg, %1 : vector<2xi64>, !llvm.ptr
  // CHECK-NOT: llvm.store %[[ARG]], %[[ALLOCA]]
  llvm.return
}

// -----

// CHECK-LABEL: llvm.func @type_consistent_vector_store
// CHECK-SAME: %[[ARG:.*]]: vector<4xi32>
llvm.func @type_consistent_vector_store(%arg: vector<4xi32>) {
  %0 = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %[[ALLOCA:.*]] = llvm.alloca %{{.*}} x !llvm.struct<"foo", (vector<4xi32>)>
  %1 = llvm.alloca %0 x !llvm.struct<"foo", (vector<4xi32>)> : (i32) -> !llvm.ptr
  // CHECK: %[[GEP:.*]] = llvm.getelementptr %[[ALLOCA]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"foo", (vector<4xi32>)>
  // CHECK: llvm.store %[[ARG]], %[[GEP]]
  llvm.store %arg, %1 : vector<4xi32>, !llvm.ptr
  llvm.return
}

// -----

// CHECK-LABEL: llvm.func @type_consistent_vector_store_other_type
// CHECK-SAME: %[[ARG:.*]]: vector<4xi32>
llvm.func @type_consistent_vector_store_other_type(%arg: vector<4xi32>) {
  %0 = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %[[ALLOCA:.*]] = llvm.alloca %{{.*}} x !llvm.struct<"foo", (vector<4xf32>)>
  %1 = llvm.alloca %0 x !llvm.struct<"foo", (vector<4xf32>)> : (i32) -> !llvm.ptr
  // CHECK: %[[GEP:.*]] = llvm.getelementptr %[[ALLOCA]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"foo", (vector<4xf32>)>
  // CHECK: %[[BIT_CAST:.*]] = llvm.bitcast %[[ARG]] : vector<4xi32> to vector<4xf32>
  // CHECK: llvm.store %[[BIT_CAST]], %[[GEP]]
  llvm.store %arg, %1 : vector<4xi32>, !llvm.ptr
  // CHECK-NOT: llvm.store %[[ARG]], %[[ALLOCA]]
  llvm.return
}

// -----

// CHECK-LABEL: llvm.func @bitcast_insertion
// CHECK-SAME: %[[ARG:.*]]: i32
llvm.func @bitcast_insertion(%arg: i32) {
  %0 = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %[[ALLOCA:.*]] = llvm.alloca %{{.*}} x f32
  %1 = llvm.alloca %0 x f32 : (i32) -> !llvm.ptr
  // CHECK: %[[BIT_CAST:.*]] = llvm.bitcast %[[ARG]] : i32 to f32
  // CHECK: llvm.store %[[BIT_CAST]], %[[ALLOCA]]
  llvm.store %arg, %1 : i32, !llvm.ptr
  // CHECK-NOT: llvm.store %[[ARG]], %[[ALLOCA]]
  llvm.return
}

// -----

// CHECK-LABEL: llvm.func @gep_split
// CHECK-SAME: %[[ARG:.*]]: i64
llvm.func @gep_split(%arg: i64) {
  %0 = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %[[ALLOCA:.*]] = llvm.alloca %{{.*}} x !llvm.array<2 x struct<"foo", (i64)>>
  %1 = llvm.alloca %0 x !llvm.array<2 x struct<"foo", (i64)>> : (i32) -> !llvm.ptr
  %3 = llvm.getelementptr %1[0, 1, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x struct<"foo", (i64)>>
  // CHECK: %[[TOP_GEP:.*]] = llvm.getelementptr %[[ALLOCA]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x struct<"foo", (i64)>>
  // CHECK: %[[GEP:.*]] = llvm.getelementptr %[[TOP_GEP]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"foo", (i64)>
  // CHECK: llvm.store %[[ARG]], %[[GEP]]
  llvm.store %arg, %3 : i64, !llvm.ptr
  // CHECK-NOT: llvm.store %[[ARG]], %[[ALLOCA]]
  llvm.return
}

// -----

// CHECK-LABEL: llvm.func @coalesced_store_ints_subaggregate
// CHECK-SAME: %[[ARG:.*]]: i64
llvm.func @coalesced_store_ints_subaggregate(%arg: i64) {
  // CHECK-DAG: %[[CST0:.*]] = llvm.mlir.constant(0 : i64) : i64
  // CHECK-DAG: %[[CST32:.*]] = llvm.mlir.constant(32 : i64) : i64
  %0 = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %[[ALLOCA:.*]] = llvm.alloca %{{.*}} x !llvm.struct<"foo", (i64, struct<(i32, i32)>)>
  %1 = llvm.alloca %0 x !llvm.struct<"foo", (i64, struct<(i32, i32)>)> : (i32) -> !llvm.ptr
  %3 = llvm.getelementptr %1[0, 1, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"foo", (i64, struct<(i32, i32)>)>

  // CHECK: %[[TOP_GEP:.*]] = llvm.getelementptr %[[ALLOCA]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"foo", (i64, struct<(i32, i32)>)>
  // CHECK: %[[GEP:.*]] = llvm.getelementptr %[[TOP_GEP]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32)>
  // CHECK: %[[SHR:.*]] = llvm.lshr %[[ARG]], %[[CST0]]
  // CHECK: %[[TRUNC:.*]] = llvm.trunc %[[SHR]] : i64 to i32
  // CHECK: llvm.store %[[TRUNC]], %[[GEP]]
  // CHECK: %[[SHR:.*]] = llvm.lshr %[[ARG]], %[[CST32]] : i64
  // CHECK: %[[TRUNC:.*]] = llvm.trunc %[[SHR]] : i64 to i32
  // CHECK: %[[GEP:.*]] = llvm.getelementptr %[[TOP_GEP]][0, 1] : (!llvm.ptr)  -> !llvm.ptr, !llvm.struct<(i32, i32)>
  // CHECK: llvm.store %[[TRUNC]], %[[GEP]]
  llvm.store %arg, %3 : i64, !llvm.ptr
  // CHECK-NOT: llvm.store %[[ARG]], %[[ALLOCA]]
  llvm.return
}

// -----

// CHECK-LABEL: llvm.func @gep_result_ptr_type_dynamic
// CHECK-SAME: %[[ARG:.*]]: i64
llvm.func @gep_result_ptr_type_dynamic(%arg: i64) {
  %0 = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %[[ALLOCA:.*]] = llvm.alloca %{{.*}} x !llvm.array<2 x struct<"foo", (i64)>>
  %1 = llvm.alloca %0 x !llvm.array<2 x struct<"foo", (i64)>> : (i32) -> !llvm.ptr
  %3 = llvm.getelementptr %1[0, %arg, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<2 x struct<"foo", (i64)>>
  // CHECK: %[[TOP_GEP:.*]] = llvm.getelementptr %[[ALLOCA]][0, %[[ARG]]] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<2 x struct<"foo", (i64)>>
  // CHECK: %[[GEP:.*]] = llvm.getelementptr %[[TOP_GEP]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"foo", (i64)>
  // CHECK: llvm.store %[[ARG]], %[[GEP]]
  llvm.store %arg, %3 : i64, !llvm.ptr
  // CHECK-NOT: llvm.store %[[ARG]], %[[ALLOCA]]
  llvm.return
}

// -----

// CHECK-LABEL: llvm.func @overlapping_int_aggregate_store
// CHECK-SAME: %[[ARG:.*]]: i64
llvm.func @overlapping_int_aggregate_store(%arg: i64) {
  // CHECK-DAG: %[[CST0:.*]] = llvm.mlir.constant(0 : i64) : i64
  // CHECK-DAG: %[[CST16:.*]] = llvm.mlir.constant(16 : i64) : i64

  %0 = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %[[ALLOCA:.*]] = llvm.alloca %{{.*}} x !llvm.struct<"foo", (i16, struct<(i16, i16, i16)>)>
  %1 = llvm.alloca %0 x !llvm.struct<"foo", (i16, struct<(i16, i16, i16)>)> : (i32) -> !llvm.ptr

  // CHECK: %[[GEP:.*]] = llvm.getelementptr %[[ALLOCA]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"foo", (i16, struct<(i16, i16, i16)>)>
  // CHECK: %[[SHR:.*]] = llvm.lshr %[[ARG]], %[[CST0]]
  // CHECK: %[[TRUNC:.*]] = llvm.trunc %[[SHR]] : i64 to i16
  // CHECK: llvm.store %[[TRUNC]], %[[GEP]]

  // CHECK: %[[SHR:.*]] = llvm.lshr %[[ARG]], %[[CST16]] : i64
  // CHECK: [[TRUNC:.*]] = llvm.trunc %[[SHR]] : i64 to i48
  // CHECK: %[[TOP_GEP:.*]] = llvm.getelementptr %[[ALLOCA]][0, 1] : (!llvm.ptr)  -> !llvm.ptr, !llvm.struct<"foo", (i16, struct<(i16, i16, i16)>)>

  // Normal integer splitting of [[TRUNC]] follows:

  // CHECK: %[[GEP:.*]] = llvm.getelementptr %[[TOP_GEP]][0, 0] : (!llvm.ptr)  -> !llvm.ptr, !llvm.struct<(i16, i16, i16)>
  // CHECK: llvm.store %{{.*}}, %[[GEP]]
  // CHECK: %[[GEP:.*]] = llvm.getelementptr %[[TOP_GEP]][0, 1] : (!llvm.ptr)  -> !llvm.ptr, !llvm.struct<(i16, i16, i16)>
  // CHECK: llvm.store %{{.*}}, %[[GEP]]
  // CHECK: %[[GEP:.*]] = llvm.getelementptr %[[TOP_GEP]][0, 2] : (!llvm.ptr)  -> !llvm.ptr, !llvm.struct<(i16, i16, i16)>
  // CHECK: llvm.store %{{.*}}, %[[GEP]]

  llvm.store %arg, %1 : i64, !llvm.ptr
  // CHECK-NOT: llvm.store %[[ARG]], %[[ALLOCA]]
  llvm.return
}

// -----

// CHECK-LABEL: llvm.func @overlapping_vector_aggregate_store
// CHECK-SAME: %[[ARG:.*]]: vector<4xi16>
llvm.func @overlapping_vector_aggregate_store(%arg: vector<4 x i16>) {
  // CHECK-DAG: %[[CST0:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK-DAG: %[[CST1:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-DAG: %[[CST2:.*]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK-DAG: %[[CST3:.*]] = llvm.mlir.constant(3 : i32) : i32

  %0 = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %[[ALLOCA:.*]] = llvm.alloca %{{.*}} x !llvm.struct<"foo", (i16, struct<(i16, i16, i16)>)>
  %1 = llvm.alloca %0 x !llvm.struct<"foo", (i16, struct<(i16, i16, i16)>)> : (i32) -> !llvm.ptr

  // CHECK: %[[GEP:.*]] = llvm.getelementptr %[[ALLOCA]][0, 0] : (!llvm.ptr)  -> !llvm.ptr, !llvm.struct<"foo", (i16, struct<(i16, i16, i16)>)>
  // CHECK: %[[EXTRACT:.*]] = llvm.extractelement %[[ARG]][%[[CST0]] : i32]
  // CHECK: llvm.store %[[EXTRACT]], %[[GEP]]

  // CHECK: %[[EXTRACT:.*]] = llvm.extractelement %[[ARG]][%[[CST1]] : i32]
  // CHECK: %[[GEP0:.*]] = llvm.getelementptr %[[ALLOCA]][0, 1] : (!llvm.ptr)  -> !llvm.ptr, !llvm.struct<"foo", (i16, struct<(i16, i16, i16)>)>
  // CHECK: %[[GEP1:.*]] = llvm.getelementptr %[[GEP0]][0, 0] : (!llvm.ptr)  -> !llvm.ptr, !llvm.struct<(i16, i16, i16)>
  // CHECK: llvm.store %[[EXTRACT]], %[[GEP1]]

  // CHECK: %[[EXTRACT:.*]] = llvm.extractelement %[[ARG]][%[[CST2]] : i32]
  // CHECK: %[[GEP0:.*]] = llvm.getelementptr %[[ALLOCA]][0, 1] : (!llvm.ptr)  -> !llvm.ptr, !llvm.struct<"foo", (i16, struct<(i16, i16, i16)>)>
  // CHECK: %[[GEP1:.*]] = llvm.getelementptr %[[GEP0]][0, 1] : (!llvm.ptr)  -> !llvm.ptr, !llvm.struct<(i16, i16, i16)>
  // CHECK: llvm.store %[[EXTRACT]], %[[GEP1]]

  // CHECK: %[[EXTRACT:.*]] = llvm.extractelement %[[ARG]][%[[CST3]] : i32]
  // CHECK: %[[GEP0:.*]] = llvm.getelementptr %[[ALLOCA]][0, 1] : (!llvm.ptr)  -> !llvm.ptr, !llvm.struct<"foo", (i16, struct<(i16, i16, i16)>)>
  // CHECK: %[[GEP1:.*]] = llvm.getelementptr %[[GEP0]][0, 2] : (!llvm.ptr)  -> !llvm.ptr, !llvm.struct<(i16, i16, i16)>
  // CHECK: llvm.store %[[EXTRACT]], %[[GEP1]]

  llvm.store %arg, %1 : vector<4 x i16>, !llvm.ptr
  // CHECK-NOT: llvm.store %[[ARG]], %[[ALLOCA]]
  llvm.return
}

// -----

// CHECK-LABEL: llvm.func @partially_overlapping_aggregate_store
// CHECK-SAME: %[[ARG:.*]]: i64
llvm.func @partially_overlapping_aggregate_store(%arg: i64) {
  // CHECK-DAG: %[[CST0:.*]] = llvm.mlir.constant(0 : i64) : i64
  // CHECK-DAG: %[[CST16:.*]] = llvm.mlir.constant(16 : i64) : i64

  %0 = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %[[ALLOCA:.*]] = llvm.alloca %{{.*}} x !llvm.struct<"foo", (i16, struct<(i16, i16, i16, i16)>)>
  %1 = llvm.alloca %0 x !llvm.struct<"foo", (i16, struct<(i16, i16, i16, i16)>)> : (i32) -> !llvm.ptr

  // CHECK: %[[GEP:.*]] = llvm.getelementptr %[[ALLOCA]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"foo", (i16, struct<(i16, i16, i16, i16)>)>
  // CHECK: %[[SHR:.*]] = llvm.lshr %[[ARG]], %[[CST0]]
  // CHECK: %[[TRUNC:.*]] = llvm.trunc %[[SHR]] : i64 to i16
  // CHECK: llvm.store %[[TRUNC]], %[[GEP]]

  // CHECK: %[[SHR:.*]] = llvm.lshr %[[ARG]], %[[CST16]] : i64
  // CHECK: [[TRUNC:.*]] = llvm.trunc %[[SHR]] : i64 to i48
  // CHECK: %[[TOP_GEP:.*]] = llvm.getelementptr %[[ALLOCA]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"foo", (i16, struct<(i16, i16, i16, i16)>)>

  // Normal integer splitting of [[TRUNC]] follows:

  // CHECK: %[[GEP:.*]] = llvm.getelementptr %[[TOP_GEP]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i16, i16, i16, i16)>
  // CHECK: llvm.store %{{.*}}, %[[GEP]]
  // CHECK: %[[GEP:.*]] = llvm.getelementptr %[[TOP_GEP]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i16, i16, i16, i16)>
  // CHECK: llvm.store %{{.*}}, %[[GEP]]
  // CHECK: %[[GEP:.*]] = llvm.getelementptr %[[TOP_GEP]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i16, i16, i16, i16)>
  // CHECK: llvm.store %{{.*}}, %[[GEP]]

  // It is important that there are no more stores at this point.
  // Specifically a store into the fourth field of %[[TOP_GEP]] would
  // incorrectly change the semantics of the code.
  // CHECK-NOT: llvm.store %{{.*}}, %{{.*}}

  llvm.store %arg, %1 : i64, !llvm.ptr

  llvm.return
}

// -----

// Here a split is undesirable since the store does a partial store into the field.

// CHECK-LABEL: llvm.func @undesirable_overlapping_aggregate_store
// CHECK-SAME: %[[ARG:.*]]: i64
llvm.func @undesirable_overlapping_aggregate_store(%arg: i64) {
  %0 = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %[[ALLOCA:.*]] = llvm.alloca %{{.*}} x !llvm.struct<"foo", (i32, i32, struct<(i64, i16, i16, i16)>)>
  %1 = llvm.alloca %0 x !llvm.struct<"foo", (i32, i32, struct<(i64, i16, i16, i16)>)> : (i32) -> !llvm.ptr
  // CHECK: %[[GEP:.*]] = llvm.getelementptr %[[ALLOCA]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"foo", (i32, i32, struct<(i64, i16, i16, i16)>)>
  %2 = llvm.getelementptr %1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"foo", (i32, i32, struct<(i64, i16, i16, i16)>)>
  // CHECK: llvm.store %[[ARG]], %[[GEP]]
  llvm.store %arg, %2 : i64, !llvm.ptr

  llvm.return
}

// -----

// CHECK-LABEL: llvm.func @coalesced_store_ints_array
// CHECK-SAME: %[[ARG:.*]]: i64
llvm.func @coalesced_store_ints_array(%arg: i64) {
  // CHECK-DAG: %[[CST0:.*]] = llvm.mlir.constant(0 : i64) : i64
  // CHECK-DAG: %[[CST32:.*]] = llvm.mlir.constant(32 : i64) : i64

  %0 = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %[[ALLOCA:.*]] = llvm.alloca %{{.*}} x !llvm.array<2 x i32>
  %1 = llvm.alloca %0 x !llvm.array<2 x i32> : (i32) -> !llvm.ptr

  // CHECK: %[[GEP:.*]] = llvm.getelementptr %[[ALLOCA]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x i32>
  // CHECK: %[[SHR:.*]] = llvm.lshr %[[ARG]], %[[CST0]]
  // CHECK: %[[TRUNC:.*]] = llvm.trunc %[[SHR]] : i64 to i32
  // CHECK: llvm.store %[[TRUNC]], %[[GEP]]
  // CHECK: %[[SHR:.*]] = llvm.lshr %[[ARG]], %[[CST32]] : i64
  // CHECK: %[[TRUNC:.*]] = llvm.trunc %[[SHR]] : i64 to i32
  // CHECK: %[[GEP:.*]] = llvm.getelementptr %[[ALLOCA]][0, 1] : (!llvm.ptr)  -> !llvm.ptr, !llvm.array<2 x i32>
  // CHECK: llvm.store %[[TRUNC]], %[[GEP]]
  llvm.store %arg, %1 : i64, !llvm.ptr
  // CHECK-NOT: llvm.store %[[ARG]], %[[ALLOCA]]
  llvm.return
}
