// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// CHECK-LABEL: @opaque_ptr_load
llvm.func @opaque_ptr_load(%arg0: !llvm.ptr) -> i32 {
  // CHECK: = ptr.load %{{.*}} : !llvm.ptr -> i32
  %0 = ptr.load %arg0 : !llvm.ptr -> i32
  llvm.return %0 : i32
}

// CHECK-LABEL: @opaque_ptr_store
llvm.func @opaque_ptr_store(%arg0: i32, %arg1: !llvm.ptr){
  // CHECK: ptr.store %{{.*}}, %{{.*}} : i32, !llvm.ptr
  ptr.store %arg0, %arg1 : i32, !llvm.ptr
  llvm.return
}

// CHECK-LABEL: @opaque_ptr_ptr_store
llvm.func @opaque_ptr_ptr_store(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
  // CHECK: ptr.store %{{.*}}, %{{.*}} : !llvm.ptr, !llvm.ptr
  ptr.store %arg0, %arg1 : !llvm.ptr, !llvm.ptr
  llvm.return
}

// CHECK-LABEL: @opaque_ptr_alloca
llvm.func @opaque_ptr_alloca(%arg0: i32) -> !llvm.ptr {
  // CHECK: llvm.alloca %{{.*}} x f32 : (i32) -> !llvm.ptr
  %0 = llvm.alloca %arg0 x f32 : (i32) -> !llvm.ptr
  llvm.return %0 : !llvm.ptr
}

// CHECK-LABEL: @opaque_ptr_gep
llvm.func @opaque_ptr_gep(%arg0: !llvm.ptr, %arg1: i32) -> !llvm.ptr {
  // CHECK: = llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr, i32) -> !llvm.ptr, f32
  %0 = llvm.getelementptr %arg0[%arg1] : (!llvm.ptr, i32) -> !llvm.ptr, f32
  llvm.return %0 : !llvm.ptr
}

// CHECK-LABEL: @opaque_ptr_gep_struct
llvm.func @opaque_ptr_gep_struct(%arg0: !llvm.ptr, %arg1: i32) -> !llvm.ptr {
  // CHECK: = llvm.getelementptr %{{.*}}[%{{.*}}, 0, 1]
  // CHECK: : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<(struct<(f32, f64)>, struct<(i32, i64)>)>
  %0 = llvm.getelementptr %arg0[%arg1, 0, 1] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<(struct<(f32, f64)>, struct<(i32, i64)>)>
  llvm.return %0 : !llvm.ptr
}

// CHECK-LABEL: @opaque_ptr_matrix_load_store
llvm.func @opaque_ptr_matrix_load_store(%ptr: !llvm.ptr, %stride: i64) -> vector<48 x f32> {
  // CHECK: = llvm.intr.matrix.column.major.load
  // CHECK: vector<48xf32> from !llvm.ptr stride i64
  %0 = llvm.intr.matrix.column.major.load %ptr, <stride=%stride>
    { isVolatile = 0: i1, rows = 3: i32, columns = 16: i32} :
    vector<48 x f32> from !llvm.ptr stride i64
  // CHECK: llvm.intr.matrix.column.major.store
  // CHECK: vector<48xf32> to !llvm.ptr stride i64
  llvm.intr.matrix.column.major.store %0, %ptr, <stride=%stride>
    { isVolatile = 0: i1, rows = 3: i32, columns = 16: i32} :
    vector<48 x f32> to !llvm.ptr stride i64
  llvm.return %0 : vector<48 x f32>
}

// CHECK-LABEL: @opaque_ptr_masked_load
llvm.func @opaque_ptr_masked_load(%arg0: !llvm.ptr, %arg1: vector<7xi1>) -> vector<7xf32> {
  // CHECK: = llvm.intr.masked.load
  // CHECK: (!llvm.ptr, vector<7xi1>) -> vector<7xf32>
  %0 = llvm.intr.masked.load %arg0, %arg1 { alignment = 1: i32} :
    (!llvm.ptr, vector<7xi1>) -> vector<7xf32>
  llvm.return %0 : vector<7 x f32>
}

// CHECK-LABEL: @opaque_ptr_gather
llvm.func @opaque_ptr_gather(%M: !llvm.vec<7 x ptr>, %mask: vector<7xi1>) -> vector<7xf32> {
  // CHECK: = llvm.intr.masked.gather
  // CHECK: (!llvm.vec<7 x ptr>, vector<7xi1>) -> vector<7xf32>
  %a = llvm.intr.masked.gather %M, %mask { alignment = 1: i32} :
      (!llvm.vec<7 x ptr>, vector<7xi1>) -> vector<7xf32>
  llvm.return %a : vector<7xf32>
}
