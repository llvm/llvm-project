// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// CHECK-LABEL: @opaque_ptr_load
llvm.func @opaque_ptr_load(%arg0: !llvm.ptr) -> i32 {
  // CHECK: load i32, ptr %{{.*}}
  %0 = llvm.load %arg0 : !llvm.ptr -> i32
  llvm.return %0 : i32
}

// CHECK-LABEL: @opaque_ptr_store
llvm.func @opaque_ptr_store(%arg0: i32, %arg1: !llvm.ptr){
  // CHECK: store i32 %{{.*}}, ptr %{{.*}}
  llvm.store %arg0, %arg1 : i32, !llvm.ptr
  llvm.return
}

// CHECK-LABEL: @opaque_ptr_ptr_store
llvm.func @opaque_ptr_ptr_store(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
  // CHECK: store ptr %{{.*}}, ptr %{{.*}}
  llvm.store %arg0, %arg1 : !llvm.ptr, !llvm.ptr
  llvm.return
}

// CHECK-LABEL: @opaque_ptr_alloca
llvm.func @opaque_ptr_alloca(%arg0: i32) -> !llvm.ptr {
  // CHECK: alloca float, i32 %{{.*}}
  %0 = llvm.alloca %arg0 x f32 : (i32) -> !llvm.ptr
  llvm.return %0 : !llvm.ptr
}

// CHECK-LABEL: @opaque_ptr_gep
llvm.func @opaque_ptr_gep(%arg0: !llvm.ptr, %arg1: i32) -> !llvm.ptr {
  // CHECK: getelementptr float, ptr %{{.*}}, i32 %{{.*}}
  %0 = llvm.getelementptr %arg0[%arg1] : (!llvm.ptr, i32) -> !llvm.ptr, f32
  llvm.return %0 : !llvm.ptr
}

// CHECK-LABEL: @opaque_ptr_gep_struct
llvm.func @opaque_ptr_gep_struct(%arg0: !llvm.ptr, %arg1: i32) -> !llvm.ptr {
  // CHECK: getelementptr { { float, double }, { i32, i64 } }, ptr %{{.*}}, i32 0, i32 1
  %0 = llvm.getelementptr %arg0[%arg1, 0, 1] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<(struct<(f32, f64)>, struct<(i32, i64)>)>
  llvm.return %0 : !llvm.ptr
}

// CHECK-LABEL: @opaque_ptr_elem_type
llvm.func @opaque_ptr_elem_type(%0: !llvm.ptr) -> !llvm.ptr {
  // CHECK: getelementptr ptr, ptr
  %1 = llvm.getelementptr %0[0] { elem_type = !llvm.ptr } : (!llvm.ptr) -> !llvm.ptr
  // CHECK: getelementptr ptr, ptr
  %2 = llvm.getelementptr %0[0] : (!llvm.ptr) -> !llvm.ptr<ptr>
  llvm.return %1 : !llvm.ptr
}

// CHECK-LABEL: @opaque_ptr_matrix_load_store
llvm.func @opaque_ptr_matrix_load_store(%ptr: !llvm.ptr, %stride: i64) -> vector<48 x f32> {
  // CHECK: call <48 x float> @llvm.matrix.column.major.load.v48f32.i64
  // CHECK: (ptr {{.*}}, i64 %{{.*}}
  %0 = llvm.intr.matrix.column.major.load %ptr, <stride=%stride>
    { isVolatile = 0: i1, rows = 3: i32, columns = 16: i32} :
    vector<48 x f32> from !llvm.ptr stride i64
  // CHECK: call void @llvm.matrix.column.major.store.v48f32.i64
  // CHECK: <48 x float> %{{.*}}, ptr {{.*}}, i64
  llvm.intr.matrix.column.major.store %0, %ptr, <stride=%stride>
    { isVolatile = 0: i1, rows = 3: i32, columns = 16: i32} :
    vector<48 x f32> to !llvm.ptr stride i64
  llvm.return %0 : vector<48 x f32>
}

// CHECK-LABEL: @opaque_ptr_masked_load
llvm.func @opaque_ptr_masked_load(%arg0: !llvm.ptr, %arg1: vector<7xi1>) -> vector<7xf32> {
  // CHECK: call <7 x float> @llvm.masked.load.v7f32.p0(ptr
  %0 = llvm.intr.masked.load %arg0, %arg1 { alignment = 1: i32} :
    (!llvm.ptr, vector<7xi1>) -> vector<7xf32>
  llvm.return %0 : vector<7 x f32>
}

// CHECK-LABEL: @opaque_ptr_gather
llvm.func @opaque_ptr_gather(%M: !llvm.vec<7 x ptr>, %mask: vector<7xi1>) -> vector<7xf32> {
  // CHECK: call <7 x float> @llvm.masked.gather.v7f32.v7p0(<7 x ptr> {{.*}}, i32
  %a = llvm.intr.masked.gather %M, %mask { alignment = 1: i32} :
      (!llvm.vec<7 x ptr>, vector<7xi1>) -> vector<7xf32>
  llvm.return %a : vector<7xf32>
}
