// Linear vars with a `linear` privatizer keep their value in the private copy.
// RUN: mlir-translate -mlir-to-llvmir -split-input-file %s | FileCheck %s

llvm.func @malloc(i64) -> !llvm.ptr
llvm.func @free(!llvm.ptr)

// A "box" holding a pointer to an i32.
omp.private {type = linear} @box.linear : !llvm.ptr init {
^bb0(%mold: !llvm.ptr, %priv: !llvm.ptr):
  %c4 = llvm.mlir.constant(4 : i64) : i64
  %mem = llvm.call @malloc(%c4) : (i64) -> !llvm.ptr
  llvm.store %mem, %priv : !llvm.ptr, !llvm.ptr
  omp.yield(%priv : !llvm.ptr)
} data_address {
^bb0(%box: !llvm.ptr):
  %data = llvm.load %box : !llvm.ptr -> !llvm.ptr
  omp.yield(%data : !llvm.ptr)
} dealloc {
^bb0(%box: !llvm.ptr):
  %data = llvm.load %box : !llvm.ptr -> !llvm.ptr
  llvm.call @free(%data) : (!llvm.ptr) -> ()
  omp.yield
}

// CHECK-LABEL: define void @simd_linear_privatizer(
// CHECK-SAME:      ptr %[[BOX:.*]], ptr %{{.*}})
// CHECK:         %[[PRIV:.*]] = alloca ptr
// CHECK-NOT:     .linear_result
// CHECK:         %[[START:.linear_var]] = alloca i32
// CHECK:       omp_loop.preheader:
// CHECK:         %[[ORIG_DATA:.*]] = load ptr, ptr %[[BOX]]
// CHECK:         %[[V:.*]] = load i32, ptr %[[ORIG_DATA]]
// CHECK:         store i32 %[[V]], ptr %[[START]]
// CHECK:       omp_loop.body:
// CHECK:         %[[PRIV_DATA:.*]] = load ptr, ptr %[[PRIV]]
// CHECK:         %[[S:.*]] = load i32, ptr %[[START]]
// CHECK:         %[[MUL:.*]] = mul i32 %{{.*}}, 1
// CHECK:         %[[ADD:.*]] = add i32 %[[S]], %[[MUL]]
// CHECK:         store i32 %[[ADD]], ptr %[[PRIV_DATA]]
// CHECK:       omp.region.cont:
// CHECK:         %[[PRIV_DATA2:.*]] = load ptr, ptr %[[PRIV]]
// CHECK:         %[[LAST:.*]] = load i32, ptr %[[PRIV_DATA2]]
// CHECK:         %[[ORIG_DATA2:.*]] = load ptr, ptr %[[BOX]]
// CHECK:         store i32 %[[LAST]], ptr %[[ORIG_DATA2]]
// CHECK:         call void @free(
llvm.func @simd_linear_privatizer(%box : !llvm.ptr, %out : !llvm.ptr) {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %c10 = llvm.mlir.constant(10 : i32) : i32
  omp.simd linear(%box : !llvm.ptr = %c1 : i32) linear_var_types([i32]) private(@box.linear %box -> %arg0 : !llvm.ptr) {
    omp.loop_nest (%iv) : i32 = (%c1) to (%c10) inclusive step (%c1) {
      %data = llvm.load %arg0 : !llvm.ptr -> !llvm.ptr
      %v = llvm.load %data : !llvm.ptr -> i32
      llvm.store %v, %out : i32, !llvm.ptr
      omp.yield
    }
  }
  llvm.return
}

// CHECK-LABEL: define void @wsloop_linear_privatizer(
// CHECK-SAME:      ptr %[[BOX:.*]], ptr %{{.*}})
// CHECK:         %[[PRIV:.*]] = alloca ptr
// CHECK-NOT:     .linear_result
// CHECK:         %[[START:.linear_var]] = alloca i32
// CHECK:       omp_loop.preheader:
// CHECK:         %[[ORIG_DATA:.*]] = load ptr, ptr %[[BOX]]
// CHECK:         %[[V:.*]] = load i32, ptr %[[ORIG_DATA]]
// CHECK:         store i32 %[[V]], ptr %[[START]]
// CHECK:       omp_loop.body:
// CHECK:         %[[PRIV_DATA:.*]] = load ptr, ptr %[[PRIV]]
// CHECK:         %[[S:.*]] = load i32, ptr %[[START]]
// CHECK:         %[[MUL:.*]] = mul i32 %{{.*}}, 2
// CHECK:         %[[ADD:.*]] = add i32 %[[S]], %[[MUL]]
// CHECK:         store i32 %[[ADD]], ptr %[[PRIV_DATA]]
// CHECK:       omp_loop.linear_lastiter_exit:
// CHECK:         %[[PRIV_DATA2:.*]] = load ptr, ptr %[[PRIV]]
// CHECK:         %[[LAST:.*]] = load i32, ptr %[[PRIV_DATA2]]
// CHECK:         %[[ORIG_DATA2:.*]] = load ptr, ptr %[[BOX]]
// CHECK:         store i32 %[[LAST]], ptr %[[ORIG_DATA2]]
// CHECK:       omp.region.cont:
// CHECK:         call void @free(
llvm.func @wsloop_linear_privatizer(%box : !llvm.ptr, %out : !llvm.ptr) {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %c2 = llvm.mlir.constant(2 : i32) : i32
  %c10 = llvm.mlir.constant(10 : i32) : i32
  omp.wsloop linear(%box : !llvm.ptr = %c2 : i32) linear_var_types([i32]) private(@box.linear %box -> %arg0 : !llvm.ptr) {
    omp.loop_nest (%iv) : i32 = (%c1) to (%c10) inclusive step (%c1) {
      %data = llvm.load %arg0 : !llvm.ptr -> !llvm.ptr
      %v = llvm.load %data : !llvm.ptr -> i32
      llvm.store %v, %out : i32, !llvm.ptr
      omp.yield
    }
  }
  llvm.return
}
