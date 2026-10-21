// Linear vars whose `linear` privatizer has a `data_address` region.
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
  omp.simd linear(@box.linear %box -> %arg0 : !llvm.ptr = %c1 : i32) linear_var_types([i32]) {
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
  omp.wsloop linear(@box.linear %box -> %arg0 : !llvm.ptr = %c2 : i32) linear_var_types([i32]) {
    omp.loop_nest (%iv) : i32 = (%c1) to (%c10) inclusive step (%c1) {
      %data = llvm.load %arg0 : !llvm.ptr -> !llvm.ptr
      %v = llvm.load %data : !llvm.ptr -> i32
      llvm.store %v, %out : i32, !llvm.ptr
      omp.yield
    }
  }
  llvm.return
}

// The loop IV is stored through the data address of the private copy; it gets
// one more step after the loop.
// CHECK-LABEL: define void @simd_linear_privatizer_iv(
// CHECK-SAME:      ptr %[[BOX:.*]])
// CHECK:         %[[PRIV:.*]] = alloca ptr
// CHECK:       omp.region.cont:
// CHECK:         %[[DATA:.*]] = load ptr, ptr %[[PRIV]]
// CHECK:         %[[LAST:.*]] = load i32, ptr %[[DATA]]
// CHECK:         %[[NEXT:.*]] = add i32 %[[LAST]], 2
// CHECK:         store i32 %[[NEXT]], ptr %[[DATA]]
// CHECK:         %[[DATA2:.*]] = load ptr, ptr %[[PRIV]]
// CHECK:         %[[FINAL:.*]] = load i32, ptr %[[DATA2]]
// CHECK:         %[[ORIG:.*]] = load ptr, ptr %[[BOX]]
// CHECK:         store i32 %[[FINAL]], ptr %[[ORIG]]
// CHECK:         call void @free(
llvm.func @simd_linear_privatizer_iv(%box : !llvm.ptr) {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %c2 = llvm.mlir.constant(2 : i32) : i32
  %c5 = llvm.mlir.constant(5 : i32) : i32
  omp.simd linear(@box.linear %box -> %arg0 : !llvm.ptr = %c2 : i32) linear_var_types([i32]) {
    omp.loop_nest (%iv) : i32 = (%c1) to (%c5) inclusive step (%c2) {
      %data = llvm.load %arg0 : !llvm.ptr -> !llvm.ptr
      llvm.store %iv, %data : i32, !llvm.ptr
      omp.yield
    }
  }
  llvm.return
}

// Same, with the private copy copied before its data address is loaded.
// CHECK-LABEL: define void @simd_linear_privatizer_iv_copy(
// CHECK:         %[[PRIV:omp.private.alloc]] = alloca ptr
// CHECK:       omp.region.cont:
// CHECK:         %[[DATA:.*]] = load ptr, ptr %[[PRIV]]
// CHECK:         %[[LAST:.*]] = load i32, ptr %[[DATA]]
// CHECK:         %[[NEXT:.*]] = add i32 %[[LAST]], 2
// CHECK:         store i32 %[[NEXT]], ptr %[[DATA]]
llvm.func @simd_linear_privatizer_iv_copy(%box : !llvm.ptr) {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %c2 = llvm.mlir.constant(2 : i32) : i32
  %c5 = llvm.mlir.constant(5 : i32) : i32
  %c8 = llvm.mlir.constant(8 : i32) : i32
  %one = llvm.mlir.constant(1 : i64) : i64
  %tmp = llvm.alloca %one x !llvm.ptr : (i64) -> !llvm.ptr
  omp.simd linear(@box.linear %box -> %arg0 : !llvm.ptr = %c2 : i32) linear_var_types([i32]) {
    omp.loop_nest (%iv) : i32 = (%c1) to (%c5) inclusive step (%c2) {
      "llvm.intr.memcpy"(%tmp, %arg0, %c8) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i32) -> ()
      %data = llvm.load %tmp : !llvm.ptr -> !llvm.ptr
      llvm.store %iv, %data : i32, !llvm.ptr
      omp.yield
    }
  }
  llvm.return
}
