// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

llvm.func @dealloc_foo_0(!llvm.ptr)

omp.private {type = private} @box.heap_privatizer0 : !llvm.ptr alloc {
^bb0(%arg0: !llvm.ptr):
  %0 = llvm.mlir.constant(1 : i32) : i32
  %7 = llvm.alloca %0 x !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>  : (i32) -> !llvm.ptr
  omp.yield(%7 : !llvm.ptr)
} dealloc {
^bb0(%arg0: !llvm.ptr):
  llvm.call @dealloc_foo_0(%arg0) : (!llvm.ptr) -> ()
  omp.yield
}

llvm.func @alloc_foo_1(!llvm.ptr)
llvm.func @dealloc_foo_1(!llvm.ptr)

omp.private {type = private} @box.heap_privatizer1 : !llvm.ptr alloc {
^bb0(%arg0: !llvm.ptr):
  %0 = llvm.mlir.constant(1 : i32) : i32
  %7 = llvm.alloca %0 x !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>  : (i32) -> !llvm.ptr
  llvm.call @alloc_foo_1(%arg0) : (!llvm.ptr) -> ()
  omp.yield(%7 : !llvm.ptr)
} dealloc {
^bb0(%arg0: !llvm.ptr):
  llvm.call @dealloc_foo_1(%arg0) : (!llvm.ptr) -> ()
  omp.yield
}

llvm.func @target_allocatable_(%arg0: !llvm.ptr {fir.bindc_name = "lb"}, %arg1: !llvm.ptr {fir.bindc_name = "ub"}, %arg2: !llvm.ptr {fir.bindc_name = "l"}) attributes {fir.internal_name = "_QPtarget_allocatable"} {
  %6 = llvm.mlir.constant(1 : i64) : i64
  %7 = llvm.alloca %6 x i32 {bindc_name = "mapped_var"} : (i64) -> !llvm.ptr
  %13 = llvm.alloca %6 x !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)> {bindc_name = "alloc_var0"} : (i64) -> !llvm.ptr
  %14 = llvm.alloca %6 x !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)> {bindc_name = "alloc_var1"} : (i64) -> !llvm.ptr
  %53 = omp.map.info var_ptr(%7 : !llvm.ptr, i32) map_clauses(implicit, exit_release_or_enter_alloc) capture(ByCopy) -> !llvm.ptr {name = "mapped_var"}
  %54 = omp.map.info var_ptr(%13 : !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>) map_clauses(to) capture(ByRef) -> !llvm.ptr
  %55 = omp.map.info var_ptr(%14 : !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>) map_clauses(to) capture(ByRef) -> !llvm.ptr
  omp.target map_entries(%53 -> %arg3, %54 -> %arg4, %55 ->%arg5 : !llvm.ptr, !llvm.ptr, !llvm.ptr) private(@box.heap_privatizer0 %13 -> %arg6 [map_idx=1], @box.heap_privatizer1 %14 -> %arg7  [map_idx=2]: !llvm.ptr, !llvm.ptr) {
    llvm.call @use_private_var0(%arg6) : (!llvm.ptr) -> ()
    llvm.call @use_private_var1(%arg7) : (!llvm.ptr) -> ()
    omp.terminator
  }
  llvm.return
}


llvm.func @use_private_var0(!llvm.ptr) -> ()
llvm.func @use_private_var1(!llvm.ptr) -> ()

// The first set of checks ensure that we are calling the offloaded function
// with the right arguments, especially the second argument which needs to
// be a memory reference to the descriptor for the privatized allocatable
// CHECK: define void @target_allocatable_
// CHECK-NOT: define internal void
// CHECK: %[[DESC_ALLOC0:.*]] = alloca { ptr, i64, i32, i8, i8, i8, i8 }, i64 1
// CHECK: %[[DESC_ALLOC1:.*]] = alloca { ptr, i64, i32, i8, i8, i8, i8 }, i64 1
// CHECK: call void @__omp_offloading_[[OFFLOADED_FUNCTION:.*]](ptr {{[^,]+}},
// CHECK-SAME: ptr %[[DESC_ALLOC0]], ptr %[[DESC_ALLOC1]])

// CHECK: define internal void @__omp_offloading_[[OFFLOADED_FUNCTION]]
// CHECK-SAME: (ptr {{[^,]+}}, ptr %[[DESCRIPTOR_ARG0:[^,]+]],
// CHECK-SAME: ptr %[[DESCRIPTOR_ARG1:.*]]) {

// `var0` privatrizer `alloc`
// CHECK: %[[PRIV_DESC0:.*]] = alloca { ptr, i64, i32, i8, i8, i8, i8 }

// `var1` privatrizer  `alloc`
// CHECK: %[[PRIV_DESC1:.*]] = alloca { ptr, i64, i32, i8, i8, i8, i8 }
// CHECK: call void @alloc_foo_1(ptr %[[DESCRIPTOR_ARG1]])

// target op body
// CHECK: call void @use_private_var0(ptr %[[PRIV_DESC0]]
// CHECK: call void @use_private_var1(ptr %[[PRIV_DESC1]]

// `var0` privatrizer `dealloc`
// CHECK:  call void @dealloc_foo_0(ptr %[[PRIV_DESC0]])

// `var1` privatrizer `dealloc`
// CHECK:  call void @dealloc_foo_1(ptr %[[PRIV_DESC1]])
