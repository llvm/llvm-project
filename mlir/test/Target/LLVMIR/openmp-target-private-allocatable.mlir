// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

llvm.func @alloc_foo_1(!llvm.ptr)
llvm.func @dealloc_foo_1(!llvm.ptr)

omp.private {type = private} @box.heap_privatizer : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)> init {
^bb0(%arg0: !llvm.ptr, %arg1 : !llvm.ptr):
  llvm.call @alloc_foo_1(%arg0) : (!llvm.ptr) -> ()
  omp.yield(%arg1 : !llvm.ptr)
} dealloc {
^bb0(%arg0: !llvm.ptr):
  llvm.call @dealloc_foo_1(%arg0) : (!llvm.ptr) -> ()
  omp.yield
}

llvm.func @target_allocatable_(%arg0: !llvm.ptr {fir.bindc_name = "lb"}, %arg1: !llvm.ptr {fir.bindc_name = "ub"}, %arg2: !llvm.ptr {fir.bindc_name = "l"}) attributes {fir.internal_name = "_QPtarget_allocatable"} {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.alloca %0 x !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %3 = llvm.alloca %0 x !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %4 = llvm.mlir.constant(1 : i64) : i64
  %5 = llvm.alloca %4 x f32 {bindc_name = "real_var"} : (i64) -> !llvm.ptr
  %7 = llvm.alloca %4 x i32 {bindc_name = "mapped_var"} : (i64) -> !llvm.ptr
  %9 = llvm.alloca %4 x !llvm.struct<(f32, f32)> {bindc_name = "comp_var"} : (i64) -> !llvm.ptr
  %11 = llvm.alloca %0 x !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %13 = llvm.alloca %4 x !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)> {bindc_name = "alloc_var"} : (i64) -> !llvm.ptr
  %39 = llvm.load %arg2 : !llvm.ptr -> i64
  %52 = llvm.alloca %39 x f32 {bindc_name = "real_arr"} : (i64) -> !llvm.ptr
  %53 = omp.map.info var_ptr(%7 : !llvm.ptr, i32) map_clauses(implicit, exit_release_or_enter_alloc) capture(ByCopy) -> !llvm.ptr {name = "mapped_var"}
  %54 = omp.map.info var_ptr(%13 : !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>) map_clauses(to) capture(ByRef) -> !llvm.ptr
  omp.target map_entries(%53 -> %arg3, %54 -> %arg4 : !llvm.ptr, !llvm.ptr) private(@box.heap_privatizer %13 -> %arg5 [map_idx=1] : !llvm.ptr) {
    llvm.call @use_private_var(%arg5) : (!llvm.ptr) -> ()
    omp.terminator
  }
  llvm.return
}

llvm.func @use_private_var(!llvm.ptr) -> ()

llvm.func @_FortranAAssign(!llvm.ptr, !llvm.ptr, !llvm.ptr, i32) -> !llvm.struct<()> attributes {fir.runtime, sym_visibility = "private"}

// The first set of checks ensure that we are calling the offloaded function
// with the right arguments, especially the second argument which needs to
// be a memory reference to the descriptor for the privatized allocatable
// CHECK: define void @target_allocatable_
// CHECK-NOT: define internal void
// CHECK: %[[DESC_ALLOC:.*]] = alloca { ptr, i64, i32, i8, i8, i8, i8 }, i64 1
// CHECK: call void @__omp_offloading_[[OFFLOADED_FUNCTION:.*]](ptr {{[^,]+}},
// CHECK-SAME: ptr %[[DESC_ALLOC]])

// The second set of checks ensure that to allocate memory for the
// allocatable, we are, in fact, using the memory reference of the descriptor
// passed as the second argument to the offloaded function.
// CHECK: define internal void @__omp_offloading_[[OFFLOADED_FUNCTION]]
// CHECK-SAME: (ptr {{[^,]+}}, ptr %[[DESCRIPTOR_ARG:.*]]) {
// CHECK: %[[DESC_TO_DEALLOC:.*]] = alloca { ptr, i64, i32, i8, i8, i8, i8 }
// CHECK: call void @alloc_foo_1(ptr %[[DESCRIPTOR_ARG]])


// CHECK: call void @use_private_var(ptr %[[DESC_TO_DEALLOC]]

// Now, check the deallocation of the private var.
// CHECK:  call void @dealloc_foo_1(ptr %[[DESC_TO_DEALLOC]])
