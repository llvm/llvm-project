// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

llvm.func @free(!llvm.ptr)
llvm.func @malloc(i64) -> !llvm.ptr
omp.private {type = private} @box.heap_privatizer0 : !llvm.ptr alloc {
^bb0(%arg0: !llvm.ptr):
  %0 = llvm.mlir.constant(1 : i32) : i32
  %10 = llvm.getelementptr %arg0[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>
  %1 = llvm.load %10 : !llvm.ptr -> i64
  %7 = llvm.alloca %0 x !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>  : (i32) -> !llvm.ptr
  %17 = llvm.call @malloc(%1) {fir.must_be_heap = true, in_type = i32} : (i64) -> !llvm.ptr
  %22 = llvm.mlir.undef : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>
  %37 = llvm.insertvalue %17, %22[0] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)> 
  llvm.store %37, %7 : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>, !llvm.ptr
  omp.yield(%7 : !llvm.ptr)
} dealloc {
^bb0(%arg0: !llvm.ptr):
  %6 = llvm.mlir.constant(0 : i64) : i64
  %8 = llvm.getelementptr %arg0[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>
  %9 = llvm.load %8 : !llvm.ptr -> !llvm.ptr
  llvm.call @free(%9) : (!llvm.ptr) -> ()
  omp.yield
}
omp.private {type = private} @box.heap_privatizer1 : !llvm.ptr alloc {
^bb0(%arg0: !llvm.ptr):
  %0 = llvm.mlir.constant(1 : i32) : i32
  %10 = llvm.getelementptr %arg0[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>
  %1 = llvm.load %10 : !llvm.ptr -> i64
  %7 = llvm.alloca %0 x !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>  : (i32) -> !llvm.ptr
  %17 = llvm.call @malloc(%1) {fir.must_be_heap = true, in_type = i32} : (i64) -> !llvm.ptr
  %22 = llvm.mlir.undef : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>
  %37 = llvm.insertvalue %17, %22[0] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)> 
  llvm.store %37, %7 : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>, !llvm.ptr
  omp.yield(%7 : !llvm.ptr)
} dealloc {
^bb0(%arg0: !llvm.ptr):
  %6 = llvm.mlir.constant(0 : i64) : i64
  %8 = llvm.getelementptr %arg0[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>
  %9 = llvm.load %8 : !llvm.ptr -> !llvm.ptr
  llvm.call @free(%9) : (!llvm.ptr) -> ()
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
  omp.target map_entries(%53 -> %arg3, %54 -> %arg4, %55 ->%arg5 : !llvm.ptr, !llvm.ptr, !llvm.ptr) private(@box.heap_privatizer0 %13 -> %arg6, @box.heap_privatizer1 %14 -> %arg7 : !llvm.ptr, !llvm.ptr) {
    %64 = llvm.mlir.constant(1 : i32) : i32
    %65 = llvm.alloca %64 x !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %67 = llvm.alloca %64 x i32 : (i32) -> !llvm.ptr
    %66 = llvm.mlir.constant(19 : i32) : i32
    %68 = llvm.mlir.constant(18 : i32) : i32
    %69 = llvm.mlir.constant(10 : i32) : i32
    %70 = llvm.mlir.constant(5 : i32) : i32
    llvm.store %70, %arg3 : i32, !llvm.ptr
    llvm.store %69, %67 : i32, !llvm.ptr
    %75 = llvm.mlir.undef : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>
    %90 = llvm.insertvalue %67, %75[0] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)> 
    llvm.store %90, %65 : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>, !llvm.ptr
    %91 = llvm.mlir.zero : !llvm.ptr
    %92 = llvm.call @_FortranAAssign(%arg6, %65, %91, %68) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32) -> !llvm.struct<()>
    %93 = llvm.call @_FortranAAssign(%arg7, %65, %91, %66) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32) -> !llvm.struct<()>
    omp.terminator
  }
  llvm.return
}


llvm.func @_FortranAAssign(!llvm.ptr, !llvm.ptr, !llvm.ptr, i32) -> !llvm.struct<()> attributes {fir.runtime, sym_visibility = "private"}

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
// CHECK: %[[I0:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8 },
// CHECK-SAME: ptr %[[DESCRIPTOR_ARG0]], i32 0, i32 1
// CHECK: %[[MALLOC_ARG0:.*]] = load i64, ptr %[[I0]]
// CHECK: %[[PRIV_DESC0:.*]] = alloca { ptr, i64, i32, i8, i8, i8, i8 }
// CHECK: %[[HEAP_PTR0:.*]] = call ptr @malloc(i64 %[[MALLOC_ARG0]])
// CHECK:  %[[TMP0:.*]] = insertvalue { ptr, i64, i32, i8, i8, i8, i8 }
// CHECK-SAME: undef, ptr %[[HEAP_PTR0]], 0
// CHECK: store { ptr, i64, i32, i8, i8, i8, i8 } %[[TMP0]], ptr %[[PRIV_DESC0]]

// CHECK: %[[I1:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8 },
// CHECK-SAME: ptr %[[DESCRIPTOR_ARG1]], i32 0, i32 1
// CHECK: %[[MALLOC_ARG1:.*]] = load i64, ptr %[[I1]]
// CHECK: %[[PRIV_DESC1:.*]] = alloca { ptr, i64, i32, i8, i8, i8, i8 }
// CHECK: %[[HEAP_PTR1:.*]] = call ptr @malloc(i64 %[[MALLOC_ARG1]])
// CHECK:  %[[TMP1:.*]] = insertvalue { ptr, i64, i32, i8, i8, i8, i8 }
// CHECK-SAME: undef, ptr %[[HEAP_PTR1]], 0
// CHECK: store { ptr, i64, i32, i8, i8, i8, i8 } %[[TMP1]], ptr %[[PRIV_DESC1]]

// CHECK: call {} @_FortranAAssign(ptr %[[PRIV_DESC0]]
// CHECK: call {} @_FortranAAssign(ptr %[[PRIV_DESC1]]

// CHECK:  %[[PTR0:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8 },
// CHECK-SAME: ptr %[[PRIV_DESC0]], i32 0, i32 0
// CHECK: %[[HEAP_MEMREF0:.*]] = load ptr, ptr %[[PTR0]]
// CHECK:  call void @free(ptr %[[HEAP_MEMREF0]])
// CHECK:  %[[PTR1:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8 },
// CHECK-SAME: ptr %[[PRIV_DESC1]], i32 0, i32 0
// CHECK: %[[HEAP_MEMREF1:.*]] = load ptr, ptr %[[PTR1]]
// CHECK:  call void @free(ptr %[[HEAP_MEMREF1]])
