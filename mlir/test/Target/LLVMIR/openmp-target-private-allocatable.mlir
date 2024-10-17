// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

llvm.func @free(!llvm.ptr)
llvm.func @malloc(i64) -> !llvm.ptr
omp.private {type = private} @box.heap_privatizer : !llvm.ptr alloc {
^bb0(%arg0: !llvm.ptr):
  %0 = llvm.mlir.constant(1 : i32) : i32
  %7 = llvm.alloca %0 x !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)> {bindc_name = "alloc_var", pinned} : (i32) -> !llvm.ptr
  %8 = llvm.mlir.constant(0 : i64) : i64
  %10 = llvm.getelementptr %arg0[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>
  %11 = llvm.load %10 : !llvm.ptr -> !llvm.ptr
  %12 = llvm.ptrtoint %11 : !llvm.ptr to i64
  %13 = llvm.icmp "ne" %12, %8 : i64
  llvm.cond_br %13, ^bb1, ^bb2
^bb1:  // pred: ^bb0
  %14 = llvm.mlir.zero : !llvm.ptr
  %16 = llvm.ptrtoint %14 : !llvm.ptr to i64
  %17 = llvm.call @malloc(%16) {fir.must_be_heap = true, in_type = i32} : (i64) -> !llvm.ptr
  %22 = llvm.mlir.undef : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>
  %37 = llvm.insertvalue %17, %22[0] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)> 
  llvm.store %37, %7 : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>, !llvm.ptr
  llvm.br ^bb3
^bb2:  // pred: ^bb0
  %39 = llvm.mlir.zero : !llvm.ptr
  %44 = llvm.mlir.undef : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>
  %59 = llvm.insertvalue %39, %44[0] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)> 
  llvm.store %59, %7 : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>, !llvm.ptr
  llvm.br ^bb3
^bb3:  // 2 preds: ^bb1, ^bb2
  omp.yield(%7 : !llvm.ptr)
} dealloc {
^bb0(%arg0: !llvm.ptr):
  %6 = llvm.mlir.constant(0 : i64) : i64
  %8 = llvm.getelementptr %arg0[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>
  %9 = llvm.load %8 : !llvm.ptr -> !llvm.ptr
  %10 = llvm.ptrtoint %9 : !llvm.ptr to i64
  %11 = llvm.icmp "ne" %10, %6 : i64
  llvm.cond_br %11, ^bb1, ^bb2
^bb1:  // pred: ^bb0
  llvm.call @free(%9) : (!llvm.ptr) -> ()
  %15 = llvm.mlir.zero : !llvm.ptr
  %16 = llvm.mlir.undef : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>
  %17 = llvm.insertvalue %15, %16[0] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)> 
  llvm.store %17, %arg0 : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>, !llvm.ptr
  llvm.br ^bb2
^bb2:  // 2 preds: ^bb0, ^bb1
  omp.yield
}
llvm.func @target_allocatable_(%arg0: !llvm.ptr {fir.bindc_name = "lb"}, %arg1: !llvm.ptr {fir.bindc_name = "ub"}, %arg2: !llvm.ptr {fir.bindc_name = "l"}) attributes {fir.internal_name = "_QPtarget_allocatable"} {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.alloca %0 x !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %2 = llvm.mlir.constant(1 : i32) : i32
  %3 = llvm.alloca %2 x !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %4 = llvm.mlir.constant(1 : i64) : i64
  %5 = llvm.alloca %4 x f32 {bindc_name = "real_var"} : (i64) -> !llvm.ptr
  %6 = llvm.mlir.constant(1 : i64) : i64
  %7 = llvm.alloca %6 x i32 {bindc_name = "mapped_var"} : (i64) -> !llvm.ptr
  %8 = llvm.mlir.constant(1 : i64) : i64
  %9 = llvm.alloca %8 x !llvm.struct<(f32, f32)> {bindc_name = "comp_var"} : (i64) -> !llvm.ptr
  %10 = llvm.mlir.constant(1 : i32) : i32
  %11 = llvm.alloca %10 x !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %12 = llvm.mlir.constant(1 : i64) : i64
  %13 = llvm.alloca %12 x !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)> {bindc_name = "alloc_var"} : (i64) -> !llvm.ptr
  %14 = llvm.mlir.constant(0 : index) : i64
  %15 = llvm.mlir.constant(1 : index) : i64
  %16 = llvm.mlir.constant(0 : i64) : i64
  %17 = llvm.mlir.zero : !llvm.ptr
  %18 = llvm.mlir.constant(9 : i32) : i32
  %19 = llvm.mlir.zero : !llvm.ptr
  %20 = llvm.getelementptr %19[1] : (!llvm.ptr) -> !llvm.ptr, i32
  %21 = llvm.ptrtoint %20 : !llvm.ptr to i64
  %22 = llvm.mlir.undef : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>
  %23 = llvm.insertvalue %21, %22[1] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)> 
  %24 = llvm.mlir.constant(20240719 : i32) : i32
  %25 = llvm.insertvalue %24, %23[2] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)> 
  %26 = llvm.mlir.constant(0 : i32) : i32
  %27 = llvm.trunc %26 : i32 to i8
  %28 = llvm.insertvalue %27, %25[3] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)> 
  %29 = llvm.trunc %18 : i32 to i8
  %30 = llvm.insertvalue %29, %28[4] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)> 
  %31 = llvm.mlir.constant(2 : i32) : i32
  %32 = llvm.trunc %31 : i32 to i8
  %33 = llvm.insertvalue %32, %30[5] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)> 
  %34 = llvm.mlir.constant(0 : i32) : i32
  %35 = llvm.trunc %34 : i32 to i8
  %36 = llvm.insertvalue %35, %33[6] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)> 
  %37 = llvm.insertvalue %17, %36[0] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)> 
  llvm.store %37, %11 : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>, !llvm.ptr
  %38 = llvm.load %11 : !llvm.ptr -> !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>
  llvm.store %38, %13 : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>, !llvm.ptr
  %39 = llvm.load %arg2 : !llvm.ptr -> i64
  %40 = llvm.icmp "sgt" %39, %16 : i64
  %41 = llvm.select %40, %39, %16 : i1, i64
  %42 = llvm.mlir.constant(1 : i64) : i64
  %43 = llvm.alloca %41 x i8 {bindc_name = "char_var"} : (i64) -> !llvm.ptr
  %44 = llvm.load %arg0 : !llvm.ptr -> i64
  %45 = llvm.load %arg1 : !llvm.ptr -> i64
  %46 = llvm.sub %45, %44 : i64
  %47 = llvm.add %46, %15 : i64
  %48 = llvm.icmp "sgt" %47, %14 : i64
  %49 = llvm.select %48, %47, %14 : i1, i64
  %50 = llvm.mlir.constant(1 : i64) : i64
  %51 = llvm.mul %49, %50 : i64
  %52 = llvm.alloca %51 x f32 {bindc_name = "real_arr"} : (i64) -> !llvm.ptr
  %53 = omp.map.info var_ptr(%7 : !llvm.ptr, i32) map_clauses(implicit, exit_release_or_enter_alloc) capture(ByCopy) -> !llvm.ptr {name = "mapped_var"}
  %54 = omp.map.info var_ptr(%13 : !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>) map_clauses(to) capture(ByRef) -> !llvm.ptr
  omp.target map_entries(%53 -> %arg3, %54 -> %arg4 : !llvm.ptr, !llvm.ptr) private(@box.heap_privatizer %13 -> %arg5 : !llvm.ptr) {
    %64 = llvm.mlir.constant(1 : i32) : i32
    %65 = llvm.alloca %64 x !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %66 = llvm.mlir.constant(1 : i64) : i64
    %67 = llvm.alloca %66 x i32 : (i64) -> !llvm.ptr
    %68 = llvm.mlir.constant(18 : i32) : i32
    %69 = llvm.mlir.constant(10 : i32) : i32
    %70 = llvm.mlir.constant(5 : i32) : i32
    llvm.store %70, %arg3 : i32, !llvm.ptr
    llvm.store %69, %67 : i32, !llvm.ptr
    %71 = llvm.mlir.constant(9 : i32) : i32
    %72 = llvm.mlir.zero : !llvm.ptr
    %73 = llvm.getelementptr %72[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %74 = llvm.ptrtoint %73 : !llvm.ptr to i64
    %75 = llvm.mlir.undef : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>
    %76 = llvm.insertvalue %74, %75[1] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)> 
    %77 = llvm.mlir.constant(20240719 : i32) : i32
    %78 = llvm.insertvalue %77, %76[2] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)> 
    %79 = llvm.mlir.constant(0 : i32) : i32
    %80 = llvm.trunc %79 : i32 to i8
    %81 = llvm.insertvalue %80, %78[3] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)> 
    %82 = llvm.trunc %71 : i32 to i8
    %83 = llvm.insertvalue %82, %81[4] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)> 
    %84 = llvm.mlir.constant(0 : i32) : i32
    %85 = llvm.trunc %84 : i32 to i8
    %86 = llvm.insertvalue %85, %83[5] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)> 
    %87 = llvm.mlir.constant(0 : i32) : i32
    %88 = llvm.trunc %87 : i32 to i8
    %89 = llvm.insertvalue %88, %86[6] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)> 
    %90 = llvm.insertvalue %67, %89[0] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)> 
    llvm.store %90, %65 : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>, !llvm.ptr
    %91 = llvm.mlir.zero : !llvm.ptr
    %92 = llvm.call @_FortranAAssign(%arg5, %65, %91, %68) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32) -> !llvm.struct<()>
    omp.terminator
  }
  %55 = llvm.load %13 : !llvm.ptr -> !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>
  llvm.store %55, %3 : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>, !llvm.ptr
  %56 = llvm.getelementptr %3[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>
  %57 = llvm.load %56 : !llvm.ptr -> !llvm.ptr
  %58 = llvm.ptrtoint %57 : !llvm.ptr to i64
  %59 = llvm.icmp "ne" %58, %16 : i64
  llvm.cond_br %59, ^bb1, ^bb2
^bb1:  // pred: ^bb0
  %60 = llvm.load %13 : !llvm.ptr -> !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>
  llvm.store %60, %1 : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>, !llvm.ptr
  %61 = llvm.getelementptr %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>
  %62 = llvm.load %61 : !llvm.ptr -> !llvm.ptr
  llvm.call @free(%62) : (!llvm.ptr) -> ()
  %63 = llvm.load %11 : !llvm.ptr -> !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>
  llvm.store %63, %13 : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>, !llvm.ptr
  llvm.br ^bb2
^bb2:  // 2 preds: ^bb0, ^bb1
  llvm.return
}


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
// CHECK: %[[HEAP_MEMREF:.*]] = call ptr @malloc(i64 0)
// CHECK: %[[DESC_SETUP_MEMREF:.*]] = insertvalue
// CHECK-SAME: { ptr, i64, i32, i8, i8, i8, i8 } undef, ptr %[[HEAP_MEMREF]], 0
// Unfortunately, in the way the blocks are arranged, the store to the
// privatized alloctables descriptor is encountered before we allocate
// device-local memory for the descriptor (PRIVATE_DESC) itself
// CHECK: store { ptr, i64, i32, i8, i8, i8, i8 } %[[DESC_SETUP_MEMREF]],
// CHECK-SAME: ptr %[[PRIVATE_DESC:.*]], align 8
// CHECK: %[[PRIVATE_DESC]] = alloca { ptr, i64, i32, i8, i8, i8, i8 }
// CHECK: %[[ORIG_DATA_PTR_MMBR_OF_DESC:.*]] = getelementptr 
// CHECK-SAME: { ptr, i64, i32, i8, i8, i8, i8 }, ptr %[[DESCRIPTOR_ARG]],
// CHECK-SAME: i32 0, i32 0
// CHECK: %[[ORIG_DATA_PTR:.*]] = load ptr, ptr %[[ORIG_DATA_PTR_MMBR_OF_DESC]]
// CHECK: %[[PTR_TO_INT:.*]] = ptrtoint ptr %[[ORIG_DATA_PTR]] to i64
// CHECK: icmp ne i64 %[[PTR_TO_INT]], 0


// CHECK: call {} @_FortranAAssign(ptr %[[DESC_TO_DEALLOC:[^,]+]],

// Now, check the deallocation of the private var.
// CHECK: call void @free(ptr %[[DATA_PTR_TO_FREE:.*]])
// CHECK:   store { ptr, i64, i32, i8, i8, i8, i8 }
// CHECK-SAME: { ptr null, i64 undef, i32 undef, i8 undef, i8 undef, i8 undef, i8 undef },
// CHECK-SAME: ptr %[[DESC_TO_DEALLOC]]

// CHECK: %[[DESC_MMBR:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8 },
// CHECK-SAME: ptr %[[DESC_TO_DEALLOC]], i32 0, i32 0
// CHECK: %[[DATA_PTR_TO_FREE]] = load ptr, ptr %[[DESC_MMBR]]
// CHECK: ptrtoint ptr %[[DATA_PTR_TO_FREE]] to i64
