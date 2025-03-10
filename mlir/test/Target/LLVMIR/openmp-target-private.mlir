// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

omp.private {type = private} @simple_var.privatizer : i32

llvm.func @target_map_single_private() attributes {fir.internal_name = "_QPtarget_map_single_private"} {
  %0 = llvm.mlir.constant(1 : i64) : i64
  %1 = llvm.alloca %0 x i32 {bindc_name = "simple_var"} : (i64) -> !llvm.ptr
  %3 = llvm.alloca %0 x i32 {bindc_name = "a"} : (i64) -> !llvm.ptr
  %4 = llvm.mlir.constant(2 : i32) : i32
  llvm.store %4, %3 : i32, !llvm.ptr
  %5 = omp.map.info var_ptr(%3 : !llvm.ptr, i32) map_clauses(to) capture(ByRef) -> !llvm.ptr {name = "a"}
  omp.target map_entries(%5 -> %arg0 : !llvm.ptr) private(@simple_var.privatizer %1 -> %arg1 : !llvm.ptr) {
    %6 = llvm.mlir.constant(10 : i32) : i32
    %7 = llvm.load %arg0 : !llvm.ptr -> i32
    %8 = llvm.add %7, %6 : i32
    llvm.store %8, %arg1 : i32, !llvm.ptr
    omp.terminator
  }
  llvm.return
}
// CHECK: define internal void @__omp_offloading_
// CHECK-NOT: define {{.*}}
// CHECK: %[[PRIV_ALLOC:.*]] = alloca i32, align 4
// CHECK: %[[ADD:.*]] = add i32 {{.*}}, 10
// CHECK: store i32 %[[ADD]], ptr %[[PRIV_ALLOC]], align 4

omp.private {type = private} @n.privatizer : f32

llvm.func @target_map_2_privates() attributes {fir.internal_name = "_QPtarget_map_2_privates"} {
  %0 = llvm.mlir.constant(1 : i64) : i64
  %1 = llvm.alloca %0 x i32 {bindc_name = "simple_var"} : (i64) -> !llvm.ptr
  %3 = llvm.alloca %0 x f32 {bindc_name = "n"} : (i64) -> !llvm.ptr
  %5 = llvm.alloca %0 x i32 {bindc_name = "a"} : (i64) -> !llvm.ptr
  %6 = llvm.mlir.constant(2 : i32) : i32
  llvm.store %6, %5 : i32, !llvm.ptr
  %7 = omp.map.info var_ptr(%5 : !llvm.ptr, i32) map_clauses(to) capture(ByRef) -> !llvm.ptr {name = "a"}
  omp.target map_entries(%7 -> %arg0 : !llvm.ptr) private(@simple_var.privatizer %1 -> %arg1, @n.privatizer %3 -> %arg2 : !llvm.ptr, !llvm.ptr) {
    %8 = llvm.mlir.constant(1.100000e+01 : f32) : f32
    %9 = llvm.mlir.constant(10 : i32) : i32
    %10 = llvm.load %arg0 : !llvm.ptr -> i32
    %11 = llvm.add %10, %9 : i32
    llvm.store %11, %arg1 : i32, !llvm.ptr
    %12 = llvm.load %arg1 : !llvm.ptr -> i32
    %13 = llvm.sitofp %12 : i32 to f32
    %14 = llvm.fadd %13, %8  {fastmathFlags = #llvm.fastmath<contract>} : f32
    llvm.store %14, %arg2 : f32, !llvm.ptr
    omp.terminator
  }
  llvm.return
}


// CHECK: define internal void @__omp_offloading_
// CHECK: %[[PRIV_I32_ALLOC:.*]] = alloca i32, align 4
// CHECK: %[[PRIV_FLOAT_ALLOC:.*]] = alloca float, align 4
// CHECK: %[[ADD_I32:.*]] = add i32 {{.*}}, 10
// CHECK: store i32 %[[ADD_I32]], ptr %[[PRIV_I32_ALLOC]], align 4
// CHECK: %[[LOAD_I32_AGAIN:.*]] = load i32, ptr %[[PRIV_I32_ALLOC]], align 4
// CHECK: %[[CAST_TO_FLOAT:.*]] = sitofp i32 %[[LOAD_I32_AGAIN]] to float
// CHECK: %[[ADD_FLOAT:.*]] = fadd contract float %[[CAST_TO_FLOAT]], 1.100000e+01
// CHECK: store float %[[ADD_FLOAT]], ptr %[[PRIV_FLOAT_ALLOC]], align 4

// An entirely artifical privatizer that is meant to check multi-block
// privatizers. The idea here is to prove that we set the correct
// insertion points for the builder when generating, first, LLVM IR for the
// privatizer and then for the actual target region.
omp.private {type = private} @multi_block.privatizer : f32 init {
^bb0(%arg0: !llvm.ptr, %arg1: !llvm.ptr):
  llvm.br ^bb1

^bb1:
  omp.yield(%arg1 : !llvm.ptr)
}

llvm.func @target_op_private_multi_block(%arg0: !llvm.ptr) {
  omp.target private(@multi_block.privatizer %arg0 -> %arg2 : !llvm.ptr) {
    %0 = llvm.load %arg2 : !llvm.ptr -> f32
    omp.terminator
  }
  llvm.return
}
// CHECK: define internal void @__omp_offloading_
// CHECK: %[[PRIV_ALLOC:.*]] = alloca float, align 4
// CHECK: %[[PHI_ALLOCA:.*]]  = phi ptr [ %[[PRIV_ALLOC]], {{.*}} ]
// CHECK: %[[RESULT:.*]] = load float, ptr %[[PHI_ALLOCA]], align 4

// Descriptors are needed for CHARACTER arrays and their type is
// !fir.boxchar<KIND>. When such arrays are used in the private construct, the
// privatizer takes a !fir.boxchar<KIND> as input. This type is lowered to
// !llvm.struct<(ptr, i64)>. This is unique because with other types of data,
// typically, the privatizer funtion takes a !llvm.ptr. Now, on the host side,
// we map the descriptor using the map clause of the omp.target op. Map clauses
// take only !llvm.ptr types. This means, we have a case where the descriptor is
// mapped by its pointer whereas the privatizer function expects the descriptor
// by value. So, we have this test to ensure that the compiler correctly loads
// from the mapped pointer before passing that to the privatizer function.
omp.private {type = private} @_QFtarget_boxcharEchar_var_private_boxchar_c8xU : !llvm.struct<(ptr, i64)> init {
^bb0(%arg0: !llvm.struct<(ptr, i64)>, %arg1: !llvm.struct<(ptr, i64)>):
  %0 = llvm.extractvalue %arg0[0] : !llvm.struct<(ptr, i64)>
  %1 = llvm.extractvalue %arg0[1] : !llvm.struct<(ptr, i64)>
  %2 = llvm.mlir.constant(1 : i64) : i64
  %3 = llvm.alloca %1 x i8 {bindc_name = "char_var", pinned} : (i64) -> !llvm.ptr
  %4 = llvm.mlir.undef : !llvm.struct<(ptr, i64)>
  %5 = llvm.insertvalue %3, %4[0] : !llvm.struct<(ptr, i64)>
  %6 = llvm.insertvalue %1, %5[1] : !llvm.struct<(ptr, i64)>
  omp.yield(%6 : !llvm.struct<(ptr, i64)>)
}
llvm.func @target_boxchar_(%arg0: !llvm.ptr {fir.bindc_name = "l"}) attributes {fir.internal_name = "_QPtarget_boxchar"} {
  %0 = llvm.mlir.constant(1 : i64) : i64
  %1 = llvm.alloca %0 x i32 {bindc_name = "mapped_var"} : (i64) -> !llvm.ptr
  %3 = llvm.alloca %0 x !llvm.struct<(ptr, i64)> : (i64) -> !llvm.ptr
  %4 = llvm.mlir.constant(0 : i64) : i64
  %5 = llvm.load %arg0 : !llvm.ptr -> i64
  %6 = llvm.icmp "sgt" %5, %4 : i64
  %7 = llvm.select %6, %5, %4 : i1, i64
  %9 = llvm.alloca %7 x i8 {bindc_name = "char_var"} : (i64) -> !llvm.ptr
  %10 = llvm.mlir.undef : !llvm.struct<(ptr, i64)>
  %11 = llvm.insertvalue %9, %10[0] : !llvm.struct<(ptr, i64)>
  %12 = llvm.insertvalue %7, %11[1] : !llvm.struct<(ptr, i64)>
  %13 = omp.map.info var_ptr(%1 : !llvm.ptr, i32) map_clauses(implicit, exit_release_or_enter_alloc) capture(ByCopy) -> !llvm.ptr {name = "mapped_var"}
  llvm.store %12, %3 : !llvm.struct<(ptr, i64)>, !llvm.ptr
  %14 = omp.map.info var_ptr(%3 : !llvm.ptr, !llvm.struct<(ptr, i64)>) map_clauses(to) capture(ByRef) -> !llvm.ptr
  omp.target map_entries(%13 -> %arg1, %14 -> %arg2 : !llvm.ptr, !llvm.ptr) private(@_QFtarget_boxcharEchar_var_private_boxchar_c8xU %12 -> %arg3 [map_idx=1] : !llvm.struct<(ptr, i64)>) {
    %15 = llvm.mlir.constant(0 : index) : i64
    %16 = llvm.mlir.constant(32 : i8) : i8
    %17 = llvm.mlir.constant(1 : index) : i64
    %18 = llvm.mlir.constant(false) : i1
    %19 = llvm.mlir.constant(5 : index) : i64
    %20 = llvm.mlir.constant(5 : i32) : i32
    %21 = llvm.extractvalue %arg3[0] : !llvm.struct<(ptr, i64)>
    %22 = llvm.extractvalue %arg3[1] : !llvm.struct<(ptr, i64)>
    llvm.store %20, %arg1 : i32, !llvm.ptr
    %23 = llvm.mlir.addressof @_QQclX68656C6C6F : !llvm.ptr
    %24 = llvm.icmp "slt" %22, %19 : i64
    %25 = llvm.select %24, %22, %19 : i1, i64
    llvm.call @llvm.memmove.p0.p0.i64(%21, %23, %25, %18) : (!llvm.ptr, !llvm.ptr, i64, i1) -> ()
    %26 = llvm.sub %22, %17 : i64
    %27 = llvm.mlir.undef : !llvm.array<1 x i8>
    %28 = llvm.insertvalue %16, %27[0] : !llvm.array<1 x i8>
    %29 = llvm.sub %26, %25 : i64
    %30 = llvm.add %29, %17 : i64
    llvm.br ^bb1(%25, %30 : i64, i64)
  ^bb1(%31: i64, %32: i64):  // 2 preds: ^bb0, ^bb2
    %33 = llvm.icmp "sgt" %32, %15 : i64
    llvm.cond_br %33, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %34 = llvm.getelementptr %21[%31] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<1 x i8>
    llvm.store %28, %34 : !llvm.array<1 x i8>, !llvm.ptr
    %35 = llvm.add %31, %17 : i64
    %36 = llvm.sub %32, %17 : i64
    llvm.br ^bb1(%35, %36 : i64, i64)
  ^bb3:  // pred: ^bb1
    omp.terminator
  }
  llvm.return
}
llvm.mlir.global linkonce constant @_QQclX68656C6C6F() comdat(@__llvm_comdat::@_QQclX68656C6C6F) {addr_space = 0 : i32} : !llvm.array<5 x i8> {
  %0 = llvm.mlir.constant("hello") : !llvm.array<5 x i8>
  llvm.return %0 : !llvm.array<5 x i8>
}
llvm.comdat @__llvm_comdat {
  llvm.comdat_selector @_QQclX68656C6C6F any
}
llvm.func @llvm.memmove.p0.p0.i64(!llvm.ptr, !llvm.ptr, i64, i1) attributes {sym_visibility = "private"}



// CHECK: define internal void @__omp_offloading_{{.*}}(ptr %{{[^,]+}}, ptr %[[MAPPED_ARG:.*]]) {
// CHECK: %[[BOXCHAR:.*]] = load { ptr, i64 }, ptr %[[MAPPED_ARG]]
// CHECK: %[[BOXCHAR_PTR:.*]] = extractvalue { ptr, i64 } %[[BOXCHAR]], 0
// CHECK: %[[BOXCHAR_i64:.*]] = extractvalue { ptr, i64 } %[[BOXCHAR]], 1
// CHECK: %[[MEM_ALLOC:.*]] = alloca i8, i64 %[[BOXCHAR_i64]]
// CHECK: %[[PRIV_BOXCHAR0:.*]] = insertvalue { ptr, i64 } undef, ptr %[[MEM_ALLOC]], 0
// CHECK: %[[PRIV_BOXCHAR1:.*]] = insertvalue { ptr, i64 } %[[PRIV_BOXCHAR0]], i64 %[[BOXCHAR_i64]], 1
