// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// CHECK-DAG: %struct.__tgt_offload_entry = type { ptr, ptr, i64, i32, i32 }
// CHECK-DAG: !omp_offload.info = !{!{{.*}}}
module attributes {llvm.target_triple = "x86_64-unknown-linux-gnu", omp.is_target_device = false} {

  // CHECK-DAG: @_QMtest_0Earray_1d = global [3 x i32] [i32 1, i32 2, i32 3]
  // CHECK-DAG: @_QMtest_0Earray_1d_decl_tgt_ref_ptr = weak global ptr @_QMtest_0Earray_1d
  // CHECK-DAG: @.omp_offloading.entry_name{{.*}} = internal unnamed_addr constant [36 x i8] c"_QMtest_0Earray_1d_decl_tgt_ref_ptr\00"
  // CHECK-DAG: @.omp_offloading.entry._QMtest_0Earray_1d_decl_tgt_ref_ptr = weak constant %struct.__tgt_offload_entry { ptr @_QMtest_0Earray_1d_decl_tgt_ref_ptr, ptr @.omp_offloading.entry_name{{.*}}, i64 8, i32 1, i32 0 }, section "omp_offloading_entries", align 1
  // CHECK-DAG: !{{.*}} = !{i32 {{.*}}, !"_QMtest_0Earray_1d_decl_tgt_ref_ptr", i32 {{.*}}, i32 {{.*}}}
  llvm.mlir.global external @_QMtest_0Earray_1d(dense<[1, 2, 3]> : tensor<3xi32>) {addr_space = 0 : i32, omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (link)>} : !llvm.array<3 x i32>

  // CHECK-DAG: @_QMtest_0Earray_2d = global [2 x [2 x i32]] {{.*}}
  // CHECK-DAG: @_QMtest_0Earray_2d_decl_tgt_ref_ptr = weak global ptr @_QMtest_0Earray_2d
  // CHECK-DAG: @.omp_offloading.entry_name{{.*}} = internal unnamed_addr constant [36 x i8] c"_QMtest_0Earray_2d_decl_tgt_ref_ptr\00"
  // CHECK-DAG: @.omp_offloading.entry._QMtest_0Earray_2d_decl_tgt_ref_ptr = weak constant %struct.__tgt_offload_entry { ptr @_QMtest_0Earray_2d_decl_tgt_ref_ptr, ptr @.omp_offloading.entry_name{{.*}}, i64 8, i32 1, i32 0 }, section "omp_offloading_entries", align 1
  // CHECK-DAG: !{{.*}} = !{i32 {{.*}}, !"_QMtest_0Earray_2d_decl_tgt_ref_ptr", i32 {{.*}}, i32 {{.*}}}
  llvm.mlir.global external @_QMtest_0Earray_2d() {addr_space = 0 : i32, omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (link)>} : !llvm.array<2 x array<2 x i32>> {
    %0 = llvm.mlir.undef : !llvm.array<2 x array<2 x i32>>
    %1 = llvm.mlir.constant(1 : i32) : i32
    %2 = llvm.insertvalue %1, %0[0, 0] : !llvm.array<2 x array<2 x i32>>
    %3 = llvm.mlir.constant(2 : i32) : i32
    %4 = llvm.insertvalue %3, %2[0, 1] : !llvm.array<2 x array<2 x i32>>
    %5 = llvm.mlir.constant(3 : i32) : i32
    %6 = llvm.insertvalue %5, %4[1, 0] : !llvm.array<2 x array<2 x i32>>
    %7 = llvm.mlir.constant(4 : i32) : i32
    %8 = llvm.insertvalue %7, %6[1, 1] : !llvm.array<2 x array<2 x i32>>
    %9 = llvm.mlir.constant(2 : index) : i64
    %10 = llvm.mlir.constant(2 : index) : i64
    llvm.return %8 : !llvm.array<2 x array<2 x i32>>
  }

  // CHECK-DAG:  @_QMtest_0Edata_extended_link_1 = global float 2.000000e+00
  // CHECK-DAG:  @_QMtest_0Edata_extended_link_1_decl_tgt_ref_ptr = weak global ptr @_QMtest_0Edata_extended_link_1
  // CHECK-DAG:  @.omp_offloading.entry_name{{.*}} = internal unnamed_addr constant [48 x i8] c"_QMtest_0Edata_extended_link_1_decl_tgt_ref_ptr\00"
  // CHECK-DAG:  @.omp_offloading.entry._QMtest_0Edata_extended_link_1_decl_tgt_ref_ptr = weak constant %struct.__tgt_offload_entry { ptr @_QMtest_0Edata_extended_link_1_decl_tgt_ref_ptr, ptr @.omp_offloading.entry_name{{.*}}, i64 8, i32 1, i32 0 }, section "omp_offloading_entries", align 1
  // CHECK-DAG:  !{{.*}} = !{i32 {{.*}}, !"_QMtest_0Edata_extended_link_1_decl_tgt_ref_ptr", i32 {{.*}}, i32 {{.*}}}
  llvm.mlir.global external @_QMtest_0Edata_extended_link_1() {addr_space = 0 : i32, omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (link)>} : f32 {
    %0 = llvm.mlir.constant(2.000000e+00 : f32) : f32
    llvm.return %0 : f32
  }

  // CHECK-DAG:  @_QMtest_0Edata_extended_link_2 = global float 3.000000e+00
  // CHECK-DAG:  @_QMtest_0Edata_extended_link_2_decl_tgt_ref_ptr = weak global ptr @_QMtest_0Edata_extended_link_2
  // CHECK-DAG:  @.omp_offloading.entry_name{{.*}} = internal unnamed_addr constant [48 x i8] c"_QMtest_0Edata_extended_link_2_decl_tgt_ref_ptr\00"
  // CHECK-DAG:  @.omp_offloading.entry._QMtest_0Edata_extended_link_2_decl_tgt_ref_ptr = weak constant %struct.__tgt_offload_entry { ptr @_QMtest_0Edata_extended_link_2_decl_tgt_ref_ptr, ptr @.omp_offloading.entry_name{{.*}}, i64 8, i32 1, i32 0 }, section "omp_offloading_entries", align 1
  // CHECK-DAG:  !{{.*}} = !{i32 {{.*}}, !"_QMtest_0Edata_extended_link_2_decl_tgt_ref_ptr", i32 {{.*}}, i32 {{.*}}}
  llvm.mlir.global external @_QMtest_0Edata_extended_link_2() {addr_space = 0 : i32, omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (link)>} : f32 {
    %0 = llvm.mlir.constant(3.000000e+00 : f32) : f32
    llvm.return %0 : f32
  }

  // CHECK-DAG: @_QMtest_0Edata_extended_to_1 = global float 2.000000e+00
  // CHECK-DAG: @.omp_offloading.entry_name{{.*}} = internal unnamed_addr constant [29 x i8] c"_QMtest_0Edata_extended_to_1\00"
  // CHECK-DAG: @.omp_offloading.entry._QMtest_0Edata_extended_to_1 = weak constant %struct.__tgt_offload_entry { ptr @_QMtest_0Edata_extended_to_1, ptr @.omp_offloading.entry_name{{.*}}, i64 4, i32 0, i32 0 }, section "omp_offloading_entries", align 1
  // CHECK-DAG: !{{.*}} = !{i32 {{.*}}, !"_QMtest_0Edata_extended_to_1", i32 {{.*}}, i32 {{.*}}}
  llvm.mlir.global external @_QMtest_0Edata_extended_to_1() {addr_space = 0 : i32, omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (to)>} : f32 {
    %0 = llvm.mlir.constant(2.000000e+00 : f32) : f32
    llvm.return %0 : f32
  }

  // CHECK-DAG: @_QMtest_0Edata_extended_enter_1 = global float 2.000000e+00
  // CHECK-DAG: @.omp_offloading.entry_name{{.*}} = internal unnamed_addr constant [32 x i8] c"_QMtest_0Edata_extended_enter_1\00"
  // CHECK-DAG: @.omp_offloading.entry._QMtest_0Edata_extended_enter_1 = weak constant %struct.__tgt_offload_entry { ptr @_QMtest_0Edata_extended_enter_1, ptr @.omp_offloading.entry_name{{.*}}, i64 4, i32 0, i32 0 }, section "omp_offloading_entries", align 1
  // CHECK-DAG: !{{.*}} = !{i32 {{.*}}, !"_QMtest_0Edata_extended_enter_1", i32 {{.*}}, i32 {{.*}}}
  llvm.mlir.global external @_QMtest_0Edata_extended_enter_1() {addr_space = 0 : i32, omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (enter)>} : f32 {
    %0 = llvm.mlir.constant(2.000000e+00 : f32) : f32
    llvm.return %0 : f32
  }

  // CHECK-DAG: @_QMtest_0Edata_extended_to_2 = global float 3.000000e+00
  // CHECK-DAG: @.omp_offloading.entry_name{{.*}} = internal unnamed_addr constant [29 x i8] c"_QMtest_0Edata_extended_to_2\00"
  // CHECK-DAG: @.omp_offloading.entry._QMtest_0Edata_extended_to_2 = weak constant %struct.__tgt_offload_entry { ptr @_QMtest_0Edata_extended_to_2, ptr @.omp_offloading.entry_name{{.*}}, i64 4, i32 0, i32 0 }, section "omp_offloading_entries", align 1
  // CHECK-DAG: !{{.*}} = !{i32 {{.*}}, !"_QMtest_0Edata_extended_to_2", i32 {{.*}}, i32 {{.*}}}
  llvm.mlir.global external @_QMtest_0Edata_extended_to_2() {addr_space = 0 : i32, omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (to)>} : f32 {
    %0 = llvm.mlir.constant(3.000000e+00 : f32) : f32
    llvm.return %0 : f32
  }

  // CHECK-DAG: @_QMtest_0Edata_extended_enter_2 = global float 3.000000e+00
  // CHECK-DAG: @.omp_offloading.entry_name{{.*}} = internal unnamed_addr constant [32 x i8] c"_QMtest_0Edata_extended_enter_2\00"
  // CHECK-DAG: @.omp_offloading.entry._QMtest_0Edata_extended_enter_2 = weak constant %struct.__tgt_offload_entry { ptr @_QMtest_0Edata_extended_enter_2, ptr @.omp_offloading.entry_name{{.*}}, i64 4, i32 0, i32 0 }, section "omp_offloading_entries", align 1
  // CHECK-DAG: !{{.*}} = !{i32 {{.*}}, !"_QMtest_0Edata_extended_enter_2", i32 {{.*}}, i32 {{.*}}}
  llvm.mlir.global external @_QMtest_0Edata_extended_enter_2() {addr_space = 0 : i32, omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (enter)>} : f32 {
    %0 = llvm.mlir.constant(3.000000e+00 : f32) : f32
    llvm.return %0 : f32
  }

  // CHECK-DAG: @_QMtest_0Edata_int = global i32 1
  // CHECK-DAG: @_QMtest_0Edata_int_decl_tgt_ref_ptr = weak global ptr @_QMtest_0Edata_int
  // CHECK-DAG: @.omp_offloading.entry_name{{.*}} = internal unnamed_addr constant [36 x i8] c"_QMtest_0Edata_int_decl_tgt_ref_ptr\00"
  // CHECK-DAG: @.omp_offloading.entry._QMtest_0Edata_int_decl_tgt_ref_ptr = weak constant %struct.__tgt_offload_entry { ptr @_QMtest_0Edata_int_decl_tgt_ref_ptr, ptr @.omp_offloading.entry_name{{.*}}, i64 8, i32 1, i32 0 }, section "omp_offloading_entries", align 1
  // CHECK-DAG: !{{.*}} = !{i32 {{.*}}, !"_QMtest_0Edata_int_decl_tgt_ref_ptr", i32 {{.*}}, i32 {{.*}}}
  llvm.mlir.global external @_QMtest_0Edata_int() {addr_space = 0 : i32, omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (link)>} : i32 {
    %0 = llvm.mlir.constant(10 : i32) : i32
    llvm.return %0 : i32
  }

  // CHECK-DAG: @_QMtest_0Edata_int_clauseless_to = global i32 1
  // CHECK-DAG: @.omp_offloading.entry_name{{.*}} = internal unnamed_addr constant [33 x i8] c"_QMtest_0Edata_int_clauseless_to\00"
  // CHECK-DAG: @.omp_offloading.entry._QMtest_0Edata_int_clauseless_to = weak constant %struct.__tgt_offload_entry { ptr @_QMtest_0Edata_int_clauseless_to, ptr @.omp_offloading.entry_name{{.*}}, i64 4, i32 0, i32 0 }, section "omp_offloading_entries", align 1
  // CHECK-DAG: !{{.*}} = !{i32 {{.*}}, !"_QMtest_0Edata_int_clauseless_to", i32 {{.*}}, i32 {{.*}}}
  llvm.mlir.global external @_QMtest_0Edata_int_clauseless_to() {addr_space = 0 : i32, omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (to)>} : i32 {
    %0 = llvm.mlir.constant(1 : i32) : i32
    llvm.return %0 : i32
  }

  // CHECK-DAG: @_QMtest_0Edata_int_clauseless_enter = global i32 1
  // CHECK-DAG: @.omp_offloading.entry_name{{.*}} = internal unnamed_addr constant [36 x i8] c"_QMtest_0Edata_int_clauseless_enter\00"
  // CHECK-DAG: @.omp_offloading.entry._QMtest_0Edata_int_clauseless_enter = weak constant %struct.__tgt_offload_entry { ptr @_QMtest_0Edata_int_clauseless_enter, ptr @.omp_offloading.entry_name{{.*}}, i64 4, i32 0, i32 0 }, section "omp_offloading_entries", align 1
  // CHECK-DAG: !{{.*}} = !{i32 {{.*}}, !"_QMtest_0Edata_int_clauseless_enter", i32 {{.*}}, i32 {{.*}}}
  llvm.mlir.global external @_QMtest_0Edata_int_clauseless_enter() {addr_space = 0 : i32, omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (enter)>} : i32 {
    %0 = llvm.mlir.constant(1 : i32) : i32
    llvm.return %0 : i32
  }

  // CHECK-DAG: @_QMtest_0Edata_int_to = global i32 5
  // CHECK-DAG: @.omp_offloading.entry_name{{.*}} = internal unnamed_addr constant [22 x i8] c"_QMtest_0Edata_int_to\00"
  // CHECK-DAG: @.omp_offloading.entry._QMtest_0Edata_int_to = weak constant %struct.__tgt_offload_entry { ptr @_QMtest_0Edata_int_to, ptr @.omp_offloading.entry_name{{.*}}, i64 4, i32 0, i32 0 }, section "omp_offloading_entries", align 1
  // CHECK-DAG: !{{.*}} = !{i32 {{.*}}, !"_QMtest_0Edata_int_to", i32 {{.*}}, i32 {{.*}}}
  llvm.mlir.global external @_QMtest_0Edata_int_to() {addr_space = 0 : i32, omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (to)>} : i32 {
    %0 = llvm.mlir.constant(5 : i32) : i32
    llvm.return %0 : i32
  }

  // CHECK-DAG: @_QMtest_0Edata_int_enter = global i32 5
  // CHECK-DAG: @.omp_offloading.entry_name{{.*}} = internal unnamed_addr constant [25 x i8] c"_QMtest_0Edata_int_enter\00"
  // CHECK-DAG: @.omp_offloading.entry._QMtest_0Edata_int_enter = weak constant %struct.__tgt_offload_entry { ptr @_QMtest_0Edata_int_enter, ptr @.omp_offloading.entry_name{{.*}}, i64 4, i32 0, i32 0 }, section "omp_offloading_entries", align 1
  // CHECK-DAG: !{{.*}} = !{i32 {{.*}}, !"_QMtest_0Edata_int_enter", i32 {{.*}}, i32 {{.*}}}
  llvm.mlir.global external @_QMtest_0Edata_int_enter() {addr_space = 0 : i32, omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (enter)>} : i32 {
    %0 = llvm.mlir.constant(5 : i32) : i32
    llvm.return %0 : i32
  }

  // CHECK-DAG: @_QMtest_0Ept1 = global { ptr, i64, i32, i8, i8, i8, i8 } { ptr null, i64 ptrtoint (ptr getelementptr (i32, ptr null, i32 1) to i64), i32 20180515, i8 0, i8 9, i8 1, i8 0 }
  // CHECK-DAG: @_QMtest_0Ept1_decl_tgt_ref_ptr = weak global ptr @_QMtest_0Ept1
  // CHECK-DAG: @.omp_offloading.entry_name{{.*}} = internal unnamed_addr constant [31 x i8] c"_QMtest_0Ept1_decl_tgt_ref_ptr\00"
  // CHECK-DAG: @.omp_offloading.entry._QMtest_0Ept1_decl_tgt_ref_ptr = weak constant %struct.__tgt_offload_entry { ptr @_QMtest_0Ept1_decl_tgt_ref_ptr, ptr @.omp_offloading.entry_name{{.*}}, i64 8, i32 1, i32 0 }, section "omp_offloading_entries", align 1
  // CHECK-DAG: !{{.*}} = !{i32 {{.*}}, !"_QMtest_0Ept1_decl_tgt_ref_ptr", i32 {{.*}}, i32 {{.*}}}
  llvm.mlir.global external @_QMtest_0Ept1() {addr_space = 0 : i32, omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (link)>} : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)> {
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = llvm.mlir.constant(9 : i32) : i32
    %2 = llvm.mlir.zero : !llvm.ptr
    %3 = llvm.getelementptr %2[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %4 = ptr.ptrtoint %3 : !llvm.ptr to i64
    %5 = llvm.mlir.undef : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>
    %6 = llvm.insertvalue %4, %5[1] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>
    %7 = llvm.mlir.constant(20180515 : i32) : i32
    %8 = llvm.insertvalue %7, %6[2] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>
    %9 = llvm.mlir.constant(0 : i32) : i32
    %10 = llvm.trunc %9 : i32 to i8
    %11 = llvm.insertvalue %10, %8[3] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>
    %12 = llvm.trunc %1 : i32 to i8
    %13 = llvm.insertvalue %12, %11[4] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>
    %14 = llvm.mlir.constant(1 : i32) : i32
    %15 = llvm.trunc %14 : i32 to i8
    %16 = llvm.insertvalue %15, %13[5] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>
    %17 = llvm.mlir.constant(0 : i32) : i32
    %18 = llvm.trunc %17 : i32 to i8
    %19 = llvm.insertvalue %18, %16[6] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>
    %20 = llvm.bitcast %0 : !llvm.ptr to !llvm.ptr
    %21 = llvm.insertvalue %20, %19[0] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>
    llvm.return %21 : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>
  }

  // CHECK-DAG: @_QMtest_0Ept2_tar = global i32 5
  // CHECK-DAG: @_QMtest_0Ept2_tar_decl_tgt_ref_ptr = weak global ptr @_QMtest_0Ept2_tar
  // CHECK-DAG: @.omp_offloading.entry_name{{.*}} = internal unnamed_addr constant [35 x i8] c"_QMtest_0Ept2_tar_decl_tgt_ref_ptr\00"
  // CHECK-DAG: @.omp_offloading.entry._QMtest_0Ept2_tar_decl_tgt_ref_ptr = weak constant %struct.__tgt_offload_entry { ptr @_QMtest_0Ept2_tar_decl_tgt_ref_ptr, ptr @.omp_offloading.entry_name{{.*}}, i64 8, i32 1, i32 0 }, section "omp_offloading_entries", align 1
  // CHECK-DAG: !{{.*}} = !{i32 {{.*}}, !"_QMtest_0Ept2_tar_decl_tgt_ref_ptr", i32 {{.*}}, i32 {{.*}}}
  llvm.mlir.global external @_QMtest_0Ept2_tar() {addr_space = 0 : i32, omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (link)>} : i32 {
    %0 = llvm.mlir.constant(5 : i32) : i32
    llvm.return %0 : i32
  }
}
