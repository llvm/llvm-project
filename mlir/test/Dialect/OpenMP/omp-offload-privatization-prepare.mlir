// RUN: mlir-opt --mlir-disable-threading -omp-offload-privatization-prepare --split-input-file %s | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>} {
  llvm.func @free(!llvm.ptr)
  llvm.func @malloc(i64) -> !llvm.ptr

  omp.private {type = firstprivate} @firstprivatizer : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> init {
  ^bb0(%arg0: !llvm.ptr, %arg1: !llvm.ptr):
    %0 = llvm.mlir.constant(48 : i64) : i64
    %1 = llvm.call @malloc(%0) : (i64) -> !llvm.ptr
    %2 = llvm.getelementptr %arg1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
    llvm.store %1, %2 : !llvm.ptr, !llvm.ptr
    omp.yield(%arg1 : !llvm.ptr)
  } copy {
  ^bb0(%arg0: !llvm.ptr, %arg1: !llvm.ptr):
    %0 = llvm.mlir.constant(48 : i32) : i32
    "llvm.intr.memcpy"(%arg1, %arg0, %0) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i32) -> ()
    omp.yield(%arg1 : !llvm.ptr)
  } dealloc {
  ^bb0(%arg0: !llvm.ptr):
    llvm.call @free(%arg0) : (!llvm.ptr) -> ()
    omp.yield
  }
  omp.private {type = firstprivate} @firstprivatizer_1 : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> init {
  ^bb0(%arg0: !llvm.ptr, %arg1: !llvm.ptr):
    %0 = llvm.mlir.constant(48 : i64) : i64
    %1 = llvm.call @malloc(%0) : (i64) -> !llvm.ptr
    %2 = llvm.getelementptr %arg1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
    llvm.store %1, %2 : !llvm.ptr, !llvm.ptr
    omp.yield(%arg1 : !llvm.ptr)
  } copy {
  ^bb0(%arg0: !llvm.ptr, %arg1: !llvm.ptr):
    %0 = llvm.mlir.constant(48 : i32) : i32
    "llvm.intr.memcpy"(%arg1, %arg0, %0) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i32) -> ()
    omp.yield(%arg1 : !llvm.ptr)
  } dealloc {
  ^bb0(%arg0: !llvm.ptr):
    llvm.call @free(%arg0) : (!llvm.ptr) -> ()
    omp.yield
  }

  llvm.func internal @firstprivate_test(%arg0: !llvm.ptr {fir.bindc_name = "ptr0"}, %arg1: !llvm.ptr {fir.bindc_name = "ptr1"}) {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(0 : index) : i64
    %5 = llvm.alloca %0 x !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %19 = llvm.alloca %0 x !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> {bindc_name = "local"} : (i32) -> !llvm.ptr
    %20 = llvm.alloca %0 x !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> {bindc_name = "glocal"} : (i32) -> !llvm.ptr
    %21 = llvm.alloca %0 x i32 {bindc_name = "i"} : (i32) -> !llvm.ptr
    %33 = llvm.mlir.undef : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
    llvm.store %33, %19 : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>, !llvm.ptr
    llvm.store %33, %20 : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>, !llvm.ptr
    llvm.store %0, %21 : i32, !llvm.ptr
    %124 = omp.map.info var_ptr(%21 : !llvm.ptr, i32) map_clauses(implicit) capture(ByCopy) -> !llvm.ptr {name = "i"}
    %150 = llvm.getelementptr %19[0, 7, %1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
    %151 = llvm.load %150 : !llvm.ptr -> i64
    %152 = llvm.getelementptr %19[0, 7, %1, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
    %153 = llvm.load %152 : !llvm.ptr -> i64
    %154 = llvm.getelementptr %19[0, 7, %1, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
    %155 = llvm.load %154 : !llvm.ptr -> i64
    %156 = llvm.sub %153, %1 : i64
    %157 = omp.map.bounds lower_bound(%1 : i64) upper_bound(%156 : i64) extent(%153 : i64) stride(%155 : i64) start_idx(%151 : i64) {stride_in_bytes = true}
    %158 = llvm.getelementptr %19[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
    %159 = omp.map.info var_ptr(%19 : !llvm.ptr, i32) map_clauses(descriptor_base_addr, to) capture(ByRef) var_ptr_ptr(%158 : !llvm.ptr) bounds(%157) -> !llvm.ptr {name = ""}
    %160 = omp.map.info var_ptr(%19 : !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>) map_clauses(always, descriptor, to) capture(ByRef) members(%159 : [0] : !llvm.ptr) -> !llvm.ptr
    %1501 = llvm.getelementptr %20[0, 7, %1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
    %1511 = llvm.load %1501 : !llvm.ptr -> i64
    %1521 = llvm.getelementptr %20[0, 7, %1, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
    %1531 = llvm.load %1521 : !llvm.ptr -> i64
    %1541 = llvm.getelementptr %20[0, 7, %1, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
    %1551 = llvm.load %1541 : !llvm.ptr -> i64
    %1561 = llvm.sub %1531, %1 : i64
    %1571 = omp.map.bounds lower_bound(%1 : i64) upper_bound(%1561 : i64) extent(%1531 : i64) stride(%1551 : i64) start_idx(%1511 : i64) {stride_in_bytes = true}
    %1581 = llvm.getelementptr %20[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
    %1591 = omp.map.info var_ptr(%20 : !llvm.ptr, i32) map_clauses(descriptor_base_addr, to) capture(ByRef) var_ptr_ptr(%1581 : !llvm.ptr) bounds(%1571) -> !llvm.ptr {name = ""}
    %1601 = omp.map.info var_ptr(%20 : !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>) map_clauses(always, descriptor, to) capture(ByRef) members(%1591 : [0] : !llvm.ptr) -> !llvm.ptr

    // Test with two firstprivate variables so that we test that even if there are multiple variables to be cleaned up
    // only one cleanup omp.task is generated.
    omp.target nowait map_entries(%124 -> %arg2, %160 -> %arg5, %159 -> %arg8, %1601 -> %arg9, %1591 -> %arg10  : !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) private(@firstprivatizer %19 -> %arg11 [map_idx=1], @firstprivatizer_1 %20 -> %arg12 [map_idx=3] : !llvm.ptr, !llvm.ptr) {
      omp.terminator
    }
    %166 = llvm.mlir.constant(48 : i32) : i32
    %167 = llvm.getelementptr %19[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
    %168 = llvm.load %167 : !llvm.ptr -> !llvm.ptr
    llvm.call @free(%168) : (!llvm.ptr) -> ()
    llvm.return
  }

}
// CHECK-LABEL:   llvm.func @free(!llvm.ptr)
// CHECK: llvm.func @malloc(i64) -> !llvm.ptr


// CHECK-LABEL:   llvm.func internal @firstprivate_test(
// CHECK-SAME:      %[[ARG0:.*]]: !llvm.ptr {fir.bindc_name = "ptr0"},
// CHECK-SAME:      %[[ARG1:.*]]: !llvm.ptr {fir.bindc_name = "ptr1"}) {
// CHECK: %[[VAL_0:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK: %[[VAL_1:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK: %[[VAL_2:.*]] = llvm.alloca %[[VAL_0]] x !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK: %[[VAL_3:.*]] = llvm.mlir.constant(48 : i64) : i64
// CHECK: %[[HEAP0:.*]] = llvm.call @malloc(%[[VAL_3]]) : (i64) -> !llvm.ptr
// CHECK: %[[VAL_5:.*]] = llvm.alloca %[[VAL_0]] x !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> {bindc_name = "local"} : (i32) -> !llvm.ptr
// CHECK: %[[VAL_6:.*]] = llvm.mlir.constant(48 : i64) : i64
// CHECK: %[[HEAP1:.*]] = llvm.call @malloc(%[[VAL_6]]) : (i64) -> !llvm.ptr
// CHECK: %[[VAL_8:.*]] = llvm.alloca %[[VAL_0]] x !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> {bindc_name = "glocal"} : (i32) -> !llvm.ptr
// CHECK: %[[VAL_9:.*]] = llvm.alloca %[[VAL_0]] x i32 {bindc_name = "i"} : (i32) -> !llvm.ptr
// CHECK: %[[VAL_10:.*]] = llvm.mlir.undef : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
// CHECK: llvm.store %[[VAL_10]], %[[VAL_5]] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>, !llvm.ptr
// CHECK: llvm.store %[[VAL_10]], %[[VAL_8]] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>, !llvm.ptr
// CHECK: llvm.store %[[VAL_0]], %[[VAL_9]] : i32, !llvm.ptr
// CHECK: %[[VAL_11:.*]] = omp.map.info var_ptr(%[[VAL_9]] : !llvm.ptr, i32) map_clauses(implicit) capture(ByCopy) -> !llvm.ptr {name = "i"}
// CHECK: %[[VAL_12:.*]] = llvm.getelementptr %[[VAL_5]][0, 7, %[[VAL_1]], 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
// CHECK: %[[VAL_13:.*]] = llvm.load %[[VAL_12]] : !llvm.ptr -> i64
// CHECK: %[[VAL_14:.*]] = llvm.getelementptr %[[VAL_5]][0, 7, %[[VAL_1]], 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
// CHECK: %[[VAL_15:.*]] = llvm.load %[[VAL_14]] : !llvm.ptr -> i64
// CHECK: %[[VAL_16:.*]] = llvm.getelementptr %[[VAL_5]][0, 7, %[[VAL_1]], 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
// CHECK: %[[VAL_17:.*]] = llvm.load %[[VAL_16]] : !llvm.ptr -> i64
// CHECK: %[[VAL_18:.*]] = llvm.sub %[[VAL_15]], %[[VAL_1]] : i64
// CHECK: %[[VAL_19:.*]] = omp.map.bounds lower_bound(%[[VAL_1]] : i64) upper_bound(%[[VAL_18]] : i64) extent(%[[VAL_15]] : i64) stride(%[[VAL_17]] : i64) start_idx(%[[VAL_13]] : i64) {stride_in_bytes = true}
// CHECK: %[[VAL_20:.*]] = llvm.call @firstprivatizer_init(%[[VAL_5]], %[[HEAP0]]) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
// CHECK: %[[VAL_21:.*]] = llvm.call @firstprivatizer_copy(%[[VAL_5]], %[[VAL_20]]) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
// CHECK: %[[VAL_22:.*]] = llvm.getelementptr %[[VAL_8]][0, 7, %[[VAL_1]], 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
// CHECK: %[[VAL_23:.*]] = llvm.load %[[VAL_22]] : !llvm.ptr -> i64
// CHECK: %[[VAL_24:.*]] = llvm.getelementptr %[[VAL_8]][0, 7, %[[VAL_1]], 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
// CHECK: %[[VAL_25:.*]] = llvm.load %[[VAL_24]] : !llvm.ptr -> i64
// CHECK: %[[VAL_26:.*]] = llvm.getelementptr %[[VAL_8]][0, 7, %[[VAL_1]], 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
// CHECK: %[[VAL_27:.*]] = llvm.load %[[VAL_26]] : !llvm.ptr -> i64
// CHECK: %[[VAL_28:.*]] = llvm.sub %[[VAL_25]], %[[VAL_1]] : i64
// CHECK: %[[VAL_29:.*]] = omp.map.bounds lower_bound(%[[VAL_1]] : i64) upper_bound(%[[VAL_28]] : i64) extent(%[[VAL_25]] : i64) stride(%[[VAL_27]] : i64) start_idx(%[[VAL_23]] : i64) {stride_in_bytes = true}
// CHECK: %[[VAL_30:.*]] = llvm.call @firstprivatizer_1_init(%[[VAL_8]], %[[HEAP1]]) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
// CHECK: %[[VAL_31:.*]] = llvm.call @firstprivatizer_1_copy(%[[VAL_8]], %[[VAL_30]]) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
// CHECK: %[[VAL_32:.*]] = llvm.getelementptr %[[HEAP0]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
// CHECK: %[[VAL_33:.*]] = omp.map.info var_ptr(%[[HEAP0]] : !llvm.ptr, i32) map_clauses({{.*}}to{{.*}}) capture(ByRef) var_ptr_ptr(%[[VAL_32]] : !llvm.ptr) bounds(%[[VAL_19]]) -> !llvm.ptr {name = ""}
// CHECK: %[[VAL_34:.*]] = omp.map.info var_ptr(%[[HEAP0]] : !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>) map_clauses(always,{{.*}}to) capture(ByRef) members(%[[VAL_33]] : [0] : !llvm.ptr) -> !llvm.ptr
// CHECK: %[[VAL_35:.*]] = llvm.getelementptr %[[HEAP1]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
// CHECK: %[[VAL_36:.*]] = omp.map.info var_ptr(%[[HEAP1]] : !llvm.ptr, i32) map_clauses({{.*}}to{{.*}}) capture(ByRef) var_ptr_ptr(%[[VAL_35]] : !llvm.ptr) bounds(%[[VAL_29]]) -> !llvm.ptr {name = ""}
// CHECK: %[[VAL_37:.*]] = omp.map.info var_ptr(%[[HEAP1]] : !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>) map_clauses(always,{{.*}}to) capture(ByRef) members(%[[VAL_36]] : [0] : !llvm.ptr) -> !llvm.ptr
// CHECK: omp.target depend(taskdependout -> %[[HEAP0]] : !llvm.ptr) nowait map_entries(%[[VAL_11]] -> %[[VAL_38:.*]], %[[VAL_34]] -> %[[VAL_39:.*]], %[[VAL_33]] -> %[[VAL_40:.*]], %[[VAL_37]] -> %[[VAL_41:.*]], %[[VAL_36]] -> %[[VAL_42:.*]] : !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) private(@firstprivatizer %[[HEAP0]] -> %[[VAL_43:.*]] [map_idx=1], @firstprivatizer_1 %[[HEAP1]] -> %[[VAL_44:.*]] [map_idx=3] : !llvm.ptr, !llvm.ptr) {
// CHECK: omp.terminator
// CHECK: }
// CHECK: omp.task depend(taskdependin -> %[[HEAP0]] : !llvm.ptr) {
// CHECK: llvm.call @firstprivatizer_1_dealloc(%[[VAL_31]]) : (!llvm.ptr) -> ()
// CHECK: llvm.call @free(%[[HEAP1]]) : (!llvm.ptr) -> ()
// CHECK: llvm.call @firstprivatizer_dealloc(%[[VAL_21]]) : (!llvm.ptr) -> ()
// CHECK: llvm.call @free(%[[HEAP0]]) : (!llvm.ptr) -> ()
// CHECK: omp.terminator
// CHECK: }
// CHECK: %[[VAL_45:.*]] = llvm.mlir.constant(48 : i32) : i32
// CHECK: %[[VAL_46:.*]] = llvm.getelementptr %[[VAL_5]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
// CHECK: %[[VAL_47:.*]] = llvm.load %[[VAL_46]] : !llvm.ptr -> !llvm.ptr
// CHECK: llvm.call @free(%[[VAL_47]]) : (!llvm.ptr) -> ()
// CHECK: llvm.return
// CHECK: }

// CHECK-LABEL:   llvm.func @firstprivatizer_init(
// CHECK-SAME:      %[[ARG0:.*]]: !llvm.ptr,
// CHECK-SAME:      %[[ARG1:.*]]: !llvm.ptr) -> !llvm.ptr attributes {always_inline} {
// CHECK: %[[VAL_0:.*]] = llvm.mlir.constant(48 : i64) : i64
// CHECK: %[[VAL_1:.*]] = llvm.call @malloc(%[[VAL_0]]) : (i64) -> !llvm.ptr
// CHECK: %[[VAL_2:.*]] = llvm.getelementptr %[[ARG1]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
// CHECK: llvm.store %[[VAL_1]], %[[VAL_2]] : !llvm.ptr, !llvm.ptr
// CHECK: llvm.return %[[ARG1]] : !llvm.ptr
// CHECK: }

// CHECK-LABEL:   llvm.func @firstprivatizer_copy(
// CHECK-SAME:      %[[ARG0:.*]]: !llvm.ptr,
// CHECK-SAME:      %[[ARG1:.*]]: !llvm.ptr) -> !llvm.ptr attributes {always_inline} {
// CHECK: %[[VAL_0:.*]] = llvm.mlir.constant(48 : i32) : i32
// CHECK: "llvm.intr.memcpy"(%[[ARG1]], %[[ARG0]], %[[VAL_0]]) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i32) -> ()
// CHECK: llvm.return %[[ARG1]] : !llvm.ptr
// CHECK: }

// CHECK-LABEL:   llvm.func @firstprivatizer_dealloc(
// CHECK-SAME:      %[[ARG0:.*]]: !llvm.ptr) attributes {always_inline} {
// CHECK: llvm.call @free(%[[ARG0]]) : (!llvm.ptr) -> ()
// CHECK: llvm.return
// CHECK: }

// CHECK-LABEL:   llvm.func @firstprivatizer_1_init(
// CHECK-SAME:      %[[ARG0:.*]]: !llvm.ptr,
// CHECK-SAME:      %[[ARG1:.*]]: !llvm.ptr) -> !llvm.ptr attributes {always_inline} {
// CHECK: %[[VAL_0:.*]] = llvm.mlir.constant(48 : i64) : i64
// CHECK: %[[VAL_1:.*]] = llvm.call @malloc(%[[VAL_0]]) : (i64) -> !llvm.ptr
// CHECK: %[[VAL_2:.*]] = llvm.getelementptr %[[ARG1]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
// CHECK: llvm.store %[[VAL_1]], %[[VAL_2]] : !llvm.ptr, !llvm.ptr
// CHECK: llvm.return %[[ARG1]] : !llvm.ptr
// CHECK: }

// CHECK-LABEL:   llvm.func @firstprivatizer_1_copy(
// CHECK-SAME:      %[[ARG0:.*]]: !llvm.ptr,
// CHECK-SAME:      %[[ARG1:.*]]: !llvm.ptr) -> !llvm.ptr attributes {always_inline} {
// CHECK: %[[VAL_0:.*]] = llvm.mlir.constant(48 : i32) : i32
// CHECK: "llvm.intr.memcpy"(%[[ARG1]], %[[ARG0]], %[[VAL_0]]) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i32) -> ()
// CHECK: llvm.return %[[ARG1]] : !llvm.ptr
// CHECK: }

// CHECK-LABEL:   llvm.func @firstprivatizer_1_dealloc(
// CHECK-SAME:      %[[ARG0:.*]]: !llvm.ptr) attributes {always_inline} {
// CHECK: llvm.call @free(%[[ARG0]]) : (!llvm.ptr) -> ()
// CHECK: llvm.return
// CHECK: }
