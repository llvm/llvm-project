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

  omp.private {type = firstprivate} @private_eye : i32 copy {
  ^bb0(%arg0: !llvm.ptr, %arg1: !llvm.ptr):
    %0 = llvm.load %arg0 : !llvm.ptr -> i32
    llvm.store %0, %arg1 : i32, !llvm.ptr
    omp.yield(%arg1 : !llvm.ptr)
  }
  omp.private {type = firstprivate} @boxchar_firstprivate : !llvm.struct<(ptr, i64)> init {
  ^bb0(%arg0: !llvm.struct<(ptr, i64)>, %arg1: !llvm.struct<(ptr, i64)>):
    %0 = llvm.extractvalue %arg0[0] : !llvm.struct<(ptr, i64)>
    %1 = llvm.extractvalue %arg0[1] : !llvm.struct<(ptr, i64)>
    %8 = llvm.call @malloc(%1) {bindc_name = "", uniq_name = ""} : (i64) -> !llvm.ptr
    %9 = llvm.mlir.undef : !llvm.struct<(ptr, i64)>
    %10 = llvm.insertvalue %8, %9[0] : !llvm.struct<(ptr, i64)>
    %11 = llvm.insertvalue %1, %10[1] : !llvm.struct<(ptr, i64)>
    omp.yield(%11 : !llvm.struct<(ptr, i64)>)
  } copy {
  ^bb0(%arg0: !llvm.struct<(ptr, i64)>, %arg1: !llvm.struct<(ptr, i64)>):
    %3 = llvm.extractvalue %arg0[0] : !llvm.struct<(ptr, i64)>
    %4 = llvm.extractvalue %arg0[1] : !llvm.struct<(ptr, i64)>
    %5 = llvm.extractvalue %arg1[0] : !llvm.struct<(ptr, i64)>
    %6 = llvm.extractvalue %arg1[1] : !llvm.struct<(ptr, i64)>
    %7 = llvm.icmp "slt" %6, %4 : i64
    %8 = llvm.select %7, %6, %4 : i1, i64
    "llvm.intr.memmove"(%5, %3, %8) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    omp.yield(%arg1 : !llvm.struct<(ptr, i64)>)
  } dealloc {
  ^bb0(%arg0: !llvm.struct<(ptr, i64)>):
    %0 = llvm.extractvalue %arg0[0] : !llvm.struct<(ptr, i64)>
    %1 = llvm.extractvalue %arg0[1] : !llvm.struct<(ptr, i64)>
    llvm.call @free(%0) : (!llvm.ptr) -> ()
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
    %124 = omp.map.info var_ptr(%21 : !llvm.ptr, i32) map_clauses(implicit, exit_release_or_enter_alloc) capture(ByCopy) -> !llvm.ptr {name = "i"}
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
    omp.target nowait map_entries(%124 -> %arg2, %160 -> %arg5, %159 -> %arg8, %1601 -> %arg9, %1591 -> %arg10  : !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) private(@firstprivatizer %19 -> %arg11 [map_idx=1], @firstprivatizer %20 -> %arg12 [map_idx=3] : !llvm.ptr, !llvm.ptr) {
      omp.terminator
    }
    %166 = llvm.mlir.constant(48 : i32) : i32
    %167 = llvm.getelementptr %19[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
    %168 = llvm.load %167 : !llvm.ptr -> !llvm.ptr
    llvm.call @free(%168) : (!llvm.ptr) -> ()
    llvm.return
  }

  llvm.func @target_boxchar_(%arg0: !llvm.ptr {fir.bindc_name = "l"}) attributes {fir.internal_name = "_QPtarget_boxchar", frame_pointer = #llvm.framePointerKind<all>, target_cpu = "x86-64"} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x i32 {bindc_name = "i"} : (i64) -> !llvm.ptr
    %2 = llvm.mlir.constant(1 : i64) : i64
    %3 = llvm.alloca %2 x !llvm.struct<(ptr, i64)> : (i64) -> !llvm.ptr
    %4 = llvm.mlir.constant(1 : index) : i64
    %5 = llvm.mlir.constant(0 : index) : i64
    %6 = llvm.mlir.constant(0 : i32) : i32
    %7 = llvm.mlir.constant(1 : i64) : i64
    %8 = llvm.mlir.constant(1 : i64) : i64
    %9 = llvm.load %arg0 : !llvm.ptr -> i32
    %10 = llvm.icmp "sgt" %9, %6 : i32
    %11 = llvm.select %10, %9, %6 : i1, i32
    %12 = llvm.mlir.constant(1 : i64) : i64
    %13 = llvm.sext %11 : i32 to i64
    %14 = llvm.alloca %13 x i8 {bindc_name = "char_var"} : (i64) -> !llvm.ptr
    %15 = llvm.mlir.undef : !llvm.struct<(ptr, i64)>
    %16 = llvm.sext %11 : i32 to i64
    %17 = llvm.insertvalue %14, %15[0] : !llvm.struct<(ptr, i64)>
    %18 = llvm.insertvalue %16, %17[1] : !llvm.struct<(ptr, i64)>
    llvm.store %18, %3 : !llvm.struct<(ptr, i64)>, !llvm.ptr
    %19 = llvm.load %3 : !llvm.ptr -> !llvm.struct<(ptr, i64)>
    %20 = llvm.extractvalue %19[0] : !llvm.struct<(ptr, i64)>
    %21 = llvm.extractvalue %19[1] : !llvm.struct<(ptr, i64)>
    %22 = llvm.sub %21, %4 : i64
    %23 = omp.map.bounds lower_bound(%5 : i64) upper_bound(%22 : i64) extent(%21 : i64) stride(%4 : i64) start_idx(%5 : i64) {stride_in_bytes = true}
    %24 = llvm.getelementptr %3[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64)>
    %25 = omp.map.info var_ptr(%3 : !llvm.ptr, i8) map_clauses(implicit, to) capture(ByRef) var_ptr_ptr(%24 : !llvm.ptr) bounds(%23) -> !llvm.ptr
    %26 = omp.map.info var_ptr(%3 : !llvm.ptr, !llvm.struct<(ptr, i64)>) map_clauses(to) capture(ByRef) members(%25 : [0] : !llvm.ptr) -> !llvm.ptr
    %27 = omp.map.info var_ptr(%1 : !llvm.ptr, i32) map_clauses(to) capture(ByCopy) -> !llvm.ptr
    omp.target nowait map_entries(%26 -> %arg1, %27 -> %arg2, %25 -> %arg3 : !llvm.ptr, !llvm.ptr, !llvm.ptr) private(@boxchar_firstprivate %18 -> %arg4 [map_idx=0], @private_eye %1 -> %arg5 [map_idx=1] : !llvm.struct<(ptr, i64)>, !llvm.ptr) {
      omp.terminator
    }
    llvm.return
  }
}

// CHECK-LABEL:       llvm.func @free(!llvm.ptr)
// CHECK: llvm.func @malloc(i64) -> !llvm.ptr

// CHECK-LABEL: llvm.func internal @firstprivate_test(
// CHECK-SAME: %[[ARG0:.*]]: !llvm.ptr {fir.bindc_name = "ptr0"},
// CHECK-SAME: %[[ARG1:.*]]: !llvm.ptr {fir.bindc_name = "ptr1"}) {
// CHECK: %[[VAL_0:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK: %[[VAL_1:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK: %[[VAL_2:.*]] = llvm.alloca %[[VAL_0]] x !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK: %[[VAL_3:.*]] = llvm.mlir.constant(48 : i64) : i64
// CHECK: %[[HEAP:.*]] = llvm.call @malloc(%[[VAL_3]]) : (i64) -> !llvm.ptr
// CHECK: %[[STACK:.*]] = llvm.alloca %[[VAL_0]] x !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> {bindc_name = "local"} : (i32) -> !llvm.ptr
// CHECK: %[[VAL_6:.*]] = llvm.alloca %[[VAL_0]] x i32 {bindc_name = "i"} : (i32) -> !llvm.ptr
// CHECK: %[[VAL_7:.*]] = llvm.mlir.undef : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
// CHECK: llvm.store %[[VAL_7]], %[[STACK]] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>, !llvm.ptr
// CHECK: llvm.store %[[VAL_0]], %[[VAL_6]] : i32, !llvm.ptr
// CHECK: %[[VAL_8:.*]] = omp.map.info var_ptr(%[[VAL_6]] : !llvm.ptr, i32) map_clauses(implicit, exit_release_or_enter_alloc) capture(ByCopy) -> !llvm.ptr {name = "i"}
// CHECK: %[[VAL_9:.*]] = llvm.getelementptr %[[STACK]][0, 7, %[[VAL_1]], 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
// CHECK: %[[VAL_10:.*]] = llvm.load %[[VAL_9]] : !llvm.ptr -> i64
// CHECK: %[[VAL_11:.*]] = llvm.getelementptr %[[STACK]][0, 7, %[[VAL_1]], 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
// CHECK: %[[VAL_12:.*]] = llvm.load %[[VAL_11]] : !llvm.ptr -> i64
// CHECK: %[[VAL_13:.*]] = llvm.getelementptr %[[STACK]][0, 7, %[[VAL_1]], 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
// CHECK: %[[VAL_14:.*]] = llvm.load %[[VAL_13]] : !llvm.ptr -> i64
// CHECK: %[[VAL_15:.*]] = llvm.sub %[[VAL_12]], %[[VAL_1]] : i64
// CHECK: %[[VAL_16:.*]] = omp.map.bounds lower_bound(%[[VAL_1]] : i64) upper_bound(%[[VAL_15]] : i64) extent(%[[VAL_12]] : i64) stride(%[[VAL_14]] : i64) start_idx(%[[VAL_10]] : i64) {stride_in_bytes = true}
// CHECK: %[[VAL_17:.*]] = llvm.call @firstprivatizer_init(%[[STACK]], %[[HEAP]]) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
// CHECK: %[[VAL_18:.*]] = llvm.call @firstprivatizer_copy(%[[STACK]], %[[VAL_17]]) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
// CHECK: %[[VAL_19:.*]] = llvm.getelementptr %[[HEAP]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
// CHECK: %[[VAL_20:.*]] = omp.map.info var_ptr(%[[HEAP]] : !llvm.ptr, i32) map_clauses({{.*}}to{{.*}}) capture(ByRef) var_ptr_ptr(%[[VAL_19]] : !llvm.ptr) bounds(%[[VAL_16]]) -> !llvm.ptr {name = ""}
// CHECK: %[[VAL_21:.*]] = omp.map.info var_ptr(%[[HEAP]] : !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>) map_clauses({{.*}}always{{.*}}to{{.*}}) capture(ByRef) members(%[[VAL_20]] : [0] : !llvm.ptr) -> !llvm.ptr
// CHECK: omp.target nowait map_entries(%[[VAL_8]] -> %[[VAL_22:.*]], %[[VAL_21]] -> %[[VAL_23:.*]], %[[VAL_20]] -> %[[VAL_24:.*]] : !llvm.ptr, !llvm.ptr, !llvm.ptr) private(@firstprivatizer %[[HEAP]] -> %[[VAL_25:.*]] [map_idx=1] : !llvm.ptr) {
// CHECK: omp.terminator
// CHECK: }
// CHECK: %[[VAL_26:.*]] = llvm.mlir.constant(48 : i32) : i32
// CHECK: %[[VAL_27:.*]] = llvm.getelementptr %[[STACK]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
// CHECK: %[[VAL_28:.*]] = llvm.load %[[VAL_27]] : !llvm.ptr -> !llvm.ptr
// CHECK: llvm.call @free(%[[VAL_28]]) : (!llvm.ptr) -> ()
// CHECK: llvm.return
// CHECK: }

// CHECK-LABEL:   llvm.func @target_boxchar_(
// CHECK-SAME:      %[[ARG0:.*]]: !llvm.ptr {fir.bindc_name = "l"}) attributes {fir.internal_name = "_QPtarget_boxchar", frame_pointer = #llvm.framePointerKind<all>, target_cpu = "x86-64"} {
// CHECK: %[[VAL_0:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK: %[[VAL_1:.*]] = llvm.alloca %[[VAL_0]] x i32 {bindc_name = "i"} : (i64) -> !llvm.ptr
// CHECK: %[[VAL_2:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK: %[[VAL_3:.*]] = llvm.mlir.constant(16 : i64) : i64
// CHECK: %[[VAL_4:.*]] = llvm.call @malloc(%[[VAL_3]]) : (i64) -> !llvm.ptr
// CHECK: %[[VAL_5:.*]] = llvm.alloca %[[VAL_2]] x !llvm.struct<(ptr, i64)> : (i64) -> !llvm.ptr
// CHECK: %[[VAL_6:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK: %[[VAL_7:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK: %[[VAL_8:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: %[[VAL_9:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK: %[[VAL_10:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK: %[[VAL_11:.*]] = llvm.load %[[ARG0]] : !llvm.ptr -> i32
// CHECK: %[[VAL_12:.*]] = llvm.icmp "sgt" %[[VAL_11]], %[[VAL_8]] : i32
// CHECK: %[[VAL_13:.*]] = llvm.select %[[VAL_12]], %[[VAL_11]], %[[VAL_8]] : i1, i32
// CHECK: %[[VAL_14:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK: %[[VAL_15:.*]] = llvm.sext %[[VAL_13]] : i32 to i64
// CHECK: %[[VAL_16:.*]] = llvm.alloca %[[VAL_15]] x i8 {bindc_name = "char_var"} : (i64) -> !llvm.ptr
// CHECK: %[[VAL_17:.*]] = llvm.mlir.undef : !llvm.struct<(ptr, i64)>
// CHECK: %[[VAL_18:.*]] = llvm.sext %[[VAL_13]] : i32 to i64
// CHECK: %[[VAL_19:.*]] = llvm.insertvalue %[[VAL_16]], %[[VAL_17]][0] : !llvm.struct<(ptr, i64)>
// CHECK: %[[VAL_20:.*]] = llvm.insertvalue %[[VAL_18]], %[[VAL_19]][1] : !llvm.struct<(ptr, i64)>
// CHECK: llvm.store %[[VAL_20]], %[[VAL_5]] : !llvm.struct<(ptr, i64)>, !llvm.ptr
// CHECK: %[[VAL_21:.*]] = llvm.load %[[VAL_5]] : !llvm.ptr -> !llvm.struct<(ptr, i64)>
// CHECK: %[[VAL_22:.*]] = llvm.extractvalue %[[VAL_21]][0] : !llvm.struct<(ptr, i64)>
// CHECK: %[[VAL_23:.*]] = llvm.extractvalue %[[VAL_21]][1] : !llvm.struct<(ptr, i64)>
// CHECK: %[[VAL_24:.*]] = llvm.sub %[[VAL_23]], %[[VAL_6]] : i64
// CHECK: %[[VAL_25:.*]] = omp.map.bounds lower_bound(%[[VAL_7]] : i64) upper_bound(%[[VAL_24]] : i64) extent(%[[VAL_23]] : i64) stride(%[[VAL_6]] : i64) start_idx(%[[VAL_7]] : i64) {stride_in_bytes = true}
// CHECK: %[[VAL_26:.*]] = llvm.load %[[VAL_5]] : !llvm.ptr -> !llvm.struct<(ptr, i64)>
// CHECK: %[[VAL_27:.*]] = llvm.load %[[VAL_4]] : !llvm.ptr -> !llvm.struct<(ptr, i64)>
// CHECK: %[[VAL_28:.*]] = llvm.call @boxchar_firstprivate_init(%[[VAL_26]], %[[VAL_27]]) : (!llvm.struct<(ptr, i64)>, !llvm.struct<(ptr, i64)>) -> !llvm.struct<(ptr, i64)>
// CHECK: %[[VAL_29:.*]] = llvm.call @boxchar_firstprivate_copy(%[[VAL_26]], %[[VAL_28]]) : (!llvm.struct<(ptr, i64)>, !llvm.struct<(ptr, i64)>) -> !llvm.struct<(ptr, i64)>
// CHECK: llvm.store %[[VAL_29]], %[[VAL_4]] : !llvm.struct<(ptr, i64)>, !llvm.ptr
// CHECK: %[[VAL_30:.*]] = omp.map.info var_ptr(%[[VAL_1]] : !llvm.ptr, i32) map_clauses(to) capture(ByCopy) -> !llvm.ptr
// CHECK: %[[VAL_31:.*]] = llvm.getelementptr %[[VAL_4]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64)>
// CHECK: %[[VAL_32:.*]] = omp.map.info var_ptr(%[[VAL_4]] : !llvm.ptr, i8) map_clauses(implicit, to) capture(ByRef) var_ptr_ptr(%[[VAL_31]] : !llvm.ptr) bounds(%[[VAL_25]]) -> !llvm.ptr
// CHECK: %[[VAL_33:.*]] = omp.map.info var_ptr(%[[VAL_4]] : !llvm.ptr, !llvm.struct<(ptr, i64)>) map_clauses(to) capture(ByRef) members(%[[VAL_32]] : [0] : !llvm.ptr) -> !llvm.ptr
// CHECK: %[[VAL_34:.*]] = llvm.load %[[VAL_4]] : !llvm.ptr -> !llvm.struct<(ptr, i64)>
// CHECK: omp.target nowait map_entries(%[[VAL_33]] -> %[[VAL_35:.*]], %[[VAL_30]] -> %[[VAL_36:.*]], %[[VAL_32]] -> %[[VAL_37:.*]] : !llvm.ptr, !llvm.ptr, !llvm.ptr) private(@boxchar_firstprivate %[[VAL_34]] -> %[[VAL_38:.*]] [map_idx=0], @private_eye %[[VAL_1]] -> %[[VAL_39:.*]] [map_idx=1] : !llvm.struct<(ptr, i64)>, !llvm.ptr) {
// CHECK: omp.terminator
// CHECK: }
// CHECK: llvm.return
// CHECK: }

// CHECK-LABEL: llvm.func @firstprivatizer_init(
// CHECK-SAME: %[[ARG0:.*]]: !llvm.ptr,
// CHECK-SAME: %[[ARG1:.*]]: !llvm.ptr) -> !llvm.ptr attributes {always_inline} {
// CHECK: %[[VAL_0:.*]] = llvm.mlir.constant(48 : i64) : i64
// CHECK: %[[VAL_1:.*]] = llvm.call @malloc(%[[VAL_0]]) : (i64) -> !llvm.ptr
// CHECK: %[[VAL_2:.*]] = llvm.getelementptr %[[ARG1]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
// CHECK: llvm.store %[[VAL_1]], %[[VAL_2]] : !llvm.ptr, !llvm.ptr
// CHECK: llvm.return %[[ARG1]] : !llvm.ptr
// CHECK: }

// CHECK-LABEL: llvm.func @firstprivatizer_copy(
// CHECK-SAME: %[[ARG0:.*]]: !llvm.ptr,
// CHECK-SAME: %[[ARG1:.*]]: !llvm.ptr) -> !llvm.ptr attributes {always_inline} {
// CHECK: %[[VAL_0:.*]] = llvm.mlir.constant(48 : i32) : i32
// CHECK: "llvm.intr.memcpy"(%[[ARG1]], %[[ARG0]], %[[VAL_0]]) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i32) -> ()
// CHECK: llvm.return %[[ARG1]] : !llvm.ptr
// CHECK: }


// CHECK-LABEL:   llvm.func @boxchar_firstprivate_init(
// CHECK-SAME:      %[[ARG0:.*]]: !llvm.struct<(ptr, i64)>,
// CHECK-SAME:      %[[ARG1:.*]]: !llvm.struct<(ptr, i64)>) -> !llvm.struct<(ptr, i64)> attributes {always_inline} {
// CHECK: %[[VAL_0:.*]] = llvm.extractvalue %[[ARG0]][0] : !llvm.struct<(ptr, i64)>
// CHECK: %[[VAL_1:.*]] = llvm.extractvalue %[[ARG0]][1] : !llvm.struct<(ptr, i64)>
// CHECK: %[[VAL_2:.*]] = llvm.call @malloc(%[[VAL_1]]) {bindc_name = "", uniq_name = ""} : (i64) -> !llvm.ptr
// CHECK: %[[VAL_3:.*]] = llvm.mlir.undef : !llvm.struct<(ptr, i64)>
// CHECK: %[[VAL_4:.*]] = llvm.insertvalue %[[VAL_2]], %[[VAL_3]][0] : !llvm.struct<(ptr, i64)>
// CHECK: %[[VAL_5:.*]] = llvm.insertvalue %[[VAL_1]], %[[VAL_4]][1] : !llvm.struct<(ptr, i64)>
// CHECK: llvm.return %[[VAL_5]] : !llvm.struct<(ptr, i64)>
// CHECK: }

// CHECK-LABEL:   llvm.func @boxchar_firstprivate_copy(
// CHECK-SAME:      %[[ARG0:.*]]: !llvm.struct<(ptr, i64)>,
// CHECK-SAME:      %[[ARG1:.*]]: !llvm.struct<(ptr, i64)>) -> !llvm.struct<(ptr, i64)> attributes {always_inline} {
// CHECK: %[[VAL_0:.*]] = llvm.extractvalue %[[ARG0]][0] : !llvm.struct<(ptr, i64)>
// CHECK: %[[VAL_1:.*]] = llvm.extractvalue %[[ARG0]][1] : !llvm.struct<(ptr, i64)>
// CHECK: %[[VAL_2:.*]] = llvm.extractvalue %[[ARG1]][0] : !llvm.struct<(ptr, i64)>
// CHECK: %[[VAL_3:.*]] = llvm.extractvalue %[[ARG1]][1] : !llvm.struct<(ptr, i64)>
// CHECK: %[[VAL_4:.*]] = llvm.icmp "slt" %[[VAL_3]], %[[VAL_1]] : i64
// CHECK: %[[VAL_5:.*]] = llvm.select %[[VAL_4]], %[[VAL_3]], %[[VAL_1]] : i1, i64
// CHECK: "llvm.intr.memmove"(%[[VAL_2]], %[[VAL_0]], %[[VAL_5]]) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
// CHECK: llvm.return %[[ARG1]] : !llvm.struct<(ptr, i64)>
// CHECK: }
