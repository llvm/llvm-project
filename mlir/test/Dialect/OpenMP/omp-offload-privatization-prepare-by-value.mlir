// RUN: mlir-opt --mlir-disable-threading -omp-offload-privatization-prepare --split-input-file %s | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>} {
  llvm.func @free(!llvm.ptr)
  llvm.func @malloc(i64) -> !llvm.ptr

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
// CHECK-LABEL:   llvm.func @target_boxchar_(
// CHECK-SAME:      %[[ARG0:.*]]: !llvm.ptr {fir.bindc_name = "l"}) attributes {fir.internal_name = "_QPtarget_boxchar", frame_pointer = #llvm.framePointerKind<all>, target_cpu = "x86-64"} {
// CHECK: %[[VAL_0:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK: %[[VAL_1:.*]] = llvm.alloca %[[VAL_0]] x i32 {bindc_name = "i"} : (i64) -> !llvm.ptr
// CHECK: %[[VAL_2:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK: %[[VAL_3:.*]] = llvm.mlir.constant(16 : i64) : i64
// CHECK: %[[HEAP0:.*]] = llvm.call @malloc(%[[VAL_3]]) : (i64) -> !llvm.ptr
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
// CHECK: %[[VAL_27:.*]] = llvm.load %[[HEAP0]] : !llvm.ptr -> !llvm.struct<(ptr, i64)>
// CHECK: %[[VAL_28:.*]] = llvm.call @boxchar_firstprivate_init(%[[VAL_26]], %[[VAL_27]]) : (!llvm.struct<(ptr, i64)>, !llvm.struct<(ptr, i64)>) -> !llvm.struct<(ptr, i64)>
// CHECK: %[[VAL_29:.*]] = llvm.call @boxchar_firstprivate_copy(%[[VAL_26]], %[[VAL_28]]) : (!llvm.struct<(ptr, i64)>, !llvm.struct<(ptr, i64)>) -> !llvm.struct<(ptr, i64)>
// CHECK: llvm.store %[[VAL_29]], %[[HEAP0]] : !llvm.struct<(ptr, i64)>, !llvm.ptr
// CHECK: %[[VAL_30:.*]] = omp.map.info var_ptr(%[[VAL_1]] : !llvm.ptr, i32) map_clauses(to) capture(ByCopy) -> !llvm.ptr
// CHECK: %[[VAL_31:.*]] = llvm.getelementptr %[[HEAP0]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64)>
// CHECK: %[[VAL_32:.*]] = omp.map.info var_ptr(%[[HEAP0]] : !llvm.ptr, i8) map_clauses(implicit, to) capture(ByRef) var_ptr_ptr(%[[VAL_31]] : !llvm.ptr) bounds(%[[VAL_25]]) -> !llvm.ptr
// CHECK: %[[VAL_33:.*]] = omp.map.info var_ptr(%[[HEAP0]] : !llvm.ptr, !llvm.struct<(ptr, i64)>) map_clauses(to) capture(ByRef) members(%[[VAL_32]] : [0] : !llvm.ptr) -> !llvm.ptr
// CHECK: %[[VAL_34:.*]] = llvm.load %[[HEAP0]] : !llvm.ptr -> !llvm.struct<(ptr, i64)>
// CHECK: omp.target depend(taskdependout -> %[[HEAP0]] : !llvm.ptr) nowait map_entries(%[[VAL_33]] -> %[[VAL_35:.*]], %[[VAL_30]] -> %[[VAL_36:.*]], %[[VAL_32]] -> %[[VAL_37:.*]] : !llvm.ptr, !llvm.ptr, !llvm.ptr) private(@boxchar_firstprivate %[[VAL_34]] -> %[[VAL_38:.*]] [map_idx=0], @private_eye %[[VAL_1]] -> %[[VAL_39:.*]] [map_idx=1] : !llvm.struct<(ptr, i64)>, !llvm.ptr) {
// CHECK: omp.terminator
// CHECK: }
// CHECK: omp.task depend(taskdependin -> %[[HEAP0]] : !llvm.ptr) {
// CHECK: llvm.call @boxchar_firstprivate_dealloc(%[[VAL_29]]) : (!llvm.struct<(ptr, i64)>) -> ()
// CHECK: llvm.call @free(%[[HEAP0]]) : (!llvm.ptr) -> ()
// CHECK: omp.terminator
// CHECK: }
// CHECK: llvm.return
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

// CHECK-LABEL:   llvm.func @boxchar_firstprivate_dealloc(
// CHECK-SAME:      %[[ARG0:.*]]: !llvm.struct<(ptr, i64)>) attributes {always_inline} {
// CHECK: %[[VAL_0:.*]] = llvm.extractvalue %[[ARG0]][0] : !llvm.struct<(ptr, i64)>
// CHECK: %[[VAL_1:.*]] = llvm.extractvalue %[[ARG0]][1] : !llvm.struct<(ptr, i64)>
// CHECK: llvm.call @free(%[[VAL_0]]) : (!llvm.ptr) -> ()
// CHECK: llvm.return
// CHECK: }
