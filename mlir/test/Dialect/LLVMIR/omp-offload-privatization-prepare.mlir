// RUN: mlir-opt --mlir-disable-threading -omp-offload-privatization-prepare --split-input-file %s | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>} {
  llvm.func @free(!llvm.ptr)
  omp.private {type = private} @privatizer : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> init {
  ^bb0(%arg0: !llvm.ptr, %arg1: !llvm.ptr):
    %0 = llvm.mlir.constant(48 : i32) : i32
    "llvm.intr.memcpy"(%arg1, %arg0, %0) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i32) -> ()
    omp.yield(%arg1 : !llvm.ptr)
  }

  omp.private {type = firstprivate} @firstprivatizer : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> copy {
  ^bb0(%arg0: !llvm.ptr, %arg1: !llvm.ptr):
    %0 = llvm.mlir.constant(48 : i32) : i32
    "llvm.intr.memcpy"(%arg1, %arg0, %0) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i32) -> ()
    omp.yield(%arg1 : !llvm.ptr)
  }

  llvm.func internal @private_test(%arg0: !llvm.ptr {fir.bindc_name = "ptr0"}, %arg1: !llvm.ptr {fir.bindc_name = "ptr1"}) {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(0 : index) : i64
    %5 = llvm.alloca %0 x !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %19 = llvm.alloca %0 x !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> {bindc_name = "local"} : (i32) -> !llvm.ptr
    %21 = llvm.alloca %0 x i32 {bindc_name = "i"} : (i32) -> !llvm.ptr
    %33 = llvm.mlir.undef : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
    llvm.store %33, %19 : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>, !llvm.ptr
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
    omp.target nowait map_entries(%124 -> %arg2, %160 -> %arg5, %159 -> %arg8 : !llvm.ptr, !llvm.ptr, !llvm.ptr) private(@privatizer %19 -> %arg9 [map_idx=1] : !llvm.ptr) {
      omp.terminator
    }
    %166 = llvm.mlir.constant(48 : i32) : i32
    %167 = llvm.getelementptr %19[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
    %168 = llvm.load %167 : !llvm.ptr -> !llvm.ptr
    llvm.call @free(%168) : (!llvm.ptr) -> ()
    llvm.return
  }

  llvm.func internal @firstprivate_test(%arg0: !llvm.ptr {fir.bindc_name = "ptr0"}, %arg1: !llvm.ptr {fir.bindc_name = "ptr1"}) {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(0 : index) : i64
    %5 = llvm.alloca %0 x !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %19 = llvm.alloca %0 x !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> {bindc_name = "local"} : (i32) -> !llvm.ptr
    %21 = llvm.alloca %0 x i32 {bindc_name = "i"} : (i32) -> !llvm.ptr
    %33 = llvm.mlir.undef : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
    llvm.store %33, %19 : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>, !llvm.ptr
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
    omp.target nowait map_entries(%124 -> %arg2, %160 -> %arg5, %159 -> %arg8 : !llvm.ptr, !llvm.ptr, !llvm.ptr) private(@firstprivatizer %19 -> %arg9 [map_idx=1] : !llvm.ptr) {
      omp.terminator
    }
    %166 = llvm.mlir.constant(48 : i32) : i32
    %167 = llvm.getelementptr %19[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
    %168 = llvm.load %167 : !llvm.ptr -> !llvm.ptr
    llvm.call @free(%168) : (!llvm.ptr) -> ()
    llvm.return
  }
}

// CHECK-LABEL:   llvm.func @malloc(i64) -> !llvm.ptr
// CHECK:         llvm.func @free(!llvm.ptr)

// CHECK-LABEL: llvm.func internal @private_test(
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
// CHECK: %[[VAL_17:.*]] = llvm.load %[[STACK]] : !llvm.ptr -> !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
// CHECK: llvm.store %[[VAL_17]], %[[HEAP]] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>, !llvm.ptr
// CHECK: %[[VAL_18:.*]] = llvm.getelementptr %[[HEAP]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
// CHECK: %[[VAL_19:.*]] = omp.map.info var_ptr(%[[HEAP]] : !llvm.ptr, i32) map_clauses(to) capture(ByRef) var_ptr_ptr(%[[VAL_18]] : !llvm.ptr) bounds(%[[VAL_16]]) -> !llvm.ptr {name = ""}
// CHECK: %[[VAL_20:.*]] = omp.map.info var_ptr(%[[HEAP]] : !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>) map_clauses(always, to) capture(ByRef) members(%[[VAL_19]] : [0] : !llvm.ptr) -> !llvm.ptr
// CHECK: omp.target nowait map_entries(%[[VAL_8]] -> %[[VAL_21:.*]], %[[VAL_20]] -> %[[VAL_22:.*]], %[[VAL_19]] -> %[[VAL_23:.*]] : !llvm.ptr, !llvm.ptr, !llvm.ptr) private(@privatizer %[[HEAP]] -> %[[VAL_24:.*]] [map_idx=1] : !llvm.ptr) {
// CHECK: omp.terminator
// CHECK: }
// CHECK: %[[VAL_26:.*]] = llvm.getelementptr %[[STACK]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
// CHECK: %[[VAL_27:.*]] = llvm.load %[[VAL_26]] : !llvm.ptr -> !llvm.ptr
// CHECK: llvm.call @free(%[[VAL_27]]) : (!llvm.ptr) -> ()
// CHECK: llvm.return
// CHECK: }

// CHECK-LABEL: llvm.func internal @firstprivate_test(
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
// CHECK: %[[VAL_17:.*]] = llvm.load %[[STACK]] : !llvm.ptr -> !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
// CHECK: llvm.store %[[VAL_17]], %[[HEAP]] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>, !llvm.ptr
// CHECK: %[[VAL_18:.*]] = llvm.getelementptr %[[STACK]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
// CHECK: %[[VAL_19:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK: %[[VAL_20:.*]] = llvm.sub %[[VAL_15]], %[[VAL_1]] : i64
// CHECK: %[[VAL_21:.*]] = llvm.add %[[VAL_20]], %[[VAL_19]] : i64
// CHECK: %[[VAL_22:.*]] = llvm.mul %[[VAL_19]], %[[VAL_21]] : i64
// CHECK: %[[VAL_23:.*]] = llvm.mlir.constant(4 : i64) : i64
// CHECK: %[[VAL_24:.*]] = llvm.mul %[[VAL_22]], %[[VAL_23]] : i64
// CHECK: %[[NEW_DATA_PTR:.*]] = llvm.call @malloc(%[[VAL_24]]) : (i64) -> !llvm.ptr
// CHECK: %[[OLD_DATA_PTR:.*]] = llvm.load %[[VAL_18]] : !llvm.ptr -> !llvm.ptr
// CHECK: "llvm.intr.memcpy"(%[[NEW_DATA_PTR]], %[[OLD_DATA_PTR]], %[[VAL_24]]) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
// CHECK: %[[VAL_27:.*]] = llvm.getelementptr %[[HEAP]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
// CHECK: llvm.store %[[NEW_DATA_PTR]], %[[VAL_27]] : !llvm.ptr, !llvm.ptr
// CHECK: %[[VAL_28:.*]] = omp.map.info var_ptr(%[[HEAP]] : !llvm.ptr, i32) map_clauses(to) capture(ByRef) var_ptr_ptr(%[[VAL_27]] : !llvm.ptr) bounds(%[[VAL_16]]) -> !llvm.ptr {name = ""}
// CHECK: %[[VAL_29:.*]] = omp.map.info var_ptr(%[[HEAP]] : !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>) map_clauses(always, to) capture(ByRef) members(%[[VAL_28]] : [0] : !llvm.ptr) -> !llvm.ptr
// CHECK: omp.target nowait map_entries(%[[VAL_8]] -> %[[VAL_30:.*]], %[[VAL_29]] -> %[[VAL_31:.*]], %[[VAL_28]] -> %[[VAL_32:.*]] : !llvm.ptr, !llvm.ptr, !llvm.ptr) private(@firstprivatizer %[[HEAP]] -> %[[VAL_33:.*]] [map_idx=1] : !llvm.ptr) {
// CHECK: omp.terminator
// CHECK: }
// CHECK: %[[VAL_34:.*]] = llvm.mlir.constant(48 : i32) : i32
// CHECK: %[[VAL_35:.*]] = llvm.getelementptr %[[STACK]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
// CHECK: %[[VAL_36:.*]] = llvm.load %[[VAL_35]] : !llvm.ptr -> !llvm.ptr
// CHECK: llvm.call @free(%[[VAL_36]]) : (!llvm.ptr) -> ()
// CHECK: llvm.return
// CHECK: }
