// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

module attributes {llvm.target_triple = "x86_64-unknown-linux-gnu", omp.is_gpu = false, omp.is_target_device = false, omp.target_triples = ["amdgcn-amd-amdhsa"]} {
  // CHECK-DAG: @_QMtest_0Ezii = global [11 x float] zeroinitializer
  // CHECK-DAG: @.offload_sizes = private unnamed_addr constant [1 x i64] [i64 48]
  // CHECK-DAG: @.offload_maptypes = private unnamed_addr constant [1 x i64] [i64 3]
  // CHECK-DAG: @.offloading.entry._QMtest_0Ezii = weak constant %struct.__tgt_offload_entry {{.*}} ptr @_QMtest_0Ezii, {{.*}}, i64 44,{{.*}}
  llvm.mlir.global external @_QMtest_0Ezii() {addr_space = 0 : i32, omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (to)>} : !llvm.array<11 x f32> {
    %0 = llvm.mlir.zero : !llvm.array<11 x f32>
    llvm.return %0 : !llvm.array<11 x f32>
  }

  // CHECK-DAG: %[[BASEPTR:.*]] = getelementptr inbounds [1 x ptr], ptr %.offload_baseptrs, i32 0, i32 0
  // CHECK-DAG: store ptr @_QMtest_0Ezii, ptr %[[BASEPTR]], align 8
  // CHECK-DAG: %[[OFFLOADPTR:.*]] = getelementptr inbounds [1 x ptr], ptr %.offload_ptrs, i32 0, i32 0
  // CHECK-DAG: store ptr @_QMtest_0Ezii, ptr %[[OFFLOADPTR]], align 8
  llvm.func @_QQmain() {
    %0 = llvm.mlir.constant(1 : index) : i64
    %1 = llvm.mlir.constant(0 : index) : i64
    %2 = llvm.mlir.constant(11 : index) : i64
    %3 = llvm.mlir.addressof @_QMtest_0Ezii : !llvm.ptr
    %4 = omp.map.bounds lower_bound(%1 : i64) upper_bound(%2 : i64) extent(%2 : i64) stride(%0 : i64) start_idx(%1 : i64) {stride_in_bytes = true}
    %5 = omp.map.info var_ptr(%3 : !llvm.ptr, !llvm.array<11 x f32>) map_clauses(tofrom) capture(ByRef) bounds(%4) -> !llvm.ptr
    omp.target map_entries(%5 -> %arg0 : !llvm.ptr) {
      %6 = llvm.mlir.constant(1.0 : f32) : f32
      %7 = llvm.mlir.constant(0 : i64) : i64
      %8 = llvm.getelementptr %arg0[%7] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      llvm.store %6, %8 : f32, !llvm.ptr
      omp.terminator
    }
    llvm.return
  }
  // CHEKC-DAG: !{{.*}} = !{i32 {{.*}}, !"_QMtest_0Ezii", i32 {{.*}}, i32 {{.*}}}
}
