// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// This tests the replacement of operations for `declare target to` with the
// generated `declare target to` global variable inside of target op regions when
// lowering to IR for device. Unfortunately, as the host file is not passed as a
// module attribute, we miss out on the metadata and entry info.

module attributes {llvm.target_triple = "amdgcn-amd-amdhsa", omp.is_gpu = true, omp.is_target_device = true} {
  // CHECK-DAG: @_QMtest_0Ezii = global [11 x float] zeroinitializer
  llvm.mlir.global external @_QMtest_0Ezii() {addr_space = 0 : i32, omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (to)>} : !llvm.array<11 x f32> {
    %0 = llvm.mlir.zero : !llvm.array<11 x f32>
    llvm.return %0 : !llvm.array<11 x f32>
  }

  // CHECK-LABEL: define weak_odr protected amdgpu_kernel void @{{.*}}(ptr %{{.*}}) {{.*}} {
  // CHECK-DAG:   omp.target:
  // CHECK-DAG: store float 1.000000e+00, ptr @_QMtest_0Ezii, align 4
  // CHECK-DAG: br label %omp.region.cont
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
}
