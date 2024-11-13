// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// This tests the fix for https://github.com/llvm/llvm-project/issues/84606
// We are only interested in ensuring that the -mlir-to-llmvir pass doesn't crash.
// CHECK: {{.*}} = add i32 {{.*}}, 5
module attributes {omp.is_target_device = true } {
  llvm.func @_QQmain() attributes {fir.bindc_name = "main", omp.declare_target = #omp.declaretarget<device_type = (host), capture_clause = (to)>} {
    %0 = llvm.mlir.constant(0 : i32) : i32
    %1 = llvm.mlir.constant(1 : i64) : i64
    %2 = llvm.alloca %1 x i32 {bindc_name = "a"} : (i64) -> !llvm.ptr<5>
    %3 = llvm.addrspacecast %2 : !llvm.ptr<5> to !llvm.ptr
    omp.task {
      llvm.store %0, %3 : i32, !llvm.ptr
      omp.terminator
    }
    %4 = omp.map.info var_ptr(%3 : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr {name = "a"}
    omp.target map_entries(%4 -> %arg0 : !llvm.ptr) {
      %5 = llvm.mlir.constant(5 : i32) : i32
      %6 = llvm.load %arg0  : !llvm.ptr -> i32
      %7 = llvm.add %6, %5  : i32
      llvm.store %7, %arg0  : i32, !llvm.ptr
      omp.terminator
    }
    llvm.return
  }
}
