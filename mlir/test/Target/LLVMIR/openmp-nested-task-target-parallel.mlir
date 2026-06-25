// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s
// This tests the fix for https://github.com/llvm/llvm-project/issues/138102
// We are only interested in ensuring that the -mlir-to-llvmir pass doesn't crash

// CHECK-LABEL: define internal void @_QQmain..omp_par

omp.private {type = private} @_QFEi_private_i32 : i32
omp.private {type = firstprivate} @_QFEc_firstprivate_i32 : i32 copy {
^bb0(%arg0: !llvm.ptr, %arg1: !llvm.ptr):
%0 = llvm.load %arg0 : !llvm.ptr -> i32
llvm.store %0, %arg1 : i32, !llvm.ptr
omp.yield(%arg1 : !llvm.ptr)
}
llvm.func @_QQmain() {
%0 = llvm.mlir.constant(1 : i64) : i64
%1 = llvm.alloca %0 x i32 {bindc_name = "i"} : (i64) -> !llvm.ptr
%2 = llvm.mlir.constant(1 : i64) : i64
%3 = llvm.alloca %2 x i32 {bindc_name = "c"} : (i64) -> !llvm.ptr
%4 = llvm.mlir.constant(10 : index) : i64
%5 = llvm.mlir.constant(0 : index) : i64
%6 = llvm.mlir.constant(10000 : index) : i64
%7 = llvm.mlir.constant(1 : index) : i64
%8 = llvm.mlir.constant(1 : i64) : i64
%9 = llvm.mlir.addressof @_QFECchunksz : !llvm.ptr
%10 = llvm.mlir.constant(1 : i64) : i64
%11 = llvm.trunc %7 : i64 to i32
llvm.br ^bb1(%11, %4 : i32, i64)
^bb1(%12: i32, %13: i64):  // 2 preds: ^bb0, ^bb2
%14 = llvm.icmp "sgt" %13, %5 : i64
llvm.store %12, %3 : i32, !llvm.ptr
omp.task private(@_QFEc_firstprivate_i32 %3 -> %arg0 : !llvm.ptr) {
  %19 = omp.map.info var_ptr(%1 : !llvm.ptr, i32) map_clauses(implicit, exit_release_or_enter_alloc) capture(ByCopy) -> !llvm.ptr {name = "i"}
  %22 = llvm.mlir.constant(9999 : i32) : i32
  %23 = llvm.mlir.constant(1 : i32) : i32
  %24 = llvm.load %arg0 : !llvm.ptr -> i32
  %25 = llvm.add %24, %22 : i32
  omp.target kernel_type(spmd) host_eval(%23 -> %arg1, %24 -> %arg2, %25 -> %arg3 : i32, i32, i32) map_entries(%19 -> %arg4 : !llvm.ptr) {
    omp.parallel {
      omp.wsloop private(@_QFEi_private_i32 %arg4 -> %arg5 : !llvm.ptr) {
        omp.loop_nest (%arg6) : i32 = (%arg2) to (%arg3) inclusive step (%arg1) {
          llvm.store %arg6, %arg5 : i32, !llvm.ptr
          omp.yield
        }
      }
      omp.terminator
    }
    omp.terminator
  }
  omp.terminator
}
llvm.return
}
llvm.mlir.global internal constant @_QFECchunksz() {addr_space = 0 : i32} : i32 {
%0 = llvm.mlir.constant(10000 : i32) : i32
llvm.return %0 : i32
}
llvm.mlir.global internal constant @_QFECn() {addr_space = 0 : i32} : i32 {
%0 = llvm.mlir.constant(100000 : i32) : i32
llvm.return %0 : i32
}
