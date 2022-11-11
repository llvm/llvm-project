; RUN: mlir-translate -import-llvm %s | FileCheck %s

; CHECK: llvm.func @foo(%arg0: !llvm.ptr {llvm.byval = i64}, %arg1: !llvm.ptr {llvm.byref = i64}, %arg2: !llvm.ptr {llvm.sret = i64}, %arg3: !llvm.ptr {llvm.inalloca = i64})
define void @foo(ptr byval(i64) %arg0, ptr byref(i64) %arg1, ptr sret(i64) %arg2, ptr inalloca(i64) %arg3) {
  ret void
}
