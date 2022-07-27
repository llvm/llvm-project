; RUN: llc < %s -mtriple=x86_64-pc-windows-msvc | FileCheck %s

define cc 92 < 9 x i64 > @clobber() {
  %1 = alloca i64
  %2 = load volatile i64, ptr %1
  ret < 9 x i64 > undef
  ; CHECK-LABEL: clobber:
  ; CHECK-NOT: popq %rsp
  ; CHECK: addq $8, %rsp
}
