; RUN: opt -S -passes=licm,simple-loop-unswitch,licm < %s | FileCheck %s

declare void @llvm.experimental.guard(i1, ...)

define void @test() {
; CHECK-LABEL: @test(
; CHECK-NOT: guard
entry:
  br label %header

header:
  br label %loop

loop:
  %0 = icmp ult i32 0, 400
  call void (i1, ...) @llvm.experimental.guard(i1 %0, i32 9) [ "deopt"() ]
  br i1 undef, label %header, label %loop
}
