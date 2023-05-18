; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64-unknown-linux-gnu -mcpu=a2 | FileCheck %s
target datalayout = "E-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f128:64:128-n32"

define i64 @foo() nounwind {
entry:
  %x = alloca [32568 x i8]
  %"alloca point" = bitcast i32 0 to i32

; Check that the RS spill slot has been allocated (because the estimate
; will fail the small-frame-size check and the function has spills).
; CHECK: @foo
; CHECK: stdu 1, -32768(1)

  %s1 = call i64 @bar(ptr %x) nounwind
  %s2 = call i64 @bar(ptr %x) nounwind
  %s3 = call i64 @bar(ptr %x) nounwind
  %s4 = call i64 @bar(ptr %x) nounwind
  %s5 = call i64 @bar(ptr %x) nounwind
  %s6 = call i64 @bar(ptr %x) nounwind
  %s7 = call i64 @bar(ptr %x) nounwind
  %s8 = call i64 @bar(ptr %x) nounwind
  %r = call i64 @can(i64 %s1, i64 %s2, i64 %s3, i64 %s4, i64 %s5, i64 %s6, i64 %s7, i64 %s8) nounwind
  br label %return

return:
  ret i64 %r
}

declare i64 @bar(ptr)
declare i64 @can(i64, i64, i64, i64, i64, i64, i64, i64)

