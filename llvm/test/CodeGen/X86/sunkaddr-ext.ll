; RUN: llc < %s | FileCheck %s

; Test to make sure that if math that can roll over has been used we don't
; use the potential overflow as the basis for an address calculation later by
; sinking it into a different basic block.

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.9.0"

; Function Attrs: nounwind ssp uwtable
define void @test_sink(ptr %arg1, i32 %arg2, i8 %arg3) #0 {
  %tmp1 = add i32 -2147483648, %arg2
  %tmp2 = add i32 -2147483648, %tmp1
  %tmp3 = getelementptr i8, ptr %arg1, i32 %arg2
  br label %bb1

bb1:
  %tmp4 = getelementptr i8, ptr %arg1, i32 %tmp2
  store i8 %arg3, ptr %tmp4
  ret void;
}

; CHECK-LABEL: test_sink:
; CHECK:   movslq  %esi, [[TEMP:%[a-z0-9]+]]
; CHECK:   movb    %dl, (%rdi,[[TEMP]])
; CHECK:   retq
