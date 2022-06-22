; RUN: llc < %s | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.9.0"

%T = type { i32, i32, i32, i32 }

define i32 @test1() nounwind {
; CHECK-LABEL:  test1
; CHECK:        movabsq $123456789012345678, %rcx
; CHECK-NEXT:   movl  4(%rcx), %eax
; CHECK-NEXT:   addl  8(%rcx), %eax
; CHECK-NEXT:   addl  12(%rcx), %eax
  %addr1 = getelementptr %T, ptr inttoptr (i64 123456789012345678 to ptr), i32 0, i32 1
  %tmp1 = load i32, ptr %addr1
  %addr2 = getelementptr %T, ptr inttoptr (i64 123456789012345678 to ptr), i32 0, i32 2
  %tmp2 = load i32, ptr %addr2
  %addr3 = getelementptr %T, ptr inttoptr (i64 123456789012345678 to ptr), i32 0, i32 3
  %tmp3 = load i32, ptr %addr3
  %tmp4 = add i32 %tmp1, %tmp2
  %tmp5 = add i32 %tmp3, %tmp4
  ret i32 %tmp5
}

