; RUN: opt -S -passes=consthoist < %s | FileCheck %s
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

%T = type { i32, i32, i32, i32 }

; Test if even cheap base addresses are hoisted.
define i32 @test1() nounwind {
; CHECK-LABEL:  @test1
; CHECK:        %const = bitcast i32 12345678 to i32
; CHECK:        %1 = inttoptr i32 %const to ptr
; CHECK:        %addr1 = getelementptr %T, ptr %1, i32 0, i32 1
  %addr1 = getelementptr %T, ptr inttoptr (i32 12345678 to ptr), i32 0, i32 1
  %tmp1 = load i32, ptr %addr1
  %addr2 = getelementptr %T, ptr inttoptr (i32 12345678 to ptr), i32 0, i32 2
  %tmp2 = load i32, ptr %addr2
  %addr3 = getelementptr %T, ptr inttoptr (i32 12345678 to ptr), i32 0, i32 3
  %tmp3 = load i32, ptr %addr3
  %tmp4 = add i32 %tmp1, %tmp2
  %tmp5 = add i32 %tmp3, %tmp4
  ret i32 %tmp5
}

