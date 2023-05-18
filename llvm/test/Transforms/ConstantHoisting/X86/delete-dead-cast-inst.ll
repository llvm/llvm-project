; RUN: opt -S -passes=consthoist < %s | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.9.0"

%T = type { i32, i32, i32, i32 }

define i32 @test1() nounwind {
; CHECK-LABEL:  @test1
; CHECK:        %const = bitcast i32 12345678 to i32
; CHECK-NOT:    %base = inttoptr i32 12345678 to ptr
; CHECK-NEXT:   %1 = inttoptr i32 %const to ptr
; CHECK-NEXT:   %addr1 = getelementptr %T, ptr %1, i32 0, i32 1
; CHECK-NEXT:   %addr2 = getelementptr %T, ptr %1, i32 0, i32 2
; CHECK-NEXT:   %addr3 = getelementptr %T, ptr %1, i32 0, i32 3
  %base = inttoptr i32 12345678 to ptr
  %addr1 = getelementptr %T, ptr %base, i32 0, i32 1
  %addr2 = getelementptr %T, ptr %base, i32 0, i32 2
  %addr3 = getelementptr %T, ptr %base, i32 0, i32 3
  ret i32 12345678
}

