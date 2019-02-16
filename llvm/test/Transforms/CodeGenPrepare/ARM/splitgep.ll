; RUN: opt -S -codegenprepare %s | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv6m-arm-none-eabi"

; Check that we have deterministic output
define void @test([65536 x i32]** %sp, [65536 x i32]* %t, i32 %n) {
; CHECK-LABEL: @test(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    %0 = bitcast [65536 x i32]* %t to i8*
; CHECK-NEXT:    %splitgep1 = getelementptr i8, i8* %0, i32 80000
; CHECK-NEXT:    %s = load [65536 x i32]*, [65536 x i32]** %sp
; CHECK-NEXT:    %1 = bitcast [65536 x i32]* %s to i8*
; CHECK-NEXT:    %splitgep = getelementptr i8, i8* %1, i32 80000
entry:
  %s = load [65536 x i32]*, [65536 x i32]** %sp
  br label %while_cond

while_cond:
  %phi = phi i32 [ 0, %entry ], [ %i, %while_body ]
  %gep0 = getelementptr [65536 x i32], [65536 x i32]* %s, i64 0, i32 20000
  %gep1 = getelementptr [65536 x i32], [65536 x i32]* %s, i64 0, i32 20001
  %gep2 = getelementptr [65536 x i32], [65536 x i32]* %t, i64 0, i32 20000
  %gep3 = getelementptr [65536 x i32], [65536 x i32]* %t, i64 0, i32 20001
  %cmp = icmp slt i32 %phi, %n
  br i1 %cmp, label %while_body, label %while_end

while_body:
  %i = add i32 %phi, 1
  %j = add i32 %phi, 2
  store i32 %i, i32* %gep0
  store i32 %phi, i32* %gep1
  store i32 %i, i32* %gep2
  store i32 %phi, i32* %gep3
  br label %while_cond

while_end:
  ret void
}

