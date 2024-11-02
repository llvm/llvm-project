; RUN: opt < %s -instsimplify -dce -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.7.2"

; This is a basic correctness check for constant propagation.  The add
; instruction should be eliminated.
define i32 @test1(i1 %B) {
        br i1 %B, label %BB1, label %BB2

BB1:      
        %Val = add i32 0, 0
        br label %BB3

BB2:      
        br label %BB3

BB3:     
; CHECK-LABEL: @test1(
; CHECK: %Ret = phi i32 [ 0, %BB1 ], [ 1, %BB2 ]
        %Ret = phi i32 [ %Val, %BB1 ], [ 1, %BB2 ] 
        ret i32 %Ret
}


; PR6197
define i1 @test2(ptr %f) nounwind {
entry:
  %V = icmp ne ptr blockaddress(@test2, %bb), null
  br label %bb
bb:
  ret i1 %V
  
; CHECK-LABEL: @test2(
; CHECK: ret i1 true
}

define i1 @TNAN() {
; CHECK-LABEL: @TNAN(
; CHECK: ret i1 true
  %A = fcmp uno double 0x7FF8000000000000, 1.000000e+00
  %B = fcmp uno double 1.230000e+02, 1.000000e+00
  %C = or i1 %A, %B
  ret i1 %C
}

define i128 @vector_to_int_cast() {
  %A = bitcast <4 x i32> <i32 1073741824, i32 1073741824, i32 1073741824, i32 1073741824> to i128
  ret i128 %A
; CHECK-LABEL: @vector_to_int_cast(
; CHECK: ret i128 85070591750041656499021422275829170176
}
  
