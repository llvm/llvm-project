; RUN: llc < %s -march=cellspu | FileCheck %s

target datalayout = "E-p:32:32:128-f64:64:128-f32:32:128-i64:32:128-i32:32:128-i16:16:128-i8:8:128-i1:8:128-a0:0:128-v128:128:128-s0:128:128"
target triple = "spu"

define i32 @main() {
entry:
  %a = call i32 @stub_1(i32 1, float 0x400921FA00000000)
  call void @extern_stub_1(i32 %a, i32 4)
  ret i32 %a
}

declare void @extern_stub_1(i32, i32)

define i32 @stub_1(i32 %x, float %y) {
 ; CHECK: il $3, 0
 ; CHECK: bi $lr 
entry:
  ret i32 0
}

; vararg call: ensure that all caller-saved registers are spilled to the
; stack:
define i32 @stub_2(...) {
entry:
  ret i32 0
}

; check that struct is passed in r3->
; assert this by changing the second field in the struct
%0 = type { i32, i32, i32 }
declare %0 @callee()
define %0 @test_structret()
{
;CHECK:	stqd	$lr, 16($sp)
;CHECK:	stqd	$sp, -48($sp)
;CHECK:	ai	$sp, $sp, -48
;CHECK:	brasl	$lr, callee
  %rv = call %0 @callee()
;CHECK: ai	$4, $4, 1
;CHECK: lqd	$lr, 64($sp)
;CHECK:	ai	$sp, $sp, 48
;CHECK:	bi	$lr
  %oldval = extractvalue %0 %rv, 1
  %newval = add i32 %oldval,1
  %newrv = insertvalue %0 %rv, i32 %newval, 1
  ret %0 %newrv
}

