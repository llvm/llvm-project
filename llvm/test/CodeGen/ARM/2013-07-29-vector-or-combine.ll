; RUN: llc < %s -mcpu=cortex-a8 -arm-atomic-cfg-tidy=0 | FileCheck %s
; ModuleID = 'bugpoint-reduced-simplified.bc'
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:64:128-a0:0:64-n32-S64"
target triple = "armv7--linux-gnueabi"

; CHECK-LABEL: function
define void @function() {
; CHECK: cmp r0, #0
; CHECK: bxne lr
entry:
  br i1 undef, label %vector.body, label %for.end

; CHECK: vld1.32 {d16, d17}, [r0]
; CHECK: vbic.i32 q8, #0xff
; CHECK: vorr q8, q8, q9
; CHECK: vst1.32 {d16, d17}, [r0]
vector.body:
  %wide.load = load <4 x i32>, <4 x i32>* undef, align 4
  %0 = and <4 x i32> %wide.load, <i32 -16711936, i32 -16711936, i32 -16711936, i32 -16711936>
  %1 = sub <4 x i32> %wide.load, zeroinitializer
  %2 = and <4 x i32> %1, <i32 16711680, i32 16711680, i32 16711680, i32 16711680>
  %3 = or <4 x i32> %0, <i32 1, i32 2, i32 3, i32 4>
  %4 = or <4 x i32> %3, %2
  store <4 x i32> %4, <4 x i32>* undef, align 4
  br label %vector.body

for.end:
  ret void
}

