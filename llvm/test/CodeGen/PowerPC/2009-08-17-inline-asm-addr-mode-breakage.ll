; RUN: llc -verify-machineinstrs < %s -disable-ppc-ilp-pref | FileCheck %s
; ModuleID = '<stdin>'
target datalayout = "E-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f128:64:128"
target triple = "powerpc-unknown-linux-gnu"
; It is wrong on powerpc to substitute reg+reg for $0; the stw opcode
; would have to change.

@x = external global [0 x i32]                    ; <[0 x i32]*> [#uses=1]

define void @foo(i32 %y) nounwind ssp {
entry:
; CHECK: foo
; CHECK: add [[REG:[0-9]+]]
; CHECK: 0([[REG]])
  %y_addr = alloca i32                            ; <i32*> [#uses=2]
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  store i32 %y, i32* %y_addr
  %0 = load i32, i32* %y_addr, align 4                 ; <i32> [#uses=1]
  %1 = getelementptr inbounds [0 x i32], [0 x i32]* @x, i32 0, i32 %0 ; <i32*> [#uses=1]
  call void asm sideeffect "isync\0A\09eieio\0A\09stw $1, $0", "=*o,r,~{memory}"(i32* elementtype(i32) %1, i32 0) nounwind
  br label %return

return:                                           ; preds = %entry
  ret void
}
