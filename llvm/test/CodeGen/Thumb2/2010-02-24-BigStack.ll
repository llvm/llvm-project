; RUN: llc < %s -O0 -relocation-model=pic -frame-pointer=all -mcpu=cortex-a8 -mattr=+vfp2
; This test creates a big stack frame without spilling any callee-saved registers.
; Make sure the whole stack frame is addrerssable wiothout scavenger crashes.
target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:64:64-v128:128:128-a0:0:32-n32"
target triple = "thumbv7-apple-darwin3.0.0-iphoneos"

define void @FindMin(ptr %panelTDEL, ptr %dclOfRow, i32 %numRows, i32 %numCols, ptr %retMin_RES_TDEL) {
entry:
  %panelTDEL.addr = alloca ptr, align 4       ; <ptr> [#uses=1]
  %panelResTDEL = alloca [2560 x double], align 4 ; <ptr> [#uses=0]
  store ptr %panelTDEL, ptr %panelTDEL.addr
  store ptr %retMin_RES_TDEL, ptr undef
  store i32 0, ptr undef
  unreachable
}
