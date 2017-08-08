; RUN: llc -march=mipsel -mcpu=mips32r2 -mattr=+micromips -asm-show-inst -verify-machineinstrs < %s | FileCheck %s

; Function Attrs: nounwind
define i32 @function1(i32 (i32)* %f) {
entry:
; CHECK-LABEL: function1:
; CHECK: SWSP_MM
; CHECK: LWSP_MM
  %call = call i32 %f(i32 0)
  ret i32 0
}
