; RUN: not llvm-as -disable-output %s 2>&1 | FileCheck %s
;
; A pointer may only be bitcast to a byte (vector) of the same size, and vice
; versa.

target datalayout = "e-p:64:64:64"

; CHECK: Invalid bitcast
; CHECK: %x = bitcast ptr %p to b1
; CHECK: Invalid bitcast
; CHECK: %y = bitcast b1 %b to ptr
define void @f(ptr %p, b1 %b) {
  %x = bitcast ptr %p to b1
  %y = bitcast b1 %b to ptr
  ret void
}
