; RUN: llc -I %p/Inputs -filetype asm -o - %s | FileCheck %s
; REQUIRES: default_triple

module asm ".include \22module.x\22"

define arm_aapcscc void @f() {
entry:
  call void asm sideeffect ".include \22function.x\22", ""()
  ret void
}

; CHECK: MODULE = 1
; CHECK: FUNCTION = 1
