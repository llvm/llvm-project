; RUN: llc -I %p/Inputs -filetype asm -o - %s | FileCheck %s
; UNSUPPORTED: -zos,target=nvptx{{.*}}
; REQUIRES: default_triple

module asm ".include \22module.x\22"

define void @f() {
entry:
  call void asm sideeffect ".include \22function.x\22", ""()
  ret void
}

; CHECK: .set MODULE, 1
; CHECK: .set FUNCTION, 1
