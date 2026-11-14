; RUN: not llvm-as %s -disable-output 2>&1 | FileCheck %s

define void @f() {
entry:
  call target("amdgpu.stridemark") () poison ()
  ret void
}
; CHECK: Return type cannot be token for indirect call!
