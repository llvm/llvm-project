; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

define void @f() {
entry:
  call target("dx.RawBuffer", half, 1, 0) () poison ()
  ret void
}
; CHECK: Return type cannot be token for indirect call!
