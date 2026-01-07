; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

define target("dx.RawBuffer", half, 1, 0) @f() {
entry:
  ret target("dx.RawBuffer", half, 1, 0) poison
}
; CHECK: Function returns a token but isn't an intrinsic
