; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s

; CHECK: Attribute 'noundef' applied to incompatible type!
; CHECK-NEXT: @noundef_void
define noundef void @noundef_void() {
  ret void
}
