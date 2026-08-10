; RUN: not llc -mtriple=i686-- -no-integrated-as < %s 2>&1 | FileCheck %s

; CHECK: error: value out of range for constraint 'I'
define void @foo() {
  call void asm sideeffect "foo $0", "I"(i32 42)
  ret void
}
