; RUN: not llc -mtriple i386-apple-- -o /dev/null < %s 2>&1| FileCheck %s
; CHECK: <inline asm>:1:13: error: register %eip is only available in 64-bit mode
; CHECK-NEXT: jmpl *_foo(%eip)

; Make sure that we emit an error if we encounter RIP-relative instructions in
; 32-bit mode.

define i32 @foo() { ret i32 0 }

define i32 @bar() {
  call void asm sideeffect "jmpl *_foo(%eip)\0A", "~{dirflag},~{fpsr},~{flags}"()
  ret i32 0
}
