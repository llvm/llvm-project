; RUN: not llc -mtriple=x86_64-unknown-unknown -no-integrated-as < %s 2>&1 | FileCheck %s

%struct.s = type { i32, i32 }

@pr40890.s = internal global %struct.s zeroinitializer, align 4

; CHECK: error: invalid operand for inline asm constraint 'e'
; CHECK: error: invalid operand for inline asm constraint 'e'

define void @pr40890() {
entry:
  ; This pointer cannot be used as an integer constant expression.
  tail call void asm sideeffect "\0A#define GLOBAL_A abcd$0\0A", "e,~{dirflag},~{fpsr},~{flags}"(ptr @pr40890.s)
  ; Floating-point is also not okay.
  tail call void asm sideeffect "\0A#define PI abcd$0\0A", "e,~{dirflag},~{fpsr},~{flags}"(float 0x40091EB860000000)
  ret void
}
