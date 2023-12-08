; RUN: llc < %s -march=xcore | FileCheck %s
; CHECK-LABEL: f1:
; CHECK: foo r0
define i32 @f1() nounwind {
entry:
  %asmtmp = tail call i32 asm sideeffect "foo $0", "=r"() nounwind
  ret i32 %asmtmp
}

; CHECK-LABEL: f2:
; CHECK: foo 5
define void @f2() nounwind {
entry:
  tail call void asm sideeffect "foo $0", "i"(i32 5) nounwind
  ret void
}

; CHECK-LABEL: f3:
; CHECK: foo 42
define void @f3() nounwind {
entry:
  tail call void asm sideeffect "foo ${0:c}", "i"(i32 42) nounwind
  ret void
}

; CHECK-LABEL: f4:
; CHECK: foo -99
define void @f4() nounwind {
entry:
  tail call void asm sideeffect "foo ${0:n}", "i"(i32 99) nounwind
  ret void
}

@x = external global i32
@y = external global i32, section ".cp.rodata"

; CHECK-LABEL: f5:
; CHECK: ldw r0, dp[x]
; CHECK: retsp 0
define i32 @f5() nounwind {
entry:
  %asmtmp = call i32 asm "ldw $0, $1", "=r,*m"(ptr elementtype(i32) @x) nounwind
  ret i32 %asmtmp
}

; CHECK-LABEL: f6:
; CHECK: ldw r0, cp[y]
; CHECK: retsp 0
define i32 @f6() nounwind {
entry:
  %asmtmp = call i32 asm "ldw $0, $1", "=r,*m"(ptr elementtype(i32) @y) nounwind
  ret i32 %asmtmp
}
