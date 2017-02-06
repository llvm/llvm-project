; RUN: llc -O0 < %s -march=avr | FileCheck %s

; CHECK-LABEL: foo
define void @foo() {
entry:
  br label %save

; CHECK-LABEL: save
; CHECK: in [[SREG1:r[0-9]+]], 61
; CHECK-NEXT: in [[SREG2:r[0-9]+]], 62
save:
  %saved = call i8* @llvm.stacksave()
  br label %restore

; CHECK-LABEL: restore
; CHECK: in r0, 63
; CHECK-NEXT: cli
; CHECK-NEXT: out 62, [[SREG2]]
; CHECK-NEXT: out 63, r0
; CHECK-NEXT: out 61, [[SREG1]]
restore:
  call void @llvm.stackrestore(i8* %saved)
  ret void
}

declare i8* @llvm.stacksave()
declare void @llvm.stackrestore(i8* %ptr)
