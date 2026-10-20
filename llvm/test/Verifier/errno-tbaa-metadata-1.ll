; RUN: not llvm-as < %s -o /dev/null 2>&1 | FileCheck %s

; CHECK: assembly parsed, but does not verify as correct!
; CHECK-NEXT: llvm.errno.tbaa must have at least one operand
!llvm.errno.tbaa = !{}
