; RUN: llc < %s -mtriple=x86_64-pc-linux -mcpu=corei7 -relocation-model=pic | FileCheck %s

; PR27502
; UNREACHABLE: "Invalid address displacement operand"

@buf = internal global [5 x ptr] zeroinitializer

declare i32 @llvm.eh.sjlj.setjmp(ptr) nounwind

define i32 @test() nounwind optsize {
  %r = tail call i32 @llvm.eh.sjlj.setjmp(ptr @buf)
  ret i32 %r
; CHECK-LABEL: test:
; CHECK:	leaq .LBB0_3(%rip), %r[[REG:[a-z]+]]
; CHECK:	movq %r[[REG]], buf+8(%rip)
; CHECK:	#EH_SjLj_Setup .LBB0_3
; CHECK:	xorl %e[[REG]], %e[[REG]]
; CHECK:	jmp .LBB0_2
; CHECK-LABEL: .LBB0_3: # Block address taken
; CHECK-LABEL: .LBB0_2:
}
