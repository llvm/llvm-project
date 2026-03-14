; RUN: llc < %s -mtriple=x86_64-- -stress-sched | FileCheck %s
; REQUIRES: asserts
; Test interference between physreg aliases during preRAsched.
; mul wants an operand in AL, but call clobbers it.

define i8 @f(i8 %v1, i8 %v2) nounwind {
entry:
; CHECK: callq
; CHECK: movl %{{.*}}, %eax
; CHECK: mulb
; CHECK: mulb
        %rval = tail call i8 @bar() nounwind
        %m1 = mul i8 %v1, %v2
        %m2 = mul i8 %m1, %rval
        ret i8 %m2
}

declare i8 @bar()
