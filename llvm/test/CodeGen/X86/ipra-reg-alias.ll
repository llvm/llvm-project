; RUN: llc -enable-ipra -print-regusage -o /dev/null 2>&1 < %s | FileCheck %s
target triple = "x86_64--"

define i8 @main(i8 %X) {
  %inc = add i8 %X, 1
  %inc2 = mul i8 %inc, 5
; Here only CL is clobbred so CH should not be clobbred, but CX, ECX and RCX
; should be clobbered.
; CHECK: main Clobbered Registers: %ah %al %ax %cl %cx %eax %ecx %eflags %rax %rcx
  ret i8 %inc2
}

