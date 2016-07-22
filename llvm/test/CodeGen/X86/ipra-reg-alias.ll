; RUN: llc -enable-ipra -print-regusage -o /dev/null 2>&1 < %s | FileCheck %s
target triple = "x86_64--"

define i8 @main(i8 %X) {
  %inc = add i8 %X, 1
  %inc2 = mul i8 %inc, 5
; Here only CL is clobbred so CH should not be clobbred, but CX, ECX and RCX
; should be clobbered.
; CHECK: main Clobbered Registers: AH AL AX CL CX EAX ECX EFLAGS RAX RCX
  ret i8 %inc2
}

