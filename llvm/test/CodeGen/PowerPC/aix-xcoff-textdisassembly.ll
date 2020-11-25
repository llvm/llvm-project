; RUN: llc -verify-machineinstrs -mcpu=pwr9 -mattr=-altivec -mtriple powerpc-ibm-aix-xcoff -filetype=obj -o %t.o  < %s
; RUN: llvm-objdump -D %t.o | FileCheck --check-prefix=CHECK %s

define i32 @foo() #0 {
entry:
  ret i32 0
}

define i32 @foo1() #0 {
entry:
  ret i32 1
}

; CHECK:     Disassembly of section .text:{{[[:space:]] *}}
; CHECK-NEXT:     00000000 <.text>:
; CHECK-NEXT:        0: 38 60 00 00                   li 3, 0
; CHECK-NEXT:        4: 4e 80 00 20                   blr
; CHECK-NEXT:        8: 60 00 00 00                   nop
; CHECK-NEXT:        c: 60 00 00 00                   nop
; CHECK:     00000010 <.foo1>:
; CHECK-NEXT:       10: 38 60 00 01                   li 3, 1
; CHECK-NEXT:       14: 4e 80 00 20                   blr
