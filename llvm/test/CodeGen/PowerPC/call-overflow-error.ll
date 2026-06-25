; RUN: not llc -verify-machineinstrs -mcpu=pwr7 -mtriple powerpc-ibm-aix \
; RUN:   2>&1 -filetype=obj < %s | FileCheck %s --check-prefix=ERROR

; RUN: llc -verify-machineinstrs -mcpu=pwr7 -mtriple powerpc-ibm-aix \
; RUN:   --function-sections -filetype=obj -o %t.o < %s
; RUN: llvm-objdump -Dr %t.o | FileCheck %s

define signext i32 @bar() {
entry:
  ret i32 42
}

define signext i32 @foo() {
entry:
  call void asm sideeffect ".space 0x2000100", ""()
  %call = call signext i32 @bar()
  ret i32 %call
}

; ERROR: error: branch target out of range (-33554732 not between -33554432 and 33554428)
; CHECK: 200012c: 49 ff fe d5   bl 0x4000000 <.bar+0x3ffffe0>
; CHECK:                        0200012c:  R_RBR        .bar
