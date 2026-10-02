; RUN: not llc -verify-machineinstrs -mcpu=pwr7 -mtriple powerpc-ibm-aix \
; RUN:   2>&1 -filetype=obj < %s | FileCheck %s --check-prefix=ERROR

; RUN: llc -verify-machineinstrs -mcpu=pwr7 -mtriple powerpc-ibm-aix \
; RUN:   --function-sections -filetype=obj -o %t.o < %s
; RUN: llvm-objdump -Dr %t.o | FileCheck %s

declare void @baz()

define i32 @padding() {
  entry:
  ret i32 55
}

define signext i32 @bar() {
entry:
  ret i32 42
}

define signext i32 @foo() {
entry:
  call void asm sideeffect ".space 0x2000100", ""()
  %call = call signext i32 @bar()
  call void @baz()
  ret i32 %call
}

; ERROR: error: branch target out of range (-33554736 not between -33554432 and 33554428)
; CHECK: 2000170: 49 ff fe d1   bl 0x4000040
; CHECK:                        02000170:  R_RBR        .bar
; CHECK: 200017c: 49 ff fe 85   bl 0x4000000
; CHECK:                        0200017c:  R_RBR        .baz
