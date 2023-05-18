; PR18354
; We actually need to use -filetype=obj in this test because if we output
; assembly, the current code path will bypass the parser and just write the
; raw text out to the Streamer. We need to actually parse the inlineasm to
; demonstrate the bug. Going the asm->obj route does not show the issue.
; RUN: llc -mtriple=arm-none-linux   < %s -filetype=obj | llvm-objdump -d - | FileCheck %s --check-prefixes=CHECK,CHECK-ELF
; RUN: llc -mtriple=arm-apple-darwin < %s -filetype=obj | llvm-objdump -d - | FileCheck %s --check-prefixes=CHECK,CHECK-MACHO
; CHECK-LABEL: <{{_?}}foo>:
; CHECK: 0:       e59f0000                                        ldr     r0, [pc]
; CHECK: 4:       e1a0f00e                                        mov     pc, lr
; Make sure the constant pool entry comes after the return
; CHECK-ELF: 8:       78 56 34 12
; CHECK-MACHO: 8:       12345678
define i32 @foo() nounwind {
entry:
  %0 = tail call i32 asm sideeffect "ldr $0,=0x12345678", "=r"() nounwind
  ret i32 %0
}
