; PR18354
; We actually need to use -filetype=obj in this test because if we output
; assembly, the current code path will bypass the parser and just write the
; raw text out to the Streamer. We need to actually parse the inlineasm to
; demonstrate the bug. Going the asm->obj route does not show the issue.
; RUN: llc -mtriple=arm-none-linux   < %s -filetype=obj | llvm-objdump -d - | FileCheck %s
; RUN: llc -mtriple=arm-apple-darwin < %s -filetype=obj | llvm-objdump -d - | FileCheck %s
; CHECK-LABEL: foo:
; CHECK: 0:       00 00 9f e5                                     ldr     r0, [pc]
; CHECK: 4:       0e f0 a0 e1                                     mov     pc, lr
; Make sure the constant pool entry comes after the return
; CHECK: 8:       78 56 34 12
define i32 @foo() nounwind {
entry:
  %0 = tail call i32 asm sideeffect "ldr $0,=0x12345678", "=r"() nounwind
  ret i32 %0
}
