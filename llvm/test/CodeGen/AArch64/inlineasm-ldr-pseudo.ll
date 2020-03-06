; We actually need to use -filetype=obj in this test because if we output
; assembly, the current code path will bypass the parser and just write the
; raw text out to the Streamer. We need to actually parse the inlineasm to
; demonstrate the bug. Going the asm->obj route does not show the issue.
; RUN: llc -mtriple=aarch64   < %s -filetype=obj | llvm-objdump -arch=aarch64 -d - | FileCheck %s

; CHECK-LABEL: <foo>:
; CHECK:       a0 79 95 d2   mov x0, #43981
; CHECK:       c0 03 5f d6   ret
define i32 @foo() nounwind {
entry:
  %0 = tail call i32 asm sideeffect "ldr $0,=0xabcd", "=r"() nounwind
  ret i32 %0
}
; CHECK-LABEL: <bar>:
; CHECK:        40 00 00 58                                      ldr    x0, #8
; CHECK:        c0 03 5f d6                                      ret
; Make sure the constant pool entry comes after the return
; CHECK-LABEL:        <$d.1>:
define i32 @bar() nounwind {
entry:
  %0 = tail call i32 asm sideeffect "ldr $0,=0x10001", "=r"() nounwind
  ret i32 %0
}
