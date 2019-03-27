; RUN: llc < %s -mtriple=x86_64-linux -remarks-section -pass-remarks-output=%/t.yaml | FileCheck -DPATH=%/t.yaml %s
; RUN: llc < %s -mtriple=x86_64-darwin -remarks-section -pass-remarks-output=%/t.yaml | FileCheck --check-prefix=CHECK-DARWIN -DPATH=%/t.yaml %s

; CHECK-LABEL: func1:

; CHECK: .section .remarks,"e",@progbits
; The magic number:
; CHECK-NEXT: .ascii "REMARKS"
; Null-terminator:
; CHECK-NEXT: .byte 0
; The version:
; CHECK-NEXT: .byte 0x00, 0x00, 0x00, 0x00
; CHECK-NEXT: .byte 0x00, 0x00, 0x00, 0x00
; The remark file path:
; CHECK-NEXT: .ascii "[[PATH]]"
; Null-terminator:
; CHECK-NEXT: .byte 0

; CHECK-DARWIN: .section __LLVM,__remarks,regular,debug
; The magic number:
; CHECK-DARWIN-NEXT: .ascii "REMARKS"
; Null-terminator:
; CHECK-DARWIN-NEXT: .byte 0
; The version:
; CHECK-DARWIN-NEXT: .byte 0x00, 0x00, 0x00, 0x00
; CHECK-DARWIN-NEXT: .byte 0x00, 0x00, 0x00, 0x00
; The remark file path:
; CHECK-DARWIN-NEXT: .ascii "[[PATH]]"
; Null-terminator:
; CHECK-DARWIN-NEXT: .byte 0
define void @func1() {
  ret void
}
