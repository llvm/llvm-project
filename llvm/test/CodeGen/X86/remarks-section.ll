; RUN: llc < %s -mtriple=x86_64-linux -remarks-section -pass-remarks-output=%/t.yaml | FileCheck -DPATH=%/t.yaml %s
; RUN: llc < %s -mtriple=x86_64-darwin -remarks-section -pass-remarks-output=%/t.yaml | FileCheck --check-prefix=CHECK-DARWIN -DPATH=%/t.yaml %s
; RUN: llc < %s -mtriple=x86_64-darwin --pass-remarks-format=yaml-strtab -remarks-section -pass-remarks-output=%/t.yaml | FileCheck --check-prefix=CHECK-DARWIN-STRTAB -DPATH=%/t.yaml %s

; CHECK-LABEL: func1:

; CHECK: .section .remarks,"e",@progbits
; CHECK-NEXT: .byte

; CHECK-DARWIN: .section __LLVM,__remarks,regular,debug
; CHECK-DARWIN-NEXT: .byte

; CHECK-DARWIN-STRTAB: .section __LLVM,__remarks,regular,debug
; CHECK-DARWIN-STRTAB-NEXT: .byte
define void @func1() {
  ret void
}
