; RUN: llc < %s -mtriple=x86_64-darwin -remarks-section -pass-remarks-output=%/t.yaml | FileCheck --check-prefix=CHECK-DARWIN -DPATH=%/t.yaml %s
; RUN: llc < %s -mtriple=x86_64-darwin --pass-remarks-format=yaml-strtab -remarks-section -pass-remarks-output=%/t.yaml | FileCheck --check-prefix=CHECK-DARWIN-STRTAB -DPATH=%/t.yaml %s

; RUN: llc < %s -mtriple=x86_64-darwin -pass-remarks-output=%/t.yaml | FileCheck --check-prefix=CHECK-DARWIN-DEFAULT %s
; RUN: llc < %s -mtriple=x86_64-darwin --pass-remarks-format=yaml-strtab -pass-remarks-output=%/t.yaml | FileCheck --check-prefix=CHECK-DARWIN-DEFAULT-YAML-STRTAB %s
; RUN: llc < %s -mtriple=x86_64-darwin --pass-remarks-format=bitstream -pass-remarks-output=%/t.yaml | FileCheck --check-prefix=CHECK-DARWIN-DEFAULT-BITSTREAM %s
; RUN: llc < %s -mtriple=x86_64-darwin --pass-remarks-format=bitstream -remarks-section=false -pass-remarks-output=%/t.yaml | FileCheck --check-prefix=CHECK-DARWIN-OVERRIDE-BITSTREAM %s
; RUN: llc < %s -mtriple=x86_64-darwin --pass-remarks-format=yaml -remarks-section=true -pass-remarks-output=%/t.yaml | FileCheck --check-prefix=CHECK-DARWIN-OVERRIDE-YAML %s

; CHECK-DARWIN: .section __LLVM,__remarks,regular,debug
; CHECK-DARWIN-NEXT: .byte

; CHECK-DARWIN-STRTAB: .section __LLVM,__remarks,regular,debug
; CHECK-DARWIN-STRTAB-NEXT: .byte

; By default, the format is YAML which does not need a section.
; CHECK-DARWIN-DEFAULT-NOT: .section __LLVM,__remarks

; yaml-strtab needs a section.
; CHECK-DARWIN-DEFAULT-YAML-STRTAB: .section __LLVM,__remarks

; bitstream needs a section.
; CHECK-DARWIN-DEFAULT-BITSTREAM: .section __LLVM,__remarks

; -remarks-section should force disable the section.
; CHECK-DARWIN-OVERRIDE-BITSTREAM-NOT: .section __LLVM,__remarks

; -remarks-section should also force enable the section.
; CHECK-DARWIN-OVERRIDE-YAML: .section __LLVM,__remarks
define void @func1() {
  ret void
}
