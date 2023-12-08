; RUN: llvm-mc < %s -triple arm64-apple-darwin -filetype=obj -o - | llvm-readobj -r - | FileCheck %s

  adrp x3, Lbar@page
; CHECK: ARM64_RELOC_PAGE21
Lbar:
  ret
