; RUN: not llvm-mc < %s -triple arm64-apple-darwin -filetype=obj -o - 2>&1 | FileCheck %s

; CHECK: error: ADR/ADRP relocations must be GOT relative
  adrp x3, Lbar
Lbar:
  ret
