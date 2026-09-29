; RUN: llvm-as %s -o /dev/null
; RUN: verify-uselistorder %s
; Test that shift instructions can be used in constant expressions.

@0 = global i32 3670016
