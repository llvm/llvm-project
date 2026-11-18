; RUN: llvm-as %s -disable-output
; RUN: verify-uselistorder %s
; Test that shift instructions can be used in constant expressions.

@0 = global i32 3670016
