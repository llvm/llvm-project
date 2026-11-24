; RUN: llvm-as %s -disable-output
; RUN: verify-uselistorder %s

@spell_order = global [4 x i8] c"\FF\00\F7\00"

