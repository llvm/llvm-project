; RUN: llvm-as < %s | llvm-dis | grep 9223372036854775808
; RUN: verify-uselistorder %s

@0 = global i64 -9223372036854775808

