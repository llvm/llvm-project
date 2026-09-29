; Bitcode with a CST_CODE_NULL with void type.

; RUN: not llvm-dis %s.bc -o - 2>&1 | FileCheck %s

; CHECK: error: Invalid type for a constant null value

