; RUN: not llvm-as < %s -o /dev/null 2>&1 | FileCheck %s

@var = internal protected global i32 0
; CHECK: symbol with local linkage must have default visibility
