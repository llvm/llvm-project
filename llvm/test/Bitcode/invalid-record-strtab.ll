; Bitcode with an invalid record that indexes a name outside of strtab.

; RUN: not llvm-dis %s.bc -o - 2>&1 | FileCheck %s

; CHECK: error: Invalid record
