; Bitcode with invalid natural stack alignment.

; RUN: not llvm-dis %s.bc -o - 2>&1 | FileCheck %s

CHECK: error: stack natural alignment must be a power of two times the byte width
