; RUN: not %python %p/Inputs/llvm-dis-and-filecheck.py llvm-dis FileCheck %s %s 2>&1 | FileCheck %s
; CHECK: stderr
; CHECK: stdout
