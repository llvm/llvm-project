; RUN: not llvm-as < %s 2>&1 | FileCheck %s

attributes
; CHECK: expected attribute group id
