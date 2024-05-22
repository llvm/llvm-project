; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s

; CHECK: expected top-level entity
@g1 = external global ptr addrspace(3) addrspace(4)
