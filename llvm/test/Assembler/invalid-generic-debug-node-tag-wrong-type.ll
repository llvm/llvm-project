; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s

; CHECK: <stdin>:[[@LINE+1]]:26: error: expected DWARF tag
!0 = !GenericDINode(tag: "string")
