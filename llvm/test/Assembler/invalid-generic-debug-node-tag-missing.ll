; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s

; CHECK: <stdin>:[[@LINE+1]]:44: error: missing required field 'tag'
!0 = !GenericDINode(header: "some\00header")
