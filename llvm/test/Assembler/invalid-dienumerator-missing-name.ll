; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s

; CHECK: [[@LINE+1]]:28: error: missing required field 'name'
!0 = !DIEnumerator(value: 7)
