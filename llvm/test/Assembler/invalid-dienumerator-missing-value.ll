; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s

; CHECK: [[@LINE+1]]:32: error: missing required field 'value'
!0 = !DIEnumerator(name: "name")
