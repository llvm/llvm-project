; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s

; CHECK: [[@LINE+1]]:34: error: missing required field 'tag'
!0 = !DIDerivedType(baseType: !{})
