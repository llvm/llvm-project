; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s

; CHECK: [[@LINE+1]]:44: error: missing required field 'type'
!0 = !DITemplateTypeParameter(name: "param")
