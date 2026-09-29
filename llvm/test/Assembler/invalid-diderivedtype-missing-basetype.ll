; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s

; CHECK: [[@LINE+1]]:45: error: missing required field 'baseType'
!0 = !DIDerivedType(tag: DW_TAG_pointer_type)
