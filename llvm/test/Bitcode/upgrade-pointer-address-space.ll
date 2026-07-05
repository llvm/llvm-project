; RUN: llvm-dis -o - %s.bc | FileCheck %s

; CHECK: !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !{{[0-9]+}}, size: {{[0-9]+}})
; CHECK: !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !{{[0-9]+}}, size: {{[0-9]+}})
