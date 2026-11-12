; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s

; Don't crash on null operands.  When we add a verify check for this, also
; require non-null in the assembler and rework this test to check for that ala
; test/Assembler/invalid-mdcompileunit-null-file.ll.
!named = !{!0}
!0 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null)

; CHECK: !named = !{!0}
; CHECK: !0 = !DIDerivedType({{.*}}baseType: null{{.*}})
