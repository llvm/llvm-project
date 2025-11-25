; RUN: llvm-as %s -o - | llvm-dis | FileCheck %s

; CHECK-DAG: ![[BASIC:[0-9]+]] = !DIBasicType
; CHECK-DAG: !DIDerivedType(tag: DW_TAG_pointer_type, baseType: ![[BASIC]], addressSpace: 1, memorySpace: DW_MSPACE_LLVM_private)
; CHECK-DAG: !DIDerivedType(tag: DW_TAG_reference_type, baseType: ![[BASIC]], addressSpace: 1, memorySpace: DW_MSPACE_LLVM_private)

!named = !{!0, !1}

!0 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !2, addressSpace: 1, memorySpace: DW_MSPACE_LLVM_private)
!1 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !2, addressSpace: 1, memorySpace: 4)
!2 = !DIBasicType(tag: DW_TAG_base_type, name: "name", size: 1, align: 2, encoding: DW_ATE_unsigned_char)
