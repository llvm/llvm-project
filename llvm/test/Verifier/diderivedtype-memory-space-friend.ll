; RUN: not opt -S < %s 2>&1 | FileCheck %s

!named = !{!0, !1}
!0 = !DIBasicType(tag: DW_TAG_base_type, name: "name", size: 1, align: 2, encoding: DW_ATE_unsigned_char)
; CHECK: DWARF memory space only applies to pointer or reference types
!1 = !DIDerivedType(tag: DW_TAG_friend, baseType: !0, size: 32, align: 32, memorySpace: DW_MSPACE_LLVM_private)
