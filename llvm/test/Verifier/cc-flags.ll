; RUN: not opt -S < %s 2>&1 | FileCheck %s

!named = !{!0}
!0 = !DICompositeType(tag: DW_TAG_structure_type, name: "A", size: 1, flags: DIFlagTypePassByReference | DIFlagTypePassByValue)
; CHECK: invalid reference flags
