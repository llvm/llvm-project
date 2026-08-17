; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s

!named = !{!0, !1, !2, !3, !4}

!0 = !DIFile(filename: "path/to/file", directory: "/path/to/dir")
!1 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)

; A getter must forward to the data member holding the backing storage.
; CHECK: property getterForward must be a member
!2 = !DIProperty(name: "x", file: !0, line: 8, type: !1, getterForward: !1)

; CHECK: property getterForward must be a member
!3 = !DIProperty(name: "x", file: !0, line: 8, type: !1, getterForward: !0)

; A pointer is a DIDerivedType, but not a DW_TAG_member.
; CHECK: property getterForward must be a member
!4 = !DIProperty(name: "x", file: !0, line: 8, type: !1, getterForward: !5)
!5 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !1, size: 64)
