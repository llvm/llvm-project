; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s

; Anchor the order of the nodes.
!named = !{!0, !1, !2, !3, !4, !5, !6, !7, !8, !9, !10, !11, !12, !13, !14, !15, !16, !17}

; Some basic building blocks.
; CHECK:      !0 = !DIBasicType
; CHECK-NEXT: !1 = !DIFile
; CHECK-NEXT: !2 = !DIFile
!0 = !DIBasicType(tag: DW_TAG_base_type, name: "name", size: 1, align: 2, encoding: DW_ATE_unsigned_char)
!1 = !DIFile(filename: "path/to/file", directory: "/path/to/dir")
!2 = !DIFile(filename: "path/to/other", directory: "/path/to/dir")

; Define an identified type with fields and functions.
; CHECK-NEXT: !3 = !DICompositeType(tag: DW_TAG_structure_type, name: "has-uuid",{{.*}}, identifier: "uuid")
; CHECK-NEXT: !4 = !DIDerivedType(tag: DW_TAG_member, name: "field1", scope: !3, file: !1
; CHECK-NEXT: !5 = !DIDerivedType(tag: DW_TAG_member, name: "field2", scope: !3, file: !1
; CHECK-NEXT: !6 = !DISubprogram(name: "foo", linkageName: "foo1", scope: !3, file: !1
; CHECK-NEXT: !7 = !DISubprogram(name: "foo", linkageName: "foo2", scope: !3, file: !1
!3 = !DICompositeType(tag: DW_TAG_structure_type, name: "has-uuid", file: !1, line: 2, size: 64, align: 32, identifier: "uuid")
!4 = !DIDerivedType(tag: DW_TAG_member, name: "field1", scope: !3, file: !1, line: 4, baseType: !0, size: 32, align: 32, offset: 32)
!5 = !DIDerivedType(tag: DW_TAG_member, name: "field2", scope: !3, file: !1, line: 4, baseType: !0, size: 32, align: 32, offset: 32)
!6 = !DISubprogram(name: "foo", linkageName: "foo1", scope: !3, file: !1, isDefinition: false)
!7 = !DISubprogram(name: "foo", linkageName: "foo2", scope: !3, file: !1, isDefinition: false)

; Define an un-identified type with fields and functions.
; CHECK-NEXT: !8 = !DICompositeType(tag: DW_TAG_structure_type, name: "no-uuid", file: !1
; CHECK-NEXT: !9 = !DIDerivedType(tag: DW_TAG_member, name: "field1", scope: !8, file: !1
; CHECK-NEXT: !10 = !DIDerivedType(tag: DW_TAG_member, name: "field2", scope: !8, file: !1
; CHECK-NEXT: !11 = !DISubprogram(name: "foo", linkageName: "foo1", scope: !8, file: !1
; CHECK-NEXT: !12 = !DISubprogram(name: "foo", linkageName: "foo2", scope: !8, file: !1
!8 = !DICompositeType(tag: DW_TAG_structure_type, name: "no-uuid", file: !1, line: 2, size: 64, align: 32)
!9 = !DIDerivedType(tag: DW_TAG_member, name: "field1", scope: !8, file: !1, line: 4, baseType: !0, size: 32, align: 32, offset: 32)
!10 = !DIDerivedType(tag: DW_TAG_member, name: "field2", scope: !8, file: !1, line: 4, baseType: !0, size: 32, align: 32, offset: 32)
!11 = !DISubprogram(name: "foo", linkageName: "foo1", scope: !8, file: !1, isDefinition: false)
!12 = !DISubprogram(name: "foo", linkageName: "foo2", scope: !8, file: !1, isDefinition: false)

; Add duplicate fields and members of "no-uuid" in a different file.  These
; should stick around, since "no-uuid" does not have an "identifier:" field.
; CHECK-NEXT: !13 = !DIDerivedType(tag: DW_TAG_member, name: "field1", scope: !8, file: !2,
; CHECK-NEXT: !14 = !DISubprogram(name: "foo", linkageName: "foo1", scope: !8, file: !2,
!13 = !DIDerivedType(tag: DW_TAG_member, name: "field1", scope: !8, file: !2, line: 4, baseType: !0, size: 32, align: 32, offset: 32)
!14 = !DISubprogram(name: "foo", linkageName: "foo1", scope: !8, file: !2, isDefinition: false)

; Add duplicate fields and members of "has-uuid" in a different file.  These
; should be merged.
!15 = !DIDerivedType(tag: DW_TAG_member, name: "field1", scope: !3, file: !2, line: 4, baseType: !0, size: 32, align: 32, offset: 32)
!16 = !DISubprogram(name: "foo", linkageName: "foo1", scope: !3, file: !2, isDefinition: false)

; CHECK-NEXT: !15 = !{!4, !6}
; CHECK-NOT: !DIDerivedType
; CHECK-NOT: !DISubprogram
!17 = !{!15, !16}
