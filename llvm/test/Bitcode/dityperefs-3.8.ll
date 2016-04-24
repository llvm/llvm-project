; RUN: llvm-dis < %s.bc | FileCheck %s
; RUN: verify-uselistorder %s.bc

; Establish a stable order.
!named = !{!0, !1, !2, !3, !4, !6, !7, !8, !9, !10, !11, !12, !13, !14, !15, !16}

; CHECK:      !0 = !DIFile(filename: "path/to/file", directory: "/path/to/dir")
; CHECK-NEXT: !1 = !DICompositeType(tag: DW_TAG_structure_type, name: "T1"{{.*}}, identifier: "T1")
; CHECK-NEXT: !2 = !DICompositeType(tag: DW_TAG_structure_type, name: "T2", scope: !1{{.*}}, baseType: !1, vtableHolder: !1, identifier: "T2")
; CHECK-NEXT: !3 = !DIDerivedType(tag: DW_TAG_member, name: "M1", scope: !1{{.*}}, baseType: !2)
; CHECK-NEXT: !4 = !DISubroutineType(types: !5)
; CHECK-NEXT: !5 = !{!1, !2}
; CHECK-NEXT: !6 = !DISubprogram(scope: !1,{{.*}} containingType: !1{{[,)]}}
; CHECK-NEXT: !7 = !DILocalVariable(name: "V1", scope: !6, type: !2)
; CHECK-NEXT: !8 = !DIObjCProperty(name: "P1", type: !1)
; CHECK-NEXT: !9 = !DITemplateTypeParameter(type: !1)
; CHECK-NEXT: !10 = !DIGlobalVariable(name: "G",{{.*}} type: !1,{{.*}} variable: i32* @G1)
; CHECK-NEXT: !11 = !DITemplateValueParameter(type: !1, value: i32* @G1)
; CHECK-NEXT: !12 = !DIImportedEntity(tag: DW_TAG_imported_module, name: "T2", scope: !0, entity: !1)
; CHECK-NEXT: !13 = !DICompositeType(tag: DW_TAG_structure_type, name: "T3", file: !0, elements: !14, identifier: "T3")
; CHECK-NEXT: !14 = !{!15}
; CHECK-NEXT: !15 = !DISubprogram(scope: !13,
; CHECK-NEXT: !16 = !DIDerivedType(tag: DW_TAG_ptr_to_member_type,{{.*}} extraData: !13)

!0 = !DIFile(filename: "path/to/file", directory: "/path/to/dir")
!1 = !DICompositeType(tag: DW_TAG_structure_type, name: "T1", file: !0, identifier: "T1")
!2 = !DICompositeType(tag: DW_TAG_structure_type, name: "T2", file: !0, scope: !"T1", baseType: !"T1", vtableHolder: !"T1", identifier: "T2")
!3 = !DIDerivedType(tag: DW_TAG_member, name: "M1", file: !0, scope: !"T1", baseType: !"T2")
!4 = !DISubroutineType(types: !5)
!5 = !{!"T1", !"T2"}
!6 = !DISubprogram(scope: !"T1", isDefinition: false, containingType: !"T1")
!7 = !DILocalVariable(name: "V1", scope: !6, type: !"T2")
!8 = !DIObjCProperty(name: "P1", type: !"T1")
!9 = !DITemplateTypeParameter(type: !"T1")
!10 = !DIGlobalVariable(name: "G", type: !"T1", isDefinition: false, variable: i32* @G1)
!11 = !DITemplateValueParameter(type: !"T1", value: i32* @G1)
!12 = !DIImportedEntity(tag: DW_TAG_imported_module, name: "T2", scope: !0, entity: !"T1")
!13 = !DICompositeType(tag: DW_TAG_structure_type, name: "T3", file: !0, elements: !14, identifier: "T3")
!14 = !{!15}
!15 = !DISubprogram(scope: !"T3", isDefinition: false)
!16 = !DIDerivedType(tag: DW_TAG_ptr_to_member_type, baseType: !4, extraData: !"T3")

@G1 = global i32 0
