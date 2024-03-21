; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s

; CHECK: !named = !{!0, !0, !1, !2, !3, !4, !5, !6, !7, !8, !8, !9, !10, !11, !12, !13, !14, !15, !16, !17, !18, !19, !20, !21, !22, !23, !24, !25, !26, !27, !27, !28, !29, !30, !31, !32, !33, !34, !35, !36, !37, !38, !39, !40, !41, !42, !43}
!named = !{!0, !1, !2, !3, !4, !5, !6, !7, !8, !9, !10, !11, !12, !13, !14, !15, !16, !17, !18, !19, !20, !21, !22, !23, !24, !25, !26, !27, !28, !29, !30, !31, !32, !33, !34, !35, !36, !37, !38, !39, !40, !41, !42, !43, !44, !45, !46}

; CHECK:      !0 = !DISubrange(count: 3, lowerBound: 0)
; CHECK-NEXT: !1 = !DISubrange(count: 3, lowerBound: 4)
; CHECK-NEXT: !2 = !DISubrange(count: 3, lowerBound: -5)
!0 = !DISubrange(count: 3, lowerBound: 0)
!1 = !DISubrange(count: 3, lowerBound: 0)

!2 = !DISubrange(count: 3, lowerBound: 4)
!3 = !DISubrange(count: 3, lowerBound: -5)

; CHECK-NEXT: !3 = !DIEnumerator(name: "seven", value: 7)
; CHECK-NEXT: !4 = !DIEnumerator(name: "negeight", value: -8)
; CHECK-NEXT: !5 = !DIEnumerator(name: "", value: 0)
!4 = !DIEnumerator(name: "seven", value: 7)
!5 = !DIEnumerator(name: "negeight", value: -8)
!6 = !DIEnumerator(name: "", value: 0)

; CHECK-NEXT: !6 = !DIBasicType(name: "name", size: 1, align: 2, encoding: DW_ATE_unsigned_char)
; CHECK-NEXT: !7 = !DIBasicType(tag: DW_TAG_unspecified_type, name: "decltype(nullptr)")
; CHECK-NEXT: !8 = !DIBasicType()
!7 = !DIBasicType(tag: DW_TAG_base_type, name: "name", size: 1, align: 2, encoding: DW_ATE_unsigned_char)
!8 = !DIBasicType(tag: DW_TAG_unspecified_type, name: "decltype(nullptr)")
!9 = !DIBasicType()
!10 = !DIBasicType(tag: DW_TAG_base_type, name: "", size: 0, align: 0, encoding: 0)

; CHECK-NEXT: !9 = !DITemplateTypeParameter(type: !6)
; CHECK-NEXT: !10 = !DIFile(filename: "path/to/file", directory: "/path/to/dir")
; CHECK-NEXT: !11 = distinct !{}
; CHECK-NEXT: !12 = !DIFile(filename: "", directory: "")
!11 = !DITemplateTypeParameter(type: !7)
!12 = !DIFile(filename: "path/to/file", directory: "/path/to/dir")
!13 = distinct !{}
!14 = !DIFile(filename: "", directory: "")

; CHECK-NEXT: !13 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !6, size: 32, align: 32, dwarfAddressSpace: 1)
!15 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !7, size: 32, align: 32, dwarfAddressSpace: 1)

; CHECK-NEXT: !14 = !DICompositeType(tag: DW_TAG_structure_type, name: "MyType", file: !10, line: 2, size: 32, align: 32, identifier: "MangledMyType")
; CHECK-NEXT: !15 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Base", scope: !14, file: !10, line: 3, size: 128, align: 32, offset: 64, flags: DIFlagPublic, elements: !16, runtimeLang: DW_LANG_C_plus_plus_11, vtableHolder: !15, templateParams: !18, identifier: "MangledBase")
; CHECK-NEXT: !16 = !{!17}
; CHECK-NEXT: !17 = !DIDerivedType(tag: DW_TAG_member, name: "field", scope: !15, file: !10, line: 4, baseType: !6, size: 32, align: 32, offset: 32, flags: DIFlagPublic)
; CHECK-NEXT: !18 = !{!9}
; CHECK-NEXT: !19 = !DICompositeType(tag: DW_TAG_structure_type, name: "Derived", scope: !14, file: !10, line: 3, baseType: !15, size: 128, align: 32, offset: 64, flags: DIFlagPublic, elements: !20, runtimeLang: DW_LANG_C_plus_plus_11, vtableHolder: !15, templateParams: !18, identifier: "MangledBase")
; CHECK-NEXT: !20 = !{!21}
; CHECK-NEXT: !21 = !DIDerivedType(tag: DW_TAG_inheritance, scope: !19, baseType: !15)
; CHECK-NEXT: !22 = !DIDerivedType(tag: DW_TAG_ptr_to_member_type, baseType: !6, size: 32, align: 32, extraData: !15)
; CHECK-NEXT: !23 = !DICompositeType(tag: DW_TAG_structure_type)
; CHECK-NEXT: !24 = !DICompositeType(tag: DW_TAG_structure_type, runtimeLang: DW_LANG_Cobol85)
!16 = !DICompositeType(tag: DW_TAG_structure_type, name: "MyType", file: !12, line: 2, size: 32, align: 32, identifier: "MangledMyType")
!17 = !DICompositeType(tag: DW_TAG_structure_type, name: "Base", scope: !16, file: !12, line: 3, size: 128, align: 32, offset: 64, flags: DIFlagPublic, elements: !18, runtimeLang: DW_LANG_C_plus_plus_11, vtableHolder: !17, templateParams: !20, identifier: "MangledBase")
!18 = !{!19}
!19 = !DIDerivedType(tag: DW_TAG_member, name: "field", scope: !17, file: !12, line: 4, baseType: !7, size: 32, align: 32, offset: 32, flags: DIFlagPublic)
!20 = !{!11}
!21 = !DICompositeType(tag: DW_TAG_structure_type, name: "Derived", scope: !16, file: !12, line: 3, baseType: !17, size: 128, align: 32, offset: 64, flags: DIFlagPublic, elements: !22, runtimeLang: DW_LANG_C_plus_plus_11, vtableHolder: !17, templateParams: !20, identifier: "MangledBase")
!22 = !{!23}
!23 = !DIDerivedType(tag: DW_TAG_inheritance, scope: !21, baseType: !17)
!24 = !DIDerivedType(tag: DW_TAG_ptr_to_member_type, baseType: !7, size: 32, align: 32, extraData: !17)
!25 = !DICompositeType(tag: DW_TAG_structure_type)
!26 = !DICompositeType(tag: DW_TAG_structure_type, runtimeLang: 6)

; CHECK-NEXT: !25 = !{!6, !6}
; CHECK-NEXT: !26 = !DISubroutineType(flags: DIFlagPublic | DIFlagStaticMember, types: !25)
; CHECK-NEXT: !27 = !DISubroutineType(types: !25)
!27 = !{!7, !7}
!28 = !DISubroutineType(flags: DIFlagPublic | DIFlagStaticMember, types: !27)
!29 = !DISubroutineType(flags: 0, types: !27)
!30 = !DISubroutineType(types: !27)

; CHECK-NEXT: !28 = !DIMacro(type: DW_MACINFO_define, line: 9, name: "Name", value: "Value")
; CHECK-NEXT: !29 = distinct !{!28}
; CHECK-NEXT: !30 = !DIMacroFile(line: 9, file: !12, nodes: !29)
; CHECK-NEXT: !31 = !DIMacroFile(line: 11, file: !12)
!31 = !DIMacro(type: DW_MACINFO_define, line: 9, name: "Name", value: "Value")
!32 = distinct !{!31}
!33 = !DIMacroFile(line: 9, file: !14, nodes: !32)
!34 = !DIMacroFile(type: DW_MACINFO_start_file, line: 11, file: !14)

; CHECK-NEXT: !32 = !DIFile(filename: "file", directory: "dir", checksumkind: CSK_MD5, checksum: "000102030405060708090a0b0c0d0e0f")
!35 = !DIFile(filename: "file", directory: "dir", checksumkind: CSK_MD5, checksum: "000102030405060708090a0b0c0d0e0f")

; CHECK-NEXT: !33 = !DICompositeType(tag: DW_TAG_variant_part, name: "A", scope: !14, size: 64, discriminator: !34)
; CHECK-NEXT: !34 = !DIDerivedType(tag: DW_TAG_member, scope: !33, baseType: !35, size: 64, align: 64, flags: DIFlagArtificial)
; CHECK-NEXT: !35 = !DIBasicType(name: "u64", size: 64, encoding: DW_ATE_unsigned)
!36 = !DICompositeType(tag: DW_TAG_variant_part, name: "A", scope: !16, size: 64, discriminator: !37)
!37 = !DIDerivedType(tag: DW_TAG_member, scope: !36, baseType: !38, size: 64, align: 64, flags: DIFlagArtificial)
!38 = !DIBasicType(name: "u64", size: 64, encoding: DW_ATE_unsigned)

; CHECK-NEXT: !36 = !DIFile(filename: "file", directory: "dir", source: "int source() { }\0A")
; CHECK-NEXT: !37 = !DIFile(filename: "file", directory: "dir", checksumkind: CSK_MD5, checksum: "3a420e2646916a475e68de8d48f779f5", source: "int source() { }\0A")
!39 = !DIFile(filename: "file", directory: "dir", source: "int source() { }\0A")
!40 = !DIFile(filename: "file", directory: "dir", checksumkind: CSK_MD5, checksum: "3a420e2646916a475e68de8d48f779f5", source: "int source() { }\0A")

; CHECK-NEXT: !38 = !DIBasicType(name: "u64.be", size: 64, align: 1, encoding: DW_ATE_unsigned, flags: DIFlagBigEndian)
; CHECK-NEXT: !39 = !DIBasicType(name: "u64.le", size: 64, align: 1, encoding: DW_ATE_unsigned, flags: DIFlagLittleEndian)
!41 = !DIBasicType(name: "u64.be", size: 64, align: 1, encoding: DW_ATE_unsigned, flags: DIFlagBigEndian)
!42 = !DIBasicType(name: "u64.le", size: 64, align: 1, encoding: DW_ATE_unsigned, flags: DIFlagLittleEndian)

; CHECK: !DIDerivedType(tag: DW_TAG_LLVM_ptrauth_type, baseType: !13, ptrAuthKey: 2, ptrAuthIsAddressDiscriminated: true, ptrAuthExtraDiscriminator: 1234, ptrAuthIsaPointer: false, ptrAuthAuthenticatesNullValues: false)
!43 = !DIDerivedType(tag: DW_TAG_LLVM_ptrauth_type, baseType: !15, ptrAuthKey: 2, ptrAuthIsAddressDiscriminated: true, ptrAuthExtraDiscriminator: 1234)

; CHECK: !DIDerivedType(tag: DW_TAG_LLVM_ptrauth_type, baseType: !13, ptrAuthKey: 2, ptrAuthIsAddressDiscriminated: true, ptrAuthExtraDiscriminator: 1234, ptrAuthIsaPointer: true, ptrAuthAuthenticatesNullValues: false)
!44 = !DIDerivedType(tag: DW_TAG_LLVM_ptrauth_type, baseType: !15, ptrAuthKey: 2, ptrAuthIsAddressDiscriminated: true, ptrAuthExtraDiscriminator: 1234, ptrAuthIsaPointer: true)

; CHECK: !DIDerivedType(tag: DW_TAG_LLVM_ptrauth_type, baseType: !13, ptrAuthKey: 2, ptrAuthIsAddressDiscriminated: true, ptrAuthExtraDiscriminator: 1234, ptrAuthIsaPointer: false, ptrAuthAuthenticatesNullValues: true)
!45 = !DIDerivedType(tag: DW_TAG_LLVM_ptrauth_type, baseType: !15, ptrAuthKey: 2, ptrAuthIsAddressDiscriminated: true, ptrAuthExtraDiscriminator: 1234, ptrAuthAuthenticatesNullValues: true)

; CHECK: !DIDerivedType(tag: DW_TAG_LLVM_ptrauth_type, baseType: !13, ptrAuthKey: 2, ptrAuthIsAddressDiscriminated: true, ptrAuthExtraDiscriminator: 1234, ptrAuthIsaPointer: true, ptrAuthAuthenticatesNullValues: true)
!46 = !DIDerivedType(tag: DW_TAG_LLVM_ptrauth_type, baseType: !15, ptrAuthKey: 2, ptrAuthIsAddressDiscriminated: true, ptrAuthExtraDiscriminator: 1234, ptrAuthIsaPointer: true, ptrAuthAuthenticatesNullValues: true)
