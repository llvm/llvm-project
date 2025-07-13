; Test if StripNonLineTableDebugInfo crashes or produces invalid IR,
; this test contains a slightly complex debug info structure,
; which may trigger the bug mentioned in pr#125116
;
; RUN: opt < %s -p=strip-nonlinetable-debuginfo -S | FileCheck %s
;
; CHECK-NOT: DIBasicType
; CHECK-NOT: DIDerivedType
; CHECK-NOT: DICompositeType
; CHECK-NOT: DILocation(line: 604, column: 1, scope: null)

define void @main() !dbg !34 {
  ret void, !dbg !68
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!31, !32}
!llvm.ident = !{!33}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 3.7 (tags/RELEASE_370/final)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !3, globals: !2)
!1 = !DIFile(filename: "no filename", directory: "")
!2 = !{}
!3 = !{!4, !22}
!4 = !DIDerivedType(tag: DW_TAG_typedef, name: "float3x3", file: !1, line: 361, baseType: !5)
!5 = !DICompositeType(tag: DW_TAG_class_type, name: "matrix<float, 3, 3>", file: !1, line: 246, size: 288, align: 32, elements: !6, templateParams: !17)
!6 = !{!7, !9, !10, !11, !12, !13, !14, !15, !16}
!7 = !DIDerivedType(tag: DW_TAG_member, name: "_11", scope: !5, file: !1, line: 246, baseType: !8, size: 32, align: 32, flags: DIFlagPublic)
!8 = !DIBasicType(name: "float", size: 32, align: 32, encoding: DW_ATE_float)
!9 = !DIDerivedType(tag: DW_TAG_member, name: "_12", scope: !5, file: !1, line: 246, baseType: !8, size: 32, align: 32, offset: 32, flags: DIFlagPublic)
!10 = !DIDerivedType(tag: DW_TAG_member, name: "_13", scope: !5, file: !1, line: 246, baseType: !8, size: 32, align: 32, offset: 64, flags: DIFlagPublic)
!11 = !DIDerivedType(tag: DW_TAG_member, name: "_21", scope: !5, file: !1, line: 246, baseType: !8, size: 32, align: 32, offset: 96, flags: DIFlagPublic)
!12 = !DIDerivedType(tag: DW_TAG_member, name: "_22", scope: !5, file: !1, line: 246, baseType: !8, size: 32, align: 32, offset: 128, flags: DIFlagPublic)
!13 = !DIDerivedType(tag: DW_TAG_member, name: "_23", scope: !5, file: !1, line: 246, baseType: !8, size: 32, align: 32, offset: 160, flags: DIFlagPublic)
!14 = !DIDerivedType(tag: DW_TAG_member, name: "_31", scope: !5, file: !1, line: 246, baseType: !8, size: 32, align: 32, offset: 192, flags: DIFlagPublic)
!15 = !DIDerivedType(tag: DW_TAG_member, name: "_32", scope: !5, file: !1, line: 246, baseType: !8, size: 32, align: 32, offset: 224, flags: DIFlagPublic)
!16 = !DIDerivedType(tag: DW_TAG_member, name: "_33", scope: !5, file: !1, line: 246, baseType: !8, size: 32, align: 32, offset: 256, flags: DIFlagPublic)
!17 = !{!18, !19, !21}
!18 = !DITemplateTypeParameter(name: "element", type: !8)
!19 = !DITemplateValueParameter(name: "row_count", type: !20, value: i32 3)
!20 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!21 = !DITemplateValueParameter(name: "col_count", type: !20, value: i32 3)
!22 = !DIDerivedType(tag: DW_TAG_typedef, name: "float4", file: !1, baseType: !23)
!23 = !DICompositeType(tag: DW_TAG_class_type, name: "vector<float, 4>", file: !1, size: 128, align: 32, elements: !24, templateParams: !29)
!24 = !{!25, !26, !27, !28}
!25 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !23, file: !1, baseType: !8, size: 32, align: 32, flags: DIFlagPublic)
!26 = !DIDerivedType(tag: DW_TAG_member, name: "y", scope: !23, file: !1, baseType: !8, size: 32, align: 32, offset: 32, flags: DIFlagPublic)
!27 = !DIDerivedType(tag: DW_TAG_member, name: "z", scope: !23, file: !1, baseType: !8, size: 32, align: 32, offset: 64, flags: DIFlagPublic)
!28 = !DIDerivedType(tag: DW_TAG_member, name: "w", scope: !23, file: !1, baseType: !8, size: 32, align: 32, offset: 96, flags: DIFlagPublic)
!29 = !{!18, !30}
!30 = !DITemplateValueParameter(name: "element_count", type: !20, value: i32 4)
!31 = !{i32 2, !"Dwarf Version", i32 4}
!32 = !{i32 2, !"Debug Info Version", i32 3}
!33 = !{!"clang version 3.7 (tags/RELEASE_370/final)"}
!34 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 581, type: !35, scopeLine: 582, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!35 = !DISubroutineType(types: !36)
!36 = !{null, !37, !58}
!37 = !DICompositeType(tag: DW_TAG_structure_type, name: "VertexInput", file: !1, line: 254, size: 416, align: 32, elements: !38)
!38 = !{!39, !40, !48, !57}
!39 = !DIDerivedType(tag: DW_TAG_member, name: "Position", scope: !37, file: !1, line: 256, baseType: !22, size: 128, align: 32)
!40 = !DIDerivedType(tag: DW_TAG_member, name: "TexCoord", scope: !37, file: !1, line: 257, baseType: !41, size: 64, align: 32, offset: 128)
!41 = !DIDerivedType(tag: DW_TAG_typedef, name: "float2", file: !1, baseType: !42)
!42 = !DICompositeType(tag: DW_TAG_class_type, name: "vector<float, 2>", file: !1, size: 64, align: 32, elements: !43, templateParams: !46)
!43 = !{!44, !45}
!44 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !42, file: !1, baseType: !8, size: 32, align: 32, flags: DIFlagPublic)
!45 = !DIDerivedType(tag: DW_TAG_member, name: "y", scope: !42, file: !1, baseType: !8, size: 32, align: 32, offset: 32, flags: DIFlagPublic)
!46 = !{!18, !47}
!47 = !DITemplateValueParameter(name: "element_count", type: !20, value: i32 2)
!48 = !DIDerivedType(tag: DW_TAG_member, name: "Normal", scope: !37, file: !1, line: 258, baseType: !49, size: 96, align: 32, offset: 192)
!49 = !DIDerivedType(tag: DW_TAG_typedef, name: "float3", file: !1, baseType: !50)
!50 = !DICompositeType(tag: DW_TAG_class_type, name: "vector<float, 3>", file: !1, size: 96, align: 32, elements: !51, templateParams: !55)
!51 = !{!52, !53, !54}
!52 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !50, file: !1, baseType: !8, size: 32, align: 32, flags: DIFlagPublic)
!53 = !DIDerivedType(tag: DW_TAG_member, name: "y", scope: !50, file: !1, baseType: !8, size: 32, align: 32, offset: 32, flags: DIFlagPublic)
!54 = !DIDerivedType(tag: DW_TAG_member, name: "z", scope: !50, file: !1, baseType: !8, size: 32, align: 32, offset: 64, flags: DIFlagPublic)
!55 = !{!18, !56}
!56 = !DITemplateValueParameter(name: "element_count", type: !20, value: i32 3)
!57 = !DIDerivedType(tag: DW_TAG_member, name: "Tangent", scope: !37, file: !1, line: 259, baseType: !22, size: 128, align: 32, offset: 288)
!58 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !59)
!59 = !DICompositeType(tag: DW_TAG_structure_type, name: "VertexOutput", file: !1, line: 269, size: 672, align: 32, elements: !60)
!60 = !{!61, !62, !63, !64, !65, !66, !67}
!61 = !DIDerivedType(tag: DW_TAG_member, name: "Position", scope: !59, file: !1, line: 271, baseType: !22, size: 128, align: 32)
!62 = !DIDerivedType(tag: DW_TAG_member, name: "TexCoord", scope: !59, file: !1, line: 272, baseType: !41, size: 64, align: 32, offset: 128)
!63 = !DIDerivedType(tag: DW_TAG_member, name: "TangentInView", scope: !59, file: !1, line: 273, baseType: !49, size: 96, align: 32, offset: 192)
!64 = !DIDerivedType(tag: DW_TAG_member, name: "BitangentInView", scope: !59, file: !1, line: 274, baseType: !49, size: 96, align: 32, offset: 288)
!65 = !DIDerivedType(tag: DW_TAG_member, name: "NormalInView", scope: !59, file: !1, line: 275, baseType: !49, size: 96, align: 32, offset: 384)
!66 = !DIDerivedType(tag: DW_TAG_member, name: "EyeDirectionInView", scope: !59, file: !1, line: 276, baseType: !49, size: 96, align: 32, offset: 480)
!67 = !DIDerivedType(tag: DW_TAG_member, name: "PositionInView", scope: !59, file: !1, line: 277, baseType: !49, size: 96, align: 32, offset: 576)
!68 = !DILocation(line: 604, column: 1, scope: !34)
