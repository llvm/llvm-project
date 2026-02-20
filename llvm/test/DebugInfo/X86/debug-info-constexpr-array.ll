; RUN: llc -mtriple=x86_64-linux-gnu -filetype=obj %s -o %t.o
; RUN: llvm-dwarfdump --debug-info %t.o | FileCheck %s
;
; Reduced from clang output for:
;   struct Test {
;     static inline constexpr char STR[] = "Hello";
;     static inline constexpr int NUMS[] = {1, 2, 3};
;     static inline constexpr unsigned char BYTES[] = {0xDE, 0xAD};
;   };
;   void use() { (void)Test::STR; (void)Test::NUMS; (void)Test::BYTES; }

; CHECK:      DW_TAG_structure_type
; CHECK:        DW_AT_name ("Test")
;
; CHECK:      DW_TAG_member
; CHECK:        DW_AT_name ("STR")
; CHECK:        DW_AT_const_value (<0x06> 48 65 6c 6c 6f 00 )
;
; CHECK:      DW_TAG_member
; CHECK:        DW_AT_name ("NUMS")
; CHECK:        DW_AT_const_value (<0x0c> 01 00 00 00 02 00 00 00 03 00 00 00 )
;
; CHECK:      DW_TAG_member
; CHECK:        DW_AT_name ("BYTES")
; CHECK:        DW_AT_const_value (<0x02> de ad )

@_ZN4Test3STRE = linkonce_odr constant [6 x i8] c"Hello\00", align 1, !dbg !0
@_ZN4Test4NUMSE = linkonce_odr constant [3 x i32] [i32 1, i32 2, i32 3], align 4, !dbg !5
@_ZN4Test5BYTESE = linkonce_odr constant [2 x i8] c"\DE\AD", align 1, !dbg !28

define dso_local void @_Z3usev() !dbg !33 {
  ret void, !dbg !36
}

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!30, !31}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "STR", linkageName: "_ZN4Test3STRE", scope: !2, file: !7, line: 5, type: !17, isLocal: false, isDefinition: true, declaration: !16)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "test.cpp", directory: "")
!4 = !{!0, !5, !28}
!5 = !DIGlobalVariableExpression(var: !6, expr: !DIExpression())
!6 = distinct !DIGlobalVariable(name: "NUMS", linkageName: "_ZN4Test4NUMSE", scope: !2, file: !7, line: 6, type: !8, isLocal: false, isDefinition: true, declaration: !13)
!7 = !DIFile(filename: "test.cpp", directory: "")
!8 = !DICompositeType(tag: DW_TAG_array_type, baseType: !9, size: 96, elements: !11)
!9 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !10)
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !{!12}
!12 = !DISubrange(count: 3)
!13 = !DIDerivedType(tag: DW_TAG_member, name: "NUMS", scope: !14, file: !7, line: 6, baseType: !8, flags: DIFlagStaticMember, extraData: [3 x i32] [i32 1, i32 2, i32 3])
!14 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Test", file: !7, line: 4, size: 8, flags: DIFlagTypePassByValue, elements: !15, identifier: "_ZTS4Test")
!15 = !{!16, !13, !22}
!16 = !DIDerivedType(tag: DW_TAG_member, name: "STR", scope: !14, file: !7, line: 5, baseType: !17, flags: DIFlagStaticMember, extraData: [6 x i8] c"Hello\00")
!17 = !DICompositeType(tag: DW_TAG_array_type, baseType: !18, size: 48, elements: !20)
!18 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !19)
!19 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!20 = !{!21}
!21 = !DISubrange(count: 6)
!22 = !DIDerivedType(tag: DW_TAG_member, name: "BYTES", scope: !14, file: !7, line: 7, baseType: !23, flags: DIFlagStaticMember, extraData: [2 x i8] c"\DE\AD")
!23 = !DICompositeType(tag: DW_TAG_array_type, baseType: !24, size: 16, elements: !26)
!24 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !25)
!25 = !DIBasicType(name: "unsigned char", size: 8, encoding: DW_ATE_unsigned_char)
!26 = !{!27}
!27 = !DISubrange(count: 2)
!28 = !DIGlobalVariableExpression(var: !29, expr: !DIExpression())
!29 = distinct !DIGlobalVariable(name: "BYTES", linkageName: "_ZN4Test5BYTESE", scope: !2, file: !7, line: 7, type: !23, isLocal: false, isDefinition: true, declaration: !22)
!30 = !{i32 2, !"Debug Info Version", i32 3}
!31 = !{i32 1, !"wchar_size", i32 4}
!33 = distinct !DISubprogram(name: "use", linkageName: "_Z3usev", scope: !7, file: !7, line: 10, type: !34, scopeLine: 10, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2)
!34 = !DISubroutineType(types: !35)
!35 = !{null}
!36 = !DILocation(line: 14, column: 1, scope: !33)