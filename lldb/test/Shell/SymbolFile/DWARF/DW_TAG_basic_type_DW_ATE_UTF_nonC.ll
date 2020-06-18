;
; This test verifies that DWARF DIE of type DW_TAG_basic_type with DW_ATE_UTF
; are matched based on their bit size (8, 16, 32) in addition to their name.
;
; This is used by languages which don't use the C(++) naming of
; `char{8,16,32}_t`, e.g. the D programming language uses `char`, `wchar`, `dchar`.
;
; The D code used to generate this IR is:
; ```
; // Compiled with `ldc2 --mtriple=x86_64-pc-linux -betterC -g -c --output-ll utftypes.d`
; __gshared string utf8 = "Hello";
; __gshared wstring utf16 = "Dlang"w;
; __gshared dstring utf32 = "World"d;
; ```
;
; Note: lldb will print types differently before and after 'run'.
;
; RUN: %clang --target=x86_64-pc-linux -c -g -o %t %s
; RUN: %lldb %t -o 'type lookup string' -o 'type lookup wstring' \
; RUN:   -o 'type lookup dstring' -o exit | FileCheck %s
;
; CHECK: struct string {
; CHECK:     unsigned long length;
; CHECK:     char8_t *ptr;
; CHECK: }
; CHECK: struct wstring {
; CHECK:     unsigned long length;
; CHECK:     char16_t *ptr;
; CHECK: }
; CHECK: struct dstring {
; CHECK:     unsigned long length;
; CHECK:     char32_t *ptr;
; CHECK: }

$_D8utftypes4utf8Aya = comdat any
$_D8utftypes5utf16Ayu = comdat any
$_D8utftypes5utf32Ayw = comdat any

@_D8utftypes4utf8Aya = global { i64, i8* } { i64 5, i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str, i32 0, i32 0) }, comdat, align 8, !dbg !0 ; [#uses = 0]
@.str = private unnamed_addr constant [6 x i8] c"Hello\00" ; [#uses = 1]
@_D8utftypes5utf16Ayu = global { i64, i16* } { i64 5, i16* getelementptr inbounds ([6 x i16], [6 x i16]* @.str.1, i32 0, i32 0) }, comdat, align 8, !dbg !11 ; [#uses = 0]
@.str.1 = private unnamed_addr constant [6 x i16] [i16 68, i16 108, i16 97, i16 110, i16 103, i16 0] ; [#uses = 1]
@_D8utftypes5utf32Ayw = global { i64, i32* } { i64 5, i32* getelementptr inbounds ([6 x i32], [6 x i32]* @.str.2, i32 0, i32 0) }, comdat, align 8, !dbg !18 ; [#uses = 0]
@.str.2 = private unnamed_addr constant [6 x i32] [i32 87, i32 111, i32 114, i32 108, i32 100, i32 0] ; [#uses = 1]

!llvm.module.flags = !{!25}
!llvm.dbg.cu = !{!26}
!llvm.ident = !{!32}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "utf8", linkageName: "_D8utftypes4utf8Aya", scope: !2, file: !3, line: 1, type: !4, isLocal: false, isDefinition: true)
!2 = !DIModule(scope: null, name: "utftypes")
!3 = !DIFile(filename: "utftypes.d", directory: "/tmp")
!4 = !DICompositeType(tag: DW_TAG_structure_type, name: "string", file: !3, size: 128, align: 64, elements: !5, identifier: "Aya")
!5 = !{!6, !8}
!6 = !DIDerivedType(tag: DW_TAG_member, name: "length", file: !3, baseType: !7, size: 64, align: 64, flags: DIFlagPublic)
!7 = !DIBasicType(name: "ulong", size: 64, encoding: DW_ATE_unsigned)
!8 = !DIDerivedType(tag: DW_TAG_member, name: "ptr", file: !3, baseType: !9, size: 64, align: 64, offset: 64, flags: DIFlagPublic)
!9 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "immutable(char)*", baseType: !10, size: 64, align: 64)
!10 = !DIBasicType(name: "immutable(char)", size: 8, encoding: DW_ATE_UTF)
!11 = !DIGlobalVariableExpression(var: !12, expr: !DIExpression())
!12 = distinct !DIGlobalVariable(name: "utf16", linkageName: "_D8utftypes5utf16Ayu", scope: !2, file: !3, line: 2, type: !13, isLocal: false, isDefinition: true)
!13 = !DICompositeType(tag: DW_TAG_structure_type, name: "wstring", file: !3, size: 128, align: 64, elements: !14, identifier: "Ayu")
!14 = !{!6, !15}
!15 = !DIDerivedType(tag: DW_TAG_member, name: "ptr", file: !3, baseType: !16, size: 64, align: 64, offset: 64, flags: DIFlagPublic)
!16 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "immutable(wchar)*", baseType: !17, size: 64, align: 64)
!17 = !DIBasicType(name: "immutable(wchar)", size: 16, encoding: DW_ATE_UTF)
!18 = !DIGlobalVariableExpression(var: !19, expr: !DIExpression())
!19 = distinct !DIGlobalVariable(name: "utf32", linkageName: "_D8utftypes5utf32Ayw", scope: !2, file: !3, line: 3, type: !20, isLocal: false, isDefinition: true)
!20 = !DICompositeType(tag: DW_TAG_structure_type, name: "dstring", file: !3, size: 128, align: 64, elements: !21, identifier: "Ayw")
!21 = !{!6, !22}
!22 = !DIDerivedType(tag: DW_TAG_member, name: "ptr", file: !3, baseType: !23, size: 64, align: 64, offset: 64, flags: DIFlagPublic)
!23 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "immutable(dchar)*", baseType: !24, size: 64, align: 64)
!24 = !DIBasicType(name: "immutable(dchar)", size: 32, encoding: DW_ATE_UTF)
!25 = !{i32 2, !"Debug Info Version", i32 3}
!26 = distinct !DICompileUnit(language: DW_LANG_D, file: !3, producer: "LDC 1.20.1 (LLVM 9.0.1)", isOptimized: false, runtimeVersion: 1, emissionKind: FullDebug, enums: !27, globals: !28, imports: !29)
!27 = !{}
!28 = !{!0, !11, !18}
!29 = !{!30}
!30 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !2, entity: !31, file: !3)
!31 = !DIModule(scope: null, name: "object")
!32 = !{!"ldc version 1.20.1"}
