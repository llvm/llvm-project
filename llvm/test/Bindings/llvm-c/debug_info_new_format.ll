; RUN: llvm-c-test --test-dibuilder-debuginfo-format | FileCheck %s
;; Duplicate of debug_info.ll using debug records instead of intrinsics.

; CHECK: ; ModuleID = 'debuginfo.c'
; CHECK-NEXT: source_filename = "debuginfo.c"

; CHECK:      define i64 @foo(i64 %0, i64 %1, <10 x i64> %2) !dbg !31 {
; CHECK-NEXT: entry:
; CHECK-NEXT:     #dbg_declare(i64 0, !38, !DIExpression(), !44)
; CHECK-NEXT:     #dbg_declare(i64 0, !39, !DIExpression(), !44)
; CHECK-NEXT:     #dbg_declare(i64 0, !40, !DIExpression(), !44)
; CHECK-NEXT:   br label %vars
; CHECK:      vars:
; CHECK-NEXT:   %p1 = phi i64 [ 0, %entry ]
; CHECK-NEXT:   %p2 = phi i64 [ 0, %entry ]
; CHECK-NEXT:     #dbg_value(i64 0, !41, !DIExpression(DW_OP_constu, 0, DW_OP_stack_value), !45)
; CHECK-NEXT:     #dbg_value(i64 1, !43, !DIExpression(DW_OP_constu, 1, DW_OP_stack_value), !45)
; CHECK-NEXT:   %a = add i64 %p1, %p2
; CHECK-NEXT:   ret i64 0
; CHECK-NEXT: }

; CHECK:      !llvm.dbg.cu = !{!0}
; CHECK-NEXT: !FooType = !{!28}
; CHECK-NEXT: !EnumTest = !{!3}

; CHECK:      !0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "llvm-c-test", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, globals: !11, imports: !19, macros: !23, splitDebugInlining: false, sysroot: "/")
; CHECK-NEXT: !1 = !DIFile(filename: "debuginfo.c", directory: ".")
; CHECK-NEXT: !2 = !{!3}
; CHECK-NEXT: !3 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "EnumTest", scope: !4, file: !1, baseType: !6, size: 64, elements: !7)
; CHECK-NEXT: !4 = !DINamespace(name: "NameSpace", scope: !5)
; CHECK-NEXT: !5 = !DIModule(scope: null, name: "llvm-c-test", includePath: "/test/include/llvm-c-test.h")
; CHECK-NEXT: !6 = !DIBasicType(name: "Int64", size: 64)
; CHECK-NEXT: !7 = !{!8, !9, !10}
; CHECK-NEXT: !8 = !DIEnumerator(name: "Test_A", value: 0, isUnsigned: true)
; CHECK-NEXT: !9 = !DIEnumerator(name: "Test_B", value: 1, isUnsigned: true)
; CHECK-NEXT: !10 = !DIEnumerator(name: "Test_B", value: 2, isUnsigned: true)
; CHECK-NEXT: !11 = !{!12, !16}
; CHECK-NEXT: !12 = !DIGlobalVariableExpression(var: !13, expr: !DIExpression(DW_OP_constu, 0, DW_OP_stack_value))
; CHECK-NEXT: !13 = distinct !DIGlobalVariable(name: "globalClass", scope: !5, file: !1, line: 1, type: !14, isLocal: true, isDefinition: true)
; CHECK-NEXT: !14 = !DICompositeType(tag: DW_TAG_structure_type, name: "TestClass", scope: !1, file: !1, line: 42, size: 64, flags: DIFlagObjcClassComplete, elements: !15)
; CHECK-NEXT: !15 = !{}
; CHECK-NEXT: !16 = !DIGlobalVariableExpression(var: !17, expr: !DIExpression(DW_OP_constu, 0, DW_OP_stack_value))
; CHECK-NEXT: !17 = distinct !DIGlobalVariable(name: "global", scope: !5, file: !1, line: 1, type: !18, isLocal: true, isDefinition: true)
; CHECK-NEXT: !18 = !DIDerivedType(tag: DW_TAG_typedef, name: "int64_t", scope: !1, file: !1, line: 42, baseType: !6)
; CHECK-NEXT: !19 = !{!20, !22}
; CHECK-NEXT: !20 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !5, entity: !21, file: !1, line: 42)
; CHECK-NEXT: !21 = !DIModule(scope: null, name: "llvm-c-test-import", includePath: "/test/include/llvm-c-test-import.h")
; CHECK-NEXT: !22 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !5, entity: !20, file: !1, line: 42)
; CHECK-NEXT: !23 = !{!24}
; CHECK-NEXT: !24 = !DIMacroFile(file: !1, nodes: !25)
; CHECK-NEXT: !25 = !{!26, !27}
; CHECK-NEXT: !26 = !DIMacro(type: DW_MACINFO_define, name: "SIMPLE_DEFINE")
; CHECK-NEXT: !27 = !DIMacro(type: DW_MACINFO_define, name: "VALUE_DEFINE", value: "1")
; CHECK-NEXT: !28 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !29, size: 192, dwarfAddressSpace: 0)
; CHECK-NEXT: !29 = !DICompositeType(tag: DW_TAG_structure_type, name: "MyStruct", scope: !4, file: !1, size: 192, elements: !30, runtimeLang: DW_LANG_C89, identifier: "MyStruct")
; CHECK-NEXT: !30 = !{!6, !6, !6}
; CHECK-NEXT: !31 = distinct !DISubprogram(name: "foo", linkageName: "foo", scope: !1, file: !1, line: 42, type: !32, scopeLine: 42, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !37)
; CHECK-NEXT: !32 = !DISubroutineType(types: !33)
; CHECK-NEXT: !33 = !{!6, !6, !34}
; CHECK-NEXT: !34 = !DICompositeType(tag: DW_TAG_array_type, baseType: !6, size: 640, flags: DIFlagVector, elements: !35)
; CHECK-NEXT: !35 = !{!36}
; CHECK-NEXT: !36 = !DISubrange(count: 10, lowerBound: 0)
; CHECK-NEXT: !37 = !{!38, !39, !40, !41, !43}
; CHECK-NEXT: !38 = !DILocalVariable(name: "a", arg: 1, scope: !31, file: !1, line: 42, type: !6)
; CHECK-NEXT: !39 = !DILocalVariable(name: "b", arg: 2, scope: !31, file: !1, line: 42, type: !6)
; CHECK-NEXT: !40 = !DILocalVariable(name: "c", arg: 3, scope: !31, file: !1, line: 42, type: !34)
; CHECK-NEXT: !41 = !DILocalVariable(name: "d", scope: !42, file: !1, line: 43, type: !6)
; CHECK-NEXT: !42 = distinct !DILexicalBlock(scope: !31, file: !1, line: 42)
; CHECK-NEXT: !43 = !DILocalVariable(name: "e", scope: !42, file: !1, line: 44, type: !6)
; CHECK-NEXT: !44 = !DILocation(line: 42, scope: !31)
; CHECK-NEXT: !45 = !DILocation(line: 43, scope: !31)
