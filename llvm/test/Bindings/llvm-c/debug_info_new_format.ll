; RUN: llvm-c-test --test-dibuilder | FileCheck %s
;; Duplicate of debug_info.ll using debug records instead of intrinsics.

; CHECK: ; ModuleID = 'debuginfo.c'
; CHECK-NEXT: source_filename = "debuginfo.c"

; CHECK:      define i64 @foo(i64 %0, i64 %1, <10 x i64> %2) !dbg !36 {
; CHECK-NEXT: entry:
; CHECK-NEXT:     #dbg_declare(i64 0, !43, !DIExpression(), !50)
; CHECK-NEXT:     #dbg_declare(i64 0, !44, !DIExpression(), !50)
; CHECK-NEXT:     #dbg_declare(i64 0, !45, !DIExpression(), !50)
; CHECK-NEXT:     #dbg_label(!51, !50)
; CHECK-NEXT:   br label %vars
; CHECK-NEXT:     #dbg_label(!52, !50)
; CHECK-NEXT:   br label %vars
; CHECK:      vars:
; CHECK-NEXT:   %p1 = phi i64 [ 0, %entry ]
; CHECK-NEXT:   %p2 = phi i64 [ 0, %entry ]
; CHECK-NEXT:     #dbg_value(i64 0, !46, !DIExpression(DW_OP_constu, 0, DW_OP_stack_value), !53)
; CHECK-NEXT:     #dbg_value(i64 1, !48, !DIExpression(DW_OP_constu, 1, DW_OP_stack_value), !53)
; CHECK-NEXT:   %a = add i64 %p1, %p2
; CHECK-NEXT:   ret i64 0
; CHECK-NEXT: }

; CHECK:      !llvm.dbg.cu = !{!0}
; CHECK-NEXT: !FooType = !{!33}
; CHECK-NEXT: !EnumTest = !{!3}
; CHECK-NEXT: !LargeEnumTest = !{!11}

; CHECK:      !0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "llvm-c-test", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, globals: !16, imports: !24, macros: !28, splitDebugInlining: false, sysroot: "/")
; CHECK-NEXT: !1 = !DIFile(filename: "debuginfo.c", directory: ".")
; CHECK-NEXT: !2 = !{!3, !11}
; CHECK-NEXT: !3 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "EnumTest", scope: !4, file: !1, baseType: !6, size: 64, elements: !7)
; CHECK-NEXT: !4 = !DINamespace(name: "NameSpace", scope: !5)
; CHECK-NEXT: !5 = !DIModule(scope: null, name: "llvm-c-test", includePath: "/test/include/llvm-c-test.h")
; CHECK-NEXT: !6 = !DIBasicType(name: "Int64", size: 64)
; CHECK-NEXT: !7 = !{!8, !9, !10}
; CHECK-NEXT: !8 = !DIEnumerator(name: "Test_A", value: 0, isUnsigned: true)
; CHECK-NEXT: !9 = !DIEnumerator(name: "Test_B", value: 1, isUnsigned: true)
; CHECK-NEXT: !10 = !DIEnumerator(name: "Test_B", value: 2, isUnsigned: true)
; CHECK-NEXT: !11 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "LargeEnumTest", scope: !4, file: !1, baseType: !12, size: 128, elements: !13)
; CHECK-NEXT: !12 = !DIBasicType(name: "UInt128", size: 128)
; CHECK-NEXT: !13 = !{!14, !15}
; CHECK-NEXT: !14 = !DIEnumerator(name: "Test_D", value: 100000000000000000000000000000000000000)
; CHECK-NEXT: !15 = !DIEnumerator(name: "Test_E", value: -1)
; CHECK-NEXT: !16 = !{!17, !21}
; CHECK-NEXT: !17 = !DIGlobalVariableExpression(var: !18, expr: !DIExpression(DW_OP_constu, 0, DW_OP_stack_value))
; CHECK-NEXT: !18 = distinct !DIGlobalVariable(name: "globalClass", scope: !5, file: !1, line: 1, type: !19, isLocal: true, isDefinition: true)
; CHECK-NEXT: !19 = !DICompositeType(tag: DW_TAG_structure_type, name: "TestClass", scope: !1, file: !1, line: 42, size: 64, flags: DIFlagObjcClassComplete, elements: !20)
; CHECK-NEXT: !20 = !{}
; CHECK-NEXT: !21 = !DIGlobalVariableExpression(var: !22, expr: !DIExpression(DW_OP_constu, 0, DW_OP_stack_value))
; CHECK-NEXT: !22 = distinct !DIGlobalVariable(name: "global", scope: !5, file: !1, line: 1, type: !23, isLocal: true, isDefinition: true)
; CHECK-NEXT: !23 = !DIDerivedType(tag: DW_TAG_typedef, name: "int64_t", scope: !1, file: !1, line: 42, baseType: !6)
; CHECK-NEXT: !24 = !{!25, !27}
; CHECK-NEXT: !25 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !5, entity: !26, file: !1, line: 42)
; CHECK-NEXT: !26 = !DIModule(scope: null, name: "llvm-c-test-import", includePath: "/test/include/llvm-c-test-import.h")
; CHECK-NEXT: !27 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !5, entity: !25, file: !1, line: 42)
; CHECK-NEXT: !28 = !{!29}
; CHECK-NEXT: !29 = !DIMacroFile(file: !1, nodes: !30)
; CHECK-NEXT: !30 = !{!31, !32}
; CHECK-NEXT: !31 = !DIMacro(type: DW_MACINFO_define, name: "SIMPLE_DEFINE")
; CHECK-NEXT: !32 = !DIMacro(type: DW_MACINFO_define, name: "VALUE_DEFINE", value: "1")
; CHECK-NEXT: !33 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !34, size: 192, dwarfAddressSpace: 0)
; CHECK-NEXT: !34 = !DICompositeType(tag: DW_TAG_structure_type, name: "MyStruct", scope: !4, file: !1, size: 192, elements: !35, runtimeLang: DW_LANG_C89, identifier: "MyStruct")
; CHECK-NEXT: !35 = !{!6, !6, !6}
; CHECK-NEXT: !36 = distinct !DISubprogram(name: "foo", linkageName: "foo", scope: !1, file: !1, line: 42, type: !37, scopeLine: 42, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !42)
; CHECK-NEXT: !37 = !DISubroutineType(types: !38)
; CHECK-NEXT: !38 = !{!6, !6, !39}
; CHECK-NEXT: !39 = !DICompositeType(tag: DW_TAG_array_type, baseType: !6, size: 640, flags: DIFlagVector, elements: !40)
; CHECK-NEXT: !40 = !{!41}
; CHECK-NEXT: !41 = !DISubrange(count: 10, lowerBound: 0)
; CHECK-NEXT: !42 = !{!43, !44, !45, !46, !48, !49}
; CHECK-NEXT: !43 = !DILocalVariable(name: "a", arg: 1, scope: !36, file: !1, line: 42, type: !6)
; CHECK-NEXT: !44 = !DILocalVariable(name: "b", arg: 2, scope: !36, file: !1, line: 42, type: !6)
; CHECK-NEXT: !45 = !DILocalVariable(name: "c", arg: 3, scope: !36, file: !1, line: 42, type: !39)
; CHECK-NEXT: !46 = !DILocalVariable(name: "d", scope: !47, file: !1, line: 43, type: !6)
; CHECK-NEXT: !47 = distinct !DILexicalBlock(scope: !36, file: !1, line: 42)
; CHECK-NEXT: !48 = !DILocalVariable(name: "e", scope: !47, file: !1, line: 44, type: !6)
; CHECK-NEXT: !49 = !DILabel(scope: !36, name: "label3", file: !1, line: 42)
; CHECK-NEXT: !50 = !DILocation(line: 42, scope: !36)
; CHECK-NEXT: !51 = !DILabel(scope: !36, name: "label1", file: !1, line: 42)
; CHECK-NEXT: !52 = !DILabel(scope: !36, name: "label2", file: !1, line: 42)
; CHECK-NEXT: !53 = !DILocation(line: 43, scope: !36)
