; RUN: llvm-c-test --test-dibuilder | FileCheck %s

; CHECK: ; ModuleID = 'debuginfo.c'
; CHECK-NEXT: source_filename = "debuginfo.c"

; CHECK:      define i64 @foo(i64, i64, <10 x i64>) !dbg !16 {
; CHECK-NEXT: entry:
; CHECK-NEXT:   call void @llvm.dbg.declare(metadata i64 0, metadata !23, metadata !DIExpression()), !dbg !28
; CHECK-NEXT:   call void @llvm.dbg.declare(metadata i64 0, metadata !24, metadata !DIExpression()), !dbg !28
; CHECK-NEXT:   call void @llvm.dbg.declare(metadata i64 0, metadata !25, metadata !DIExpression()), !dbg !28
; CHECK:      vars:
; CHECK-NEXT:   call void @llvm.dbg.value(metadata i64 0, metadata !26, metadata !DIExpression(DW_OP_constu, 0, DW_OP_stack_value)), !dbg !29
; CHECK-NEXT: }

; CHECK: declare void @llvm.dbg.declare(metadata, metadata, metadata) #0
; CHECK: declare void @llvm.dbg.value(metadata, metadata, metadata) #0

; CHECK: !llvm.dbg.cu = !{!0}
; CHECK: !FooType = !{!12}

; CHECK:      !0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "llvm-c-test", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, globals: !3, imports: !8, splitDebugInlining: false)
; CHECK-NEXT: !1 = !DIFile(filename: "debuginfo.c", directory: ".")
; CHECK-NEXT: !2 = !{}
; CHECK-NEXT: !3 = !{!4}
; CHECK-NEXT: !4 = !DIGlobalVariableExpression(var: !5, expr: !DIExpression(DW_OP_constu, 0, DW_OP_stack_value))
; CHECK-NEXT: !5 = distinct !DIGlobalVariable(name: "global", scope: !6, file: !1, line: 1, type: !7, isLocal: true, isDefinition: true)
; CHECK-NEXT: !6 = !DIModule(scope: null, name: "llvm-c-test", includePath: "/test/include/llvm-c-test.h")
; CHECK-NEXT: !7 = !DIBasicType(name: "Int64", size: 64)
; CHECK-NEXT: !8 = !{!9, !11}
; CHECK-NEXT: !9 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !6, entity: !10, file: !1, line: 42)
; CHECK-NEXT: !10 = !DIModule(scope: null, name: "llvm-c-test-import", includePath: "/test/include/llvm-c-test-import.h")
; CHECK-NEXT: !11 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !6, entity: !9, file: !1, line: 42)
; CHECK-NEXT: !12 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !13, size: 192, dwarfAddressSpace: 0)
; CHECK-NEXT: !13 = !DICompositeType(tag: DW_TAG_structure_type, name: "MyStruct", scope: !14, file: !1, size: 192, elements: !15, runtimeLang: DW_LANG_C89, identifier: "MyStruct")
; CHECK-NEXT: !14 = !DINamespace(name: "NameSpace", scope: !6)
; CHECK-NEXT: !15 = !{!7, !7, !7}
; CHECK-NEXT: !16 = distinct !DISubprogram(name: "foo", linkageName: "foo", scope: !1, file: !1, line: 42, type: !17, isLocal: true, isDefinition: true, scopeLine: 42, isOptimized: false, unit: !0, variables: !22)
; CHECK-NEXT: !17 = !DISubroutineType(types: !18)
; CHECK-NEXT: !18 = !{!7, !7, !19}
; CHECK-NEXT: !19 = !DICompositeType(tag: DW_TAG_array_type, baseType: !7, size: 640, flags: DIFlagVector, elements: !20)
; CHECK-NEXT: !20 = !{!21}
; CHECK-NEXT: !21 = !DISubrange(count: 10)
; CHECK-NEXT: !22 = !{!23, !24, !25, !26}
; CHECK-NEXT: !23 = !DILocalVariable(name: "a", arg: 1, scope: !16, file: !1, line: 42, type: !7)
; CHECK-NEXT: !24 = !DILocalVariable(name: "b", arg: 2, scope: !16, file: !1, line: 42, type: !7)
; CHECK-NEXT: !25 = !DILocalVariable(name: "c", arg: 3, scope: !16, file: !1, line: 42, type: !19)
; CHECK-NEXT: !26 = !DILocalVariable(name: "d", scope: !27, file: !1, line: 43, type: !7)
; CHECK-NEXT: !27 = distinct !DILexicalBlock(scope: !16, file: !1, line: 42)
; CHECK-NEXT: !28 = !DILocation(line: 42, scope: !16)
; CHECK-NEXT: !29 = !DILocation(line: 43, scope: !16)
