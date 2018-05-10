; RUN: llvm-c-test --test-dibuilder | FileCheck %s

; CHECK: ; ModuleID = 'debuginfo.c'
; CHECK-NEXT: source_filename = "debuginfo.c"

; CHECK:      define i64 @foo(i64, i64, <10 x i64>) !dbg !17 {
; CHECK-NEXT: entry:
; CHECK-NEXT:   call void @llvm.dbg.declare(metadata i64 0, metadata !24, metadata !DIExpression()), !dbg !29
; CHECK-NEXT:   call void @llvm.dbg.declare(metadata i64 0, metadata !25, metadata !DIExpression()), !dbg !29
; CHECK-NEXT:   call void @llvm.dbg.declare(metadata i64 0, metadata !26, metadata !DIExpression()), !dbg !29
; CHECK:      vars:
; CHECK-NEXT:   call void @llvm.dbg.value(metadata i64 0, metadata !27, metadata !DIExpression(DW_OP_constu, 0, DW_OP_stack_value)), !dbg !30
; CHECK-NEXT: }

; CHECK: declare void @llvm.dbg.declare(metadata, metadata, metadata) #0
; CHECK: declare void @llvm.dbg.value(metadata, metadata, metadata) #0

; CHECK: !llvm.dbg.cu = !{!0}
; CHECK: !FooType = !{!13}

; CHECK:      !0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "llvm-c-test", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, globals: !3, imports: !9, splitDebugInlining: false)
; CHECK-NEXT: !1 = !DIFile(filename: "debuginfo.c", directory: ".")
; CHECK-NEXT: !2 = !{}
; CHECK-NEXT: !3 = !{!4}
; CHECK-NEXT: !4 = !DIGlobalVariableExpression(var: !5, expr: !DIExpression(DW_OP_constu, 0, DW_OP_stack_value))
; CHECK-NEXT: !5 = distinct !DIGlobalVariable(name: "global", scope: !6, file: !1, line: 1, type: !7, isLocal: true, isDefinition: true)
; CHECK-NEXT: !6 = !DIModule(scope: null, name: "llvm-c-test", includePath: "/test/include/llvm-c-test.h")
; CHECK-NEXT: !7 = !DIDerivedType(tag: DW_TAG_typedef, name: "int64_t", scope: !1, file: !1, line: 42, baseType: !8)
; CHECK-NEXT: !8 = !DIBasicType(name: "Int64", size: 64)
; CHECK-NEXT: !9 = !{!10, !12}
; CHECK-NEXT: !10 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !6, entity: !11, file: !1, line: 42)
; CHECK-NEXT: !11 = !DIModule(scope: null, name: "llvm-c-test-import", includePath: "/test/include/llvm-c-test-import.h")
; CHECK-NEXT: !12 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !6, entity: !10, file: !1, line: 42)
; CHECK-NEXT: !13 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !14, size: 192, dwarfAddressSpace: 0)
; CHECK-NEXT: !14 = !DICompositeType(tag: DW_TAG_structure_type, name: "MyStruct", scope: !15, file: !1, size: 192, elements: !16, runtimeLang: DW_LANG_C89, identifier: "MyStruct")
; CHECK-NEXT: !15 = !DINamespace(name: "NameSpace", scope: !6)
; CHECK-NEXT: !16 = !{!8, !8, !8}
; CHECK-NEXT: !17 = distinct !DISubprogram(name: "foo", linkageName: "foo", scope: !1, file: !1, line: 42, type: !18, isLocal: true, isDefinition: true, scopeLine: 42, isOptimized: false, unit: !0, retainedNodes: !23)
; CHECK-NEXT: !18 = !DISubroutineType(types: !19)
; CHECK-NEXT: !19 = !{!8, !8, !20}
; CHECK-NEXT: !20 = !DICompositeType(tag: DW_TAG_array_type, baseType: !8, size: 640, flags: DIFlagVector, elements: !21)
; CHECK-NEXT: !21 = !{!22}
; CHECK-NEXT: !22 = !DISubrange(count: 10)
; CHECK-NEXT: !23 = !{!24, !25, !26, !27}
; CHECK-NEXT: !24 = !DILocalVariable(name: "a", arg: 1, scope: !17, file: !1, line: 42, type: !8)
; CHECK-NEXT: !25 = !DILocalVariable(name: "b", arg: 2, scope: !17, file: !1, line: 42, type: !8)
; CHECK-NEXT: !26 = !DILocalVariable(name: "c", arg: 3, scope: !17, file: !1, line: 42, type: !20)
; CHECK-NEXT: !27 = !DILocalVariable(name: "d", scope: !28, file: !1, line: 43, type: !8)
; CHECK-NEXT: !28 = distinct !DILexicalBlock(scope: !17, file: !1, line: 42)
; CHECK-NEXT: !29 = !DILocation(line: 42, scope: !17)
; CHECK-NEXT: !30 = !DILocation(line: 43, scope: !17)
