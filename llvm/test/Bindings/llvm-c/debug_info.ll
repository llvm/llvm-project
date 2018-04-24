; RUN: llvm-c-test --test-dibuilder | FileCheck %s

; CHECK: ; ModuleID = 'debuginfo.c'
; CHECK-NEXT: source_filename = "debuginfo.c"

; CHECK:      define i64 @foo(i64, i64, <10 x i64>) !dbg !12 {
; CHECK-NEXT: entry:
; CHECK-NEXT:   call void @llvm.dbg.declare(metadata i64 0, metadata !19, metadata !DIExpression()), !dbg !24
; CHECK-NEXT:   call void @llvm.dbg.declare(metadata i64 0, metadata !20, metadata !DIExpression()), !dbg !24
; CHECK-NEXT:   call void @llvm.dbg.declare(metadata i64 0, metadata !21, metadata !DIExpression()), !dbg !24
; CHECK:      vars:
; CHECK-NEXT:   call void @llvm.dbg.value(metadata i64 0, metadata !22, metadata !DIExpression(DW_OP_constu, 0, DW_OP_stack_value)), !dbg !25
; CHECK-NEXT: }

; CHECK: declare void @llvm.dbg.declare(metadata, metadata, metadata) #0
; CHECK: declare void @llvm.dbg.value(metadata, metadata, metadata) #0

; CHECK: !llvm.dbg.cu = !{!0}
; CHECK: !FooType = !{!8}

; CHECK:      !0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "llvm-c-test", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, globals: !3, splitDebugInlining: false)
; CHECK-NEXT: !1 = !DIFile(filename: "debuginfo.c", directory: ".")
; CHECK-NEXT: !2 = !{}
; CHECK-NEXT: !3 = !{!4}
; CHECK-NEXT: !4 = !DIGlobalVariableExpression(var: !5, expr: !DIExpression(DW_OP_constu, 0, DW_OP_stack_value))
; CHECK-NEXT: !5 = distinct !DIGlobalVariable(name: "global", scope: !6, file: !1, line: 1, type: !7, isLocal: true, isDefinition: true)
; CHECK-NEXT: !6 = !DIModule(scope: null, name: "llvm-c-test", includePath: "/test/include/llvm-c-test.h")
; CHECK-NEXT: !7 = !DIBasicType(name: "Int64", size: 64)
; CHECK-NEXT: !8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !9, size: 192, dwarfAddressSpace: 0)
; CHECK-NEXT: !9 = !DICompositeType(tag: DW_TAG_structure_type, name: "MyStruct", scope: !10, file: !1, size: 192, elements: !11, runtimeLang: DW_LANG_C89, identifier: "MyStruct")
; CHECK-NEXT: !10 = !DINamespace(name: "NameSpace", scope: !6)
; CHECK-NEXT: !11 = !{!7, !7, !7}
; CHECK-NEXT: !12 = distinct !DISubprogram(name: "foo", linkageName: "foo", scope: !1, file: !1, line: 42, type: !13, isLocal: true, isDefinition: true, scopeLine: 42, isOptimized: false, unit: !0, variables: !18)
; CHECK-NEXT: !13 = !DISubroutineType(types: !14)
; CHECK-NEXT: !14 = !{!7, !7, !15}
; CHECK-NEXT: !15 = !DICompositeType(tag: DW_TAG_array_type, baseType: !7, size: 640, flags: DIFlagVector, elements: !16)
; CHECK-NEXT: !16 = !{!17}
; CHECK-NEXT: !17 = !DISubrange(count: 10)
; CHECK-NEXT: !18 = !{!19, !20, !21, !22}
; CHECK-NEXT: !19 = !DILocalVariable(name: "a", arg: 1, scope: !12, file: !1, line: 42, type: !7)
; CHECK-NEXT: !20 = !DILocalVariable(name: "b", arg: 2, scope: !12, file: !1, line: 42, type: !7)
; CHECK-NEXT: !21 = !DILocalVariable(name: "c", arg: 3, scope: !12, file: !1, line: 42, type: !15)
; CHECK-NEXT: !22 = !DILocalVariable(name: "d", scope: !23, file: !1, line: 43, type: !7)
; CHECK-NEXT: !23 = distinct !DILexicalBlock(scope: !12, file: !1, line: 42)
; CHECK-NEXT: !24 = !DILocation(line: 42, scope: !12)
; CHECK-NEXT: !25 = !DILocation(line: 43, scope: !12)
