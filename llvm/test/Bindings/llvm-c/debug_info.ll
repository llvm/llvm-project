; RUN: llvm-c-test --test-dibuilder | FileCheck %s

; CHECK: ; ModuleID = 'debuginfo.c'
; CHECK-NEXT: source_filename = "debuginfo.c"

; CHECK: declare !dbg !7 i64 @foo(i64, i64)

; CHECK: declare !dbg !10 i64 @foo_inner_scope(i64, i64)

; CHECK: !llvm.dbg.cu = !{!0}
; CHECK-NEXT: !FooType = !{!3}

; CHECK: !0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "llvm-c-test", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false)
; CHECK-NEXT: !1 = !DIFile(filename: "debuginfo.c", directory: ".")
; CHECK-NEXT: !2 = !{}
; CHECK-NEXT: !3 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !4, size: 192, dwarfAddressSpace: 0)
; CHECK-NEXT: !4 = !DICompositeType(tag: DW_TAG_structure_type, name: "MyStruct", file: !1, size: 192, elements: !5, runtimeLang: DW_LANG_C89, identifier: "MyStruct")
; CHECK-NEXT: !5 = !{!6, !6, !6}
; CHECK-NEXT: !6 = !DIBasicType(name: "Int64", size: 64)
; CHECK-NEXT: !7 = distinct !DISubprogram(name: "foo", linkageName: "foo", scope: !1, file: !1, line: 42, type: !8, isLocal: true, isDefinition: true, scopeLine: 42, isOptimized: false, unit: !0, variables: !2)
; CHECK-NEXT: !8 = !DISubroutineType(types: !9)
; CHECK-NEXT: !9 = !{!6, !6}
; CHECK-NEXT: !10 = distinct !DISubprogram(name: "foo_inner_scope", linkageName: "foo_inner_scope", scope: !11, file: !1, line: 42, type: !8, isLocal: true, isDefinition: true, scopeLine: 42, isOptimized: false, unit: !0, variables: !2)
; CHECK-NEXT: !11 = distinct !DILexicalBlock(scope: !7, file: !1, line: 42)
