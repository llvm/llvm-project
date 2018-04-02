; RUN: llvm-c-test --test-dibuilder | FileCheck %s

; CHECK: ; ModuleID = 'debuginfo.c'
; CHECK-NEXT: source_filename = "debuginfo.c"

; CHECK: !llvm.dbg.cu = !{!0}
; CHECK-NEXT: !FooType = !{!2}

; CHECK: !0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "llvm-c-test", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false)
; CHECK-NEXT: !1 = !DIFile(filename: "debuginfo.c", directory: ".")
; CHECK-NEXT: !2 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !3, size: 192, dwarfAddressSpace: 0)
; CHECK-NEXT: !3 = !DICompositeType(tag: DW_TAG_structure_type, name: "MyStruct", file: !1, size: 192, elements: !4, runtimeLang: DW_LANG_C89, identifier: "MyStruct")
; CHECK-NEXT: !4 = !{!5, !5, !5}
; CHECK-NEXT: !5 = !DIBasicType(name: "Int64", size: 64)