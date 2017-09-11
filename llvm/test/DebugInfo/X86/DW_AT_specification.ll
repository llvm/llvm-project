; RUN: llc -mtriple=x86_64-apple-darwin %s -o %t -filetype=obj
; RUN: llvm-dwarfdump -debug-info %t | FileCheck %s

; test that the DW_AT_specification is a back edge in the file.

; CHECK: DW_TAG_subprogram
; CHECK-NEXT: DW_AT_linkage_name {{.*}} "_ZN3foo3barEv"
; CHECK: DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_specification {{.*}} "_ZN3foo3barEv"

source_filename = "test/DebugInfo/X86/DW_AT_specification.ll"

@_ZZN3foo3barEvE1x = constant i32 0, align 4, !dbg !0

define void @_ZN3foo3barEv() !dbg !2 {
entry:
  ret void, !dbg !17
}

!llvm.dbg.cu = !{!8}
!llvm.module.flags = !{!16}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "x", scope: !2, file: !3, line: 5, type: !14, isLocal: true, isDefinition: true)
!2 = distinct !DISubprogram(name: "bar", linkageName: "_ZN3foo3barEv", scope: null, file: !3, line: 4, type: !4, isLocal: false, isDefinition: true, scopeLine: 4, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !8, declaration: !11)
!3 = !DIFile(filename: "nsNativeAppSupportBase.ii", directory: "/Users/espindola/mozilla-central/obj-x86_64-apple-darwin11.2.0/toolkit/library")
!4 = !DISubroutineType(types: !5)
!5 = !{null, !6}
!6 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !7, size: 64, align: 64, flags: DIFlagArtificial)
!7 = !DICompositeType(tag: DW_TAG_structure_type, name: "foo", file: !3, line: 1, flags: DIFlagFwdDecl)
!8 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "clang version 3.0 ()", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !9, retainedTypes: !9, globals: !10, imports: !9)
!9 = !{}
!10 = !{!0}
!11 = !DISubprogram(name: "bar", linkageName: "_ZN3foo3barEv", scope: !12, file: !3, line: 2, type: !4, isLocal: false, isDefinition: false, scopeLine: 2, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false)
!12 = !DICompositeType(tag: DW_TAG_class_type, name: "foo", file: !3, line: 1, size: 8, align: 8, elements: !13)
!13 = !{!11}
!14 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !15)
!15 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!16 = !{i32 1, !"Debug Info Version", i32 3}
!17 = !DILocation(line: 6, column: 1, scope: !18)
!18 = distinct !DILexicalBlock(scope: !2, file: !3, line: 4, column: 17)

