; ModuleID = 'test/ThinLTO/X86/Inputs/crash_debuginfo.ll'
source_filename = "src.bc"
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.7.0"

define void @bar(i32 %arg) {
  %tmp = add i32 %arg, 0, !dbg !8
  unreachable
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "Apple LLVM version 8.0.0 (clang-800.0.25.1)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !2, globals: !3, imports: !2)
!1 = !DIFile(filename: "2.cpp", directory: "some_dir")
!2 = !{}
!3 = !{!4}
!4 = !DIGlobalVariableExpression(var: !5)
!5 = !DIGlobalVariable(name: "a_global", linkageName: "a_global", scope: null, line: 52, type: !6, isLocal: true, isDefinition: true)
!6 = !DISubroutineType(types: !2)
!7 = !{i32 2, !"Debug Info Version", i32 3}
!8 = !DILocation(line: 728, column: 71, scope: !9, inlinedAt: !15)
!9 = distinct !DISubprogram(name: "baz", linkageName: "baz", scope: !10, file: !1, line: 726, type: !6, isLocal: false, isDefinition: true, scopeLine: 727, flags: DIFlagPrototyped, isOptimized: true, unit: !0, declaration: !11, variables: !12)
!10 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "some_other_class", scope: !1, file: !1, line: 197, size: 192, align: 64, elements: !2, templateParams: !2, identifier: "some_other_class")
!11 = !DISubprogram(name: "baz", linkageName: "baz", scope: !10, file: !1, line: 726, type: !6, isLocal: false, isDefinition: false, scopeLine: 726, flags: DIFlagPrototyped, isOptimized: true)
!12 = !{!13}
!13 = !DILocalVariable(name: "caster", scope: !9, file: !1, line: 728, type: !14)
!14 = distinct !DICompositeType(tag: DW_TAG_union_type, scope: !9, file: !1, line: 728, size: 64, align: 64, elements: !2, identifier: "someclass")
!15 = distinct !DILocation(line: 795, column: 16, scope: !16)
!16 = distinct !DILexicalBlock(scope: !17, file: !1, line: 794, column: 7)
!17 = distinct !DISubprogram(name: "operator()", linkageName: "some_special_function", scope: null, file: !1, line: 783, type: !6, isLocal: true, isDefinition: true, scopeLine: 784, flags: DIFlagPrototyped, isOptimized: true, unit: !0, declaration: !18, variables: !2)
!18 = !DISubprogram(name: "operator()", linkageName: "some_special_function", scope: null, file: !1, line: 783, type: !6, isLocal: false, isDefinition: false, scopeLine: 783, flags: DIFlagPrototyped, isOptimized: true)

