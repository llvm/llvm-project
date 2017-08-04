; RUN: opt -module-summary -o %t-dst.bc %s
; RUN: opt -module-summary -o %t-src.bc %p/Inputs/crash_debuginfo.ll
; RUN: llvm-lto -thinlto -o %t-index %t-dst.bc %t-src.bc
; RUN: opt -function-import -inline -summary-file %t-index.thinlto.bc %t-dst.bc -o %t.out
; RUN: llvm-nm %t.out | FileCheck %s

; Verify that we import bar and inline it. It use to crash importing due to ODR type uniquing
; CHECK-NOT: bar
; CHECK: foo
; CHECK-NOT: bar

; ModuleID = 'test/ThinLTO/X86/crash_debuginfo.ll'
source_filename = "test/ThinLTO/X86/crash_debuginfo.ll"
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.7.0"

define void @foo(i32 %arg) {
  call void @bar(i32 %arg), !dbg !8
  unreachable
}

declare void @bar(i32)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "Apple LLVM version 8.0.0 (clang-800.0.24.1)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, globals: !2)
!1 = !DIFile(filename: "1.cpp", directory: "/another_dir")
!2 = !{!3}
!3 = !DIGlobalVariableExpression(var: !4)
!4 = !DIGlobalVariable(name: "_", linkageName: "some_global", scope: null, file: !1, line: 20, type: !5, isLocal: true, isDefinition: true)
!5 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "slice_nil", file: !1, line: 13, size: 64, align: 64, elements: !6, identifier: "_ZTSN5boost6python3api9slice_nilE")
!6 = !{}
!7 = !{i32 2, !"Debug Info Version", i32 3}
!8 = !DILocation(line: 728, column: 71, scope: !9, inlinedAt: !16)
!9 = distinct !DISubprogram(name: "baz", linkageName: "baz", scope: !10, file: !1, line: 726, type: !11, isLocal: false, isDefinition: true, scopeLine: 727, flags: DIFlagPrototyped, isOptimized: true, unit: !0, declaration: !12, variables: !13)
!10 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "some_other_class", file: !1, line: 197, size: 192, align: 64, elements: !6, templateParams: !6, identifier: "some_other_class")
!11 = !DISubroutineType(types: !6)
!12 = !DISubprogram(name: "baz", linkageName: "baz", scope: !10, file: !1, line: 726, type: !11, isLocal: false, isDefinition: false, scopeLine: 726, flags: DIFlagPrototyped, isOptimized: true)
!13 = !{!14}
!14 = !DILocalVariable(name: "caster", scope: !9, file: !1, line: 728, type: !15)
!15 = distinct !DICompositeType(tag: DW_TAG_union_type, scope: !9, file: !1, line: 728, size: 64, align: 64, elements: !6, identifier: "someclass")
!16 = distinct !DILocation(line: 87, column: 9, scope: !17)
!17 = distinct !DISubprogram(name: "foo", linkageName: "foo", scope: !10, file: !1, line: 73, type: !11, isLocal: false, isDefinition: true, scopeLine: 74, flags: DIFlagPrototyped, isOptimized: true, unit: !0, declaration: !18, variables: !6)
!18 = !DISubprogram(name: "foo", linkageName: "foo", scope: !10, file: !1, line: 83, type: !11, isLocal: false, isDefinition: false, scopeLine: 83, flags: DIFlagPrototyped, isOptimized: true)
