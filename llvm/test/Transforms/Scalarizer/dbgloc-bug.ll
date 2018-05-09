; RUN: opt -S -march=x86 -scalarizer %s | FileCheck %s

; Reproducer for pr27938
; https://llvm.org/bugs/show_bug.cgi?id=27938

define i16 @f1() !dbg !5 {
  ret i16 undef, !dbg !9
}

define void @f2() !dbg !10 {
bb1:
  %_tmp7 = tail call i16 @f1(), !dbg !13
; CHECK: call i16 @f1(), !dbg !13
  %broadcast.splatinsert5 = insertelement <4 x i16> undef, i16 %_tmp7, i32 0
  %broadcast.splat6 = shufflevector <4 x i16> %broadcast.splatinsert5, <4 x i16> undef, <4 x i32> zeroinitializer
  br label %vector.body

vector.body:
  br i1 undef, label %middle.block, label %vector.body

middle.block:
  ret void, !dbg !15
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, enums: !2, retainedTypes: !2)
!1 = !DIFile(filename: "dbgloc-bug.c", directory: ".")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "f1", scope: !1, file: !1, line: 9, type: !6, isLocal: false, isDefinition: true, scopeLine: 10, isOptimized: true, unit: !0, retainedNodes: !2)
!6 = !DISubroutineType(types: !7)
!7 = !{!8}
!8 = !DIBasicType(name: "short", size: 16, align: 16, encoding: DW_ATE_signed)
!9 = !DILocation(line: 11, column: 5, scope: !5)
!10 = distinct !DISubprogram(name: "f2", scope: !1, file: !1, line: 14, type: !11, isLocal: false, isDefinition: true, scopeLine: 15, isOptimized: true, unit: !0, retainedNodes: !2)
!11 = !DISubroutineType(types: !12)
!12 = !{null}
!13 = !DILocation(line: 24, column: 9, scope: !14)
!14 = !DILexicalBlock(scope: !10, file: !1, line: 17, column: 5)
!15 = !DILocation(line: 28, column: 1, scope: !10)
