; RUN: llc -mtriple=aarch64 -O1 -opt-bisect-limit=2 -o - %s  2> /dev/null | FileCheck %s
; RUN: llc --try-experimental-debuginfo-iterators -mtriple=aarch64 -O1 -opt-bisect-limit=2 -o - %s  2> /dev/null | FileCheck %s

define dso_local i32 @a() #0 !dbg !7 {
entry:
; CHECK:    b       .LBB0_1
; CHECK:  .LBB0_1:
  call void @llvm.dbg.value(metadata i32 0, metadata !12, metadata !DIExpression()), !dbg !13
  br label %for.cond, !dbg !14

; CHECK:    b       .LBB0_1
; CHECK:  .Lfunc_end0:
for.cond:
  br label %for.cond, !dbg !15, !llvm.loop !18
}
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture) #1

declare void @llvm.dbg.value(metadata, metadata, metadata) #2

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "fast-isel-branch-uncond-debug.ll", directory: "/tmp")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!""}
!7 = distinct !DISubprogram(name: "a", scope: !1, file: !1, line: 1, type: !8, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !11)
!8 = !DISubroutineType(types: !9)
!9 = !{!10}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !{!12}
!12 = !DILocalVariable(name: "b", scope: !7, file: !1, line: 2, type: !10)
!13 = !DILocation(line: 0, scope: !7)
!14 = !DILocation(line: 3, column: 3, scope: !7)
!15 = !DILocation(line: 3, column: 3, scope: !16)
!16 = distinct !DILexicalBlock(scope: !17, file: !1, line: 3, column: 3)
!17 = distinct !DILexicalBlock(scope: !7, file: !1, line: 3, column: 3)
!18 = distinct !{!18, !19, !20}
!19 = !DILocation(line: 3, column: 3, scope: !17)
!20 = !DILocation(line: 4, column: 5, scope: !17)

