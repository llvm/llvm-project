; RUN: opt -S -simplifycfg < %s | FileCheck %s --match-full-lines

; Make sure we reset the debug location when folding instructions.
; CHECK: [[VAL:%.*]] = and i32 %c2, %k
; CHECK-NEXT: [[VAL2:%.*]] icmp eq i32 [[VAL]], 0

declare i32 @bar(...)

define i32 @patatino(i32 %k, i32 %c1, i32 %c2) !dbg !6 {
  %1 = and i32 %c1, %k, !dbg !8
  %2 = icmp eq i32 %1, 0, !dbg !9
  br i1 %2, label %8, label %3, !dbg !10

3:
  %4 = and i32 %c2, %k, !dbg !11
  %5 = icmp eq i32 %4, 0, !dbg !12
  br i1 %5, label %8, label %6, !dbg !13

6:
  %7 = tail call i32 (...) @bar(), !dbg !14
  br label %8, !dbg !15

8:
  ret i32 undef, !dbg !16
}

!llvm.dbg.cu = !{!0}
!llvm.debugify = !{!3, !4}
!llvm.module.flags = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "a.ll", directory: "/")
!2 = !{}
!3 = !{i32 9}
!4 = !{i32 0}
!5 = !{i32 2, !"Debug Info Version", i32 3}
!6 = distinct !DISubprogram(name: "patatino", linkageName: "patatino", scope: null, file: !1, line: 1, type: !7, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!7 = !DISubroutineType(types: !2)
!8 = !DILocation(line: 1, column: 1, scope: !6)
!9 = !DILocation(line: 2, column: 1, scope: !6)
!10 = !DILocation(line: 3, column: 1, scope: !6)
!11 = !DILocation(line: 4, column: 1, scope: !6)
!12 = !DILocation(line: 5, column: 1, scope: !6)
!13 = !DILocation(line: 6, column: 1, scope: !6)
!14 = !DILocation(line: 7, column: 1, scope: !6)
!15 = !DILocation(line: 8, column: 1, scope: !6)
!16 = !DILocation(line: 9, column: 1, scope: !6)
