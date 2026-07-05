; RUN: opt < %s -S -passes=jump-threading | FileCheck %s

; Test that JumpThreading's `simplifyPartiallyRedundantLoad` propagates
; the debug location to the `bitcast` from the LoadInst it replaces (`%b`).

declare void @f1(...)

define void @test8(ptr %0, ptr %1, ptr %2) !dbg !5 {
; CHECK: @test8
; CHECK:    [[TMP4:%.*]] = bitcast float [[A:%.*]] to i32, !dbg [[DBG9:![0-9]+]]
; CHECK: [[DBG9]] = !DILocation(line: 2,
;
  %a = load float, ptr %0, align 4, !dbg !8
  %b = load i32, ptr %0, align 4, !dbg !9
  store float %a, ptr %1, align 4, !dbg !10
  %c = icmp eq i32 %b, 8, !dbg !11
  br i1 %c, label %ret1, label %ret2, !dbg !12

ret1:                                             ; preds = %3
  ret void, !dbg !13

ret2:                                             ; preds = %3
  %xxx = tail call i32 (...) @f1() #0, !dbg !14
  ret void, !dbg !15
}

!llvm.dbg.cu = !{!0}
!llvm.debugify = !{!2, !3}
!llvm.module.flags = !{!4}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "test.ll", directory: "/")
!2 = !{i32 8}
!3 = !{i32 0}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "test8", linkageName: "test8", scope: null, file: !1, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!6 = !DISubroutineType(types: !7)
!7 = !{}
!8 = !DILocation(line: 1, column: 1, scope: !5)
!9 = !DILocation(line: 2, column: 1, scope: !5)
!10 = !DILocation(line: 3, column: 1, scope: !5)
!11 = !DILocation(line: 4, column: 1, scope: !5)
!12 = !DILocation(line: 5, column: 1, scope: !5)
!13 = !DILocation(line: 6, column: 1, scope: !5)
!14 = !DILocation(line: 7, column: 1, scope: !5)
!15 = !DILocation(line: 8, column: 1, scope: !5)
