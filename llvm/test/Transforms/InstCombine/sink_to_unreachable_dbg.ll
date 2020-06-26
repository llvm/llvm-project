; RUN: opt -instcombine -S < %s | FileCheck %s

; CHECK-LABEL: @test(
; CHECK: [[phi:%.*]] = phi i32
; CHECK-NEXT: [[add:%.*]] = add i32 %x, 1{{$}}
; CHECK-NEXT: add i32 [[phi]], [[add]], !dbg
define i32 @test(i32 %x, i1 %c) !dbg !6 {
bb0:
  %a = add i32 %x, 1, !dbg !8
  br i1 %c, label %bb1, label %bb2, !dbg !9

bb1:                                              ; preds = %bb0
  br label %bb3, !dbg !10

bb2:                                              ; preds = %bb0
  br label %bb3, !dbg !11

bb3:                                              ; preds = %bb2, %bb1
  %p = phi i32 [ 0, %bb1 ], [ 1, %bb2 ], !dbg !12
  %r = add i32 %p, %a, !dbg !13
  ret i32 %r, !dbg !14
}

!llvm.dbg.cu = !{!0}
!llvm.debugify = !{!3, !4}
!llvm.module.flags = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "sink_to_unreachable_dbg.ll", directory: "/")
!2 = !{}
!3 = !{i32 7}
!4 = !{i32 0}
!5 = !{i32 2, !"Debug Info Version", i32 3}
!6 = distinct !DISubprogram(name: "test", linkageName: "test", scope: null, file: !1, line: 1, type: !7, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!7 = !DISubroutineType(types: !2)
!8 = !DILocation(line: 1, column: 1, scope: !6)
!9 = !DILocation(line: 2, column: 1, scope: !6)
!10 = !DILocation(line: 3, column: 2, scope: !6)
!11 = !DILocation(line: 4, column: 3, scope: !6)
!12 = !DILocation(line: 5, column: 4, scope: !6)
!13 = !DILocation(line: 6, column: 4, scope: !6)
!14 = !DILocation(line: 7, column: 4, scope: !6)
