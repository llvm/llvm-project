; RUN: opt -S -o - %s -simplifycfg -simplifycfg-require-and-preserve-domtree=1 | FileCheck %s

declare i1 @make_condition()

; CHECK-LABEL: @test1(
; CHECK: and i1 [[COND:%.*]], [[COND]]{{$}}
; CHECK: select i1 [[COND]]
define void @test1() !dbg !6 {
start:
  %cond = call i1 @make_condition(), !dbg !8
  br i1 %cond, label %then, label %else, !dbg !9

then:                                             ; preds = %start
  %and = and i1 %cond, %cond, !dbg !10
  br label %else, !dbg !11

else:                                             ; preds = %then, %start
  %phi = phi i1 [ %cond, %start ], [ %and, %then ], !dbg !12
  ret void, !dbg !13
}

!llvm.dbg.cu = !{!0}
!llvm.debugify = !{!3, !4}
!llvm.module.flags = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "test.ll", directory: "/")
!2 = !{}
!3 = !{i32 6}
!4 = !{i32 0}
!5 = !{i32 2, !"Debug Info Version", i32 3}
!6 = distinct !DISubprogram(name: "test1", linkageName: "test1", scope: null, file: !1, line: 1, type: !7, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!7 = !DISubroutineType(types: !2)
!8 = !DILocation(line: 1, column: 1, scope: !6)
!9 = !DILocation(line: 2, column: 1, scope: !6)
!10 = !DILocation(line: 3, column: 2, scope: !6)
!11 = !DILocation(line: 4, column: 2, scope: !6)
!12 = !DILocation(line: 5, column: 3, scope: !6)
!13 = !DILocation(line: 6, column: 3, scope: !6)
