; RUN: opt < %s -passes=indvars -S | FileCheck %s

; This testcase checks the preservation of debug locations of newly created 
; phi, sitofp, add and icmp instructions in IndVarSimplify Pass.

define void @test1() !dbg !5 {
; CHECK-LABEL: @test1(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label [[BB:%.*]], !dbg
; CHECK:  bb:
; CHECK:    [[IV_INT:%.*]] = phi i32 [ 0, [[ENTRY:%.*]] ], [ [[DOTINT:%.*]], [[BB]] ], !dbg ![[DBG1:[0-9]+]]
; CHECK:    [[INDVAR_CONV:%.*]] = sitofp i32 [[IV_INT]] to double, !dbg ![[DBG1]]
; CHECK:    [[DOTINT]] = add nuw nsw i32 [[IV_INT]], 1, !dbg ![[DBG2:[0-9]+]]
; CHECK:    [[TMP1:%.*]] = icmp ult i32 [[DOTINT]], 10000, !dbg ![[DBG3:[0-9]+]]
; CHECK: ![[DBG1]] = !DILocation(line: 2
; CHECK: ![[DBG2]] = !DILocation(line: 4
; CHECK: ![[DBG3]] = !DILocation(line: 5
;
entry:
  br label %bb, !dbg !16

bb:                                               ; preds = %bb, %entry
  %iv = phi double [ 0.000000e+00, %entry ], [ %1, %bb ], !dbg !17
  %0 = tail call i32 @foo(double %iv), !dbg !18
  %1 = fadd double %iv, 1.000000e+00, !dbg !19
  %2 = fcmp olt double %1, 1.000000e+04, !dbg !20
  br i1 %2, label %bb, label %return, !dbg !21

return:                                           ; preds = %bb
  ret void, !dbg !22
}

declare i32 @foo(double)

!llvm.dbg.cu = !{!0}
!llvm.debugify = !{!2, !3}
!llvm.module.flags = !{!4}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "indvars-preserving.ll", directory: "/")
!2 = !{i32 7}
!3 = !{i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "test1", linkageName: "test1", scope: null, file: !1, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !8)
!6 = !DISubroutineType(types: !7)
!7 = !{}
!8 = !{!9, !11, !13, !14}
!9 = !DILocalVariable(name: "1", scope: !5, file: !1, line: 2, type: !10)
!10 = !DIBasicType(name: "ty64", size: 64, encoding: DW_ATE_unsigned)
!11 = !DILocalVariable(name: "2", scope: !5, file: !1, line: 3, type: !12)
!12 = !DIBasicType(name: "ty32", size: 32, encoding: DW_ATE_unsigned)
!13 = !DILocalVariable(name: "3", scope: !5, file: !1, line: 4, type: !10)
!14 = !DILocalVariable(name: "4", scope: !5, file: !1, line: 5, type: !15)
!15 = !DIBasicType(name: "ty8", size: 8, encoding: DW_ATE_unsigned)
!16 = !DILocation(line: 1, column: 1, scope: !5)
!17 = !DILocation(line: 2, column: 1, scope: !5)
!18 = !DILocation(line: 3, column: 1, scope: !5)
!19 = !DILocation(line: 4, column: 1, scope: !5)
!20 = !DILocation(line: 5, column: 1, scope: !5)
!21 = !DILocation(line: 6, column: 1, scope: !5)
!22 = !DILocation(line: 7, column: 1, scope: !5)
