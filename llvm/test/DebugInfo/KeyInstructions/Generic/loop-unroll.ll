; RUN: opt %s --passes=loop-unroll -S -o - -S | FileCheck %s

; CHECK: store i32 %a, ptr %p, align 4, !dbg [[G1R1:!.*]]
; CHECK: %v.1 = add i32 1, %a, !dbg [[G2R2:!.*]]
; CHECK: store i32 %v.1, ptr %p, align 4, !dbg [[G2R1:!.*]]
; CHECK: %v.2 = add i32 2, %a, !dbg [[G3R2:!.*]]
; CHECK: store i32 %v.2, ptr %p, align 4, !dbg [[G3R1:!.*]]

; CHECK: [[G1R1]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 1)
; CHECK: [[G2R2]] = !DILocation({{.*}}, atomGroup: 2, atomRank: 2)
; CHECK: [[G2R1]] = !DILocation({{.*}}, atomGroup: 2, atomRank: 1)
; CHECK: [[G3R2]] = !DILocation({{.*}}, atomGroup: 3, atomRank: 2)
; CHECK: [[G3R1]] = !DILocation({{.*}}, atomGroup: 3, atomRank: 1)

define void @f(ptr %p, i32 %a) local_unnamed_addr #0 !dbg !5 {
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %for.body, %entry
  %i = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %v = add i32 %i, %a, !dbg !11
  store i32 %v, ptr %p, align 4, !dbg !12
  %inc = add i32 %i, 1
  %cmp = icmp slt i32 %inc, 3
  br i1 %cmp, label %for.body, label %for.cond.cleanup
}

declare i32 @bar(...) local_unnamed_addr

attributes #0 = { nounwind ssp uwtable }

!llvm.dbg.cu = !{!0}
!llvm.debugify = !{!2, !3}
!llvm.module.flags = !{!4}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "/home/och/scratch/test.ll", directory: "/")
!2 = !{i32 8}
!3 = !{i32 0}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "f", linkageName: "f", scope: null, file: !1, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!6 = !DISubroutineType(types: !7)
!7 = !{}
!11 = !DILocation(line: 4, column: 1, scope: !5, atomGroup: 1, atomRank: 2)
!12 = !DILocation(line: 5, column: 1, scope: !5, atomGroup: 1, atomRank: 1)
