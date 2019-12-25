; RUN: opt -loop-vectorize -pass-remarks-analysis=loop-vectorize < %s 2>&1 | FileCheck %s

;  1     extern int map[];
;  2     extern int out[];
;  3
;  4     void f(int a, int n) {
;  5       for (int i = 0; i < n; ++i) {
;  6         out[i] = a;
;  7         a = map[a];
;  8       }
;  9     }

; CHECK: remark: /tmp/s.c:5:3: loop not vectorized: value that could not be identified as reduction is used outside the loop

; %a.addr.08 is the phi corresponding to the remark.  It does not have debug
; location attached.  In this case we should use the debug location of the
; loop rather than emitting <unknown>:0:0:

; ModuleID = '/tmp/s.c'
source_filename = "/tmp/s.c"
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

@out = external local_unnamed_addr global [0 x i32], align 4
@map = external local_unnamed_addr global [0 x i32], align 4

; Function Attrs: norecurse nounwind ssp uwtable
define void @f(i32 %a, i32 %n) local_unnamed_addr #0 !dbg !6 {
entry:
  %cmp7 = icmp sgt i32 %n, 0, !dbg !8
  br i1 %cmp7, label %for.body.preheader, label %for.cond.cleanup, !dbg !9

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64, !dbg !9
  br label %for.body, !dbg !10

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void, !dbg !11

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %for.body.preheader ]
  %a.addr.08 = phi i32 [ %0, %for.body ], [ %a, %for.body.preheader ]

  %arrayidx = getelementptr inbounds [0 x i32], [0 x i32]* @out, i64 0, i64 %indvars.iv, !dbg !10
  store i32 %a.addr.08, i32* %arrayidx, align 4, !dbg !12, !tbaa !13
  %idxprom1 = sext i32 %a.addr.08 to i64, !dbg !17
  %arrayidx2 = getelementptr inbounds [0 x i32], [0 x i32]* @map, i64 0, i64 %idxprom1, !dbg !17
  %0 = load i32, i32* %arrayidx2, align 4, !dbg !17, !tbaa !13
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !9
  %exitcond = icmp eq i64 %indvars.iv.next, %wide.trip.count, !dbg !9
  br i1 %exitcond, label %for.cond.cleanup, label %for.body, !dbg !9, !llvm.loop !18
}

attributes #0 = { norecurse nounwind ssp uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="core2" "target-features"="+cx16,+fxsr,+mmx,+sse,+sse2,+sse3,+ssse3,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 4.0.0 (trunk 281293) (llvm/trunk 281290)", isOptimized: true, runtimeVersion: 0, emissionKind: NoDebug, enums: !2)
!1 = !DIFile(filename: "/tmp/s.c", directory: "/tmp")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"PIC Level", i32 2}
!5 = !{!"clang version 4.0.0 (trunk 281293) (llvm/trunk 281290)"}
!6 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 4, type: !7, isLocal: false, isDefinition: true, scopeLine: 4, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !2)
!7 = !DISubroutineType(types: !2)
!8 = !DILocation(line: 5, column: 21, scope: !6)
!9 = !DILocation(line: 5, column: 3, scope: !6)
!10 = !DILocation(line: 6, column: 5, scope: !6)
!11 = !DILocation(line: 9, column: 1, scope: !6)
!12 = !DILocation(line: 6, column: 12, scope: !6)
!13 = !{!14, !14, i64 0}
!14 = !{!"int", !15, i64 0}
!15 = !{!"omnipotent char", !16, i64 0}
!16 = !{!"Simple C/C++ TBAA"}
!17 = !DILocation(line: 7, column: 9, scope: !6)
!18 = distinct !{!18, !9}
