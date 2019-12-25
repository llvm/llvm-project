; RUN: opt -S -loop-vectorize -pass-remarks-missed=loop-vectorize \
; RUN:     -pass-remarks-with-hotness < %s 2>&1 | \
; RUN:     FileCheck -check-prefix=HOTNESS -check-prefix=BOTH %s

; RUN: opt -S -loop-vectorize -pass-remarks-missed=loop-vectorize < %s 2>&1 | \
; RUN:     FileCheck -check-prefix=NO_HOTNESS -check-prefix=BOTH %s


; RUN: opt -S -passes=loop-vectorize -pass-remarks-missed=loop-vectorize \
; RUN:     -pass-remarks-with-hotness < %s 2>&1 | \
; RUN:     FileCheck -check-prefix=HOTNESS -check-prefix=BOTH %s

; RUN: opt -S -passes=loop-vectorize \
; RUN:     -pass-remarks-missed=loop-vectorize < %s 2>&1 | \
; RUN:     FileCheck -check-prefix=NO_HOTNESS -check-prefix=BOTH %s


;   1	void cold(char *A, char *B, char *C, char *D, char *E, int N) {
;   2	  for(int i = 0; i < N; i++) {
;   3	    A[i + 1] = A[i] + B[i];
;   4	    C[i] = D[i] * E[i];
;   5	  }
;   6	}
;   7
;   8	void hot(char *A, char *B, char *C, char *D, char *E, int N) {
;   9	  for(int i = 0; i < N; i++) {
;  10	    A[i + 1] = A[i] + B[i];
;  11	    C[i] = D[i] * E[i];
;  12	  }
;  13	}
;  14
;  15	void unknown(char *A, char *B, char *C, char *D, char *E, int N) {
;  16	  for(int i = 0; i < N; i++) {
;  17	    A[i + 1] = A[i] + B[i];
;  18	    C[i] = D[i] * E[i];
;  19	  }
;  20	}

; HOTNESS: remark: /tmp/s.c:2:3: loop not vectorized (hotness: 300)
; NO_HOTNESS: remark: /tmp/s.c:2:3: loop not vectorized{{$}}
; HOTNESS: remark: /tmp/s.c:9:3: loop not vectorized (hotness: 5000)
; NO_HOTNESS: remark: /tmp/s.c:9:3: loop not vectorized{{$}}
; BOTH: remark: /tmp/s.c:16:3: loop not vectorized{{$}}

; ModuleID = '/tmp/s.c'
source_filename = "/tmp/s.c"
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

; Function Attrs: norecurse nounwind ssp uwtable
define void @cold(i8* nocapture %A, i8* nocapture readonly %B, i8* nocapture %C, i8* nocapture readonly %D, i8* nocapture readonly %E, i32 %N) local_unnamed_addr #0 !dbg !7 !prof !56 {
entry:
  %cmp28 = icmp sgt i32 %N, 0, !dbg !9
  br i1 %cmp28, label %ph, label %for.cond.cleanup, !dbg !10, !prof !58

ph:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %ph ]
  %arrayidx = getelementptr inbounds i8, i8* %A, i64 %indvars.iv, !dbg !12
  %0 = load i8, i8* %arrayidx, align 1, !dbg !12, !tbaa !13
  %arrayidx2 = getelementptr inbounds i8, i8* %B, i64 %indvars.iv, !dbg !16
  %1 = load i8, i8* %arrayidx2, align 1, !dbg !16, !tbaa !13
  %add = add i8 %1, %0, !dbg !17
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !10
  %arrayidx7 = getelementptr inbounds i8, i8* %A, i64 %indvars.iv.next, !dbg !18
  store i8 %add, i8* %arrayidx7, align 1, !dbg !19, !tbaa !13
  %arrayidx9 = getelementptr inbounds i8, i8* %D, i64 %indvars.iv, !dbg !20
  %2 = load i8, i8* %arrayidx9, align 1, !dbg !20, !tbaa !13
  %arrayidx12 = getelementptr inbounds i8, i8* %E, i64 %indvars.iv, !dbg !21
  %3 = load i8, i8* %arrayidx12, align 1, !dbg !21, !tbaa !13
  %mul = mul i8 %3, %2, !dbg !22
  %arrayidx16 = getelementptr inbounds i8, i8* %C, i64 %indvars.iv, !dbg !23
  store i8 %mul, i8* %arrayidx16, align 1, !dbg !24, !tbaa !13
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32, !dbg !10
  %exitcond = icmp eq i32 %lftr.wideiv, %N, !dbg !10
  br i1 %exitcond, label %for.cond.cleanup, label %for.body, !dbg !10, !llvm.loop !25, !prof !59

for.cond.cleanup:
  ret void, !dbg !11
}

; Function Attrs: norecurse nounwind ssp uwtable
define void @hot(i8* nocapture %A, i8* nocapture readonly %B, i8* nocapture %C, i8* nocapture readonly %D, i8* nocapture readonly %E, i32 %N) local_unnamed_addr #0 !dbg !26 !prof !57 {
entry:
  %cmp28 = icmp sgt i32 %N, 0, !dbg !27
  br i1 %cmp28, label %ph, label %for.cond.cleanup, !dbg !28, !prof !58

ph:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %ph ]
  %arrayidx = getelementptr inbounds i8, i8* %A, i64 %indvars.iv, !dbg !30
  %0 = load i8, i8* %arrayidx, align 1, !dbg !30, !tbaa !13
  %arrayidx2 = getelementptr inbounds i8, i8* %B, i64 %indvars.iv, !dbg !31
  %1 = load i8, i8* %arrayidx2, align 1, !dbg !31, !tbaa !13
  %add = add i8 %1, %0, !dbg !32
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !28
  %arrayidx7 = getelementptr inbounds i8, i8* %A, i64 %indvars.iv.next, !dbg !33
  store i8 %add, i8* %arrayidx7, align 1, !dbg !34, !tbaa !13
  %arrayidx9 = getelementptr inbounds i8, i8* %D, i64 %indvars.iv, !dbg !35
  %2 = load i8, i8* %arrayidx9, align 1, !dbg !35, !tbaa !13
  %arrayidx12 = getelementptr inbounds i8, i8* %E, i64 %indvars.iv, !dbg !36
  %3 = load i8, i8* %arrayidx12, align 1, !dbg !36, !tbaa !13
  %mul = mul i8 %3, %2, !dbg !37
  %arrayidx16 = getelementptr inbounds i8, i8* %C, i64 %indvars.iv, !dbg !38
  store i8 %mul, i8* %arrayidx16, align 1, !dbg !39, !tbaa !13
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32, !dbg !28
  %exitcond = icmp eq i32 %lftr.wideiv, %N, !dbg !28
  br i1 %exitcond, label %for.cond.cleanup, label %for.body, !dbg !28, !llvm.loop !40, !prof !59

for.cond.cleanup:
  ret void, !dbg !29
}

; Function Attrs: norecurse nounwind ssp uwtable
define void @unknown(i8* nocapture %A, i8* nocapture readonly %B, i8* nocapture %C, i8* nocapture readonly %D, i8* nocapture readonly %E, i32 %N) local_unnamed_addr #0 !dbg !41 {
entry:
  %cmp28 = icmp sgt i32 %N, 0, !dbg !42
  br i1 %cmp28, label %for.body, label %for.cond.cleanup, !dbg !43

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void, !dbg !44

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i8, i8* %A, i64 %indvars.iv, !dbg !45
  %0 = load i8, i8* %arrayidx, align 1, !dbg !45, !tbaa !13
  %arrayidx2 = getelementptr inbounds i8, i8* %B, i64 %indvars.iv, !dbg !46
  %1 = load i8, i8* %arrayidx2, align 1, !dbg !46, !tbaa !13
  %add = add i8 %1, %0, !dbg !47
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !43
  %arrayidx7 = getelementptr inbounds i8, i8* %A, i64 %indvars.iv.next, !dbg !48
  store i8 %add, i8* %arrayidx7, align 1, !dbg !49, !tbaa !13
  %arrayidx9 = getelementptr inbounds i8, i8* %D, i64 %indvars.iv, !dbg !50
  %2 = load i8, i8* %arrayidx9, align 1, !dbg !50, !tbaa !13
  %arrayidx12 = getelementptr inbounds i8, i8* %E, i64 %indvars.iv, !dbg !51
  %3 = load i8, i8* %arrayidx12, align 1, !dbg !51, !tbaa !13
  %mul = mul i8 %3, %2, !dbg !52
  %arrayidx16 = getelementptr inbounds i8, i8* %C, i64 %indvars.iv, !dbg !53
  store i8 %mul, i8* %arrayidx16, align 1, !dbg !54, !tbaa !13
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32, !dbg !43
  %exitcond = icmp eq i32 %lftr.wideiv, %N, !dbg !43
  br i1 %exitcond, label %for.cond.cleanup, label %for.body, !dbg !43, !llvm.loop !55
}

attributes #0 = { norecurse nounwind ssp uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="core2" "target-features"="+cx16,+fxsr,+mmx,+sse,+sse2,+sse3,+ssse3,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 3.9.0 (trunk 273572) (llvm/trunk 273585)", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, enums: !2)
!1 = !DIFile(filename: "/tmp/s.c", directory: "/tmp")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 2}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"PIC Level", i32 2}
!6 = !{!"clang version 3.9.0 (trunk 273572) (llvm/trunk 273585)"}
!7 = distinct !DISubprogram(name: "cold", scope: !1, file: !1, line: 1, type: !8, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !2)
!9 = !DILocation(line: 2, column: 20, scope: !7)
!10 = !DILocation(line: 2, column: 3, scope: !7)
!11 = !DILocation(line: 6, column: 1, scope: !7)
!12 = !DILocation(line: 3, column: 16, scope: !7)
!13 = !{!14, !14, i64 0}
!14 = !{!"omnipotent char", !15, i64 0}
!15 = !{!"Simple C/C++ TBAA"}
!16 = !DILocation(line: 3, column: 23, scope: !7)
!17 = !DILocation(line: 3, column: 21, scope: !7)
!18 = !DILocation(line: 3, column: 5, scope: !7)
!19 = !DILocation(line: 3, column: 14, scope: !7)
!20 = !DILocation(line: 4, column: 12, scope: !7)
!21 = !DILocation(line: 4, column: 19, scope: !7)
!22 = !DILocation(line: 4, column: 17, scope: !7)
!23 = !DILocation(line: 4, column: 5, scope: !7)
!24 = !DILocation(line: 4, column: 10, scope: !7)
!25 = distinct !{!25, !10}
!26 = distinct !DISubprogram(name: "hot", scope: !1, file: !1, line: 8, type: !8, isLocal: false, isDefinition: true, scopeLine: 8, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !2)
!27 = !DILocation(line: 9, column: 20, scope: !26)
!28 = !DILocation(line: 9, column: 3, scope: !26)
!29 = !DILocation(line: 13, column: 1, scope: !26)
!30 = !DILocation(line: 10, column: 16, scope: !26)
!31 = !DILocation(line: 10, column: 23, scope: !26)
!32 = !DILocation(line: 10, column: 21, scope: !26)
!33 = !DILocation(line: 10, column: 5, scope: !26)
!34 = !DILocation(line: 10, column: 14, scope: !26)
!35 = !DILocation(line: 11, column: 12, scope: !26)
!36 = !DILocation(line: 11, column: 19, scope: !26)
!37 = !DILocation(line: 11, column: 17, scope: !26)
!38 = !DILocation(line: 11, column: 5, scope: !26)
!39 = !DILocation(line: 11, column: 10, scope: !26)
!40 = distinct !{!40, !28}
!41 = distinct !DISubprogram(name: "unknown", scope: !1, file: !1, line: 15, type: !8, isLocal: false, isDefinition: true, scopeLine: 15, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !2)
!42 = !DILocation(line: 16, column: 20, scope: !41)
!43 = !DILocation(line: 16, column: 3, scope: !41)
!44 = !DILocation(line: 20, column: 1, scope: !41)
!45 = !DILocation(line: 17, column: 16, scope: !41)
!46 = !DILocation(line: 17, column: 23, scope: !41)
!47 = !DILocation(line: 17, column: 21, scope: !41)
!48 = !DILocation(line: 17, column: 5, scope: !41)
!49 = !DILocation(line: 17, column: 14, scope: !41)
!50 = !DILocation(line: 18, column: 12, scope: !41)
!51 = !DILocation(line: 18, column: 19, scope: !41)
!52 = !DILocation(line: 18, column: 17, scope: !41)
!53 = !DILocation(line: 18, column: 5, scope: !41)
!54 = !DILocation(line: 18, column: 10, scope: !41)
!55 = distinct !{!55, !43}
!56 = !{!"function_entry_count", i64 3}
!57 = !{!"function_entry_count", i64 50}
!58 = !{!"branch_weights", i32 99, i32 1}
!59 = !{!"branch_weights", i32 1, i32 99}
