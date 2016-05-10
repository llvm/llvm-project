; RUN: opt -loop-vectorize -force-vector-width=2 -pass-remarks-analysis=loop-vectorize < %s 2>&1 | FileCheck %s

; ModuleID = '/tmp/kk.c'
source_filename = "/tmp/kk.c"
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.11.0"

;     1	void success (char *A, char *B, char *C, char *D, char *E, int N) {
;     2	  for(int i = 0; i < N; i++) {
;     3	    A[i + 1] = A[i] + B[i];
;     4	    C[i] = D[i] * E[i];
;     5	  }
;     6	}

; CHECK: remark: /tmp/kk.c:3:16: loop not vectorized: unsafe dependent memory operations in loop. Use #pragma loop distribute(enable) to allow loop distribution to attempt to isolate the offending operations into a separate loop

define void @success(i8* nocapture %A, i8* nocapture readonly %B, i8* nocapture %C, i8* nocapture readonly %D, i8* nocapture readonly %E, i32 %N) !dbg !6 {
entry:
  %cmp28 = icmp sgt i32 %N, 0, !dbg !8
  br i1 %cmp28, label %for.body, label %for.cond.cleanup, !dbg !9

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i8, i8* %A, i64 %indvars.iv, !dbg !11
  %0 = load i8, i8* %arrayidx, align 1, !dbg !11, !tbaa !12
  %arrayidx2 = getelementptr inbounds i8, i8* %B, i64 %indvars.iv, !dbg !15
  %1 = load i8, i8* %arrayidx2, align 1, !dbg !15, !tbaa !12
  %add = add i8 %1, %0, !dbg !16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !9
  %arrayidx7 = getelementptr inbounds i8, i8* %A, i64 %indvars.iv.next, !dbg !17
  store i8 %add, i8* %arrayidx7, align 1, !dbg !18, !tbaa !12
  %arrayidx9 = getelementptr inbounds i8, i8* %D, i64 %indvars.iv, !dbg !19
  %2 = load i8, i8* %arrayidx9, align 1, !dbg !19, !tbaa !12
  %arrayidx12 = getelementptr inbounds i8, i8* %E, i64 %indvars.iv, !dbg !20
  %3 = load i8, i8* %arrayidx12, align 1, !dbg !20, !tbaa !12
  %mul = mul i8 %3, %2, !dbg !21
  %arrayidx16 = getelementptr inbounds i8, i8* %C, i64 %indvars.iv, !dbg !22
  store i8 %mul, i8* %arrayidx16, align 1, !dbg !23, !tbaa !12
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32, !dbg !9
  %exitcond = icmp eq i32 %lftr.wideiv, %N, !dbg !9
  br i1 %exitcond, label %for.cond.cleanup, label %for.body, !dbg !9

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void, !dbg !10
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 3.9.0 ", isOptimized: true, runtimeVersion: 0, emissionKind: NoDebug, enums: !2)
!1 = !DIFile(filename: "/tmp/kk.c", directory: "/tmp")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"PIC Level", i32 2}
!5 = !{!"clang version 3.9.0 "}
!6 = distinct !DISubprogram(name: "success", scope: !1, file: !1, line: 1, type: !7, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: true, unit: !0, variables: !2)
!7 = !DISubroutineType(types: !2)
!8 = !DILocation(line: 2, column: 20, scope: !6)
!9 = !DILocation(line: 2, column: 3, scope: !6)
!10 = !DILocation(line: 6, column: 1, scope: !6)
!11 = !DILocation(line: 3, column: 16, scope: !6)
!12 = !{!13, !13, i64 0}
!13 = !{!"omnipotent char", !14, i64 0}
!14 = !{!"Simple C/C++ TBAA"}
!15 = !DILocation(line: 3, column: 23, scope: !6)
!16 = !DILocation(line: 3, column: 21, scope: !6)
!17 = !DILocation(line: 3, column: 5, scope: !6)
!18 = !DILocation(line: 3, column: 14, scope: !6)
!19 = !DILocation(line: 4, column: 12, scope: !6)
!20 = !DILocation(line: 4, column: 19, scope: !6)
!21 = !DILocation(line: 4, column: 17, scope: !6)
!22 = !DILocation(line: 4, column: 5, scope: !6)
!23 = !DILocation(line: 4, column: 10, scope: !6)
