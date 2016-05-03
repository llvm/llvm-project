; RUN: opt -loop-distribute -S < %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=ALWAYS --check-prefix=NO_REMARKS
; RUN: opt -loop-distribute -S -pass-remarks-missed=loop-distribute < %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=ALWAYS --check-prefix=MISSED_REMARKS
; RUN: opt -loop-distribute -S -pass-remarks-analysis=loop-distribute < %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=ALWAYS --check-prefix=ANALYSIS_REMARKS

; This is the input program:
;
;     1	void forced (char *A, char *B, char *C, int N) {
;     2	#pragma clang loop distribute(enable)
;     3	  for(int i = 0; i < N; i++) {
;     4	    A[i] = B[i] * C[i];
;     5	  }
;     6	}
;     7
;     8	void not_forced (char *A, char *B, char *C, int N) {
;     9	  for(int i = 0; i < N; i++) {
;    10	    A[i] = B[i] * C[i];
;    11	  }
;    12	}

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.11.0"

; MISSED_REMARKS: remark:  /tmp/t.c:3:3: loop not distributed: use -Rpass-analysis=loop-distribute for more info
; ALWAYS:         remark: /tmp/t.c:3:3: loop not distributed: memory operations are safe for vectorization
; ALWAYS:         warning: /tmp/t.c:3:3: loop not disributed: failed explicitly specified loop distribution

define void @forced(i8* %A, i8* %B, i8* %C, i32 %N) !dbg !7 {
entry:
  %cmp12 = icmp sgt i32 %N, 0, !dbg !9
  br i1 %cmp12, label %ph, label %for.cond.cleanup, !dbg !10

ph:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %ph ]
  %arrayidx = getelementptr inbounds i8, i8* %B, i64 %indvars.iv, !dbg !12
  %0 = load i8, i8* %arrayidx, align 1, !dbg !12, !tbaa !13
  %arrayidx2 = getelementptr inbounds i8, i8* %C, i64 %indvars.iv, !dbg !16
  %1 = load i8, i8* %arrayidx2, align 1, !dbg !16, !tbaa !13
  %mul = mul i8 %1, %0, !dbg !17
  %arrayidx6 = getelementptr inbounds i8, i8* %A, i64 %indvars.iv, !dbg !18
  store i8 %mul, i8* %arrayidx6, align 1, !dbg !19, !tbaa !13
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !10
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32, !dbg !10
  %exitcond = icmp eq i32 %lftr.wideiv, %N, !dbg !10
  br i1 %exitcond, label %for.cond.cleanup, label %for.body, !dbg !10, !llvm.loop !20

for.cond.cleanup:
  ret void, !dbg !11
}

; NO_REMARKS-NOT: remark: /tmp/t.c:9:3: loop not distributed: memory operations are safe for vectorization
; MISSED_REMARKS: remark: /tmp/t.c:9:3: loop not distributed: use -Rpass-analysis=loop-distribute for more info
; ANALYSIS_REMARKS: remark: /tmp/t.c:9:3: loop not distributed: memory operations are safe for vectorization
; ALWAYS-NOT: warning: /tmp/t.c:9:3: loop not disributed: failed explicitly specified loop distribution

define void @not_forced(i8* %A, i8* %B, i8* %C, i32 %N) !dbg !22 {
entry:
  %cmp12 = icmp sgt i32 %N, 0, !dbg !23
  br i1 %cmp12, label %ph, label %for.cond.cleanup, !dbg !24

ph:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %ph ]
  %arrayidx = getelementptr inbounds i8, i8* %B, i64 %indvars.iv, !dbg !26
  %0 = load i8, i8* %arrayidx, align 1, !dbg !26, !tbaa !13
  %arrayidx2 = getelementptr inbounds i8, i8* %C, i64 %indvars.iv, !dbg !27
  %1 = load i8, i8* %arrayidx2, align 1, !dbg !27, !tbaa !13
  %mul = mul i8 %1, %0, !dbg !28
  %arrayidx6 = getelementptr inbounds i8, i8* %A, i64 %indvars.iv, !dbg !29
  store i8 %mul, i8* %arrayidx6, align 1, !dbg !30, !tbaa !13
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !24
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32, !dbg !24
  %exitcond = icmp eq i32 %lftr.wideiv, %N, !dbg !24
  br i1 %exitcond, label %for.cond.cleanup, label %for.body, !dbg !24

for.cond.cleanup:
  ret void, !dbg !25
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 3.9.0 (trunk 267633) (llvm/trunk 267675)", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, enums: !2)
!1 = !DIFile(filename: "/tmp/t.c", directory: "/tmp")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 2}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!7 = distinct !DISubprogram(name: "forced", scope: !1, file: !1, line: 1, type: !8, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: true, unit: !0, variables: !2)
!8 = !DISubroutineType(types: !2)
!9 = !DILocation(line: 3, column: 20, scope: !7)
!10 = !DILocation(line: 3, column: 3, scope: !7)
!11 = !DILocation(line: 6, column: 1, scope: !7)
!12 = !DILocation(line: 4, column: 12, scope: !7)
!13 = !{!14, !14, i64 0}
!14 = !{!"omnipotent char", !15, i64 0}
!15 = !{!"Simple C/C++ TBAA"}
!16 = !DILocation(line: 4, column: 19, scope: !7)
!17 = !DILocation(line: 4, column: 17, scope: !7)
!18 = !DILocation(line: 4, column: 5, scope: !7)
!19 = !DILocation(line: 4, column: 10, scope: !7)
!20 = distinct !{!20, !21}
!21 = !{!"llvm.loop.distribute.enable", i1 true}
!22 = distinct !DISubprogram(name: "not_forced", scope: !1, file: !1, line: 8, type: !8, isLocal: false, isDefinition: true, scopeLine: 8, flags: DIFlagPrototyped, isOptimized: true, unit: !0, variables: !2)
!23 = !DILocation(line: 9, column: 20, scope: !22)
!24 = !DILocation(line: 9, column: 3, scope: !22)
!25 = !DILocation(line: 12, column: 1, scope: !22)
!26 = !DILocation(line: 10, column: 12, scope: !22)
!27 = !DILocation(line: 10, column: 19, scope: !22)
!28 = !DILocation(line: 10, column: 17, scope: !22)
!29 = !DILocation(line: 10, column: 5, scope: !22)
!30 = !DILocation(line: 10, column: 10, scope: !22)
