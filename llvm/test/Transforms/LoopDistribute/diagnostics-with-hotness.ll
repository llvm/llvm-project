; RUN: opt -loop-distribute -S -pass-remarks-missed=loop-distribute \
; RUN:     -pass-remarks-with-hotness < %s 2>&1 | FileCheck %s --check-prefix=HOTNESS
; RUN: opt -loop-distribute -S -pass-remarks-missed=loop-distribute \
; RUN:                                < %s 2>&1 | FileCheck %s --check-prefix=NO_HOTNESS

; RUN: opt -passes='require<aa>,loop-distribute' -S -pass-remarks-missed=loop-distribute \
; RUN:     -pass-remarks-with-hotness < %s 2>&1 | FileCheck %s --check-prefix=HOTNESS
; RUN: opt -passes='require<aa>,loop-distribute' -S -pass-remarks-missed=loop-distribute \
; RUN:                                < %s 2>&1 | FileCheck %s --check-prefix=NO_HOTNESS

; This is the input program:
;
;     1	void forced (char *A, char *B, char *C, int N) {
;     2	#pragma clang loop distribute(enable)
;     3	  for(int i = 0; i < N; i++) {
;     4	    A[i] = B[i] * C[i];
;     5	  }
;     6	}

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.11.0"

; HOTNESS: remark: /tmp/t.c:3:3: loop not distributed: use -Rpass-analysis=loop-distribute for more info (hotness: 300)
; NO_HOTNESS: remark: /tmp/t.c:3:3: loop not distributed: use -Rpass-analysis=loop-distribute for more info{{$}}

define void @forced(i8* %A, i8* %B, i8* %C, i32 %N) !dbg !7 !prof !22 {
entry:
  %cmp12 = icmp sgt i32 %N, 0, !dbg !9
  br i1 %cmp12, label %ph, label %for.cond.cleanup, !dbg !10, !prof !23

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
  br i1 %exitcond, label %for.cond.cleanup, label %for.body, !dbg !10, !llvm.loop !20, !prof !24

for.cond.cleanup:
  ret void, !dbg !11
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
!22 = !{!"function_entry_count", i64 3}
!23 = !{!"branch_weights", i32 99, i32 1}
!24 = !{!"branch_weights", i32 1, i32 99}
