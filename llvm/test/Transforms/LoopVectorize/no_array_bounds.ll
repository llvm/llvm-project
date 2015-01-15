; RUN: opt < %s -loop-vectorize -S 2>&1 | FileCheck %s

; Verify warning is generated when vectorization/ interleaving is explicitly specified and fails to occur.
; CHECK: warning: no_array_bounds.cpp:5:5: loop not vectorized: failed explicitly specified loop vectorization
; CHECK: warning: no_array_bounds.cpp:10:5: loop not interleaved: failed explicitly specified loop interleaving

;  #pragma clang loop vectorize(enable)
;  for (int i = 0; i < number; i++) {
;    A[B[i]]++;
;  }

;  #pragma clang loop vectorize(disable) interleave(enable)
;  for (int i = 0; i < number; i++) {
;    B[A[i]]++;
;  }

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

; Function Attrs: nounwind ssp uwtable
define void @_Z4testPiS_i(i32* nocapture %A, i32* nocapture %B, i32 %number) #0 {
entry:
  %cmp25 = icmp sgt i32 %number, 0, !dbg !10
  br i1 %cmp25, label %for.body.preheader, label %for.end15, !dbg !10, !llvm.loop !12

for.body.preheader:                               ; preds = %entry
  br label %for.body, !dbg !14

for.cond5.preheader:                              ; preds = %for.body
  br i1 %cmp25, label %for.body7.preheader, label %for.end15, !dbg !16, !llvm.loop !18

for.body7.preheader:                              ; preds = %for.cond5.preheader
  br label %for.body7, !dbg !20

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv27 = phi i64 [ %indvars.iv.next28, %for.body ], [ 0, %for.body.preheader ]
  %arrayidx = getelementptr inbounds i32* %B, i64 %indvars.iv27, !dbg !14
  %0 = load i32* %arrayidx, align 4, !dbg !14, !tbaa !22
  %idxprom1 = sext i32 %0 to i64, !dbg !14
  %arrayidx2 = getelementptr inbounds i32* %A, i64 %idxprom1, !dbg !14
  %1 = load i32* %arrayidx2, align 4, !dbg !14, !tbaa !22
  %inc = add nsw i32 %1, 1, !dbg !14
  store i32 %inc, i32* %arrayidx2, align 4, !dbg !14, !tbaa !22
  %indvars.iv.next28 = add nuw nsw i64 %indvars.iv27, 1, !dbg !10
  %lftr.wideiv29 = trunc i64 %indvars.iv.next28 to i32, !dbg !10
  %exitcond30 = icmp eq i32 %lftr.wideiv29, %number, !dbg !10
  br i1 %exitcond30, label %for.cond5.preheader, label %for.body, !dbg !10, !llvm.loop !12

for.body7:                                        ; preds = %for.body7.preheader, %for.body7
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body7 ], [ 0, %for.body7.preheader ]
  %arrayidx9 = getelementptr inbounds i32* %A, i64 %indvars.iv, !dbg !20
  %2 = load i32* %arrayidx9, align 4, !dbg !20, !tbaa !22
  %idxprom10 = sext i32 %2 to i64, !dbg !20
  %arrayidx11 = getelementptr inbounds i32* %B, i64 %idxprom10, !dbg !20
  %3 = load i32* %arrayidx11, align 4, !dbg !20, !tbaa !22
  %inc12 = add nsw i32 %3, 1, !dbg !20
  store i32 %inc12, i32* %arrayidx11, align 4, !dbg !20, !tbaa !22
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !16
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32, !dbg !16
  %exitcond = icmp eq i32 %lftr.wideiv, %number, !dbg !16
  br i1 %exitcond, label %for.end15.loopexit, label %for.body7, !dbg !16, !llvm.loop !18

for.end15.loopexit:                               ; preds = %for.body7
  br label %for.end15

for.end15:                                        ; preds = %for.end15.loopexit, %entry, %for.cond5.preheader
  ret void, !dbg !26
}

attributes #0 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!7, !8}
!llvm.ident = !{!9}

!0 = !{!"0x11\004\00clang version 3.5.0\001\00\000\00\002", !1, !2, !2, !3, !2, !2} ; [ DW_TAG_compile_unit ]
!1 = !{!"no_array_bounds.cpp", !"."}
!2 = !{}
!3 = !{!4}
!4 = !{!"0x2e\00test\00test\00\001\000\001\000\006\00256\001\002", !1, !5, !6, null, void (i32*, i32*, i32)* @_Z4testPiS_i, null, null, !2} ; [ DW_TAG_subprogram ]
!5 = !{!"0x29", !1} ; [ DW_TAG_file_type ]
!6 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !2, null, null, null} ; [ DW_TAG_subroutine_type ]
!7 = !{i32 2, !"Dwarf Version", i32 2}
!8 = !{i32 2, !"Debug Info Version", i32 2}
!9 = !{!"clang version 3.5.0"}
!10 = !MDLocation(line: 4, column: 8, scope: !11)
!11 = !{!"0xb\004\003\000", !1, !4} ; [ DW_TAG_lexical_block ]
!12 = !{!12, !13}
!13 = !{!"llvm.loop.vectorize.enable", i1 true}
!14 = !MDLocation(line: 5, column: 5, scope: !15)
!15 = !{!"0xb\004\0036\000", !1, !11} ; [ DW_TAG_lexical_block ]
!16 = !MDLocation(line: 9, column: 8, scope: !17)
!17 = !{!"0xb\009\003\000", !1, !4} ; [ DW_TAG_lexical_block ]
!18 = !{!18, !13, !19}
!19 = !{!"llvm.loop.vectorize.width", i32 1}
!20 = !MDLocation(line: 10, column: 5, scope: !21)
!21 = !{!"0xb\009\0036\000", !1, !17} ; [ DW_TAG_lexical_block ]
!22 = !{!23, !23, i64 0}
!23 = !{!"int", !24, i64 0}
!24 = !{!"omnipotent char", !25, i64 0}
!25 = !{!"Simple C/C++ TBAA"}
!26 = !MDLocation(line: 12, column: 1, scope: !4)
