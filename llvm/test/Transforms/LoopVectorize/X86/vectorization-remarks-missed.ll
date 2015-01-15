; RUN: opt < %s -loop-vectorize -S -pass-remarks-missed='loop-vectorize' -pass-remarks-analysis='loop-vectorize' 2>&1 | FileCheck %s

; C/C++ code for tests
; void test(int *A, int Length) {
; #pragma clang loop vectorize(enable) interleave(enable)
;   for (int i = 0; i < Length; i++) {
;     A[i] = i;
;     if (A[i] > Length)
;       break;
;   }
; }

; void test_disabled(int *A, int Length) {
; #pragma clang loop vectorize(disable) interleave(disable)
;   for (int i = 0; i < Length; i++)
;     A[i] = i;
; }

; void test_array_bounds(int *A, int *B, int Length) {
; #pragma clang loop vectorize(enable)
;   for (int i = 0; i < Length; i++)
;     A[i] = A[B[i]];
; }

; File, line, and column should match those specified in the metadata
; CHECK: remark: source.cpp:4:5: loop not vectorized: could not determine number of loop iterations
; CHECK: remark: source.cpp:4:5: loop not vectorized: use -Rpass-analysis=loop-vectorize for more info
; CHECK: remark: source.cpp:13:5: loop not vectorized: vector width and interleave count are explicitly set to 1
; CHECK: remark: source.cpp:19:5: loop not vectorized: cannot identify array bounds
; CHECK: remark: source.cpp:19:5: loop not vectorized: use -Rpass-analysis=loop-vectorize for more info
; CHECK: warning: source.cpp:19:5: loop not vectorized: failed explicitly specified loop vectorization

; CHECK: _Z4testPii
; CHECK-NOT: x i32>
; CHECK: ret

; CHECK: _Z13test_disabledPii
; CHECK-NOT: x i32>
; CHECK: ret

; CHECK: _Z17test_array_boundsPiS_i
; CHECK-NOT: x i32>
; CHECK: ret

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

; Function Attrs: nounwind optsize ssp uwtable
define void @_Z4testPii(i32* nocapture %A, i32 %Length) #0 {
entry:
  %cmp10 = icmp sgt i32 %Length, 0, !dbg !12
  br i1 %cmp10, label %for.body, label %for.end, !dbg !12, !llvm.loop !14

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32* %A, i64 %indvars.iv, !dbg !16
  %0 = trunc i64 %indvars.iv to i32, !dbg !16
  store i32 %0, i32* %arrayidx, align 4, !dbg !16, !tbaa !18
  %cmp3 = icmp sle i32 %0, %Length, !dbg !22
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !12
  %1 = trunc i64 %indvars.iv.next to i32
  %cmp = icmp slt i32 %1, %Length, !dbg !12
  %or.cond = and i1 %cmp3, %cmp, !dbg !22
  br i1 %or.cond, label %for.body, label %for.end, !dbg !22

for.end:                                          ; preds = %for.body, %entry
  ret void, !dbg !24
}

; Function Attrs: nounwind optsize ssp uwtable
define void @_Z13test_disabledPii(i32* nocapture %A, i32 %Length) #0 {
entry:
  %cmp4 = icmp sgt i32 %Length, 0, !dbg !25
  br i1 %cmp4, label %for.body, label %for.end, !dbg !25, !llvm.loop !27

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32* %A, i64 %indvars.iv, !dbg !30
  %0 = trunc i64 %indvars.iv to i32, !dbg !30
  store i32 %0, i32* %arrayidx, align 4, !dbg !30, !tbaa !18
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !25
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32, !dbg !25
  %exitcond = icmp eq i32 %lftr.wideiv, %Length, !dbg !25
  br i1 %exitcond, label %for.end, label %for.body, !dbg !25, !llvm.loop !27

for.end:                                          ; preds = %for.body, %entry
  ret void, !dbg !31
}

; Function Attrs: nounwind optsize ssp uwtable
define void @_Z17test_array_boundsPiS_i(i32* nocapture %A, i32* nocapture readonly %B, i32 %Length) #0 {
entry:
  %cmp9 = icmp sgt i32 %Length, 0, !dbg !32
  br i1 %cmp9, label %for.body.preheader, label %for.end, !dbg !32, !llvm.loop !34

for.body.preheader:                               ; preds = %entry
  br label %for.body, !dbg !35

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %for.body.preheader ]
  %arrayidx = getelementptr inbounds i32* %B, i64 %indvars.iv, !dbg !35
  %0 = load i32* %arrayidx, align 4, !dbg !35, !tbaa !18
  %idxprom1 = sext i32 %0 to i64, !dbg !35
  %arrayidx2 = getelementptr inbounds i32* %A, i64 %idxprom1, !dbg !35
  %1 = load i32* %arrayidx2, align 4, !dbg !35, !tbaa !18
  %arrayidx4 = getelementptr inbounds i32* %A, i64 %indvars.iv, !dbg !35
  store i32 %1, i32* %arrayidx4, align 4, !dbg !35, !tbaa !18
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !32
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32, !dbg !32
  %exitcond = icmp eq i32 %lftr.wideiv, %Length, !dbg !32
  br i1 %exitcond, label %for.end.loopexit, label %for.body, !dbg !32, !llvm.loop !34

for.end.loopexit:                                 ; preds = %for.body
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %entry
  ret void, !dbg !36
}

attributes #0 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!9, !10}
!llvm.ident = !{!11}

!0 = !{!"0x11\004\00clang version 3.5.0\001\00\006\00\002", !1, !2, !2, !3, !2, !2} ; [ DW_TAG_compile_unit ] [./source.cpp] [DW_LANG_C_plus_plus]
!1 = !{!"source.cpp", !"."}
!2 = !{}
!3 = !{!4, !7, !8}
!4 = !{!"0x2e\00test\00test\00\001\000\001\000\006\00256\001\001", !1, !5, !6, null, void (i32*, i32)* @_Z4testPii, null, null, !2} ; [ DW_TAG_subprogram ] [line 1] [def] [test]
!5 = !{!"0x29", !1}          ; [ DW_TAG_file_type ] [./source.cpp]
!6 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !2, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = !{!"0x2e\00test_disabled\00test_disabled\00\0010\000\001\000\006\00256\001\0010", !1, !5, !6, null, void (i32*, i32)* @_Z13test_disabledPii, null, null, !2} ; [ DW_TAG_subprogram ] [line 10] [def] [test_disabled]
!8 = !{!"0x2e\00test_array_bounds\00test_array_bounds\00\0016\000\001\000\006\00256\001\0016", !1, !5, !6, null, void (i32*, i32*, i32)* @_Z17test_array_boundsPiS_i, null, null, !2} ; [ DW_TAG_subprogram ] [line 16] [def] [test_array_bounds]
!9 = !{i32 2, !"Dwarf Version", i32 2}
!10 = !{i32 2, !"Debug Info Version", i32 2}
!11 = !{!"clang version 3.5.0"}
!12 = !MDLocation(line: 3, column: 8, scope: !13)
!13 = !{!"0xb\003\003\000", !1, !4} ; [ DW_TAG_lexical_block ]
!14 = !{!14, !15, !15}
!15 = !{!"llvm.loop.vectorize.enable", i1 true}
!16 = !MDLocation(line: 4, column: 5, scope: !17)
!17 = !{!"0xb\003\0036\000", !1, !13} ; [ DW_TAG_lexical_block ]
!18 = !{!19, !19, i64 0}
!19 = !{!"int", !20, i64 0}
!20 = !{!"omnipotent char", !21, i64 0}
!21 = !{!"Simple C/C++ TBAA"}
!22 = !MDLocation(line: 5, column: 9, scope: !23)
!23 = !{!"0xb\005\009\000", !1, !17} ; [ DW_TAG_lexical_block ]
!24 = !MDLocation(line: 8, column: 1, scope: !4)
!25 = !MDLocation(line: 12, column: 8, scope: !26)
!26 = !{!"0xb\0012\003\000", !1, !7} ; [ DW_TAG_lexical_block ]
!27 = !{!27, !28, !29}
!28 = !{!"llvm.loop.interleave.count", i32 1}
!29 = !{!"llvm.loop.vectorize.width", i32 1}
!30 = !MDLocation(line: 13, column: 5, scope: !26)
!31 = !MDLocation(line: 14, column: 1, scope: !7)
!32 = !MDLocation(line: 18, column: 8, scope: !33)
!33 = !{!"0xb\0018\003\000", !1, !8} ; [ DW_TAG_lexical_block ]
!34 = !{!34, !15}
!35 = !MDLocation(line: 19, column: 5, scope: !33)
!36 = !MDLocation(line: 20, column: 1, scope: !8)
