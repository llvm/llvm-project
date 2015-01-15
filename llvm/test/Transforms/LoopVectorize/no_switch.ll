; RUN: opt < %s -loop-vectorize -force-vector-width=4 -S -pass-remarks-missed='loop-vectorize' -pass-remarks-analysis='loop-vectorize' 2>&1 | FileCheck %s

; CHECK: remark: source.cpp:4:5: loop not vectorized: loop contains a switch statement
; CHECK: remark: source.cpp:4:5: loop not vectorized: use -Rpass-analysis=loop-vectorize for more info (Force=true, Vector Width=4)
; CHECK: warning: source.cpp:4:5: loop not vectorized: failed explicitly specified loop vectorization

; CHECK: _Z11test_switchPii
; CHECK-NOT: x i32>
; CHECK: ret

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

; Function Attrs: nounwind optsize ssp uwtable
define void @_Z11test_switchPii(i32* nocapture %A, i32 %Length) #0 {
entry:
  %cmp18 = icmp sgt i32 %Length, 0, !dbg !10
  br i1 %cmp18, label %for.body.preheader, label %for.end, !dbg !10, !llvm.loop !12

for.body.preheader:                               ; preds = %entry
  br label %for.body, !dbg !14

for.body:                                         ; preds = %for.body.preheader, %for.inc
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %for.body.preheader ]
  %arrayidx = getelementptr inbounds i32* %A, i64 %indvars.iv, !dbg !14
  %0 = load i32* %arrayidx, align 4, !dbg !14, !tbaa !16
  switch i32 %0, label %for.inc [
    i32 0, label %sw.bb
    i32 1, label %sw.bb3
  ], !dbg !14

sw.bb:                                            ; preds = %for.body
  %1 = trunc i64 %indvars.iv to i32, !dbg !20
  %mul = shl nsw i32 %1, 1, !dbg !20
  br label %for.inc, !dbg !22

sw.bb3:                                           ; preds = %for.body
  %2 = trunc i64 %indvars.iv to i32, !dbg !23
  store i32 %2, i32* %arrayidx, align 4, !dbg !23, !tbaa !16
  br label %for.inc, !dbg !23

for.inc:                                          ; preds = %sw.bb3, %for.body, %sw.bb
  %storemerge = phi i32 [ %mul, %sw.bb ], [ 0, %for.body ], [ 0, %sw.bb3 ]
  store i32 %storemerge, i32* %arrayidx, align 4, !dbg !20, !tbaa !16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !10
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32, !dbg !10
  %exitcond = icmp eq i32 %lftr.wideiv, %Length, !dbg !10
  br i1 %exitcond, label %for.end.loopexit, label %for.body, !dbg !10, !llvm.loop !12

for.end.loopexit:                                 ; preds = %for.inc
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %entry
  ret void, !dbg !24
}

attributes #0 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!7, !8}
!llvm.ident = !{!9}

!0 = !{!"0x11\004\00clang version 3.5.0\001\00\006\00\002", !1, !2, !2, !3, !2, !2} ; [ DW_TAG_compile_unit ] [./source.cpp] [DW_LANG_C_plus_plus]
!1 = !{!"source.cpp", !"."}
!2 = !{}
!3 = !{!4}
!4 = !{!"0x2e\00test_switch\00test_switch\00\001\000\001\000\006\00256\001\001", !1, !5, !6, null, void (i32*, i32)* @_Z11test_switchPii, null, null, !2} ; [ DW_TAG_subprogram ] [line 1] [def] [test_switch]
!5 = !{!"0x29", !1}          ; [ DW_TAG_file_type ] [./source.cpp]
!6 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !2, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = !{i32 2, !"Dwarf Version", i32 2}
!8 = !{i32 2, !"Debug Info Version", i32 2}
!9 = !{!"clang version 3.5.0"}
!10 = !MDLocation(line: 3, column: 8, scope: !11)
!11 = !{!"0xb\003\003\000", !1, !4} ; [ DW_TAG_lexical_block ]
!12 = !{!12, !13, !13}
!13 = !{!"llvm.loop.vectorize.enable", i1 true}
!14 = !MDLocation(line: 4, column: 5, scope: !15)
!15 = !{!"0xb\003\0036\000", !1, !11} ; [ DW_TAG_lexical_block ]
!16 = !{!17, !17, i64 0}
!17 = !{!"int", !18, i64 0}
!18 = !{!"omnipotent char", !19, i64 0}
!19 = !{!"Simple C/C++ TBAA"}
!20 = !MDLocation(line: 6, column: 7, scope: !21)
!21 = !{!"0xb\004\0018\000", !1, !15} ; [ DW_TAG_lexical_block ]
!22 = !MDLocation(line: 7, column: 5, scope: !21)
!23 = !MDLocation(line: 9, column: 7, scope: !21)
!24 = !MDLocation(line: 14, column: 1, scope: !4)
