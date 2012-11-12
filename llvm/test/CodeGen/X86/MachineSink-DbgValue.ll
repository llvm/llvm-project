; RUN: llc < %s | FileCheck %s
; Should sink matching DBG_VALUEs also.
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-macosx10.7.0"

define i32 @foo(i32 %i, i32* nocapture %c) nounwind uwtable readonly ssp {
  tail call void @llvm.dbg.value(metadata !{i32 %i}, i64 0, metadata !6), !dbg !12
  %ab = load i32* %c, align 1, !dbg !14
  tail call void @llvm.dbg.value(metadata !{i32* %c}, i64 0, metadata !7), !dbg !13
  tail call void @llvm.dbg.value(metadata !{i32 %ab}, i64 0, metadata !10), !dbg !14
  %cd = icmp eq i32 %i, 42, !dbg !15
  br i1 %cd, label %bb1, label %bb2, !dbg !15

bb1:                                     ; preds = %0
;CHECK: DEBUG_VALUE: a
;CHECK-NEXT: 	.loc	1 5 5
;CHECK-NEXT:	addl
  %gh = add nsw i32 %ab, 2, !dbg !16
  br label %bb2, !dbg !16

bb2:
  %.0 = phi i32 [ %gh, %bb1 ], [ 0, %0 ]
  ret i32 %.0, !dbg !17
}

declare void @llvm.dbg.value(metadata, i64, metadata) nounwind readnone

!llvm.dbg.cu = !{!0}
!llvm.dbg.sp = !{!1}
!llvm.dbg.lv.foo = !{!6, !7, !10}

!0 = metadata !{i32 589841, i32 0, i32 12, metadata !"a.c", metadata !"/private/tmp", metadata !"Apple clang version 3.0 (tags/Apple/clang-211.10.1) (based on LLVM 3.0svn)", i1 true, i1 true, metadata !"", i32 0} ; [ DW_TAG_compile_unit ]
!1 = metadata !{i32 589870, i32 0, metadata !2, metadata !"foo", metadata !"foo", metadata !"", metadata !2, i32 2, metadata !3, i1 false, i1 true, i32 0, i32 0, i32 0, i32 256, i1 true, i32 (i32, i32*)* @foo, null, null} ; [ DW_TAG_subprogram ]
!2 = metadata !{i32 589865, metadata !"a.c", metadata !"/private/tmp", metadata !0} ; [ DW_TAG_file_type ]
!3 = metadata !{i32 589845, metadata !2, metadata !"", metadata !2, i32 0, i64 0, i64 0, i32 0, i32 0, i32 0, metadata !4, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!4 = metadata !{metadata !5}
!5 = metadata !{i32 589860, metadata !0, metadata !"int", null, i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!6 = metadata !{i32 590081, metadata !1, metadata !"i", metadata !2, i32 16777218, metadata !5, i32 0} ; [ DW_TAG_arg_variable ]
!7 = metadata !{i32 590081, metadata !1, metadata !"c", metadata !2, i32 33554434, metadata !8, i32 0} ; [ DW_TAG_arg_variable ]
!8 = metadata !{i32 589839, metadata !0, metadata !"", null, i32 0, i64 64, i64 64, i64 0, i32 0, metadata !9} ; [ DW_TAG_pointer_type ]
!9 = metadata !{i32 589860, metadata !0, metadata !"char", null, i32 0, i64 8, i64 8, i64 0, i32 0, i32 6} ; [ DW_TAG_base_type ]
!10 = metadata !{i32 590080, metadata !11, metadata !"a", metadata !2, i32 3, metadata !9, i32 0} ; [ DW_TAG_auto_variable ]
!11 = metadata !{i32 589835, metadata !1, i32 2, i32 25, metadata !2, i32 0} ; [ DW_TAG_lexical_block ]
!12 = metadata !{i32 2, i32 13, metadata !1, null}
!13 = metadata !{i32 2, i32 22, metadata !1, null}
!14 = metadata !{i32 3, i32 14, metadata !11, null}
!15 = metadata !{i32 4, i32 3, metadata !11, null}
!16 = metadata !{i32 5, i32 5, metadata !11, null}
!17 = metadata !{i32 7, i32 1, metadata !11, null}
