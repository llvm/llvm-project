; RUN: llc < %s | FileCheck %s
; RUN: llc < %s -regalloc=basic | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin8"

;CHECK: Ldebug_loc0:
;CHECK-NEXT:	.quad	Lfunc_begin0
;CHECK-NEXT:	.quad	L
;CHECK-NEXT: Lset{{.*}} = Ltmp{{.*}}-Ltmp{{.*}}          ## Loc expr size
;CHECK-NEXT:    .short  Lset
;CHECK-NEXT: Ltmp
;CHECK-NEXT:	.byte	85                      ## DW_OP_reg5
;CHECK-NEXT: Ltmp
;CHECK-NEXT:	.quad	0
;CHECK-NEXT:	.quad	0

%0 = type { i64, i1 }

@__clz_tab = external constant [256 x i8]

define hidden i128 @__divti3(i128 %u, i128 %v) nounwind readnone {
entry:
  tail call void @llvm.dbg.value(metadata !{i128 %u}, i64 0, metadata !14), !dbg !15
  tail call void @llvm.dbg.value(metadata !16, i64 0, metadata !17), !dbg !21
  br i1 undef, label %bb2, label %bb4, !dbg !22

bb2:                                              ; preds = %entry
  br label %bb4, !dbg !23

bb4:                                              ; preds = %bb2, %entry
  br i1 undef, label %__udivmodti4.exit, label %bb82.i, !dbg !24

bb82.i:                                           ; preds = %bb4
  unreachable

__udivmodti4.exit:                                ; preds = %bb4
  ret i128 undef, !dbg !27
}

declare void @llvm.dbg.declare(metadata, metadata) nounwind readnone

declare void @llvm.dbg.value(metadata, i64, metadata) nounwind readnone

declare %0 @llvm.uadd.with.overflow.i64(i64, i64) nounwind readnone

!llvm.dbg.sp = !{!0, !9}

!0 = metadata !{i32 589870, i32 0, metadata !1, metadata !"__udivmodti4", metadata !"__udivmodti4", metadata !"", metadata !1, i32 879, metadata !3, i1 true, i1 true, i32 0, i32 0, null, i32 256, i1 true, null} ; [ DW_TAG_subprogram ]
!1 = metadata !{i32 589865, metadata !"foobar.c", metadata !"/tmp", metadata !2} ; [ DW_TAG_file_type ]
!2 = metadata !{i32 589841, i32 0, i32 1, metadata !"foobar.c", metadata !"/tmp", metadata !"4.2.1 (Based on Apple Inc. build 5658) (LLVM build)", i1 true, i1 true, metadata !"", i32 0} ; [ DW_TAG_compile_unit ]
!3 = metadata !{i32 589845, metadata !1, metadata !"", metadata !1, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !4, i32 0, null} ; [ DW_TAG_subroutine_type ]
!4 = metadata !{metadata !5, metadata !5, metadata !5, metadata !8}
!5 = metadata !{i32 589846, metadata !6, metadata !"UTItype", metadata !6, i32 166, i64 0, i64 0, i64 0, i32 0, metadata !7} ; [ DW_TAG_typedef ]
!6 = metadata !{i32 589865, metadata !"foobar.h", metadata !"/tmp", metadata !2} ; [ DW_TAG_file_type ]
!7 = metadata !{i32 589860, metadata !1, metadata !"", metadata !1, i32 0, i64 128, i64 128, i64 0, i32 0, i32 7} ; [ DW_TAG_base_type ]
!8 = metadata !{i32 589839, metadata !1, metadata !"", metadata !1, i32 0, i64 64, i64 64, i64 0, i32 0, metadata !5} ; [ DW_TAG_pointer_type ]
!9 = metadata !{i32 589870, i32 0, metadata !1, metadata !"__divti3", metadata !"__divti3", metadata !"__divti3", metadata !1, i32 1094, metadata !10, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 true, i128 (i128, i128)* @__divti3} ; [ DW_TAG_subprogram ]
!10 = metadata !{i32 589845, metadata !1, metadata !"", metadata !1, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !11, i32 0, null} ; [ DW_TAG_subroutine_type ]
!11 = metadata !{metadata !12, metadata !12, metadata !12}
!12 = metadata !{i32 589846, metadata !6, metadata !"TItype", metadata !6, i32 160, i64 0, i64 0, i64 0, i32 0, metadata !13} ; [ DW_TAG_typedef ]
!13 = metadata !{i32 589860, metadata !1, metadata !"", metadata !1, i32 0, i64 128, i64 128, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!14 = metadata !{i32 590081, metadata !9, metadata !"u", metadata !1, i32 1093, metadata !12, i32 0} ; [ DW_TAG_arg_variable ]
!15 = metadata !{i32 1093, i32 0, metadata !9, null}
!16 = metadata !{i64 0}
!17 = metadata !{i32 590080, metadata !18, metadata !"c", metadata !1, i32 1095, metadata !19, i32 0} ; [ DW_TAG_auto_variable ]
!18 = metadata !{i32 589835, metadata !9, i32 1094, i32 0, metadata !1, i32 13} ; [ DW_TAG_lexical_block ]
!19 = metadata !{i32 589846, metadata !6, metadata !"word_type", metadata !6, i32 424, i64 0, i64 0, i64 0, i32 0, metadata !20} ; [ DW_TAG_typedef ]
!20 = metadata !{i32 589860, metadata !1, metadata !"long int", metadata !1, i32 0, i64 64, i64 64, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!21 = metadata !{i32 1095, i32 0, metadata !18, null}
!22 = metadata !{i32 1103, i32 0, metadata !18, null}
!23 = metadata !{i32 1104, i32 0, metadata !18, null}
!24 = metadata !{i32 1003, i32 0, metadata !25, metadata !26}
!25 = metadata !{i32 589835, metadata !0, i32 879, i32 0, metadata !1, i32 0} ; [ DW_TAG_lexical_block ]
!26 = metadata !{i32 1107, i32 0, metadata !18, null}
!27 = metadata !{i32 1111, i32 0, metadata !18, null}
