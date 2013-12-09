; RUN: opt -loop-idiom < %s -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin10.0.0"


define void @foo(double* nocapture %a) nounwind ssp {
entry:
  tail call void @llvm.dbg.value(metadata !{double* %a}, i64 0, metadata !5), !dbg !8
  tail call void @llvm.dbg.value(metadata !9, i64 0, metadata !10), !dbg !14
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %indvar = phi i64 [ 0, %entry ], [ %indvar.next, %for.body ]
  %arrayidx = getelementptr double* %a, i64 %indvar
; CHECK: call void @llvm.memset{{.+}} !dbg 
  store double 0.000000e+00, double* %arrayidx, align 8, !dbg !15
  %indvar.next = add i64 %indvar, 1
  %exitcond = icmp ne i64 %indvar.next, 1000
  br i1 %exitcond, label %for.body, label %for.end, !dbg !14

for.end:                                          ; preds = %for.body
  tail call void @llvm.dbg.value(metadata !{null}, i64 0, metadata !10), !dbg !16
  ret void, !dbg !17
}

declare void @llvm.dbg.declare(metadata, metadata) nounwind readnone

declare void @llvm.dbg.value(metadata, i64, metadata) nounwind readnone

!llvm.module.flags = !{!19}
!llvm.dbg.sp = !{!0}

!0 = metadata !{i32 589870, metadata !18, metadata !1, metadata !"foo", metadata !"foo", metadata !"", i32 2, metadata !3, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void (double*)* @foo, null, null, null, i32 0} ; [ DW_TAG_subprogram ] [line 2] [def] [scope 0] [foo]
!1 = metadata !{i32 589865, metadata !18} ; [ DW_TAG_file_type ]
!2 = metadata !{i32 589841, metadata !18, i32 12, metadata !"clang version 2.9 (trunk 127165:127174)", i1 true, metadata !"", i32 0, metadata !9, metadata !9, null, null, null, metadata !""} ; [ DW_TAG_compile_unit ]
!3 = metadata !{i32 589845, metadata !18, metadata !1, metadata !"", i32 0, i64 0, i64 0, i32 0, i32 0, null, metadata !4, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!4 = metadata !{null}
!5 = metadata !{i32 590081, metadata !0, metadata !"a", metadata !1, i32 16777218, metadata !6, i32 0} ; [ DW_TAG_arg_variable ]
!6 = metadata !{i32 589839, null, metadata !2, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, metadata !7} ; [ DW_TAG_pointer_type ]
!7 = metadata !{i32 589860, null, metadata !2, metadata !"double", i32 0, i64 64, i64 64, i64 0, i32 0, i32 4} ; [ DW_TAG_base_type ]
!8 = metadata !{i32 2, i32 18, metadata !0, null}
!9 = metadata !{i32 0}
!10 = metadata !{i32 590080, metadata !11, metadata !"i", metadata !1, i32 3, metadata !13, i32 0} ; [ DW_TAG_auto_variable ]
!11 = metadata !{i32 589835, metadata !18, metadata !12, i32 3, i32 3, i32 1} ; [ DW_TAG_lexical_block ]
!12 = metadata !{i32 589835, metadata !18, metadata !0, i32 2, i32 21, i32 0} ; [ DW_TAG_lexical_block ]
!13 = metadata !{i32 589860, null, metadata !2, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!14 = metadata !{i32 3, i32 3, metadata !12, null}
!15 = metadata !{i32 4, i32 5, metadata !11, null}
!16 = metadata !{i32 3, i32 29, metadata !11, null}
!17 = metadata !{i32 5, i32 1, metadata !12, null}
!18 = metadata !{metadata !"li.c", metadata !"/private/tmp"}
!19 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
