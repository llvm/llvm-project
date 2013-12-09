; RUN: opt -licm -basicaa < %s -S | FileCheck %s

define void @dgefa() nounwind ssp {
entry:
  br label %for.body

for.body:                                         ; preds = %for.cond.backedge, %entry
  br i1 undef, label %if.then, label %for.cond.backedge, !dbg !11

for.cond.backedge:                                ; preds = %for.body61, %for.body61.us, %for.body
  br i1 undef, label %for.end104, label %for.body, !dbg !15

if.then:                                          ; preds = %for.body
  br i1 undef, label %if.then27, label %if.end.if.end.split_crit_edge.critedge, !dbg !16

if.then27:                                        ; preds = %if.then
; CHECK: tail call void @llvm.dbg.value
  tail call void @llvm.dbg.value(metadata !18, i64 0, metadata !19), !dbg !21
  br label %for.body61.us

if.end.if.end.split_crit_edge.critedge:           ; preds = %if.then
  br label %for.body61

for.body61.us:                                    ; preds = %for.body61.us, %if.then27
  br i1 undef, label %for.cond.backedge, label %for.body61.us, !dbg !23

for.body61:                                       ; preds = %for.body61, %if.end.if.end.split_crit_edge.critedge
  br i1 undef, label %for.cond.backedge, label %for.body61, !dbg !23

for.end104:                                       ; preds = %for.cond.backedge
  ret void, !dbg !24
}

declare void @llvm.dbg.value(metadata, i64, metadata) nounwind readnone

!llvm.module.flags = !{!26}
!llvm.dbg.sp = !{!0, !6, !9, !10}

!0 = metadata !{i32 589870, metadata !25, metadata !1, metadata !"idamax", metadata !"idamax", metadata !"", i32 112, metadata !3, i1 false, i1 true, i32 0, i32 0, i32 0, i32 256, i1 false, null, null, null, null, i32 0} ; [ DW_TAG_subprogram ]
!1 = metadata !{i32 589865, metadata !25} ; [ DW_TAG_file_type ]
!2 = metadata !{i32 589841, metadata !25, i32 12, metadata !"clang version 2.9 (trunk 127169)", i1 true, metadata !"", i32 0, metadata !8, metadata !8, metadata !8, null, null, metadata !""} ; [ DW_TAG_compile_unit ]
!3 = metadata !{i32 589845, metadata !25, metadata !1, metadata !"", i32 0, i64 0, i64 0, i32 0, i32 0, null, metadata !4, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!4 = metadata !{metadata !5}
!5 = metadata !{i32 589860, null, metadata !2, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!6 = metadata !{i32 589870, metadata !25, metadata !1, metadata !"dscal", metadata !"dscal", metadata !"", i32 206, metadata !7, i1 false, i1 true, i32 0, i32 0, i32 0, i32 256, i1 false, null, null, null, null, i32 0} ; [ DW_TAG_subprogram ]
!7 = metadata !{i32 589845, metadata !25, metadata !1, metadata !"", i32 0, i64 0, i64 0, i32 0, i32 0, null, metadata !8, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = metadata !{null}
!9 = metadata !{i32 589870, metadata !25, metadata !1, metadata !"daxpy", metadata !"daxpy", metadata !"", i32 230, metadata !7, i1 false, i1 true, i32 0, i32 0, i32 0, i32 256, i1 false, null, null, null, null, i32 0} ; [ DW_TAG_subprogram ]
!10 = metadata !{i32 589870, metadata !25, metadata !1, metadata !"dgefa", metadata !"dgefa", metadata !"", i32 267, metadata !7, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, null, null, null, null, i32 0} ; [ DW_TAG_subprogram ] [line 267] [def] [scope 0] [dgefa]
!11 = metadata !{i32 281, i32 9, metadata !12, null}
!12 = metadata !{i32 589835, metadata !25, metadata !13, i32 272, i32 5, i32 32} ; [ DW_TAG_lexical_block ]
!13 = metadata !{i32 589835, metadata !25, metadata !14, i32 271, i32 5, i32 31} ; [ DW_TAG_lexical_block ]
!14 = metadata !{i32 589835, metadata !25, metadata !10, i32 267, i32 1, i32 30} ; [ DW_TAG_lexical_block ]
!15 = metadata !{i32 271, i32 5, metadata !14, null}
!16 = metadata !{i32 284, i32 10, metadata !17, null}
!17 = metadata !{i32 589835, metadata !25, metadata !12, i32 282, i32 9, i32 33} ; [ DW_TAG_lexical_block ]
!18 = metadata !{double undef}
!19 = metadata !{i32 590080, metadata !14, metadata !"temp", metadata !1, i32 268, metadata !20, i32 0} ; [ DW_TAG_auto_variable ]
!20 = metadata !{i32 589860, null, metadata !2, metadata !"double", i32 0, i64 64, i64 64, i64 0, i32 0, i32 4} ; [ DW_TAG_base_type ]
!21 = metadata !{i32 286, i32 14, metadata !22, null}
!22 = metadata !{i32 589835, metadata !25, metadata !17, i32 285, i32 13, i32 34} ; [ DW_TAG_lexical_block ]
!23 = metadata !{i32 296, i32 13, metadata !17, null}
!24 = metadata !{i32 313, i32 1, metadata !14, null}
!25 = metadata !{metadata !"/Volumes/Lalgate/work/llvm/projects/llvm-test/SingleSource/Benchmarks/CoyoteBench/lpbench.c", metadata !"/private/tmp"}
!26 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
