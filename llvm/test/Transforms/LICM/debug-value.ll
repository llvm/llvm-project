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
  tail call void @llvm.dbg.value(metadata double undef, i64 0, metadata !19, metadata !{}), !dbg !21
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

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

!llvm.module.flags = !{!26}
!llvm.dbg.sp = !{!0, !6, !9, !10}

!0 = !{!"0x2e\00idamax\00idamax\00\00112\000\001\000\006\00256\000\000", !25, !1, !3, i32 0, null, null, null, null} ; [ DW_TAG_subprogram ]
!1 = !{!"0x29", !25} ; [ DW_TAG_file_type ]
!2 = !{!"0x11\0012\00clang version 2.9 (trunk 127169)\001\00\000\00\000", !25, !8, !8, !8, null, null} ; [ DW_TAG_compile_unit ]
!3 = !{!"0x15\00\000\000\000\000\000\000", !25, !1, null, !4, i32 0} ; [ DW_TAG_subroutine_type ]
!4 = !{!5}
!5 = !{!"0x24\00int\000\0032\0032\000\000\005", null, !2} ; [ DW_TAG_base_type ]
!6 = !{!"0x2e\00dscal\00dscal\00\00206\000\001\000\006\00256\000\000", !25, !1, !7, i32 0, null, null, null, null} ; [ DW_TAG_subprogram ]
!7 = !{!"0x15\00\000\000\000\000\000\000", !25, !1, null, !8, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = !{null}
!9 = !{!"0x2e\00daxpy\00daxpy\00\00230\000\001\000\006\00256\000\000", !25, !1, !7, i32 0, null, null, null, null} ; [ DW_TAG_subprogram ]
!10 = !{!"0x2e\00dgefa\00dgefa\00\00267\000\001\000\006\00256\000\000", !25, !1, !7, null, null, null, null, null} ; [ DW_TAG_subprogram ] [line 267] [def] [scope 0] [dgefa]
!11 = !MDLocation(line: 281, column: 9, scope: !12)
!12 = !{!"0xb\00272\005\0032", !25, !13} ; [ DW_TAG_lexical_block ]
!13 = !{!"0xb\00271\005\0031", !25, !14} ; [ DW_TAG_lexical_block ]
!14 = !{!"0xb\00267\001\0030", !25, !10} ; [ DW_TAG_lexical_block ]
!15 = !MDLocation(line: 271, column: 5, scope: !14)
!16 = !MDLocation(line: 284, column: 10, scope: !17)
!17 = !{!"0xb\00282\009\0033", !25, !12} ; [ DW_TAG_lexical_block ]
!18 = !{double undef}
!19 = !{!"0x100\00temp\00268\000", !14, !1, !20} ; [ DW_TAG_auto_variable ]
!20 = !{!"0x24\00double\000\0064\0064\000\000\004", null, !2} ; [ DW_TAG_base_type ]
!21 = !MDLocation(line: 286, column: 14, scope: !22)
!22 = !{!"0xb\00285\0013\0034", !25, !17} ; [ DW_TAG_lexical_block ]
!23 = !MDLocation(line: 296, column: 13, scope: !17)
!24 = !MDLocation(line: 313, column: 1, scope: !14)
!25 = !{!"/Volumes/Lalgate/work/llvm/projects/llvm-test/SingleSource/Benchmarks/CoyoteBench/lpbench.c", !"/private/tmp"}
!26 = !{i32 1, !"Debug Info Version", i32 2}
