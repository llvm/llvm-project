; RUN: opt -strip-dead-debug-info -disable-output %s
define i32 @foo() nounwind ssp {
entry:
  ret i32 0, !dbg !8
}

!llvm.dbg.sp = !{!0}
!llvm.dbg.gv = !{!6}

!0 = metadata !{i32 524334, i32 0, metadata !1, metadata !"foo", metadata !"foo", metadata !"foo", metadata !1, i32 3, metadata !3, i1 false, i1 true, i32 0, i32 0, null, i1 false, i1 false, i32 ()* @foo} ; [ DW_TAG_subprogram ]
!1 = metadata !{i32 524329, metadata !"/tmp/a.c", metadata !"/Volumes/Lalgate/clean/D.CW", metadata !2} ; [ DW_TAG_file_type ]
!2 = metadata !{i32 524305, i32 0, i32 12, metadata !"/tmp/a.c", metadata !"/Volumes/Lalgate/clean/D.CW", metadata !"clang version 2.8 (trunk 112062)", i1 true, i1 false, metadata !"", i32 0} ; [ DW_TAG_compile_unit ]
!3 = metadata !{i32 524309, metadata !1, metadata !"", metadata !1, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !4, i32 0, null} ; [ DW_TAG_subroutine_type ]
!4 = metadata !{metadata !5}
!5 = metadata !{i32 524324, metadata !1, metadata !"int", metadata !1, i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!6 = metadata !{i32 524340, i32 0, metadata !1, metadata !"i", metadata !"i", metadata !"i", metadata !1, i32 2, metadata !7, i1 true, i1 true, i32 0} ; [ DW_TAG_variable ]
!7 = metadata !{i32 524326, metadata !1, metadata !"", metadata !1, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !5} ; [ DW_TAG_const_type ]
!8 = metadata !{i32 3, i32 13, metadata !9, null}
!9 = metadata !{i32 524299, metadata !0, i32 3, i32 11, metadata !1, i32 0} ; [ DW_TAG_lexical_block ]
