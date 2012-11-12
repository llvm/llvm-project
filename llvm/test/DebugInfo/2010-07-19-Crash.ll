; RUN: llc -o /dev/null < %s
; PR7662
; Do not add variables to !11 because it is a declaration entry.

define i32 @bar() nounwind readnone ssp {
entry:
  ret i32 42, !dbg !9
}

!llvm.dbg.sp = !{!0, !6, !11}
!llvm.dbg.lv.foo = !{!7}

!0 = metadata !{i32 524334, i32 0, metadata !1, metadata !"bar", metadata !"bar", metadata !"bar", metadata !1, i32 3, metadata !3, i1 false, i1 true, i32 0, i32 0, null, i1 false, i1 true, i32 ()* @bar} ; [ DW_TAG_subprogram ]
!1 = metadata !{i32 524329, metadata !"one.c", metadata !"/private/tmp", metadata !2} ; [ DW_TAG_file_type ]
!2 = metadata !{i32 524305, i32 0, i32 12, metadata !"one.c", metadata !".", metadata !"clang 2.8", i1 true, i1 true, metadata !"", i32 0} ; [ DW_TAG_compile_unit ]
!3 = metadata !{i32 524309, metadata !1, metadata !"", metadata !1, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !4, i32 0, null} ; [ DW_TAG_subroutine_type ]
!4 = metadata !{metadata !5}
!5 = metadata !{i32 524324, metadata !1, metadata !"int", metadata !1, i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!6 = metadata !{i32 524334, i32 0, metadata !1, metadata !"foo", metadata !"foo", metadata !"foo", metadata !1, i32 7, metadata !3, i1 true, i1 true, i32 0, i32 0, null, i1 false, i1 true, null} ; [ DW_TAG_subprogram ]
!11 = metadata !{i32 524334, i32 0, metadata !1, metadata !"foo", metadata !"foo", metadata !"foo", metadata !1, i32 7, metadata !3, i1 true, i1 false, i32 0, i32 0, null, i1 false, i1 true, null} ; [ DW_TAG_subprogram ]
!7 = metadata !{i32 524544, metadata !8, metadata !"one", metadata !1, i32 8, metadata !5} ; [ DW_TAG_auto_variable ]
!8 = metadata !{i32 524299, metadata !6, i32 7, i32 18} ; [ DW_TAG_lexical_block ]
!9 = metadata !{i32 4, i32 3, metadata !10, null}
!10 = metadata !{i32 524299, metadata !0, i32 3, i32 11} ; [ DW_TAG_lexical_block ]
