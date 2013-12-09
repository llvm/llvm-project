; RUN: llc -o /dev/null < %s
; PR7662
; Do not add variables to !11 because it is a declaration entry.

define i32 @bar() nounwind readnone ssp {
entry:
  ret i32 42, !dbg !9
}

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!15}
!llvm.dbg.sp = !{!0, !6, !11}
!llvm.dbg.lv.foo = !{!7}

!0 = metadata !{i32 524334, metadata !12, metadata !1, metadata !"bar", metadata !"bar", metadata !"bar", i32 3, metadata !3, i1 false, i1 true, i32 0, i32 0, null, i1 false, i1 true, i32 ()* @bar, null, null, null, i32 0} ; [ DW_TAG_subprogram ]
!1 = metadata !{i32 524329, metadata !12} ; [ DW_TAG_file_type ]
!2 = metadata !{i32 524305, metadata !12, i32 12, metadata !"clang 2.8", i1 true, metadata !"", i32 0, metadata !14, metadata !14, metadata !13, null, null, metadata !""} ; [ DW_TAG_compile_unit ]
!3 = metadata !{i32 524309, metadata !12, metadata !1, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !4, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!4 = metadata !{metadata !5}
!5 = metadata !{i32 524324, metadata !12, metadata !1, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!6 = metadata !{i32 524334, metadata !12, metadata !1, metadata !"foo", metadata !"foo", metadata !"foo", i32 7, metadata !3, i1 true, i1 true, i32 0, i32 0, null, i1 false, i1 true, null, null, null, null, i32 0} ; [ DW_TAG_subprogram ]
!7 = metadata !{i32 524544, metadata !8, metadata !"one", metadata !1, i32 8, metadata !5} ; [ DW_TAG_auto_variable ]
!8 = metadata !{i32 524299, metadata !12, metadata !6, i32 7, i32 18, i32 0} ; [ DW_TAG_lexical_block ]
!9 = metadata !{i32 4, i32 3, metadata !10, null}
!10 = metadata !{i32 524299, metadata !12, metadata !0, i32 3, i32 11, i32 0} ; [ DW_TAG_lexical_block ]
!11 = metadata !{i32 524334, metadata !12, metadata !1, metadata !"foo", metadata !"foo", metadata !"foo", i32 7, metadata !3, i1 true, i1 false, i32 0, i32 0, null, i1 false, i1 true, null, null, null, null, i32 0} ; [ DW_TAG_subprogram ]
!12 = metadata !{metadata !"one.c", metadata !"/private/tmp"}
!13 = metadata !{metadata !0, metadata !6, metadata !11}
!14 = metadata !{i32 0}
!15 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
