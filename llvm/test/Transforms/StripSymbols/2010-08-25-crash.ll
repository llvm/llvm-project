; RUN: opt -strip-dead-debug-info -disable-output < %s
define i32 @foo() nounwind ssp {
entry:
  ret i32 0, !dbg !8
}

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!14}

!0 = metadata !{i32 524334, metadata !10, metadata !1, metadata !"foo", metadata !"foo", metadata !"foo", i32 3, metadata !3, i1 false, i1 true, i32 0, i32 0, null, i1 false, i1 false, i32 ()* @foo, null, null, null, i32 0} ; [ DW_TAG_subprogram ]
!1 = metadata !{i32 524329, metadata !10} ; [ DW_TAG_file_type ]
!2 = metadata !{i32 524305, metadata !10, i32 12, metadata !"clang version 2.8 (trunk 112062)", i1 true, metadata !"", i32 0, metadata !11, metadata !11, metadata !12, metadata !13, null, metadata !""} ; [ DW_TAG_compile_unit ]
!3 = metadata !{i32 524309, metadata !10, metadata !1, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !4, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!4 = metadata !{metadata !5}
!5 = metadata !{i32 524324, metadata !10, metadata !1, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!6 = metadata !{i32 524340, i32 0, metadata !1, metadata !"i", metadata !"i", metadata !"i", metadata !1, i32 2, metadata !7, i1 true, i1 true, i32 0, null} ; [ DW_TAG_variable ]
!7 = metadata !{i32 524326, metadata !10, metadata !1, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, metadata !5} ; [ DW_TAG_const_type ]
!8 = metadata !{i32 3, i32 13, metadata !9, null}
!9 = metadata !{i32 524299, metadata !10, metadata !0, i32 3, i32 11, i32 0} ; [ DW_TAG_lexical_block ]
!10 = metadata !{metadata !"/tmp/a.c", metadata !"/Volumes/Lalgate/clean/D.CW"}
!11 = metadata !{i32 0}
!12 = metadata !{metadata !0}
!13 = metadata !{metadata !6}
!14 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
