; RUN: llc %s -o /dev/null
; Here variable bar is optimzied away. Do not trip over while trying to generate debug info.


define i32 @foo() nounwind uwtable readnone ssp {
entry:
  ret i32 42, !dbg !15
}

!llvm.dbg.cu = !{!0}

!0 = metadata !{i32 720913, i32 0, i32 12, metadata !"fb.c", metadata !"/private/tmp", metadata !"clang version 3.0 (trunk 139632)", i1 true, i1 true, metadata !"", i32 0, metadata !1, metadata !1, metadata !3, metadata !12} ; [ DW_TAG_compile_unit ]
!1 = metadata !{metadata !2}
!2 = metadata !{i32 0}
!3 = metadata !{metadata !4}
!4 = metadata !{metadata !5}
!5 = metadata !{i32 720942, i32 0, metadata !6, metadata !"foo", metadata !"foo", metadata !"", metadata !6, i32 1, metadata !7, i1 false, i1 true, i32 0, i32 0, i32 0, i32 0, i1 true, i32 ()* @foo, null, null, metadata !10} ; [ DW_TAG_subprogram ]
!6 = metadata !{i32 720937, metadata !"fb.c", metadata !"/private/tmp", null} ; [ DW_TAG_file_type ]
!7 = metadata !{i32 720917, i32 0, metadata !"", i32 0, i32 0, i64 0, i64 0, i32 0, i32 0, i32 0, metadata !8, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!8 = metadata !{metadata !9}
!9 = metadata !{i32 720932, null, metadata !"int", null, i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!10 = metadata !{metadata !11}
!11 = metadata !{i32 720932}                      ; [ DW_TAG_base_type ]
!12 = metadata !{metadata !13}
!13 = metadata !{metadata !14}
!14 = metadata !{i32 720948, i32 0, metadata !5, metadata !"bar", metadata !"bar", metadata !"", metadata !6, i32 2, metadata !9, i32 1, i32 1, null} ; [ DW_TAG_variable ]
!15 = metadata !{i32 3, i32 3, metadata !16, null}
!16 = metadata !{i32 720907, metadata !5, i32 1, i32 11, metadata !6, i32 0} ; [ DW_TAG_lexical_block ]
