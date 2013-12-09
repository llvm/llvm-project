; RUN: llc %s -o /dev/null
; Here variable bar is optimzied away. Do not trip over while trying to generate debug info.


define i32 @foo() nounwind uwtable readnone ssp {
entry:
  ret i32 42, !dbg !15
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!18}

!0 = metadata !{i32 720913, metadata !17, i32 12, metadata !"clang version 3.0 (trunk 139632)", i1 true, metadata !"", i32 0, metadata !1, metadata !1, metadata !3, metadata !12, null, metadata !""} ; [ DW_TAG_compile_unit ]
!1 = metadata !{i32 0}
!3 = metadata !{metadata !5}
!5 = metadata !{i32 720942, metadata !17, metadata !6, metadata !"foo", metadata !"foo", metadata !"", i32 1, metadata !7, i1 false, i1 true, i32 0, i32 0, null, i32 0, i1 true, i32 ()* @foo, null, null, metadata !10, i32 0} ; [ DW_TAG_subprogram ] [line 1] [def] [scope 0] [foo]
!6 = metadata !{i32 720937, metadata !17} ; [ DW_TAG_file_type ]
!7 = metadata !{i32 720917, i32 0, null, i32 0, i32 0, i64 0, i64 0, i32 0, i32 0, null, metadata !8, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = metadata !{metadata !9}
!9 = metadata !{i32 720932, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!10 = metadata !{metadata !11}
!11 = metadata !{i32 720932}                      ; [ DW_TAG_base_type ]
!12 = metadata !{metadata !14}
!14 = metadata !{i32 720948, i32 0, metadata !5, metadata !"bar", metadata !"bar", metadata !"", metadata !6, i32 2, metadata !9, i32 1, i32 1, null, null} ; [ DW_TAG_variable ]
!15 = metadata !{i32 3, i32 3, metadata !16, null}
!16 = metadata !{i32 720907, metadata !17, metadata !5, i32 1, i32 11, i32 0} ; [ DW_TAG_lexical_block ]
!17 = metadata !{metadata !"fb.c", metadata !"/private/tmp"}
!18 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
