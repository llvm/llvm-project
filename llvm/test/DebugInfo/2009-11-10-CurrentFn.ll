; RUN: llc < %s -o /dev/null

define void @bar(i32 %i) nounwind uwtable ssp {
entry:
  tail call void (...)* @foo() nounwind, !dbg !14
  ret void, !dbg !16
}

declare void @foo(...)

declare void @llvm.dbg.value(metadata, i64, metadata) nounwind readnone

!llvm.dbg.cu = !{!0}

!0 = metadata !{i32 720913, i32 0, i32 12, metadata !"cf.c", metadata !"/private/tmp", metadata !"clang version 3.0 (trunk 139632)", i1 true, i1 true, metadata !"", i32 0, metadata !1, metadata !1, metadata !3, metadata !1} ; [ DW_TAG_compile_unit ]
!1 = metadata !{metadata !2}
!2 = metadata !{i32 0}
!3 = metadata !{metadata !4}
!4 = metadata !{metadata !5}
!5 = metadata !{i32 720942, i32 0, metadata !6, metadata !"bar", metadata !"bar", metadata !"", metadata !6, i32 3, metadata !7, i1 false, i1 true, i32 0, i32 0, i32 0, i32 256, i1 true, void (i32)* @bar, null, null, metadata !9} ; [ DW_TAG_subprogram ]
!6 = metadata !{i32 720937, metadata !"cf.c", metadata !"/private/tmp", null} ; [ DW_TAG_file_type ]
!7 = metadata !{i32 720917, i32 0, metadata !"", i32 0, i32 0, i64 0, i64 0, i32 0, i32 0, i32 0, metadata !8, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!8 = metadata !{null}
!9 = metadata !{metadata !10}
!10 = metadata !{metadata !11}
!11 = metadata !{i32 721153, metadata !5, metadata !"i", metadata !6, i32 16777219, metadata !12, i32 0, i32 0} ; [ DW_TAG_arg_variable ]
!12 = metadata !{i32 720932, null, metadata !"int", null, i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!13 = metadata !{i32 3, i32 14, metadata !5, null}
!14 = metadata !{i32 4, i32 3, metadata !15, null}
!15 = metadata !{i32 720907, metadata !5, i32 3, i32 17, metadata !6, i32 0} ; [ DW_TAG_lexical_block ]
!16 = metadata !{i32 5, i32 1, metadata !15, null}

