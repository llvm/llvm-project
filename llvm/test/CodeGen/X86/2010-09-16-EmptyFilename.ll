; RUN: llc -O0 -mtriple=x86_64-apple-darwin10 < %s - | FileCheck %s
; Radar 8286101
; CHECK: .file   2 "<stdin>"

define i32 @foo() nounwind ssp {
entry:
  ret i32 42, !dbg !8
}

define i32 @bar() nounwind ssp {
entry:
  ret i32 21, !dbg !10
}

!llvm.dbg.sp = !{!0, !6}

!0 = metadata !{i32 524334, i32 0, metadata !1, metadata !"foo", metadata !"foo", metadata !"foo", metadata !1, i32 53, metadata !3, i1 false, i1 true, i32 0, i32 0, null, i1 false, i1 false, i32 ()* @foo} ; [ DW_TAG_subprogram ]
!1 = metadata !{i32 524329, metadata !"", metadata !"/private/tmp", metadata !2} ; [ DW_TAG_file_type ]
!2 = metadata !{i32 524305, i32 0, i32 12, metadata !"bug.c", metadata !"/private/tmp", metadata !"clang version 2.9 (trunk 114084)", i1 true, i1 false, metadata !"", i32 0} ; [ DW_TAG_compile_unit ]
!3 = metadata !{i32 524309, metadata !1, metadata !"", metadata !1, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !4, i32 0, null} ; [ DW_TAG_subroutine_type ]
!4 = metadata !{metadata !5}
!5 = metadata !{i32 524324, metadata !1, metadata !"int", metadata !1, i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!6 = metadata !{i32 524334, i32 0, metadata !7, metadata !"bar", metadata !"bar", metadata !"bar", metadata !7, i32 4, metadata !3, i1 false, i1 true, i32 0, i32 0, null, i1 false, i1 false, i32 ()* @bar} ; [ DW_TAG_subprogram ]
!7 = metadata !{i32 524329, metadata !"bug.c", metadata !"/private/tmp", metadata !2} ; [ DW_TAG_file_type ]
!8 = metadata !{i32 53, i32 13, metadata !9, null}
!9 = metadata !{i32 524299, metadata !0, i32 53, i32 11, metadata !1, i32 0} ; [ DW_TAG_lexical_block ]
!10 = metadata !{i32 4, i32 13, metadata !11, null}
!11 = metadata !{i32 524299, metadata !12, i32 4, i32 13, metadata !7, i32 2} ; [ DW_TAG_lexical_block ]
!12 = metadata !{i32 524299, metadata !6, i32 4, i32 11, metadata !7, i32 1} ; [ DW_TAG_lexical_block ]
