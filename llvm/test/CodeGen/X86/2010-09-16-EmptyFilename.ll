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

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!17}

!0 = metadata !{i32 786478, metadata !14, metadata !1, metadata !"foo", metadata !"foo", metadata !"foo", i32 53, metadata !3, i1 false, i1 true, i32 0, i32 0, null, i1 false, i1 false, i32 ()* @foo, null, null, null, i32 0} ; [ DW_TAG_subprogram ]
!1 = metadata !{i32 786473, metadata !14} ; [ DW_TAG_file_type ]
!2 = metadata !{i32 786449, metadata !15, i32 12, metadata !"clang version 2.9 (trunk 114084)", i1 false, metadata !"", i32 0, metadata !16, metadata !16, metadata !13, null, null, metadata !""} ; [ DW_TAG_compile_unit ]
!3 = metadata !{i32 786453, metadata !14, metadata !1, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !4, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!4 = metadata !{metadata !5}
!5 = metadata !{i32 786468, metadata !14, metadata !1, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!6 = metadata !{i32 786478, metadata !15, metadata !7, metadata !"bar", metadata !"bar", metadata !"bar", i32 4, metadata !3, i1 false, i1 true, i32 0, i32 0, null, i1 false, i1 false, i32 ()* @bar, null, null, null, i32 0} ; [ DW_TAG_subprogram ]
!7 = metadata !{i32 786473, metadata !15} ; [ DW_TAG_file_type ]
!8 = metadata !{i32 53, i32 13, metadata !9, null}
!9 = metadata !{i32 786443, metadata !14, metadata !0, i32 53, i32 11, i32 0} ; [ DW_TAG_lexical_block ]
!10 = metadata !{i32 4, i32 13, metadata !11, null}
!11 = metadata !{i32 786443, metadata !15, metadata !12, i32 4, i32 13, i32 2} ; [ DW_TAG_lexical_block ]
!12 = metadata !{i32 786443, metadata !15, metadata !6, i32 4, i32 11, i32 1} ; [ DW_TAG_lexical_block ]
!13 = metadata !{metadata !0, metadata !6}
!14 = metadata !{metadata !"", metadata !"/private/tmp"}
!15 = metadata !{metadata !"bug.c", metadata !"/private/tmp"}
!16 = metadata !{i32 0}
!17 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
