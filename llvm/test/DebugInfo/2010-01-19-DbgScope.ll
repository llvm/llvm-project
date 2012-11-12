; RUN: llc -O0 < %s -o /dev/null
; Ignore unreachable scopes.
declare void @foo(i32) noreturn

define i32 @bar() nounwind ssp {
entry:
  br i1 undef, label %bb, label %bb11, !dbg !0

bb:                                               ; preds = %entry
  call void @foo(i32 0) noreturn nounwind, !dbg !7
  unreachable, !dbg !7

bb11:                                             ; preds = %entry
  ret i32 1, !dbg !11
}

!0 = metadata !{i32 8647, i32 0, metadata !1, null}
!1 = metadata !{i32 458763, metadata !2}          ; [ DW_TAG_lexical_block ]
!2 = metadata !{i32 458798, i32 0, metadata !3, metadata !"bar", metadata !"bar", metadata !"bar", metadata !3, i32 8639, metadata !4, i1 true, i1 true, i32 0, i32 0, null} ; [ DW_TAG_subprogram ]
!3 = metadata !{i32 458769, i32 0, i32 1, metadata !"c-parser.c", metadata !"llvmgcc", metadata !"LLVM build 00", i1 true, i1 false, metadata !"", i32 0} ; [ DW_TAG_compile_unit ]
!4 = metadata !{i32 458773, metadata !3, metadata !"", metadata !3, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !5, i32 0} ; [ DW_TAG_subroutine_type ]
!5 = metadata !{metadata !6}
!6 = metadata !{i32 458788, metadata !3, metadata !"char", metadata !3, i32 0, i64 8, i64 8, i64 0, i32 0, i32 6} ; [ DW_TAG_base_type ]
!7 = metadata !{i32 8648, i32 0, metadata !8, null}
!8 = metadata !{i32 458763, metadata !9}          ; [ DW_TAG_lexical_block ]
!9 = metadata !{i32 458763, metadata !10}         ; [ DW_TAG_lexical_block ]
!10 = metadata !{i32 458798, i32 0, metadata !3, metadata !"bar2", metadata !"bar2", metadata !"bar2", metadata !3, i32 8639, metadata !4, i1 true, i1 true, i32 0, i32 0, null} ; [ DW_TAG_subprogram ]
!11 = metadata !{i32 8652, i32 0, metadata !1, null}
