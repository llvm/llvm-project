; RUN: llc < %s -o /dev/null
; PR 5942
define i8* @foo() nounwind {
entry:
  %0 = load i32* undef, align 4, !dbg !0          ; <i32> [#uses=1]
  %1 = inttoptr i32 %0 to i8*, !dbg !0            ; <i8*> [#uses=1]
  ret i8* %1, !dbg !10

}

!llvm.dbg.cu = !{!3}
!llvm.module.flags = !{!14}

!0 = metadata !{i32 571, i32 3, metadata !1, null}
!1 = metadata !{i32 458763, metadata !11, metadata !2, i32 1, i32 1, i32 0}; [DW_TAG_lexical_block ]
!2 = metadata !{i32 458798, i32 0, metadata !3, metadata !"foo", metadata !"foo", metadata !"foo", i32 561, metadata !4, i1 false, i1 true, i32 0, i32 0, null, i32 0, i32 0, null, null, null, null, i32 0}; [DW_TAG_subprogram ]
!3 = metadata !{i32 458769, metadata !11, i32 12, metadata !"clang 1.1", i1 true, metadata !"", i32 0, metadata !12, metadata !12, metadata !13, null, null, metadata !""}; [DW_TAG_compile_unit ]
!4 = metadata !{i32 458773, null, metadata !3, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !5, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!5 = metadata !{metadata !6}
!6 = metadata !{i32 458788, null, metadata !3, metadata !"char", i32 0, i64 8, i64 8, i64 0, i32 0, i32 6} ; [ DW_TAG_base_type ]
!10 = metadata !{i32 588, i32 1, metadata !2, null}
!11 = metadata !{metadata !"hashtab.c", metadata !"/usr/src/gnu/usr.bin/cc/cc_tools/../../../../contrib/gcclibs/libiberty"}
!12 = metadata !{i32 0}
!13 = metadata !{metadata !2}
!14 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
