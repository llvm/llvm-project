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

!0 = !MDLocation(line: 571, column: 3, scope: !1)
!1 = !{!"0xb\001\001\000", !11, !2}; [DW_TAG_lexical_block ]
!2 = !{!"0x2e\00foo\00foo\00foo\00561\000\001\000\006\000\000\000", i32 0, !3, !4, null, null, null, null, null}; [DW_TAG_subprogram ]
!3 = !{!"0x11\0012\00clang 1.1\001\00\000\00\000", !11, !12, !12, !13, null, null}; [DW_TAG_compile_unit ]
!4 = !{!"0x15\00\000\000\000\000\000\000", null, !3, null, !5, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!5 = !{!6}
!6 = !{!"0x24\00char\000\008\008\000\000\006", null, !3} ; [ DW_TAG_base_type ]
!10 = !MDLocation(line: 588, column: 1, scope: !2)
!11 = !{!"hashtab.c", !"/usr/src/gnu/usr.bin/cc/cc_tools/../../../../contrib/gcclibs/libiberty"}
!12 = !{i32 0}
!13 = !{!2}
!14 = !{i32 1, !"Debug Info Version", i32 2}
