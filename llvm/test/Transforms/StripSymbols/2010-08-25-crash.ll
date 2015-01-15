; RUN: opt -strip-dead-debug-info -disable-output < %s
define i32 @foo() nounwind ssp {
entry:
  ret i32 0, !dbg !8
}

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!14}

!0 = !{!"0x2e\00foo\00foo\00foo\003\000\001\000\006\000\000\000", !10, !1, !3, null, i32 ()* @foo, null, null, null} ; [ DW_TAG_subprogram ]
!1 = !{!"0x29", !10} ; [ DW_TAG_file_type ]
!2 = !{!"0x11\0012\00clang version 2.8 (trunk 112062)\001\00\000\00\001", !10, !11, !11, !12, !13, null} ; [ DW_TAG_compile_unit ]
!3 = !{!"0x15\00\000\000\000\000\000\000", !10, !1, null, !4, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!4 = !{!5}
!5 = !{!"0x24\00int\000\0032\0032\000\000\005", !10, !1} ; [ DW_TAG_base_type ]
!6 = !{!"0x34\00i\00i\00i\002\001\001", !1, !1, !7, i32 0, null} ; [ DW_TAG_variable ]
!7 = !{!"0x26\00\000\000\000\000\000", !10, !1, !5} ; [ DW_TAG_const_type ]
!8 = !MDLocation(line: 3, column: 13, scope: !9)
!9 = !{!"0xb\003\0011\000", !10, !0} ; [ DW_TAG_lexical_block ]
!10 = !{!"/tmp/a.c", !"/Volumes/Lalgate/clean/D.CW"}
!11 = !{i32 0}
!12 = !{!0}
!13 = !{!6}
!14 = !{i32 1, !"Debug Info Version", i32 2}
