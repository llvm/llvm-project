; RUN: llc -o /dev/null < %s
; PR7662
; Do not add variables to !11 because it is a declaration entry.

define i32 @bar() nounwind readnone ssp {
entry:
  ret i32 42, !dbg !9
}

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!15}
!llvm.dbg.sp = !{!0, !6, !11}
!llvm.dbg.lv.foo = !{!7}

!0 = !{!"0x2e\00bar\00bar\00bar\003\000\001\000\006\000\001\000", !12, !1, !3, null, i32 ()* @bar, null, null, null} ; [ DW_TAG_subprogram ]
!1 = !{!"0x29", !12} ; [ DW_TAG_file_type ]
!2 = !{!"0x11\0012\00clang 2.8\001\00\000\00\000", !12, !14, !14, !13, null, null} ; [ DW_TAG_compile_unit ]
!3 = !{!"0x15\00\000\000\000\000\000\000", !12, !1, null, !4, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!4 = !{!5}
!5 = !{!"0x24\00int\000\0032\0032\000\000\005", !12, !1} ; [ DW_TAG_base_type ]
!6 = !{!"0x2e\00foo\00foo\00foo\007\001\001\000\006\000\001\000", !12, !1, !3, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!7 = !{!"0x100\00one\008\000", !8, !1, !5} ; [ DW_TAG_auto_variable ]
!8 = !{!"0xb\007\0018\000", !12, !6} ; [ DW_TAG_lexical_block ]
!9 = !MDLocation(line: 4, column: 3, scope: !10)
!10 = !{!"0xb\003\0011\000", !12, !0} ; [ DW_TAG_lexical_block ]
!11 = !{!"0x2e\00foo\00foo\00foo\007\001\000\000\006\000\001\000", !12, !1, !3, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!12 = !{!"one.c", !"/private/tmp"}
!13 = !{!0}
!14 = !{i32 0}
!15 = !{i32 1, !"Debug Info Version", i32 2}
