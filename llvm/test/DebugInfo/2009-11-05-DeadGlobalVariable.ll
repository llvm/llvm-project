; RUN: llc %s -o /dev/null
; Here variable bar is optimized away. Do not trip over while trying to generate debug info.


define i32 @foo() nounwind uwtable readnone ssp {
entry:
  ret i32 42, !dbg !15
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!18}

!0 = !{!"0x11\0012\00clang version 3.0 (trunk 139632)\001\00\000\00\000", !17, !1, !1, !3, !12, null} ; [ DW_TAG_compile_unit ]
!1 = !{i32 0}
!3 = !{!5}
!5 = !{!"0x2e\00foo\00foo\00\001\000\001\000\006\000\001\000", !17, !6, !7, null, i32 ()* @foo, null, null, null} ; [ DW_TAG_subprogram ] [line 1] [def] [scope 0] [foo]
!6 = !{!"0x29", !17} ; [ DW_TAG_file_type ]
!7 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !8, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = !{!9}
!9 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ]
!12 = !{!14}
!14 = !{!"0x34\00bar\00bar\00\002\001\001", !5, !6, !9, null, null} ; [ DW_TAG_variable ]
!15 = !MDLocation(line: 3, column: 3, scope: !16)
!16 = !{!"0xb\001\0011\000", !17, !5} ; [ DW_TAG_lexical_block ]
!17 = !{!"fb.c", !"/private/tmp"}
!18 = !{i32 1, !"Debug Info Version", i32 2}
