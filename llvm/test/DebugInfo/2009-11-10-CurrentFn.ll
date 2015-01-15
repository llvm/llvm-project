; RUN: llc < %s -o /dev/null

define void @bar(i32 %i) nounwind uwtable ssp {
entry:
  tail call void (...)* @foo() nounwind, !dbg !14
  ret void, !dbg !16
}

declare void @foo(...)

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!18}

!0 = !{!"0x11\0012\00clang version 3.0 (trunk 139632)\001\00\000\00\000", !17, !1, !1, !3, !1, null} ; [ DW_TAG_compile_unit ]
!1 = !{}
!3 = !{!5}
!5 = !{!"0x2e\00bar\00bar\00\003\000\001\000\006\00256\001\000", !17, !6, !7, null, void (i32)* @bar, null, null, !9} ; [ DW_TAG_subprogram ] [line 3] [def] [scope 0] [bar]
!6 = !{!"0x29", !17} ; [ DW_TAG_file_type ]
!7 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !8, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = !{null}
!9 = !{!11}
!11 = !{!"0x101\00i\0016777219\000", !17, !5, !12} ; [ DW_TAG_arg_variable ]
!12 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ]
!13 = !MDLocation(line: 3, column: 14, scope: !5)
!14 = !MDLocation(line: 4, column: 3, scope: !15)
!15 = !{!"0xb\003\0017\000", !17, !5} ; [ DW_TAG_lexical_block ]
!16 = !MDLocation(line: 5, column: 1, scope: !15)
!17 = !{!"cf.c", !"/private/tmp"}
!18 = !{i32 1, !"Debug Info Version", i32 2}
