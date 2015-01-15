; Check that DEBUG_VALUE comments come through on a variety of targets.

define i32 @main() nounwind ssp {
entry:
; CHECK: DEBUG_VALUE
  call void @llvm.dbg.value(metadata i32 0, i64 0, metadata !7, metadata !{!"0x102"}), !dbg !9
  ret i32 0, !dbg !10
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!13}

!0 = !{!"0x2e\00main\00main\00\002\000\001\000\006\000\000\000", !12, !1, !3, null, i32 ()* @main, null, null, null} ; [ DW_TAG_subprogram ]
!1 = !{!"0x29", !12} ; [ DW_TAG_file_type ]
!2 = !{!"0x11\0012\00clang version 2.9 (trunk 120996)\000\00\000\00\000", !12, !6, !6, !11, null, null} ; [ DW_TAG_compile_unit ]
!3 = !{!"0x15\00\000\000\000\000\000\000", !12, !1, null, !4, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!4 = !{!5}
!5 = !{!"0x24\00int\000\0032\0032\000\000\005", !12, !2} ; [ DW_TAG_base_type ]
!6 = !{i32 0}
!7 = !{!"0x100\00i\003\000", !8, !1, !5} ; [ DW_TAG_auto_variable ]
!8 = !{!"0xb\002\0012\000", !12, !0} ; [ DW_TAG_lexical_block ]
!9 = !MDLocation(line: 3, column: 11, scope: !8)
!10 = !MDLocation(line: 4, column: 2, scope: !8)
!11 = !{!0}
!12 = !{!"/tmp/x.c", !"/Users/manav"}
!13 = !{i32 1, !"Debug Info Version", i32 2}
