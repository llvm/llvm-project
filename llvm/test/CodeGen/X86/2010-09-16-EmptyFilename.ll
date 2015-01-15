; RUN: llc -O0 -mtriple=x86_64-apple-darwin10 < %s - | FileCheck %s
; Radar 8286101
; CHECK: .file   {{[0-9]+}} "<stdin>"

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

!0 = !{!"0x2e\00foo\00foo\00foo\0053\000\001\000\006\000\000\000", !14, !1, !3, null, i32 ()* @foo, null, null, null} ; [ DW_TAG_subprogram ]
!1 = !{!"0x29", !14} ; [ DW_TAG_file_type ]
!2 = !{!"0x11\0012\00clang version 2.9 (trunk 114084)\000\00\000\00\000", !15, !16, !16, !13, null, null} ; [ DW_TAG_compile_unit ]
!3 = !{!"0x15\00\000\000\000\000\000\000", !14, !1, null, !4, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!4 = !{!5}
!5 = !{!"0x24\00int\000\0032\0032\000\000\005", !14, !1} ; [ DW_TAG_base_type ]
!6 = !{!"0x2e\00bar\00bar\00bar\004\000\001\000\006\000\000\000", !15, !7, !3, null, i32 ()* @bar, null, null, null} ; [ DW_TAG_subprogram ]
!7 = !{!"0x29", !15} ; [ DW_TAG_file_type ]
!8 = !MDLocation(line: 53, column: 13, scope: !9)
!9 = !{!"0xb\0053\0011\000", !14, !0} ; [ DW_TAG_lexical_block ]
!10 = !MDLocation(line: 4, column: 13, scope: !11)
!11 = !{!"0xb\004\0013\002", !15, !12} ; [ DW_TAG_lexical_block ]
!12 = !{!"0xb\004\0011\001", !15, !6} ; [ DW_TAG_lexical_block ]
!13 = !{!0, !6}
!14 = !{!"", !"/private/tmp"}
!15 = !{!"bug.c", !"/private/tmp"}
!16 = !{i32 0}
!17 = !{i32 1, !"Debug Info Version", i32 2}
