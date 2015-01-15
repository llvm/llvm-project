; RUN: llc  -mtriple=i686-linux -O0 -filetype=obj -o %t %s
; RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck %s
; CHECK: DW_TAG_constant
; CHECK-NEXT: DW_AT_name [DW_FORM_strp] ( .debug_str[0x{{[0-9a-f]*}}] = "ro")

define void @foo() nounwind ssp {
entry:
  call void @bar(i32 201), !dbg !8
  ret void, !dbg !8
}

declare void @bar(i32)

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!13}

!0 = !{!"0x2e\00foo\00foo\00foo\003\000\001\000\006\000\000\003", !12, !1, !3, null, void ()* @foo, null, null, null} ; [ DW_TAG_subprogram ]
!1 = !{!"0x29", !12} ; [ DW_TAG_file_type ]
!2 = !{!"0x11\0012\00clang 2.8\000\00\000\00\000", !12, !4, !4, !10, !11,  !14} ; [ DW_TAG_compile_unit ]
!3 = !{!"0x15\00\000\000\000\000\000\000", !12, !1, null, !4, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!4 = !{null}
!5 = !{!"0x27\00ro\00ro\00ro\001\001\001", !1, !1, !6, i32 201, null} ; [ DW_TAG_constant ]
!6 = !{!"0x26\00\000\000\000\000\000", !12, !1, !7} ; [ DW_TAG_const_type ]
!7 = !{!"0x24\00unsigned int\000\0032\0032\000\000\007", !12, !1} ; [ DW_TAG_base_type ]
!8 = !MDLocation(line: 3, column: 14, scope: !9)
!9 = !{!"0xb\003\0012\000", !12, !0} ; [ DW_TAG_lexical_block ]
!10 = !{!0}
!11 = !{!5}
!12 = !{!"/tmp/l.c", !"/Volumes/Lalgate/clean/D"}
!13 = !{i32 1, !"Debug Info Version", i32 2}
!14 = !{}
