; REQUIRES: object-emission

; RUN: %llc_dwarf -O0 -filetype=obj < %s | llvm-dwarfdump -debug-dump=info - | FileCheck %s

; Check that two compile units are generated

; CHECK: Compile Unit:
; CHECK: Compile Unit:

define i32 @foo() nounwind readnone ssp {
return:
  ret i32 42, !dbg !0
}

define i32 @bar() nounwind readnone ssp {
return:
  ret i32 21, !dbg !8
}

!llvm.dbg.cu = !{!4, !12}
!llvm.module.flags = !{!21}
!16 = !{!2}
!17 = !{!10}

!0 = !MDLocation(line: 3, scope: !1)
!1 = !{!"0xb\002\000\000", !18, !2} ; [ DW_TAG_lexical_block ]
!2 = !{!"0x2e\00foo\00foo\00foo\002\000\001\000\006\000\000\000", !18, !3, !5, null, i32 ()* @foo, null, null, null} ; [ DW_TAG_subprogram ]
!3 = !{!"0x29", !18} ; [ DW_TAG_file_type ]
!4 = !{!"0x11\001\004.2.1 (Based on Apple Inc. build 5658) (LLVM build)\000\00\000\00\000", !18, !19, !19, !16, null, null} ; [ DW_TAG_compile_unit ]
!5 = !{!"0x15\00\000\000\000\000\000\000", !18, !3, null, !6, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!6 = !{!7}
!7 = !{!"0x24\00int\000\0032\0032\000\000\005", !18, !3} ; [ DW_TAG_base_type ]
!8 = !MDLocation(line: 3, scope: !9)
!9 = !{!"0xb\002\000\000", !20, !10} ; [ DW_TAG_lexical_block ]
!10 = !{!"0x2e\00bar\00bar\00bar\002\000\001\000\006\000\000\000", !20, !11, !13, null, i32 ()* @bar, null, null, null} ; [ DW_TAG_subprogram ]
!11 = !{!"0x29", !20} ; [ DW_TAG_file_type ]
!12 = !{!"0x11\001\004.2.1 (Based on Apple Inc. build 5658) (LLVM build)\000\00\000\00\000", !20, !19, !19, !17, null, null} ; [ DW_TAG_compile_unit ]
!13 = !{!"0x15\00\000\000\000\000\000\000", !20, !11, null, !14, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!14 = !{!15}
!15 = !{!"0x24\00int\000\0032\0032\000\000\005", !20, !11} ; [ DW_TAG_base_type ]
!18 = !{!"a.c", !"/tmp/"}
!19 = !{i32 0}
!20 = !{!"b.c", !"/tmp/"}
!21 = !{i32 1, !"Debug Info Version", i32 2}
