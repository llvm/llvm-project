; RUN: %llc_dwarf -O2 %s -o - | FileCheck %s
; Check struct X for dead variable xyz from inlined function foo.

; CHECK:	DW_TAG_structure_type
; CHECK-NEXT:	info_string
 

@i = common global i32 0                          ; <i32*> [#uses=2]

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

define i32 @bar() nounwind ssp {
entry:
  %0 = load i32* @i, align 4, !dbg !17            ; <i32> [#uses=2]
  tail call void @llvm.dbg.value(metadata i32 %0, i64 0, metadata !9, metadata !{!"0x102"}), !dbg !19
  tail call void @llvm.dbg.declare(metadata !29, metadata !10, metadata !{!"0x102"}), !dbg !21
  %1 = mul nsw i32 %0, %0, !dbg !22               ; <i32> [#uses=2]
  store i32 %1, i32* @i, align 4, !dbg !17
  ret i32 %1, !dbg !23
}

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!28}

!0 = !{!"0x2e\00foo\00foo\00\009\001\001\000\006\000\001\009", !27, !1, !3, null, null, null, null, !24} ; [ DW_TAG_subprogram ]
!1 = !{!"0x29", !27} ; [ DW_TAG_file_type ]
!2 = !{!"0x11\001\004.2.1 (Based on Apple Inc. build 5658) (LLVM build)\001\00\000\00\000", !27, !20, !20, !25, !26,  !20} ; [ DW_TAG_compile_unit ]
!3 = !{!"0x15\00\000\000\000\000\000\000", !27, !1, null, !4, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!4 = !{!5, !5}
!5 = !{!"0x24\00int\000\0032\0032\000\000\005", !27, !1} ; [ DW_TAG_base_type ]
!6 = !{!"0x2e\00bar\00bar\00bar\0014\000\001\000\006\000\001\000", !27, !1, !7, null, i32 ()* @bar, null, null, null} ; [ DW_TAG_subprogram ]
!7 = !{!"0x15\00\000\000\000\000\000\000", !27, !1, null, !8, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = !{!5}
!9 = !{!"0x101\00j\009\000", !0, !1, !5} ; [ DW_TAG_arg_variable ]
!10 = !{!"0x100\00xyz\0010\000", !11, !1, !12} ; [ DW_TAG_auto_variable ]
!11 = !{!"0xb\009\000\000", !1, !0} ; [ DW_TAG_lexical_block ]
!12 = !{!"0x13\00X\0010\0064\0032\000\000\000", !27, !0, null, !13, null, null, null} ; [ DW_TAG_structure_type ] [X] [line 10, size 64, align 32, offset 0] [def] [from ]
!13 = !{!14, !15}
!14 = !{!"0xd\00a\0010\0032\0032\000\000", !27, !12, !5} ; [ DW_TAG_member ]
!15 = !{!"0xd\00b\0010\0032\0032\0032\000", !27, !12, !5} ; [ DW_TAG_member ]
!16 = !{!"0x34\00i\00i\00\005\000\001", !1, !1, !5, i32* @i, null} ; [ DW_TAG_variable ]
!17 = !MDLocation(line: 15, scope: !18)
!18 = !{!"0xb\0014\000\001", !1, !6} ; [ DW_TAG_lexical_block ]
!19 = !MDLocation(line: 9, scope: !0, inlinedAt: !17)
!20 = !{}
!21 = !MDLocation(line: 9, scope: !11, inlinedAt: !17)
!22 = !MDLocation(line: 11, scope: !11, inlinedAt: !17)
!23 = !MDLocation(line: 16, scope: !18)
!24 = !{!9, !10}
!25 = !{!0, !6}
!26 = !{!16}
!27 = !{!"bar.c", !"/tmp/"}
!28 = !{i32 1, !"Debug Info Version", i32 2}
!29 = !{null}
