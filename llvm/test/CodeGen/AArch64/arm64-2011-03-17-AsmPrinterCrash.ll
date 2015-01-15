; RUN: llc < %s -mtriple=arm64-apple-darwin

; rdar://9146594

define void @drt_vsprintf() nounwind ssp {
entry:
  %do_tab_convert = alloca i32, align 4
  br i1 undef, label %if.then24, label %if.else295, !dbg !13

if.then24:                                        ; preds = %entry
  unreachable

if.else295:                                       ; preds = %entry
  call void @llvm.dbg.declare(metadata i32* %do_tab_convert, metadata !16, metadata !{!"0x102"}), !dbg !18
  store i32 0, i32* %do_tab_convert, align 4, !dbg !19
  unreachable
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

!llvm.dbg.gv = !{!0}
!llvm.dbg.sp = !{!1, !7, !10, !11, !12}

!0 = !{!"0x34\00vsplive\00vsplive\00\00617\001\001", !1, !2, !6, null, null} ; [ DW_TAG_variable ]
!1 = !{!"0x2e\00drt_vsprintf\00drt_vsprintf\00\00616\000\001\000\006\00256\000\000", !20, !2, !4, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!2 = !{!"0x29", !20} ; [ DW_TAG_file_type ]
!3 = !{!"0x11\0012\00clang version 3.0 (http://llvm.org/git/clang.git git:/git/puzzlebox/clang.git/ c4d1aea01c4444eb81bdbf391f1be309127c3cf1)\001\00\000\00\000", !20, !21, !21, null, null, null} ; [ DW_TAG_compile_unit ]
!4 = !{!"0x15\00\000\000\000\000\000\000", !20, !2, null, !5, i32 0} ; [ DW_TAG_subroutine_type ]
!5 = !{!6}
!6 = !{!"0x24\00int\000\0032\0032\000\000\005", null, !3} ; [ DW_TAG_base_type ]
!7 = !{!"0x2e\00putc_mem\00putc_mem\00\0030\001\001\000\006\00256\000\000", !20, !2, !8, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!8 = !{!"0x15\00\000\000\000\000\000\000", !20, !2, null, !9, i32 0} ; [ DW_TAG_subroutine_type ]
!9 = !{null}
!10 = !{!"0x2e\00print_double\00print_double\00\00203\001\001\000\006\00256\000\000", !20, !2, !4, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!11 = !{!"0x2e\00print_number\00print_number\00\0075\001\001\000\006\00256\000\000", !20, !2, !4, i32 0, null, null, null, null} ; [ DW_TAG_subprogram ]
!12 = !{!"0x2e\00get_flags\00get_flags\00\00508\001\001\000\006\00256\000\000", !20, !2, !8, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!13 = !MDLocation(line: 653, column: 5, scope: !14)
!14 = !{!"0xb\00652\0035\002", !20, !15} ; [ DW_TAG_lexical_block ]
!15 = !{!"0xb\00616\001\000", !20, !1} ; [ DW_TAG_lexical_block ]
!16 = !{!"0x100\00do_tab_convert\00853\000", !17, !2, !6} ; [ DW_TAG_auto_variable ]
!17 = !{!"0xb\00850\0012\0033", !20, !14} ; [ DW_TAG_lexical_block ]
!18 = !MDLocation(line: 853, column: 11, scope: !17)
!19 = !MDLocation(line: 853, column: 29, scope: !17)
!20 = !{!"print.i", !"/Volumes/Ebi/echeng/radars/r9146594"}
!21 = !{i32 0}
