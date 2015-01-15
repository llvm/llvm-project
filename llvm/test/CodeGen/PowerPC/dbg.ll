; RUN: llc < %s -break-anti-dependencies=all -march=ppc64 -mcpu=g5 | FileCheck %s
; CHECK-LABEL: main:

target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

define i32 @main(i32 %argc, i8** nocapture %argv) nounwind readnone {
entry:
  tail call void @llvm.dbg.value(metadata i32 %argc, i64 0, metadata !15, metadata !{!"0x102"}), !dbg !17
  tail call void @llvm.dbg.value(metadata i8** %argv, i64 0, metadata !16, metadata !{!"0x102"}), !dbg !18
  %add = add nsw i32 %argc, 1, !dbg !19
  ret i32 %add, !dbg !19
}

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!22}

!0 = !{!"0x11\0012\00clang version 3.1\001\00\000\00\000", !21, !1, !1, !3, !1, !""} ; [ DW_TAG_compile_unit ]
!1 = !{}
!3 = !{!5}
!5 = !{!"0x2e\00main\00main\00\001\000\001\000\006\00256\001\000", !21, null, !7, null, i32 (i32, i8**)* @main, null, null, !13} ; [ DW_TAG_subprogram ]
!6 = !{!"0x29", !21} ; [ DW_TAG_file_type ]
!7 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !8, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = !{!9, !9, !10}
!9 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ]
!10 = !{!"0xf\00\000\0064\0064\000\000", null, null, !11} ; [ DW_TAG_pointer_type ]
!11 = !{!"0xf\00\000\0064\0064\000\000", null, null, !12} ; [ DW_TAG_pointer_type ]
!12 = !{!"0x24\00char\000\008\008\000\000\008", null, null} ; [ DW_TAG_base_type ]
!13 = !{!15, !16}
!15 = !{!"0x101\00argc\0016777217\000", !5, !6, !9} ; [ DW_TAG_arg_variable ]
!16 = !{!"0x101\00argv\0033554433\000", !5, !6, !10} ; [ DW_TAG_arg_variable ]
!17 = !MDLocation(line: 1, column: 14, scope: !5)
!18 = !MDLocation(line: 1, column: 26, scope: !5)
!19 = !MDLocation(line: 2, column: 3, scope: !20)
!20 = !{!"0xb\001\0034\000", !21, !5} ; [ DW_TAG_lexical_block ]
!21 = !{!"dbg.c", !"/src"}
!22 = !{i32 1, !"Debug Info Version", i32 2}
