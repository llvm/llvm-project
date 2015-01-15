; RUN: llc -mtriple=x86_64-linux < %s | FileCheck %s

; CHECK: DW_AT_const_value
; CHECK-NEXT: 42

define i128 @__foo(i128 %a, i128 %b) nounwind {
entry:
  tail call void @llvm.dbg.value(metadata i128 42 , i64 0, metadata !1, metadata !{!"0x102"}), !dbg !11
  %add = add i128 %a, %b, !dbg !11
  ret i128 %add, !dbg !11
}

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!5}
!llvm.module.flags = !{!16}

!0 = !{i128 42 }
!1 = !{!"0x100\00MAX\0029\000", !2, !4, !8} ; [ DW_TAG_auto_variable ]
!2 = !{!"0xb\0026\000\000", !13, !3} ; [ DW_TAG_lexical_block ]
!3 = !{!"0x2e\00__foo\00__foo\00__foo\0026\000\001\000\006\000\000\0026", !13, !4, !6, null, i128 (i128, i128)* @__foo, null, null, null} ; [ DW_TAG_subprogram ]
!4 = !{!"0x29", !13} ; [ DW_TAG_file_type ]
!5 = !{!"0x11\001\00clang\001\00\000\00\000", !13, !15, !15, !12, null,  null} ; [ DW_TAG_compile_unit ]
!6 = !{!"0x15\00\000\000\000\000\000\000", !13, !4, null, !7, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = !{!8, !8, !8}
!8 = !{!"0x16\00ti_int\0078\000\000\000\000", !14, !4, !10} ; [ DW_TAG_typedef ]
!9 = !{!"0x29", !14} ; [ DW_TAG_file_type ]
!10 = !{!"0x24\00\000\00128\00128\000\000\005", !13, !4} ; [ DW_TAG_base_type ]
!11 = !MDLocation(line: 29, scope: !2)
!12 = !{!3}
!13 = !{!"foo.c", !"/tmp"}
!14 = !{!"myint.h", !"/tmp"}
!15 = !{i32 0}
!16 = !{i32 1, !"Debug Info Version", i32 2}
