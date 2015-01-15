; RUN: %llc_dwarf -O0 < %s | FileCheck %s
; Do not emit AT_upper_bound for an unbounded array.
; radar 9241695
define i32 @main() nounwind ssp {
entry:
  %retval = alloca i32, align 4
  %a = alloca [0 x i32], align 4
  store i32 0, i32* %retval
  call void @llvm.dbg.declare(metadata [0 x i32]* %a, metadata !6, metadata !{!"0x102"}), !dbg !11
  ret i32 0, !dbg !12
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!16}

!0 = !{!"0x2e\00main\00main\00\003\000\001\000\006\000\000\003", !14, !1, !3, null, i32 ()* @main, null, null, null} ; [ DW_TAG_subprogram ] [line 3] [def] [main]
!1 = !{!"0x29", !14} ; [ DW_TAG_file_type ]
!2 = !{!"0x11\0012\00clang version 3.0 (trunk 129138)\000\00\000\00\000", !14, !15, !15, !13, null,  null} ; [ DW_TAG_compile_unit ]
!3 = !{!"0x15\00\000\000\000\000\000\000", !14, !1, null, !4, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!4 = !{!5}
!5 = !{!"0x24\00int\000\0032\0032\000\000\005", null, !2} ; [ DW_TAG_base_type ]
!6 = !{!"0x100\00a\004\000", !7, !1, !8} ; [ DW_TAG_auto_variable ]
!7 = !{!"0xb\003\0012\000", !14, !0} ; [ DW_TAG_lexical_block ]
!8 = !{!"0x1\00\000\000\0032\000\000", !14, !2, !5, !9, i32 0, null, null, null} ; [ DW_TAG_array_type ] [line 0, size 0, align 32, offset 0] [from int]
!9 = !{!10}
;CHECK: DW_TAG_subrange_type
;CHECK-NEXT: DW_AT_type
;CHECK-NOT: DW_AT_lower_bound
;CHECK-NOT: DW_AT_upper_bound
;CHECK-NEXT: End Of Children Mark
!10 = !{!"0x21\000\00-1"}        ; [ DW_TAG_subrange_type ]
!11 = !MDLocation(line: 4, column: 7, scope: !7)
!12 = !MDLocation(line: 5, column: 3, scope: !7)
!13 = !{!0}
!14 = !{!"array.c", !"/private/tmp"}
!15 = !{i32 0}
!16 = !{i32 1, !"Debug Info Version", i32 2}
