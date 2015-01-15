; RUN: llc %s -o %t -filetype=obj -mtriple=x86_64-unknown-linux-gnu
; RUN: llvm-dwarfdump -debug-dump=line %t | FileCheck %s
; PR13303

; Check that the prologue ends with is_stmt here.
; CHECK: 0x0000000000000000 {{.*}} is_stmt

define i32 @main() nounwind uwtable {
entry:
  %retval = alloca i32, align 4
  store i32 0, i32* %retval
  ret i32 0, !dbg !10
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!13}

!0 = !{!"0x11\0012\00clang version 3.2 (trunk 160143)\000\00\000\00\000", !12, !1, !1, !3, !1,  !1} ; [ DW_TAG_compile_unit ] [/home/probinson/PR13303.c] [DW_LANG_C99]
!1 = !{}
!3 = !{!5}
!5 = !{!"0x2e\00main\00main\00\001\000\001\000\006\000\000\001", !12, !6, !7, null, i32 ()* @main, null, null, !1} ; [ DW_TAG_subprogram ] [line 1] [def] [main]
!6 = !{!"0x29", !12} ; [ DW_TAG_file_type ]
!7 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !8, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = !{!9}
!9 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!10 = !MDLocation(line: 1, column: 14, scope: !11)
!11 = !{!"0xb\001\0012\000", !12, !5} ; [ DW_TAG_lexical_block ] [/home/probinson/PR13303.c]
!12 = !{!"PR13303.c", !"/home/probinson"}
!13 = !{i32 1, !"Debug Info Version", i32 2}
