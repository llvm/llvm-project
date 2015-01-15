; This file is for use with 2011-08-10-unique-debug-type.ll
; RUN: true

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-macosx10.7.0"

define i32 @bar() nounwind uwtable ssp {
entry:
  ret i32 2, !dbg !10
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!13}

!0 = !{!"0x11\0012\00clang version 3.0 (trunk 137954)\001\00\000\00\000", !12, !2, !2, !3, !2, null} ; [ DW_TAG_compile_unit ]
!1 = !{!2}
!2 = !{i32 0}
!3 = !{!5}
!5 = !{!"0x2e\00bar\00bar\00\001\000\001\000\006\000\000\000", !12, !6, !7, null, i32 ()* @bar, null, null, null} ; [ DW_TAG_subprogram ] [line 1] [def] [scope 0] [bar]
!6 = !{!"0x29", !12} ; [ DW_TAG_file_type ]
!7 = !{!"0x15\00\000\000\000\000\000\000", !12, !6, null, !8, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = !{!9}
!9 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ]
!10 = !MDLocation(line: 1, column: 13, scope: !11)
!11 = !{!"0xb\001\0011\000", !12, !5} ; [ DW_TAG_lexical_block ]
!12 = !{!"two.c", !"/private/tmp"}
!13 = !{i32 1, !"Debug Info Version", i32 2}
