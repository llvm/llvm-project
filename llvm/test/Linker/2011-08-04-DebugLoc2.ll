; This file is used by 2011-08-04-DebugLoc.ll, so it doesn't actually do anything itself
;
; RUN: true


target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-macosx10.7.0"

define i32 @bar() nounwind ssp {
  ret i32 21, !dbg !6
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!11}
!llvm.dbg.sp = !{!1}

!0 = !{!"0x11\0012\00Apple clang version 3.0 (tags/Apple/clang-209.11) (based on LLVM 3.0svn)\001\00\000\00\000", !8, !9, !9, !10, null, null} ; [ DW_TAG_compile_unit ]
!1 = !{!"0x2e\00bar\00bar\00\001\000\001\000\006\000\000\000", !8, !2, !3, null, i32 ()* @bar, null, null, null} ; [ DW_TAG_subprogram ] [line 1] [def] [scope 0] [bar]
!2 = !{!"0x29", !8} ; [ DW_TAG_file_type ]
!3 = !{!"0x15\00\000\000\000\000\000\000", !8, !2, null, !4, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!4 = !{!5}
!5 = !{!"0x24\00int\000\0032\0032\000\000\005", null, !0} ; [ DW_TAG_base_type ]
!6 = !MDLocation(line: 1, column: 13, scope: !7)
!7 = !{!"0xb\001\0011\000", !8, !1} ; [ DW_TAG_lexical_block ]
!8 = !{!"b.c", !"/private/tmp"}
!9 = !{i32 0}
!10 = !{!1}
!11 = !{i32 1, !"Debug Info Version", i32 2}
