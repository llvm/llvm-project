; RUN: llc < %s -disable-dot-loc -mtriple=x86_64-apple-darwin -O0 | FileCheck %s


define void @foo() nounwind ssp {
entry:
  ret void, !dbg !5
}

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!10}
!7 = metadata !{metadata !0}

!0 = metadata !{i32 786478, metadata !9, metadata !1, metadata !"foo", metadata !"foo", metadata !"", i32 3, metadata !3, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void ()* @foo, null, null, null, i32 0} ; [ DW_TAG_subprogram ]
!1 = metadata !{i32 786473, metadata !9} ; [ DW_TAG_file_type ]
!2 = metadata !{i32 786449, metadata !9, i32 12, metadata !"clang version 2.9 (trunk 120563)", i1 false, metadata !"", i32 0, metadata !8, metadata !8, metadata !7, null, null, metadata !""} ; [ DW_TAG_compile_unit ]
!3 = metadata !{i32 786453, metadata !9, metadata !1, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !4, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!4 = metadata !{null}
!5 = metadata !{i32 5, i32 1, metadata !6, null}
!6 = metadata !{i32 786443, metadata !9, metadata !0, i32 3, i32 16, i32 0} ; [ DW_TAG_lexical_block ]
!8 = metadata !{i32 0}
!9 = metadata !{metadata !"e.c", metadata !"/private/tmp"}

; CHECK: .subsections_via_symbols
; CHECK-NEXT: __debug_line
; CHECK-NEXT: Lline_table_start0
; CHECK-NEXT: Ltmp{{[0-9]}} = (Ltmp
!10 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
