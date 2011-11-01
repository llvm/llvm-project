; RUN: llc -mtriple=x86_64-linux < %s | FileCheck %s

; CHECK: DW_AT_const_value
; CHECK-NEXT: 42

define i128 @__foo(i128 %a, i128 %b) nounwind {
entry:
  tail call void @llvm.dbg.value(metadata !0, i64 0, metadata !1), !dbg !11
  %add = add i128 %a, %b, !dbg !11
  ret i128 %add, !dbg !11
}

declare void @llvm.dbg.value(metadata, i64, metadata) nounwind readnone

!0 = metadata !{i128 42 }
!1 = metadata !{i32 524544, metadata !2, metadata !"MAX", metadata !4, i32 29, metadata !8} ; [ DW_TAG_auto_variable ]
!2 = metadata !{i32 524299, metadata !3, i32 26, i32 0} ; [ DW_TAG_lexical_block ]
!3 = metadata !{i32 524334, i32 0, metadata !4, metadata !"__foo", metadata !"__foo", metadata !"__foo", metadata !4, i32 26, metadata !6, i1 false, i1 true, i32 0, i32 0, null, i1 false} ; [ DW_TAG_subprogram ]
!4 = metadata !{i32 524329, metadata !"foo.c", metadata !"/tmp", metadata !5} ; [ DW_TAG_file_type ]
!5 = metadata !{i32 524305, i32 0, i32 1, metadata !"foo.c", metadata !"/tmp", metadata !"clang", i1 true, i1 true, metadata !"", i32 0} ; [ DW_TAG_compile_unit ]
!6 = metadata !{i32 524309, metadata !4, metadata !"", metadata !4, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !7, i32 0, null} ; [ DW_TAG_subroutine_type ]
!7 = metadata !{metadata !8, metadata !8, metadata !8}
!8 = metadata !{i32 524310, metadata !4, metadata !"ti_int", metadata !9, i32 78, i64 0, i64 0, i64 0, i32 0, metadata !10} ; [ DW_TAG_typedef ]
!9 = metadata !{i32 524329, metadata !"myint.h", metadata !"/tmp", metadata !5} ; [ DW_TAG_file_type ]
!10 = metadata !{i32 524324, metadata !4, metadata !"", metadata !4, i32 0, i64 128, i64 128, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!11 = metadata !{i32 29, i32 0, metadata !2, null}
