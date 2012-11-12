; RUN: llc  -mtriple=i686-linux -O0 < %s | FileCheck %s
; CHECK: DW_TAG_constant
; CHECK-NEXT: .long .Lstring3 #{{#?}} DW_AT_name

define void @foo() nounwind ssp {
entry:
  call void @bar(i32 201), !dbg !8
  ret void, !dbg !8
}

declare void @bar(i32)

!llvm.dbg.sp = !{!0}
!llvm.dbg.gv = !{!5}

!0 = metadata !{i32 524334, i32 0, metadata !1, metadata !"foo", metadata !"foo", metadata !"foo", metadata !1, i32 3, metadata !3, i1 false, i1 true, i32 0, i32 0, null, i1 false, i1 false, void ()* @foo} ; [ DW_TAG_subprogram ]
!1 = metadata !{i32 524329, metadata !"/tmp/l.c", metadata !"/Volumes/Lalgate/clean/D", metadata !2} ; [ DW_TAG_file_type ]
!2 = metadata !{i32 524305, i32 0, i32 12, metadata !"/tmp/l.c", metadata !"/Volumes/Lalgate/clean/D", metadata !"clang 2.8", i1 true, i1 false, metadata !"", i32 0} ; [ DW_TAG_compile_unit ]
!3 = metadata !{i32 524309, metadata !1, metadata !"", metadata !1, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !4, i32 0, null} ; [ DW_TAG_subroutine_type ]
!4 = metadata !{null}
!5 = metadata !{i32 524327, i32 0, metadata !1, metadata !"ro", metadata !"ro", metadata !"ro", metadata !1, i32 1, metadata !6, i1 true, i1 true, i32 201} ; [ DW_TAG_constant ]
!6 = metadata !{i32 524326, metadata !1, metadata !"", metadata !1, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !7} ; [ DW_TAG_const_type ]
!7 = metadata !{i32 524324, metadata !1, metadata !"unsigned int", metadata !1, i32 0, i64 32, i64 32, i64 0, i32 0, i32 7} ; [ DW_TAG_base_type ]
!8 = metadata !{i32 3, i32 14, metadata !9, null}
!9 = metadata !{i32 524299, metadata !0, i32 3, i32 12, metadata !1, i32 0} ; [ DW_TAG_lexical_block ]
