; RUN: llc -O0 < %s -o /dev/null
; llc should not crash on this invalid input.
; PR6588
declare void @llvm.dbg.declare(metadata, metadata) nounwind readnone

define void @foo() {
entry:
  call void @llvm.dbg.declare(metadata !{i32* undef}, metadata !0)
  ret void
}

!0 = metadata !{i32 524545, metadata !1, metadata !"sy", metadata !2, i32 890, metadata !7} ; [ DW_TAG_arg_variable ]
!1 = metadata !{i32 524334, i32 0, metadata !2, metadata !"foo", metadata !"foo", metadata !"foo", metadata !2, i32 892, metadata !4, i1 false, i1 true, i32 0, i32 0, null, i1 false} ; [ DW_TAG_subprogram ]
!2 = metadata !{i32 524329, metadata !"qpainter.h", metadata !"QtGui", metadata !3} ; [ DW_TAG_file_type ]
!3 = metadata !{i32 524305, i32 0, i32 4, metadata !"splineeditor.cpp", metadata !"editor", metadata !"clang 1.1", i1 true, i1 false, metadata !"", i32 0} ; [ DW_TAG_compile_unit ]
!4 = metadata !{i32 524309, metadata !5, metadata !"", metadata !5, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !6, i32 0, null} ; [ DW_TAG_subroutine_type ]
!5 = metadata !{i32 524329, metadata !"splineeditor.cpp", metadata !"src", metadata !3} ; [ DW_TAG_file_type ]
!6 = metadata !{null}
!7 = metadata !{i32 524324, metadata !5, metadata !"int", metadata !5, i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]

