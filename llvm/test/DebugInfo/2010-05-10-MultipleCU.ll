; RUN: llc -O0 -asm-verbose < %s | FileCheck %s
; One for a.c, second one for b.c and third one for abbrev.

; CHECK: info_begin
; CHECK: DW_TAG_compile_unit
; CHECK-NOT: DW_TAG_compile_unit
; CHECK: info_end

; CHECK: info_begin
; CHECK: DW_TAG_compile_unit
; CHECK-NOT: DW_TAG_compile_unit
; CHECK: info_end

; CHECK: abbrev_begin
; CHECK: DW_TAG_compile_unit
; CHECK-NOT: DW_TAG_compile_unit
; CHECK: abbrev_end

define i32 @foo() nounwind readnone ssp {
return:
  ret i32 42, !dbg !0
}

define i32 @bar() nounwind readnone ssp {
return:
  ret i32 21, !dbg !8
}

!0 = metadata !{i32 3, i32 0, metadata !1, null}
!1 = metadata !{i32 524299, metadata !2, i32 2, i32 0} ; [ DW_TAG_lexical_block ]
!2 = metadata !{i32 524334, i32 0, metadata !3, metadata !"foo", metadata !"foo", metadata !"foo", metadata !3, i32 2, metadata !5, i1 false, i1 true, i32 0, i32 0, null, i1 false, i1 false} ; [ DW_TAG_subprogram ]
!3 = metadata !{i32 524329, metadata !"a.c", metadata !"/tmp/", metadata !4} ; [ DW_TAG_file_type ]
!4 = metadata !{i32 524305, i32 0, i32 1, metadata !"a.c", metadata !"/tmp/", metadata !"4.2.1 (Based on Apple Inc. build 5658) (LLVM build)", i1 true, i1 false, metadata !"", i32 0} ; [ DW_TAG_compile_unit ]
!5 = metadata !{i32 524309, metadata !3, metadata !"", metadata !3, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !6, i32 0, null} ; [ DW_TAG_subroutine_type ]
!6 = metadata !{metadata !7}
!7 = metadata !{i32 524324, metadata !3, metadata !"int", metadata !3, i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!8 = metadata !{i32 3, i32 0, metadata !9, null}
!9 = metadata !{i32 524299, metadata !10, i32 2, i32 0} ; [ DW_TAG_lexical_block ]
!10 = metadata !{i32 524334, i32 0, metadata !11, metadata !"bar", metadata !"bar", metadata !"bar", metadata !11, i32 2, metadata !13, i1 false, i1 true, i32 0, i32 0, null, i1 false, i1 false} ; [ DW_TAG_subprogram ]
!11 = metadata !{i32 524329, metadata !"b.c", metadata !"/tmp/", metadata !12} ; [ DW_TAG_file_type ]
!12 = metadata !{i32 524305, i32 0, i32 1, metadata !"b.c", metadata !"/tmp/", metadata !"4.2.1 (Based on Apple Inc. build 5658) (LLVM build)", i1 true, i1 false, metadata !"", i32 0} ; [ DW_TAG_compile_unit ]
!13 = metadata !{i32 524309, metadata !11, metadata !"", metadata !11, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !14, i32 0, null} ; [ DW_TAG_subroutine_type ]
!14 = metadata !{metadata !15}
!15 = metadata !{i32 524324, metadata !11, metadata !"int", metadata !11, i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
