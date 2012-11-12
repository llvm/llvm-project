; RUN: llc -mtriple=x86_64-apple-darwin %s -o %t -filetype=obj
; RUN: llvm-dwarfdump %t | FileCheck %s

; Check that we use DW_AT_low_pc

; CHECK: DW_TAG_compile_unit [1]
; CHECK: DW_AT_low_pc [DW_FORM_addr]       (0x0000000000000000)
; CHECK: DW_TAG_subprogram [2]

define i32 @_Z1qv() nounwind uwtable readnone ssp {
entry:
  ret i32 undef, !dbg !13
}

!llvm.dbg.cu = !{!0}

!0 = metadata !{i32 786449, i32 0, i32 4, metadata !"foo.cpp", metadata !"/Users/echristo/tmp", metadata !"clang version 3.1 (trunk 153454) (llvm/trunk 153471)", i1 true, i1 false, metadata !"", i32 0, metadata !1, metadata !1, metadata !3, metadata !1} ; [ DW_TAG_compile_unit ]
!1 = metadata !{metadata !2}
!2 = metadata !{i32 0}
!3 = metadata !{metadata !4}
!4 = metadata !{metadata !5, metadata !12}
!5 = metadata !{i32 786478, i32 0, metadata !6, metadata !"q", metadata !"q", metadata !"_Z1qv", metadata !6, i32 5, metadata !7, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, i32 ()* @_Z1qv, null, null, metadata !10} ; [ DW_TAG_subprogram ]
!6 = metadata !{i32 786473, metadata !"foo.cpp", metadata !"/Users/echristo/tmp", null} ; [ DW_TAG_file_type ]
!7 = metadata !{i32 786453, i32 0, metadata !"", i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !8, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!8 = metadata !{metadata !9}
!9 = metadata !{i32 786468, null, metadata !"int", null, i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!10 = metadata !{metadata !11}
!11 = metadata !{i32 786468}                      ; [ DW_TAG_base_type ]
!12 = metadata !{i32 786478, i32 0, metadata !6, metadata !"t", metadata !"t", metadata !"", metadata !6, i32 2, metadata !7, i1 true, i1 true, i32 0, i32 0, null, i32 256, i1 false, null, null, null, metadata !10} ; [ DW_TAG_subprogram ]
!13 = metadata !{i32 7, i32 1, metadata !14, null}
!14 = metadata !{i32 786443, metadata !5, i32 5, i32 1, metadata !6, i32 0} ; [ DW_TAG_lexical_block ]
