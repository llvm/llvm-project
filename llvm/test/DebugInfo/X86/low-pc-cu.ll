; RUN: llc -mtriple=x86_64-apple-darwin %s -o %t -filetype=obj
; RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck %s

; Check that we use DW_AT_low_pc

; CHECK: DW_TAG_compile_unit [1]
; CHECK: DW_AT_low_pc [DW_FORM_addr]       (0x0000000000000000)
; CHECK: DW_TAG_subprogram [2]

define i32 @_Z1qv() nounwind uwtable readnone ssp {
entry:
  ret i32 undef, !dbg !13
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!16}

!0 = metadata !{i32 786449, metadata !15, i32 4, metadata !"clang version 3.1 (trunk 153454) (llvm/trunk 153471)", i1 false, metadata !"", i32 0, metadata !1, metadata !1, metadata !3, metadata !1,  metadata !1, metadata !""} ; [ DW_TAG_compile_unit ]
!1 = metadata !{i32 0}
!3 = metadata !{metadata !5, metadata !12}
!5 = metadata !{i32 786478, metadata !6, null, metadata !"q", metadata !"q", metadata !"_Z1qv", i32 5, metadata !7, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, i32 ()* @_Z1qv, null, null, metadata !10, i32 0} ; [ DW_TAG_subprogram ] [line 5] [def] [scope 0] [q]
!6 = metadata !{i32 786473, metadata !15} ; [ DW_TAG_file_type ]
!7 = metadata !{i32 786453, i32 0, null, i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !8, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = metadata !{metadata !9}
!9 = metadata !{i32 786468, metadata !15, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!10 = metadata !{metadata !11}
!11 = metadata !{i32 786468}                      ; [ DW_TAG_base_type ]
!12 = metadata !{i32 786478, metadata !15, metadata !6, metadata !"t", metadata !"t", metadata !"", i32 2, metadata !7, i1 true, i1 true, i32 0, i32 0, null, i32 256, i1 false, null, null, null, metadata !10, i32 0} ; [ DW_TAG_subprogram ]
!13 = metadata !{i32 7, i32 1, metadata !14, null}
!14 = metadata !{i32 786443, metadata !5, i32 5, i32 1, metadata !6, i32 0} ; [ DW_TAG_lexical_block ]
!15 = metadata !{metadata !"foo.cpp", metadata !"/Users/echristo/tmp"}
!16 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
