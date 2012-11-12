; RUN: llc -mtriple=x86_64-apple-darwin %s -o %t -filetype=obj
; RUN: llvm-dwarfdump %t | FileCheck %s

; test that the DW_AT_specification is a back edge in the file.

; CHECK: 0x0000003a: DW_TAG_subprogram [5] *
; CHECK: 0x00000060: DW_AT_specification [DW_FORM_ref4]      (cu + 0x003a => {0x0000003a})


@_ZZN3foo3barEvE1x = constant i32 0, align 4

define void @_ZN3foo3barEv()  {
entry:
  ret void, !dbg !25
}

!llvm.dbg.cu = !{!0}

!0 = metadata !{i32 720913, i32 0, i32 4, metadata !"<unknown>", metadata !"/Users/espindola/mozilla-central/obj-x86_64-apple-darwin11.2.0/toolkit/library", metadata !"clang version 3.0 ()", i1 true, i1 false, metadata !"", i32 0, metadata !1, metadata !1, metadata !3, metadata !18} ; [ DW_TAG_compile_unit ]
!1 = metadata !{metadata !2}
!2 = metadata !{i32 0}
!3 = metadata !{metadata !4}
!4 = metadata !{metadata !5}
!5 = metadata !{i32 720942, i32 0, null, metadata !"bar", metadata !"bar", metadata !"_ZN3foo3barEv", metadata !6, i32 4, metadata !7, i1 false, i1 true, i32 0, i32 0, i32 0, i32 256, i1 false, void ()* @_ZN3foo3barEv, null, metadata !11, metadata !16} ; [ DW_TAG_subprogram ]
!6 = metadata !{i32 720937, metadata !"nsNativeAppSupportBase.ii", metadata !"/Users/espindola/mozilla-central/obj-x86_64-apple-darwin11.2.0/toolkit/library", null} ; [ DW_TAG_file_type ]
!7 = metadata !{i32 720917, i32 0, metadata !"", i32 0, i32 0, i64 0, i64 0, i32 0, i32 0, i32 0, metadata !8, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!8 = metadata !{null, metadata !9}
!9 = metadata !{i32 720911, i32 0, metadata !"", i32 0, i32 0, i64 64, i64 64, i64 0, i32 64, metadata !10} ; [ DW_TAG_pointer_type ]
!10 = metadata !{i32 720915, null, metadata !"foo", metadata !6, i32 1, i64 0, i64 0, i32 0, i32 4, i32 0, null, i32 0, i32 0} ; [ DW_TAG_structure_type ]
!11 = metadata !{i32 720942, i32 0, metadata !12, metadata !"bar", metadata !"bar", metadata !"_ZN3foo3barEv", metadata !6, i32 2, metadata !7, i1 false, i1 false, i32 0, i32 0, null, i32 256, i1 false, null, null, i32 0, metadata !14} ; [ DW_TAG_subprogram ]
!12 = metadata !{i32 720898, null, metadata !"foo", metadata !6, i32 1, i64 8, i64 8, i32 0, i32 0, null, metadata !13, i32 0, null, null} ; [ DW_TAG_class_type ]
!13 = metadata !{metadata !11}
!14 = metadata !{metadata !15}
!15 = metadata !{i32 720932}                      ; [ DW_TAG_base_type ]
!16 = metadata !{metadata !17}
!17 = metadata !{i32 720932}                      ; [ DW_TAG_base_type ]
!18 = metadata !{metadata !19}
!19 = metadata !{metadata !20}
!20 = metadata !{i32 720948, i32 0, metadata !5, metadata !"x", metadata !"x", metadata !"", metadata !6, i32 5, metadata !21, i32 1, i32 1, i32* @_ZZN3foo3barEvE1x} ; [ DW_TAG_variable ]
!21 = metadata !{i32 720934, null, metadata !"", null, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !22} ; [ DW_TAG_const_type ]
!22 = metadata !{i32 720932, null, metadata !"int", null, i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!25 = metadata !{i32 6, i32 1, metadata !26, null}
!26 = metadata !{i32 720907, metadata !5, i32 4, i32 17, metadata !6, i32 0} ; [ DW_TAG_lexical_block ]
