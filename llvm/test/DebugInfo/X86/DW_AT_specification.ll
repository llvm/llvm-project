; RUN: llc -mtriple=x86_64-apple-darwin %s -o %t -filetype=obj
; RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck %s

; test that the DW_AT_specification is a back edge in the file.

; CHECK: DW_TAG_subprogram [{{[0-9]+}}] *
; CHECK: DW_AT_specification [DW_FORM_ref4]      (cu + 0x[[OFFSET:[0-9a-f]*]] => {0x0000[[OFFSET]]})
; CHECK: 0x0000[[OFFSET]]: DW_TAG_subprogram [{{[0-9]+}}] *
; CHECK: DW_AT_name [DW_FORM_strp]	( .debug_str[0x{{[0-9a-f]*}}] = "bar")


@_ZZN3foo3barEvE1x = constant i32 0, align 4

define void @_ZN3foo3barEv()  {
entry:
  ret void, !dbg !25
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!28}

!0 = metadata !{i32 786449, metadata !27, i32 4, metadata !"clang version 3.0 ()", i1 false, metadata !"", i32 0, metadata !1, metadata !1, metadata !3, metadata !18,  metadata !18, metadata !""} ; [ DW_TAG_compile_unit ]
!1 = metadata !{i32 0}
!3 = metadata !{metadata !5}
!5 = metadata !{i32 720942, metadata !6, null, metadata !"bar", metadata !"bar", metadata !"_ZN3foo3barEv", i32 4, metadata !7, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void ()* @_ZN3foo3barEv, null, metadata !11, metadata !16, i32 4} ; [ DW_TAG_subprogram ] [line 4] [def] [bar]
!6 = metadata !{i32 720937, metadata !27} ; [ DW_TAG_file_type ]
!7 = metadata !{i32 720917, i32 0, null, i32 0, i32 0, i64 0, i64 0, i32 0, i32 0, null, metadata !8, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = metadata !{null, metadata !9}
!9 = metadata !{i32 786447, i32 0, null, i32 0, i32 0, i64 64, i64 64, i64 0, i32 64, metadata !10} ; [ DW_TAG_pointer_type ]
!10 = metadata !{i32 786451, metadata !27, null, metadata !"foo", i32 1, i64 0, i64 0, i32 0, i32 4, null, null, i32 0, null, null, null} ; [ DW_TAG_structure_type ] [foo] [line 1, size 0, align 0, offset 0] [decl] [from ]
!11 = metadata !{i32 720942, metadata !6, metadata !12, metadata !"bar", metadata !"bar", metadata !"_ZN3foo3barEv", i32 2, metadata !7, i1 false, i1 false, i32 0, i32 0, null, i32 256, i1 false, null, null, i32 0, metadata !14, i32 2} ; [ DW_TAG_subprogram ]
!12 = metadata !{i32 720898, metadata !27, null, metadata !"foo", i32 1, i64 8, i64 8, i32 0, i32 0, null, metadata !13, i32 0, null, null} ; [ DW_TAG_class_type ]
!13 = metadata !{metadata !11}
!14 = metadata !{metadata !15}
!15 = metadata !{i32 720932}                      ; [ DW_TAG_base_type ]
!16 = metadata !{metadata !17}
!17 = metadata !{i32 720932}                      ; [ DW_TAG_base_type ]
!18 = metadata !{metadata !20}
!20 = metadata !{i32 720948, i32 0, metadata !5, metadata !"x", metadata !"x", metadata !"", metadata !6, i32 5, metadata !21, i32 1, i32 1, i32* @_ZZN3foo3barEvE1x, null} ; [ DW_TAG_variable ]
!21 = metadata !{i32 720934, null, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, metadata !22} ; [ DW_TAG_const_type ]
!22 = metadata !{i32 720932, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!25 = metadata !{i32 6, i32 1, metadata !26, null}
!26 = metadata !{i32 786443, metadata !5, i32 4, i32 17, metadata !6, i32 0} ; [ DW_TAG_lexical_block ]
!27 = metadata !{metadata !"nsNativeAppSupportBase.ii", metadata !"/Users/espindola/mozilla-central/obj-x86_64-apple-darwin11.2.0/toolkit/library"}
!28 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
