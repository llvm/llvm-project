; RUN: llc < %s -o /dev/null

define void @baz(i32 %i) nounwind ssp {
entry:
  call void @llvm.dbg.declare(metadata !0, metadata !1), !dbg !0
  ret void, !dbg !0
}

declare void @llvm.dbg.declare(metadata, metadata) nounwind readnone

!0 = metadata !{{ [0 x i8] }** undef}
!1 = metadata !{i32 524544, metadata !2, metadata !"x", metadata !4, i32 11, metadata !9} ; [ DW_TAG_auto_variable ]
!2 = metadata !{i32 524299, metadata !3, i32 8, i32 0} ; [ DW_TAG_lexical_block ]
!3 = metadata !{i32 524334, i32 0, metadata !4, metadata !"baz", metadata !"baz", metadata !"baz", metadata !4, i32 8, metadata !6, i1 true, i1 true, i32 0, i32 0, null, i1 false} ; [ DW_TAG_subprogram ]
!4 = metadata !{i32 524329, metadata !"2007-12-VarArrayDebug.c", metadata !"/Users/sabre/llvm/test/FrontendC/", metadata !5} ; [ DW_TAG_file_type ]
!5 = metadata !{i32 524305, i32 0, i32 1, metadata !"2007-12-VarArrayDebug.c", metadata !"/Users/sabre/llvm/test/FrontendC/", metadata !"4.2.1 (Based on Apple Inc. build 5658) (LLVM build)", i1 true, i1 false, metadata !"", i32 0} ; [ DW_TAG_compile_unit ]
!6 = metadata !{i32 524309, metadata !4, metadata !"", metadata !4, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !7, i32 0, null} ; [ DW_TAG_subroutine_type ]
!7 = metadata !{null, metadata !8}
!8 = metadata !{i32 524324, metadata !4, metadata !"int", metadata !4, i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!9 = metadata !{i32 524303, metadata !4, metadata !"", metadata !4, i32 0, i64 64, i64 64, i64 0, i32 0, metadata !10} ; [ DW_TAG_pointer_type ]
!10 = metadata !{i32 524307, metadata !3, metadata !"", metadata !4, i32 11, i64 8, i64 8, i64 0, i32 0, null, metadata !11, i32 0, null} ; [ DW_TAG_structure_type ]
!11 = metadata !{metadata !12}
!12 = metadata !{i32 524301, metadata !10, metadata !"b", metadata !4, i32 11, i64 8, i64 8, i64 0, i32 0, metadata !13} ; [ DW_TAG_member ]
!13 = metadata !{i32 524310, metadata !3, metadata !"A", metadata !4, i32 11, i64 0, i64 0, i64 0, i32 0, metadata !14} ; [ DW_TAG_typedef ]
!14 = metadata !{i32 524289, metadata !4, metadata !"", metadata !4, i32 0, i64 8, i64 8, i64 0, i32 0, metadata !15, metadata !16, i32 0, null} ; [ DW_TAG_array_type ]
!15 = metadata !{i32 524324, metadata !4, metadata !"char", metadata !4, i32 0, i64 8, i64 8, i64 0, i32 0, i32 6} ; [ DW_TAG_base_type ]
!16 = metadata !{metadata !17}
!17 = metadata !{i32 524321, i64 0, i64 0}        ; [ DW_TAG_subrange_type ]
!18 = metadata !{metadata !"llvm.mdnode.fwdref.19"}
!19 = metadata !{metadata !"llvm.mdnode.fwdref.23"}
