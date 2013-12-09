; RUN: llc < %s -o /dev/null

define void @baz(i32 %i) nounwind ssp {
entry:
  call void @llvm.dbg.declare(metadata !0, metadata !1), !dbg !0
  ret void, !dbg !0
}

declare void @llvm.dbg.declare(metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!5}
!llvm.module.flags = !{!22}

!0 = metadata !{{ [0 x i8] }** undef}
!1 = metadata !{i32 524544, metadata !2, metadata !"x", metadata !4, i32 11, metadata !9} ; [ DW_TAG_auto_variable ]
!2 = metadata !{i32 524299, metadata !20, metadata !3, i32 8, i32 0, i32 0} ; [ DW_TAG_lexical_block ]
!3 = metadata !{i32 524334, metadata !20, null, metadata !"baz", metadata !"baz", metadata !"baz", i32 8, metadata !6, i1 true, i1 true, i32 0, i32 0, null, i1 false, i32 0, null, null, null, null, i32 0} ; [ DW_TAG_subprogram ]
!4 = metadata !{i32 524329, metadata !20} ; [ DW_TAG_file_type ]
!5 = metadata !{i32 524305, metadata !20, i32 1, metadata !"4.2.1 (Based on Apple Inc. build 5658) (LLVM build)", i1 true, metadata !"", i32 0, metadata !21, metadata !21, null, null, null, metadata !""} ; [ DW_TAG_compile_unit ]
!6 = metadata !{i32 524309, metadata !20, metadata !4, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !7, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{null, metadata !8}
!8 = metadata !{i32 524324, metadata !20, metadata !4, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!9 = metadata !{i32 524303, metadata !20, metadata !4, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, metadata !10} ; [ DW_TAG_pointer_type ]
!10 = metadata !{i32 524307, metadata !20, metadata !3, metadata !"", i32 11, i64 8, i64 8, i64 0, i32 0, null, metadata !11, i32 0, null, null, null} ; [ DW_TAG_structure_type ] [line 11, size 8, align 8, offset 0] [def] [from ]
!11 = metadata !{metadata !12}
!12 = metadata !{i32 524301, metadata !20, metadata !10, metadata !"b", i32 11, i64 8, i64 8, i64 0, i32 0, metadata !13} ; [ DW_TAG_member ]
!13 = metadata !{i32 524310, metadata !20, metadata !3, metadata !"A", i32 11, i64 0, i64 0, i64 0, i32 0, metadata !14} ; [ DW_TAG_typedef ]
!14 = metadata !{i32 524289, metadata !20, metadata !4, metadata !"", i32 0, i64 8, i64 8, i64 0, i32 0, metadata !15, metadata !16, i32 0, null, null, null} ; [ DW_TAG_array_type ] [line 0, size 8, align 8, offset 0] [from char]
!15 = metadata !{i32 524324, metadata !20, metadata !4, metadata !"char", i32 0, i64 8, i64 8, i64 0, i32 0, i32 6} ; [ DW_TAG_base_type ]
!16 = metadata !{metadata !17}
!17 = metadata !{i32 524321, i64 0, i64 1}        ; [ DW_TAG_subrange_type ]
!18 = metadata !{metadata !"llvm.mdnode.fwdref.19"}
!19 = metadata !{metadata !"llvm.mdnode.fwdref.23"}
!20 = metadata !{metadata !"2007-12-VarArrayDebug.c", metadata !"/Users/sabre/llvm/test/FrontendC/"}
!21 = metadata !{i32 0}
!22 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
