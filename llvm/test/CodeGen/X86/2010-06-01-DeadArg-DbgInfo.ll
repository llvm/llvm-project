; RUN: llc -O2 < %s | FileCheck %s
; RUN: llc -O2 -regalloc=basic < %s | FileCheck %s
; Test to check that unused argument 'this' is not undefined in debug info.

target triple = "x86_64-apple-darwin10.2"
%struct.foo = type { i32 }

@llvm.used = appending global [1 x i8*] [i8* bitcast (i32 (%struct.foo*, i32)* @_ZN3foo3bazEi to i8*)], section "llvm.metadata" ; <[1 x i8*]*> [#uses=0]

define i32 @_ZN3foo3bazEi(%struct.foo* nocapture %this, i32 %x) nounwind readnone optsize noinline ssp align 2 {
;CHECK: DEBUG_VALUE: baz:this <- RDI+0
entry:
  tail call void @llvm.dbg.value(metadata !{%struct.foo* %this}, i64 0, metadata !15)
  tail call void @llvm.dbg.value(metadata !{i32 %x}, i64 0, metadata !16)
  %0 = mul nsw i32 %x, 7, !dbg !29                ; <i32> [#uses=1]
  %1 = add nsw i32 %0, 1, !dbg !29                ; <i32> [#uses=1]
  ret i32 %1, !dbg !29
}

declare void @llvm.dbg.value(metadata, i64, metadata) nounwind readnone

!llvm.dbg.lv = !{!0, !14, !15, !16, !17, !24, !25, !28}

!0 = metadata !{i32 524545, metadata !1, metadata !"this", metadata !3, i32 11, metadata !12} ; [ DW_TAG_arg_variable ]
!1 = metadata !{i32 524334, i32 0, metadata !2, metadata !"bar", metadata !"bar", metadata !"_ZN3foo3barEi", metadata !3, i32 11, metadata !9, i1 false, i1 true, i32 0, i32 0, null, i1 false, i1 true} ; [ DW_TAG_subprogram ]
!2 = metadata !{i32 524307, metadata !3, metadata !"foo", metadata !3, i32 3, i64 32, i64 32, i64 0, i32 0, null, metadata !5, i32 0, null} ; [ DW_TAG_structure_type ]
!3 = metadata !{i32 524329, metadata !"foo.cp", metadata !"/tmp/", metadata !4} ; [ DW_TAG_file_type ]
!4 = metadata !{i32 524305, i32 0, i32 4, metadata !"foo.cp", metadata !"/tmp/", metadata !"4.2.1 LLVM build", i1 true, i1 true, metadata !"", i32 0} ; [ DW_TAG_compile_unit ]
!5 = metadata !{metadata !6, metadata !1, metadata !8}
!6 = metadata !{i32 524301, metadata !2, metadata !"y", metadata !3, i32 8, i64 32, i64 32, i64 0, i32 0, metadata !7} ; [ DW_TAG_member ]
!7 = metadata !{i32 524324, metadata !3, metadata !"int", metadata !3, i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!8 = metadata !{i32 524334, i32 0, metadata !2, metadata !"baz", metadata !"baz", metadata !"_ZN3foo3bazEi", metadata !3, i32 15, metadata !9, i1 false, i1 true, i32 0, i32 0, null, i1 false, i1 true} ; [ DW_TAG_subprogram ]
!9 = metadata !{i32 524309, metadata !3, metadata !"", metadata !3, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !10, i32 0, null} ; [ DW_TAG_subroutine_type ]
!10 = metadata !{metadata !7, metadata !11, metadata !7}
!11 = metadata !{i32 524303, metadata !3, metadata !"", metadata !3, i32 0, i64 64, i64 64, i64 0, i32 64, metadata !2} ; [ DW_TAG_pointer_type ]
!12 = metadata !{i32 524326, metadata !3, metadata !"", metadata !3, i32 0, i64 64, i64 64, i64 0, i32 64, metadata !13} ; [ DW_TAG_const_type ]
!13 = metadata !{i32 524303, metadata !3, metadata !"", metadata !3, i32 0, i64 64, i64 64, i64 0, i32 0, metadata !2} ; [ DW_TAG_pointer_type ]
!14 = metadata !{i32 524545, metadata !1, metadata !"x", metadata !3, i32 11, metadata !7} ; [ DW_TAG_arg_variable ]
!15 = metadata !{i32 524545, metadata !8, metadata !"this", metadata !3, i32 15, metadata !12} ; [ DW_TAG_arg_variable ]
!16 = metadata !{i32 524545, metadata !8, metadata !"x", metadata !3, i32 15, metadata !7} ; [ DW_TAG_arg_variable ]
!17 = metadata !{i32 524545, metadata !18, metadata !"argc", metadata !3, i32 19, metadata !7} ; [ DW_TAG_arg_variable ]
!18 = metadata !{i32 524334, i32 0, metadata !3, metadata !"main", metadata !"main", metadata !"main", metadata !3, i32 19, metadata !19, i1 false, i1 true, i32 0, i32 0, null, i1 false, i1 true} ; [ DW_TAG_subprogram ]
!19 = metadata !{i32 524309, metadata !3, metadata !"", metadata !3, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !20, i32 0, null} ; [ DW_TAG_subroutine_type ]
!20 = metadata !{metadata !7, metadata !7, metadata !21}
!21 = metadata !{i32 524303, metadata !3, metadata !"", metadata !3, i32 0, i64 64, i64 64, i64 0, i32 0, metadata !22} ; [ DW_TAG_pointer_type ]
!22 = metadata !{i32 524303, metadata !3, metadata !"", metadata !3, i32 0, i64 64, i64 64, i64 0, i32 0, metadata !23} ; [ DW_TAG_pointer_type ]
!23 = metadata !{i32 524324, metadata !3, metadata !"char", metadata !3, i32 0, i64 8, i64 8, i64 0, i32 0, i32 6} ; [ DW_TAG_base_type ]
!24 = metadata !{i32 524545, metadata !18, metadata !"argv", metadata !3, i32 19, metadata !21} ; [ DW_TAG_arg_variable ]
!25 = metadata !{i32 524544, metadata !26, metadata !"a", metadata !3, i32 20, metadata !2} ; [ DW_TAG_auto_variable ]
!26 = metadata !{i32 524299, metadata !27, i32 19, i32 0} ; [ DW_TAG_lexical_block ]
!27 = metadata !{i32 524299, metadata !18, i32 19, i32 0} ; [ DW_TAG_lexical_block ]
!28 = metadata !{i32 524544, metadata !26, metadata !"b", metadata !3, i32 21, metadata !7} ; [ DW_TAG_auto_variable ]
!29 = metadata !{i32 16, i32 0, metadata !30, null}
!30 = metadata !{i32 524299, metadata !8, i32 15, i32 0} ; [ DW_TAG_lexical_block ]
