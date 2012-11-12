; RUN: llc -O2 %s -o - | FileCheck %s
; Check struct X for dead variable xyz from inlined function foo.

; CHECK:	DW_TAG_structure_type
; CHECK-NEXT:	DW_AT_name
 

@i = common global i32 0                          ; <i32*> [#uses=2]

declare void @llvm.dbg.declare(metadata, metadata) nounwind readnone

declare void @llvm.dbg.value(metadata, i64, metadata) nounwind readnone

define i32 @bar() nounwind ssp {
entry:
  %0 = load i32* @i, align 4, !dbg !17            ; <i32> [#uses=2]
  tail call void @llvm.dbg.value(metadata !{i32 %0}, i64 0, metadata !9), !dbg !19
  tail call void @llvm.dbg.declare(metadata !20, metadata !10), !dbg !21
  %1 = mul nsw i32 %0, %0, !dbg !22               ; <i32> [#uses=2]
  store i32 %1, i32* @i, align 4, !dbg !17
  ret i32 %1, !dbg !23
}

!llvm.dbg.sp = !{!0, !6}
!llvm.dbg.lv.foo = !{!9, !10}
!llvm.dbg.gv = !{!16}

!0 = metadata !{i32 524334, i32 0, metadata !1, metadata !"foo", metadata !"foo", metadata !"", metadata !1, i32 9, metadata !3, i1 true, i1 true, i32 0, i32 0, null, i1 false, i1 true, null, null, null, metadata !24} ; [ DW_TAG_subprogram ]
!1 = metadata !{i32 524329, metadata !"bar.c", metadata !"/tmp/", metadata !2} ; [ DW_TAG_file_type ]
!2 = metadata !{i32 524305, i32 0, i32 1, metadata !"bar.c", metadata !"/tmp/", metadata !"4.2.1 (Based on Apple Inc. build 5658) (LLVM build)", i1 true, i1 true, metadata !"", i32 0} ; [ DW_TAG_compile_unit ]
!3 = metadata !{i32 524309, metadata !1, metadata !"", metadata !1, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !4, i32 0, null} ; [ DW_TAG_subroutine_type ]
!4 = metadata !{metadata !5, metadata !5}
!5 = metadata !{i32 524324, metadata !1, metadata !"int", metadata !1, i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!6 = metadata !{i32 524334, i32 0, metadata !1, metadata !"bar", metadata !"bar", metadata !"bar", metadata !1, i32 14, metadata !7, i1 false, i1 true, i32 0, i32 0, null, i1 false, i1 true, i32 ()* @bar} ; [ DW_TAG_subprogram ]
!7 = metadata !{i32 524309, metadata !1, metadata !"", metadata !1, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !8, i32 0, null} ; [ DW_TAG_subroutine_type ]
!8 = metadata !{metadata !5}
!9 = metadata !{i32 524545, metadata !0, metadata !"j", metadata !1, i32 9, metadata !5} ; [ DW_TAG_arg_variable ]
!10 = metadata !{i32 524544, metadata !11, metadata !"xyz", metadata !1, i32 10, metadata !12} ; [ DW_TAG_auto_variable ]
!11 = metadata !{i32 524299, metadata !0, i32 9, i32 0} ; [ DW_TAG_lexical_block ]
!12 = metadata !{i32 524307, metadata !0, metadata !"X", metadata !1, i32 10, i64 64, i64 32, i64 0, i32 0, null, metadata !13, i32 0, null} ; [ DW_TAG_structure_type ]
!13 = metadata !{metadata !14, metadata !15}
!14 = metadata !{i32 524301, metadata !12, metadata !"a", metadata !1, i32 10, i64 32, i64 32, i64 0, i32 0, metadata !5} ; [ DW_TAG_member ]
!15 = metadata !{i32 524301, metadata !12, metadata !"b", metadata !1, i32 10, i64 32, i64 32, i64 32, i32 0, metadata !5} ; [ DW_TAG_member ]
!16 = metadata !{i32 524340, i32 0, metadata !1, metadata !"i", metadata !"i", metadata !"", metadata !1, i32 5, metadata !5, i1 false, i1 true, i32* @i} ; [ DW_TAG_variable ]
!17 = metadata !{i32 15, i32 0, metadata !18, null}
!18 = metadata !{i32 524299, metadata !6, i32 14, i32 0} ; [ DW_TAG_lexical_block ]
!19 = metadata !{i32 9, i32 0, metadata !0, metadata !17}
!20 = metadata !{null}
!21 = metadata !{i32 9, i32 0, metadata !11, metadata !17}
!22 = metadata !{i32 11, i32 0, metadata !11, metadata !17}
!23 = metadata !{i32 16, i32 0, metadata !18, null}
!24 = metadata !{metadata !25}
!25 = metadata !{metadata !9, metadata !10}

