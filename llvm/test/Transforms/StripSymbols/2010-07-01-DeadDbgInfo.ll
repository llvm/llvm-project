; RUN: opt -strip-dead-debug-info < %s | llvm-dis -o %t.ll
; RUN: grep -v bar %t.ll
; RUN: grep -v abcd %t.ll

@xyz = global i32 2                               ; <i32*> [#uses=1]

declare void @llvm.dbg.value(metadata, i64, metadata) nounwind readnone

define i32 @fn() nounwind readnone ssp {
entry:
  ret i32 0, !dbg !17
}

define i32 @foo(i32 %i) nounwind readonly ssp {
entry:
  tail call void @llvm.dbg.value(metadata !{i32 %i}, i64 0, metadata !14), !dbg !19
  %.0 = load i32* @xyz, align 4                   ; <i32> [#uses=1]
  ret i32 %.0, !dbg !20
}

!llvm.dbg.sp = !{!0, !5, !9}
!llvm.dbg.lv.bar = !{!12}
!llvm.dbg.lv.foo = !{!14}
!llvm.dbg.gv = !{!15, !16}

!0 = metadata !{i32 524334, i32 0, metadata !1, metadata !"bar", metadata !"bar", metadata !"", metadata !1, i32 5, metadata !3, i1 true, i1 true, i32 0, i32 0, null, i1 false, i1 true, null} ; [ DW_TAG_subprogram ]
!1 = metadata !{i32 524329, metadata !"g.c", metadata !"/tmp/", metadata !2} ; [ DW_TAG_file_type ]
!2 = metadata !{i32 524305, i32 0, i32 1, metadata !"g.c", metadata !"/tmp/", metadata !"4.2.1 (Based on Apple Inc. build 5658) (LLVM build)", i1 true, i1 true, metadata !"", i32 0} ; [ DW_TAG_compile_unit ]
!3 = metadata !{i32 524309, metadata !1, metadata !"", metadata !1, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !4, i32 0, null} ; [ DW_TAG_subroutine_type ]
!4 = metadata !{null}
!5 = metadata !{i32 524334, i32 0, metadata !1, metadata !"fn", metadata !"fn", metadata !"fn", metadata !1, i32 6, metadata !6, i1 false, i1 true, i32 0, i32 0, null, i1 false, i1 true, i32 ()* @fn} ; [ DW_TAG_subprogram ]
!6 = metadata !{i32 524309, metadata !1, metadata !"", metadata !1, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !7, i32 0, null} ; [ DW_TAG_subroutine_type ]
!7 = metadata !{metadata !8}
!8 = metadata !{i32 524324, metadata !1, metadata !"int", metadata !1, i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!9 = metadata !{i32 524334, i32 0, metadata !1, metadata !"foo", metadata !"foo", metadata !"foo", metadata !1, i32 7, metadata !10, i1 false, i1 true, i32 0, i32 0, null, i1 false, i1 true, i32 (i32)* @foo} ; [ DW_TAG_subprogram ]
!10 = metadata !{i32 524309, metadata !1, metadata !"", metadata !1, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !11, i32 0, null} ; [ DW_TAG_subroutine_type ]
!11 = metadata !{metadata !8, metadata !8}
!12 = metadata !{i32 524544, metadata !13, metadata !"bb", metadata !1, i32 5, metadata !8} ; [ DW_TAG_auto_variable ]
!13 = metadata !{i32 524299, metadata !0, i32 5, i32 0} ; [ DW_TAG_lexical_block ]
!14 = metadata !{i32 524545, metadata !9, metadata !"i", metadata !1, i32 7, metadata !8} ; [ DW_TAG_arg_variable ]
!15 = metadata !{i32 524340, i32 0, metadata !1, metadata !"abcd", metadata !"abcd", metadata !"", metadata !1, i32 2, metadata !8, i1 true, i1 true, null} ; [ DW_TAG_variable ]
!16 = metadata !{i32 524340, i32 0, metadata !1, metadata !"xyz", metadata !"xyz", metadata !"", metadata !1, i32 3, metadata !8, i1 false, i1 true, i32* @xyz} ; [ DW_TAG_variable ]
!17 = metadata !{i32 6, i32 0, metadata !18, null}
!18 = metadata !{i32 524299, metadata !5, i32 6, i32 0} ; [ DW_TAG_lexical_block ]
!19 = metadata !{i32 7, i32 0, metadata !9, null}
!20 = metadata !{i32 10, i32 0, metadata !21, null}
!21 = metadata !{i32 524299, metadata !9, i32 7, i32 0} ; [ DW_TAG_lexical_block ]
