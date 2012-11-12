; RUN: llc -O1 < %s
; ModuleID = 'pr6157.bc'
; formerly crashed in SelectionDAGBuilder

%tart.reflect.ComplexType = type { double, double }

@.type.SwitchStmtTest = constant %tart.reflect.ComplexType { double 3.0, double 2.0 }

define i32 @"main(tart.core.String[])->int32"(i32 %args) {
entry:
  tail call void @llvm.dbg.value(metadata !14, i64 0, metadata !8)
  tail call void @"tart.reflect.ComplexType.create->tart.core.Object"(%tart.reflect.ComplexType* @.type.SwitchStmtTest) ; <%tart.core.Object*> [#uses=2]
  ret i32 3
}

declare void @llvm.dbg.value(metadata, i64, metadata) nounwind readnone
declare void @"tart.reflect.ComplexType.create->tart.core.Object"(%tart.reflect.ComplexType*) nounwind readnone

!0 = metadata !{i32 458769, i32 0, i32 1, metadata !"sm.c", metadata !"/Volumes/MacOS9/tests/", metadata !"4.2.1 (Based on Apple Inc. build 5658) (LLVM build)", i1 true, i1 false, metadata !"", i32 0} ; [ DW_TAG_compile_unit ]
!1 = metadata !{i32 458790, metadata !0, metadata !"", metadata !0, i32 0, i64 192, i64 64, i64 0, i32 0, metadata !2} ; [ DW_TAG_const_type ]
!2 = metadata !{i32 458771, metadata !0, metadata !"C", metadata !0, i32 1, i64 192, i64 64, i64 0, i32 0, null, metadata !3, i32 0, null} ; [ DW_TAG_structure_type ]
!3 = metadata !{metadata !4, metadata !6, metadata !7}
!4 = metadata !{i32 458765, metadata !2, metadata !"x", metadata !0, i32 1, i64 64, i64 64, i64 0, i32 0, metadata !5} ; [ DW_TAG_member ]
!5 = metadata !{i32 458788, metadata !0, metadata !"double", metadata !0, i32 0, i64 64, i64 64, i64 0, i32 0, i32 4} ; [ DW_TAG_base_type ]
!6 = metadata !{i32 458765, metadata !2, metadata !"y", metadata !0, i32 1, i64 64, i64 64, i64 64, i32 0, metadata !5} ; [ DW_TAG_member ]
!7 = metadata !{i32 458765, metadata !2, metadata !"z", metadata !0, i32 1, i64 64, i64 64, i64 128, i32 0, metadata !5} ; [ DW_TAG_member ]
!8 = metadata !{i32 459008, metadata !9, metadata !"t", metadata !0, i32 5, metadata !2} ; [ DW_TAG_auto_variable ]
!9 = metadata !{i32 458763, metadata !10}        ; [ DW_TAG_lexical_block ]
!10 = metadata !{i32 458798, i32 0, metadata !0, metadata !"foo", metadata !"foo", metadata !"foo", metadata !0, i32 4, metadata !11, i1 false, i1 true, i32 0, i32 0, null} ; [ DW_TAG_subprogram ]
!11 = metadata !{i32 458773, metadata !0, metadata !"", metadata !0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !12, i32 0, null} ; [ DW_TAG_subroutine_type ]
!12 = metadata !{metadata !13}
!13 = metadata !{i32 458788, metadata !0, metadata !"int", metadata !0, i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!14 = metadata !{%tart.reflect.ComplexType* @.type.SwitchStmtTest}
