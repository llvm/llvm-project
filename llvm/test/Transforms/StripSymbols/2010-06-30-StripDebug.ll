; RUN: opt -strip-debug < %s -S | FileCheck %s

; CHECK-NOT: llvm.dbg

@x = common global i32 0                          ; <i32*> [#uses=0]

define void @foo() nounwind readnone optsize ssp {
entry:
  tail call void @llvm.dbg.value(metadata !9, i64 0, metadata !5), !dbg !10
  ret void, !dbg !11
}

declare void @llvm.dbg.value(metadata, i64, metadata) nounwind readnone

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!13}
!llvm.dbg.sp = !{!0}
!llvm.dbg.lv.foo = !{!5}
!llvm.dbg.gv = !{!8}

!0 = metadata !{i32 524334, metadata !12, metadata !1, metadata !"foo", metadata !"foo", metadata !"foo", i32 2, metadata !3, i1 false, i1 true, i32 0, i32 0, null, i1 false, i1 true, void ()* @foo, null, null, null, i32 0} ; [ DW_TAG_subprogram ]
!1 = metadata !{i32 524329, metadata !12} ; [ DW_TAG_file_type ]
!2 = metadata !{i32 524305, metadata !12, i32 1, metadata !"4.2.1 (Based on Apple Inc. build 5658) (LLVM build)", i1 true, metadata !"", i32 0, metadata !4, metadata !4, null, null, null, metadata !""} ; [ DW_TAG_compile_unit ]
!3 = metadata !{i32 524309, metadata !12, metadata !1, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !4, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!4 = metadata !{null}
!5 = metadata !{i32 524544, metadata !6, metadata !"y", metadata !1, i32 3, metadata !7} ; [ DW_TAG_auto_variable ]
!6 = metadata !{i32 524299, metadata !12, metadata !0, i32 2, i32 0, i32 0} ; [ DW_TAG_lexical_block ]
!7 = metadata !{i32 524324, metadata !12, metadata !1, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!8 = metadata !{i32 524340, i32 0, metadata !1, metadata !"x", metadata !"x", metadata !"", metadata !1, i32 1, metadata !7, i1 false, i1 true, i32* @x} ; [ DW_TAG_variable ]
!9 = metadata !{i32 0}
!10 = metadata !{i32 3, i32 0, metadata !6, null}
!11 = metadata !{i32 4, i32 0, metadata !6, null}
!12 = metadata !{metadata !"b.c", metadata !"/tmp"}
!13 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
