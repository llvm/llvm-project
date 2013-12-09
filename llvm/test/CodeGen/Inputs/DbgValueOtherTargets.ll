; Check that DEBUG_VALUE comments come through on a variety of targets.

define i32 @main() nounwind ssp {
entry:
; CHECK: DEBUG_VALUE
  call void @llvm.dbg.value(metadata !6, i64 0, metadata !7), !dbg !9
  ret i32 0, !dbg !10
}

declare void @llvm.dbg.declare(metadata, metadata) nounwind readnone

declare void @llvm.dbg.value(metadata, i64, metadata) nounwind readnone

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!13}

!0 = metadata !{i32 786478, metadata !12, metadata !1, metadata !"main", metadata !"main", metadata !"", i32 2, metadata !3, i1 false, i1 true, i32 0, i32 0, null, i32 0, i1 false, i32 ()* @main, null, null, null, i32 0} ; [ DW_TAG_subprogram ]
!1 = metadata !{i32 786473, metadata !12} ; [ DW_TAG_file_type ]
!2 = metadata !{i32 786449, metadata !12, i32 12, metadata !"clang version 2.9 (trunk 120996)", i1 false, metadata !"", i32 0, metadata !6, metadata !6, metadata !11, null, null, metadata !""} ; [ DW_TAG_compile_unit ]
!3 = metadata !{i32 786453, metadata !12, metadata !1, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !4, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!4 = metadata !{metadata !5}
!5 = metadata !{i32 786468, metadata !12, metadata !2, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!6 = metadata !{i32 0}
!7 = metadata !{i32 786688, metadata !8, metadata !"i", metadata !1, i32 3, metadata !5, i32 0, null} ; [ DW_TAG_auto_variable ]
!8 = metadata !{i32 786443, metadata !12, metadata !0, i32 2, i32 12, i32 0} ; [ DW_TAG_lexical_block ]
!9 = metadata !{i32 3, i32 11, metadata !8, null}
!10 = metadata !{i32 4, i32 2, metadata !8, null}
!11 = metadata !{metadata !0}
!12 = metadata !{metadata !"/tmp/x.c", metadata !"/Users/manav"}
!13 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
