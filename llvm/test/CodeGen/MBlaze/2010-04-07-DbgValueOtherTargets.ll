; RUN: llc -O0 -march=mblaze -asm-verbose < %s | FileCheck %s
; Check that DEBUG_VALUE comments come through on a variety of targets.

define i32 @main() nounwind ssp {
entry:
; CHECK: DEBUG_VALUE
  call void @llvm.dbg.value(metadata !6, i64 0, metadata !7), !dbg !9
  ret i32 0, !dbg !10
}

declare void @llvm.dbg.declare(metadata, metadata) nounwind readnone

declare void @llvm.dbg.value(metadata, i64, metadata) nounwind readnone

!llvm.dbg.sp = !{!0}

!0 = metadata !{i32 589870, i32 0, metadata !1, metadata !"main", metadata !"main", metadata !"", metadata !1, i32 2, metadata !3, i1 false, i1 true, i32 0, i32 0, null, i32 0, i1 false, i32 ()* @main} ; [ DW_TAG_subprogram ]
!1 = metadata !{i32 589865, metadata !"/tmp/x.c", metadata !"/Users/manav", metadata !2} ; [ DW_TAG_file_type ]
!2 = metadata !{i32 589841, i32 0, i32 12, metadata !"/tmp/x.c", metadata !"/Users/manav", metadata !"clang version 2.9 (trunk 120996)", i1 true, i1 false, metadata !"", i32 0} ; [ DW_TAG_compile_unit ]
!3 = metadata !{i32 589845, metadata !1, metadata !"", metadata !1, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !4, i32 0, null} ; [ DW_TAG_subroutine_type ]
!4 = metadata !{metadata !5}
!5 = metadata !{i32 589860, metadata !2, metadata !"int", metadata !1, i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!6 = metadata !{i32 0}
!7 = metadata !{i32 590080, metadata !8, metadata !"i", metadata !1, i32 3, metadata !5, i32 0} ; [ DW_TAG_auto_variable ]
!8 = metadata !{i32 589835, metadata !0, i32 2, i32 12, metadata !1, i32 0} ; [ DW_TAG_lexical_block ]
!9 = metadata !{i32 3, i32 11, metadata !8, null}
!10 = metadata !{i32 4, i32 2, metadata !8, null}

