; RUN: llc  < %s | FileCheck %s
; Radar 10147769
; Do not unnecessarily use AT_specification DIE.
; CHECK-NOT: AT_specification

@a = common global [10 x i32] zeroinitializer, align 16

!llvm.dbg.cu = !{!0}

!0 = metadata !{i32 720913, i32 0, i32 12, metadata !"x.c", metadata !"/private/tmp", metadata !"clang version 3.0 (trunk 140253)", i1 true, i1 false, metadata !"", i32 0, metadata !1, metadata !1, metadata !1, metadata !3} ; [ DW_TAG_compile_unit ]
!1 = metadata !{metadata !2}
!2 = metadata !{i32 0}
!3 = metadata !{metadata !4}
!4 = metadata !{metadata !5}
!5 = metadata !{i32 720948, i32 0, null, metadata !"a", metadata !"a", metadata !"", metadata !6, i32 1, metadata !7, i32 0, i32 1, [10 x i32]* @a} ; [ DW_TAG_variable ]
!6 = metadata !{i32 720937, metadata !"x.c", metadata !"/private/tmp", null} ; [ DW_TAG_file_type ]
!7 = metadata !{i32 720897, null, metadata !"", null, i32 0, i64 320, i64 32, i32 0, i32 0, metadata !8, metadata !9, i32 0, i32 0} ; [ DW_TAG_array_type ]
!8 = metadata !{i32 720932, null, metadata !"int", null, i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!9 = metadata !{metadata !10}
!10 = metadata !{i32 720929, i64 0, i64 9}        ; [ DW_TAG_subrange_type ]
