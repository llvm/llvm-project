; RUN: llc  < %s | FileCheck %s
; Radar 10147769
; Do not unnecessarily use AT_specification DIE.
; CHECK-NOT: AT_specification

@a = common global [10 x i32] zeroinitializer, align 16

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!12}

!0 = metadata !{i32 720913, metadata !11, i32 12, metadata !"clang version 3.0 (trunk 140253)", i1 true, metadata !"", i32 0, metadata !2, metadata !2, metadata !2, metadata !3, null, i32 0} ; [ DW_TAG_compile_unit ]
!2 = metadata !{i32 0}
!3 = metadata !{metadata !5}
!5 = metadata !{i32 720948, i32 0, null, metadata !"a", metadata !"a", metadata !"", metadata !6, i32 1, metadata !7, i32 0, i32 1, [10 x i32]* @a, null} ; [ DW_TAG_variable ]
!6 = metadata !{i32 720937, metadata !11} ; [ DW_TAG_file_type ]
!7 = metadata !{i32 720897, null, null, null, i32 0, i64 320, i64 32, i32 0, i32 0, metadata !8, metadata !9, i32 0, null, null, null} ; [ DW_TAG_array_type ] [line 0, size 320, align 32, offset 0] [from int]
!8 = metadata !{i32 720932, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!9 = metadata !{metadata !10}
!10 = metadata !{i32 720929, i64 0, i64 10}        ; [ DW_TAG_subrange_type ]
!11 = metadata !{metadata !"x.c", metadata !"/private/tmp"}
!12 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
