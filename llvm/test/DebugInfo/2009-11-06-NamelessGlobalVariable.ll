; RUN: llc %s -o /dev/null
@0 = internal constant i32 1

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!9}

!0 = metadata !{i32 720913, metadata !8, i32 12, metadata !"clang version 3.0 (trunk 139632)", i1 true, metadata !"", i32 0, metadata !2, metadata !2, metadata !2, metadata !3, null, metadata !""} ; [ DW_TAG_compile_unit ]
!2 = metadata !{i32 0}
!3 = metadata !{metadata !5}
!5 = metadata !{i32 720948, i32 0, null, metadata !"a", metadata !"a", metadata !"", metadata !6, i32 2, metadata !7, i32 0, i32 1, i32* @0, null} ; [ DW_TAG_variable ]
!6 = metadata !{i32 720937, metadata !8} ; [ DW_TAG_file_type ]
!7 = metadata !{i32 720932, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!8 = metadata !{metadata !"g.c", metadata !"/private/tmp"}
!9 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
