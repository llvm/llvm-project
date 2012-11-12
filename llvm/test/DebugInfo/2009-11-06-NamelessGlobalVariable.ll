; RUN: llc %s -o /dev/null
@0 = internal constant i32 1

!llvm.dbg.cu = !{!0}

!0 = metadata !{i32 720913, i32 0, i32 12, metadata !"g.c", metadata !"/private/tmp", metadata !"clang version 3.0 (trunk 139632)", i1 true, i1 true, metadata !"", i32 0, metadata !1, metadata !1, metadata !1, metadata !3} ; [ DW_TAG_compile_unit ]
!1 = metadata !{metadata !2}
!2 = metadata !{i32 0}
!3 = metadata !{metadata !4}
!4 = metadata !{metadata !5}
!5 = metadata !{i32 720948, i32 0, null, metadata !"", metadata !"", metadata !"", metadata !6, i32 2, metadata !7, i32 0, i32 1, i32* @0} ; [ DW_TAG_variable ]
!6 = metadata !{i32 720937, metadata !"g.c", metadata !"/private/tmp", null} ; [ DW_TAG_file_type ]
!7 = metadata !{i32 720932, null, metadata !"int", null, i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
