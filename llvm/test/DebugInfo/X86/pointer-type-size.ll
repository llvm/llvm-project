; RUN: llc -mtriple=x86_64-apple-macosx10.7 %s -o %t -filetype=obj
; RUN: llvm-dwarfdump %t | FileCheck %s

; CHECK: ptr
; CHECK-NOT: AT_bit_size

%struct.crass = type { i8* }

@crass = common global %struct.crass zeroinitializer, align 8

!llvm.dbg.cu = !{!0}

!0 = metadata !{i32 720913, i32 0, i32 12, metadata !"foo.c", metadata !"/Users/echristo/tmp", metadata !"clang version 3.1 (trunk 147882)", i1 true, i1 false, metadata !"", i32 0, metadata !1, metadata !1, metadata !1, metadata !3} ; [ DW_TAG_compile_unit ]
!1 = metadata !{metadata !2}
!2 = metadata !{i32 0}
!3 = metadata !{metadata !4}
!4 = metadata !{metadata !5}
!5 = metadata !{i32 720948, i32 0, null, metadata !"crass", metadata !"crass", metadata !"", metadata !6, i32 1, metadata !7, i32 0, i32 1, %struct.crass* @crass} ; [ DW_TAG_variable ]
!6 = metadata !{i32 720937, metadata !"foo.c", metadata !"/Users/echristo/tmp", null} ; [ DW_TAG_file_type ]
!7 = metadata !{i32 720915, null, metadata !"crass", metadata !6, i32 1, i64 64, i64 64, i32 0, i32 0, null, metadata !8, i32 0, i32 0} ; [ DW_TAG_structure_type ]
!8 = metadata !{metadata !9}
!9 = metadata !{i32 720909, metadata !7, metadata !"ptr", metadata !6, i32 1, i64 64, i64 64, i64 0, i32 0, metadata !10} ; [ DW_TAG_member ]
!10 = metadata !{i32 720934, null, metadata !"", null, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !11} ; [ DW_TAG_const_type ]
!11 = metadata !{i32 720911, null, metadata !"", null, i32 0, i64 64, i64 64, i64 0, i32 0, metadata !12} ; [ DW_TAG_pointer_type ]
!12 = metadata !{i32 720932, null, metadata !"char", null, i32 0, i64 8, i64 8, i64 0, i32 0, i32 6} ; [ DW_TAG_base_type ]
