; RUN: llc -mtriple=x86_64-apple-macosx10.7 %s -o %t -filetype=obj
; RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck %s

; CHECK: ptr
; CHECK-NOT: AT_bit_size

%struct.crass = type { i8* }

@crass = common global %struct.crass zeroinitializer, align 8

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!14}

!0 = metadata !{i32 786449, metadata !13, i32 12, metadata !"clang version 3.1 (trunk 147882)", i1 false, metadata !"", i32 0, metadata !1, metadata !1, metadata !1, metadata !3,  metadata !3, metadata !""} ; [ DW_TAG_compile_unit ]
!1 = metadata !{i32 0}
!3 = metadata !{metadata !5}
!5 = metadata !{i32 720948, i32 0, null, metadata !"crass", metadata !"crass", metadata !"", metadata !6, i32 1, metadata !7, i32 0, i32 1, %struct.crass* @crass, null} ; [ DW_TAG_variable ]
!6 = metadata !{i32 720937, metadata !13} ; [ DW_TAG_file_type ]
!7 = metadata !{i32 786451, metadata !13, null, metadata !"crass", i32 1, i64 64, i64 64, i32 0, i32 0, null, metadata !8, i32 0, null, null, null} ; [ DW_TAG_structure_type ] [crass] [line 1, size 64, align 64, offset 0] [def] [from ]
!8 = metadata !{metadata !9}
!9 = metadata !{i32 786445, metadata !13, metadata !7, metadata !"ptr", i32 1, i64 64, i64 64, i64 0, i32 0, metadata !10} ; [ DW_TAG_member ]
!10 = metadata !{i32 720934, null, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, metadata !11} ; [ DW_TAG_const_type ]
!11 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, metadata !12} ; [ DW_TAG_pointer_type ]
!12 = metadata !{i32 720932, null, null, metadata !"char", i32 0, i64 8, i64 8, i64 0, i32 0, i32 6} ; [ DW_TAG_base_type ]
!13 = metadata !{metadata !"foo.c", metadata !"/Users/echristo/tmp"}
!14 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
