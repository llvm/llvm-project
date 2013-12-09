; RUN: llc -mtriple=x86_64-linux-gnu -O0 -filetype=obj -o %t %s
; RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck %s

; Generated from:
; clang -g -S -emit-llvm -o foo.ll foo.c
; typedef int v4si __attribute__((__vector_size__(16)));
;
; v4si a

@a = common global <4 x i32> zeroinitializer, align 16

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!13}

!0 = metadata !{i32 786449, metadata !12, i32 12, metadata !"clang version 3.3 (trunk 171825) (llvm/trunk 171822)", i1 false, metadata !"", i32 0, metadata !1, metadata !1, metadata !1, metadata !3,  metadata !3, metadata !""} ; [ DW_TAG_compile_unit ] [/Users/echristo/foo.c] [DW_LANG_C99]
!1 = metadata !{i32 0}
!3 = metadata !{metadata !5}
!5 = metadata !{i32 786484, i32 0, null, metadata !"a", metadata !"a", metadata !"", metadata !6, i32 3, metadata !7, i32 0, i32 1, <4 x i32>* @a, null} ; [ DW_TAG_variable ] [a] [line 3] [def]
!6 = metadata !{i32 786473, metadata !12} ; [ DW_TAG_file_type ]
!7 = metadata !{i32 786454, metadata !12, null, metadata !"v4si", i32 1, i64 0, i64 0, i64 0, i32 0, metadata !8} ; [ DW_TAG_typedef ] [v4si] [line 1, size 0, align 0, offset 0] [from ]
!8 = metadata !{i32 786433, null, null, metadata !"", i32 0, i64 128, i64 128, i32 0, i32 2048, metadata !9, metadata !10, i32 0, null, null, null} ; [ DW_TAG_array_type ] [line 0, size 128, align 128, offset 0] [vector] [from int]
!9 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!10 = metadata !{metadata !11}
!11 = metadata !{i32 786465, i64 0, i64 4}        ; [ DW_TAG_subrange_type ] [0, 3]
!12 = metadata !{metadata !"foo.c", metadata !"/Users/echristo"}

; Check that we get an array type with a vector attribute.
; CHECK: DW_TAG_array_type
; CHECK-NEXT: DW_AT_GNU_vector
!13 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
