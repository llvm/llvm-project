
; RUN: llvm-link %s %p/2011-08-18-unique-debug-type2.ll -S -o - | grep "int" | grep -v "^; ModuleID" | count 1
; Test to check only one MDNode for "int" after linking.
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-macosx10.7.0"

define i32 @foo() nounwind uwtable ssp {
entry:
  ret i32 1, !dbg !10
}

!llvm.dbg.cu = !{!0}

!0 = metadata !{i32 720913, i32 0, i32 12, metadata !"one.c", metadata !"/private/tmp", metadata !"clang version 3.0 (trunk 137954)", i1 true, i1 false, metadata !"", i32 0, metadata !1, metadata !1, metadata !3, metadata !1} ; [ DW_TAG_compile_unit ]
!1 = metadata !{metadata !2}
!2 = metadata !{i32 0}
!3 = metadata !{metadata !4}
!4 = metadata !{metadata !5}
!5 = metadata !{i32 720942, i32 0, metadata !6, metadata !"foo", metadata !"foo", metadata !"", metadata !6, i32 1, metadata !7, i1 false, i1 true, i32 0, i32 0, i32 0, i32 0, i1 false, i32 ()* @foo, null, null} ; [ DW_TAG_subprogram ]
!6 = metadata !{i32 720937, metadata !"one.c", metadata !"/private/tmp", null} ; [ DW_TAG_file_type ]
!7 = metadata !{i32 720917, metadata !6, metadata !"", metadata !6, i32 0, i64 0, i64 0, i32 0, i32 0, i32 0, metadata !8, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!8 = metadata !{metadata !9}
!9 = metadata !{i32 720932, null, metadata !"int", null, i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!10 = metadata !{i32 1, i32 13, metadata !11, null}
!11 = metadata !{i32 720907, metadata !5, i32 1, i32 11, metadata !6, i32 0} ; [ DW_TAG_lexical_block ]

