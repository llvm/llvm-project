; RUN: llc < %s -asm-verbose=false -mtriple=x86_64-apple-darwin10 -use-unknown-locations | FileCheck %s

; The divide instruction does not have a debug location. CodeGen should
; represent this in the debug information. This is done by setting line
; and column to 0

;      CHECK:         leal
; CHECK-NEXT:         .loc 1 0 0
;      CHECK:         cltd
; CHECK-NEXT:         idivl
; CHECK-NEXT:         .loc 2 4 3

define i32 @foo(i32 %w, i32 %x, i32 %y, i32 %z) nounwind {
entry:
  %a = add  i32 %w, %x, !dbg !8
  %b = sdiv i32 %a, %y
  %c = add  i32 %b, %z, !dbg !8
  ret i32 %c, !dbg !8
}

!llvm.dbg.cu = !{!3}
!llvm.module.flags = !{!12}

!0 = metadata !{i32 786689, metadata !1, metadata !"x", metadata !2, i32 1, metadata !6} ; [ DW_TAG_arg_variable ]
!1 = metadata !{i32 786478, metadata !10, metadata !2, metadata !"foo", metadata !"foo", metadata !"foo", i32 1, metadata !4, i1 false, i1 true, i32 0, i32 0, null, i1 false, i1 false, i32 (i32, i32, i32, i32)* @foo, null, null, null, i32 1} ; [ DW_TAG_subprogram ]
!2 = metadata !{i32 786473, metadata !10} ; [ DW_TAG_file_type ]
!3 = metadata !{i32 786449, metadata !10, i32 12, metadata !"producer", i1 false, metadata !"", i32 0, metadata !11, metadata !11, metadata !9, null, null, metadata !""} ; [ DW_TAG_compile_unit ]
!4 = metadata !{i32 786453, metadata !10, metadata !2, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !5, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!5 = metadata !{metadata !6}
!6 = metadata !{i32 786468, metadata !10, metadata !2, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!7 = metadata !{i32 786443, metadata !2, metadata !1, i32 1, i32 30, i32 0} ; [ DW_TAG_lexical_block ]
!8 = metadata !{i32 4, i32 3, metadata !7, null}
!9 = metadata !{metadata !1}
!10 = metadata !{metadata !"test.c", metadata !"/dir"}
!11 = metadata !{i32 0}
!12 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
