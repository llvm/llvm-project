; RUN: llc -disable-fp-elim -O0 %s -mtriple x86_64-unknown-linux-gnu -o - | FileCheck %s

; int callme(int);
; int isel_line_test2() {
;   callme(400);
;   return 0;
; }

define i32 @isel_line_test2() nounwind uwtable {
  ; The stack adjustment should be part of the prologue.
  ; CHECK: isel_line_test2:
  ; CHECK: {{subq|leaq}} {{.*}}, %rsp
  ; CHECK: .loc 1 5 3 prologue_end
entry:
  %call = call i32 @callme(i32 400), !dbg !10
  ret i32 0, !dbg !12
}

declare i32 @callme(i32)

!llvm.dbg.cu = !{!0}

!0 = metadata !{i32 786449, i32 0, i32 12, metadata !"bar.c", metadata !"/usr/local/google/home/echristo/tmp", metadata !"clang version 3.2 (trunk 164980) (llvm/trunk 164979)", i1 true, i1 false, metadata !"", i32 0, metadata !1, metadata !1, metadata !3, metadata !1} ; [ DW_TAG_compile_unit ] [/usr/local/google/home/echristo/tmp/bar.c] [DW_LANG_C99]
!1 = metadata !{metadata !2}
!2 = metadata !{i32 0}
!3 = metadata !{metadata !4}
!4 = metadata !{metadata !5}
!5 = metadata !{i32 786478, i32 0, metadata !6, metadata !"isel_line_test2", metadata !"isel_line_test2", metadata !"", metadata !6, i32 3, metadata !7, i1 false, i1 true, i32 0, i32 0, null, i32 0, i1 false, i32 ()* @isel_line_test2, null, null, metadata !1, i32 4} ; [ DW_TAG_subprogram ] [line 3] [def] [scope 4] [isel_line_test2]
!6 = metadata !{i32 786473, metadata !"bar.c", metadata !"/usr/local/google/home/echristo/tmp", null} ; [ DW_TAG_file_type ]
!7 = metadata !{i32 786453, i32 0, metadata !"", i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !8, i32 0, i32 0} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = metadata !{metadata !9}
!9 = metadata !{i32 786468, null, metadata !"int", null, i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!10 = metadata !{i32 5, i32 3, metadata !11, null}
!11 = metadata !{i32 786443, metadata !5, i32 4, i32 1, metadata !6, i32 0} ; [ DW_TAG_lexical_block ] [/usr/local/google/home/echristo/tmp/bar.c]
!12 = metadata !{i32 6, i32 3, metadata !11, null}
