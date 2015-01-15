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
!llvm.module.flags = !{!14}

!0 = !{!"0x11\0012\00clang version 3.2 (trunk 164980) (llvm/trunk 164979)\000\00\000\00\000", !13, !1, !1, !3, !1,  !1} ; [ DW_TAG_compile_unit ] [/usr/local/google/home/echristo/tmp/bar.c] [DW_LANG_C99]
!1 = !{}
!3 = !{!5}
!5 = !{!"0x2e\00isel_line_test2\00isel_line_test2\00\003\000\001\000\006\000\000\004", !13, !6, !7, null, i32 ()* @isel_line_test2, null, null, !1} ; [ DW_TAG_subprogram ] [line 3] [def] [scope 4] [isel_line_test2]
!6 = !{!"0x29", !13} ; [ DW_TAG_file_type ]
!7 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !8, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = !{!9}
!9 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!10 = !MDLocation(line: 5, column: 3, scope: !11)
!11 = !{!"0xb\004\001\000", !13, !5} ; [ DW_TAG_lexical_block ] [/usr/local/google/home/echristo/tmp/bar.c]
!12 = !MDLocation(line: 6, column: 3, scope: !11)
!13 = !{!"bar.c", !"/usr/local/google/home/echristo/tmp"}
!14 = !{i32 1, !"Debug Info Version", i32 2}
