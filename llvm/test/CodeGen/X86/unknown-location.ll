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

!0 = !{!"0x101\00x\001\000", !1, !2, !6} ; [ DW_TAG_arg_variable ]
!1 = !{!"0x2e\00foo\00foo\00foo\001\000\001\000\006\000\000\001", !10, !2, !4, null, i32 (i32, i32, i32, i32)* @foo, null, null, null} ; [ DW_TAG_subprogram ]
!2 = !{!"0x29", !10} ; [ DW_TAG_file_type ]
!3 = !{!"0x11\0012\00producer\000\00\000\00\000", !10, !11, !11, !9, null, null} ; [ DW_TAG_compile_unit ]
!4 = !{!"0x15\00\000\000\000\000\000\000", !10, !2, null, !5, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!5 = !{!6}
!6 = !{!"0x24\00int\000\0032\0032\000\000\005", !10, !2} ; [ DW_TAG_base_type ]
!7 = !{!"0xb\001\0030\000", !2, !1} ; [ DW_TAG_lexical_block ]
!8 = !MDLocation(line: 4, column: 3, scope: !7)
!9 = !{!1}
!10 = !{!"test.c", !"/dir"}
!11 = !{i32 0}
!12 = !{i32 1, !"Debug Info Version", i32 2}
