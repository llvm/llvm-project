; RUN: not llvm-as -disable-output -verify-debug-info < %s 2>&1 | FileCheck %s
; ModuleID = 'test.c'
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.10.0"

; Function Attrs: nounwind ssp uwtable
define i32 @foo() #0 {
entry:
  ret i32 42, !dbg !13
}

attributes #0 = { nounwind ssp uwtable }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!9, !10, !11}
!llvm.ident = !{!12}

!0 = !{!"0x11\0012\00clang version 3.7.0 \000\00\000\00\001", !1, !2, !2, !3, !2, !2} ; [ DW_TAG_compile_unit ] [/test.c] [DW_LANG_C99]
!1 = !{!"test.c", !""}
!2 = !{}
!3 = !{!4}
!4 = !{!"0x2e\00foo\00foo\00\001\000\001\000\000\000\000\001", !1, !5, !6, null, i32 ()* @foo, null, null, !2} ; [ DW_TAG_subprogram ] [line 1] [def] [foo]
!5 = !{!"0x29", !1}                               ; [ DW_TAG_file_type ] [/test.c]
!6 = !{!"0x15\00\000\000\000\000\000\000", null, null, null, !7, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = !{!8}
!8 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!9 = !{i32 2, !"Dwarf Version", i32 2}
!10 = !{i32 2, !"Debug Info Version", i32 2}
!11 = !{i32 1, !"PIC Level", i32 2}
!12 = !{!"clang version 3.7.0 "}
; An old-style MDLocation should not pass verify.
; CHECK: DISubprogram does not Verify
!13 = !{i32 2, i32 2, !4, null}
