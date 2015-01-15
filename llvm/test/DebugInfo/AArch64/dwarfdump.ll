; RUN: llc -mtriple=aarch64-non-linux-gnu -dwarf-version=4 < %s -filetype=obj \
; RUN:    | llvm-dwarfdump - | FileCheck -check-prefix=CHECK -check-prefix=CHECK-4 %s
; RUN: llc -mtriple=aarch64-non-linux-gnu -dwarf-version=3 < %s -filetype=obj \
; RUN:    | llvm-dwarfdump - | FileCheck -check-prefix=CHECK -check-prefix=CHECK-3 %s

; We're mostly checking that relocations are applied correctly
; here. Currently R_AARCH64_ABS32 is used for references to debug data
; and R_AARCH64_ABS64 is used for program addresses.

; A couple of ABS32s, both at 0 and elsewhere, interpreted correctly:

; CHECK: DW_AT_producer [DW_FORM_strp] ( .debug_str[0x00000000] = "clang version 3.3 ")
; CHECK: DW_AT_name [DW_FORM_strp] ( .debug_str[0x00000013] = "tmp.c")

; A couple of ABS64s similarly:

; CHECK: DW_AT_low_pc [DW_FORM_addr] (0x0000000000000000)
; CHECK-4: DW_AT_high_pc [DW_FORM_data4] (0x00000008)
; CHECK-3: DW_AT_high_pc [DW_FORM_addr] (0x0000000000000008)

define i32 @main() nounwind {
  ret i32 0, !dbg !8
}

attributes #0 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!10}

!0 = !{!"0x11\0012\00clang version 3.3 \000\00\000\00\000", !9, !1, !1, !2, !1,  !1} ; [ DW_TAG_compile_unit ] [/home/timnor01/llvm/build/tmp.c] [DW_LANG_C99]
!1 = !{}
!2 = !{!3}
!3 = !{!"0x2e\00main\00main\00\001\000\001\000\006\000\000\001", !9, !4, !5, null, i32 ()* @main, null, null, !1} ; [ DW_TAG_subprogram ] [line 1] [def] [main]
!4 = !{!"0x29", !9} ; [ DW_TAG_file_type ]
!5 = !{!"0x15\00\000\000\000\000\000\000", null, null, null, !6, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!6 = !{!7}
!7 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!8 = !MDLocation(line: 2, scope: !3)
!9 = !{!"tmp.c", !"/home/tim/llvm/build"}
!10 = !{i32 1, !"Debug Info Version", i32 2}
