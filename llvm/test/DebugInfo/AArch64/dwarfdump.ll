; RUN: llc -mtriple=aarch64-non-linux-gnu < %s -filetype=obj | llvm-dwarfdump - | FileCheck %s

; We're mostly checking that relocations are applied correctly
; here. Currently R_AARCH64_ABS32 is used for references to debug data
; and R_AARCH64_ABS64 is used for program addresses.

; A couple of ABS32s, both at 0 and elsewhere, interpreted correctly:

; CHECK: DW_AT_producer [DW_FORM_strp] ( .debug_str[0x00000000] = "clang version 3.3 ")
; CHECK: DW_AT_name [DW_FORM_strp] ( .debug_str[0x00000013] = "tmp.c")

; A couple of ABS64s similarly:

; CHECK: DW_AT_low_pc [DW_FORM_addr] (0x0000000000000000)
; CHECK: DW_AT_high_pc [DW_FORM_addr] (0x0000000000000008)

define i32 @main() nounwind {
  ret i32 0, !dbg !8
}

attributes #0 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!10}

!0 = metadata !{i32 786449, metadata !9, i32 12, metadata !"clang version 3.3 ", i1 false, metadata !"", i32 0, metadata !1, metadata !1, metadata !2, metadata !1,  metadata !1, metadata !""} ; [ DW_TAG_compile_unit ] [/home/timnor01/llvm/build/tmp.c] [DW_LANG_C99]
!1 = metadata !{i32 0}
!2 = metadata !{metadata !3}
!3 = metadata !{i32 786478, metadata !9, metadata !4, metadata !"main", metadata !"main", metadata !"", i32 1, metadata !5, i1 false, i1 true, i32 0, i32 0, null, i32 0, i1 false, i32 ()* @main, null, null, metadata !1, i32 1} ; [ DW_TAG_subprogram ] [line 1] [def] [main]
!4 = metadata !{i32 786473, metadata !9} ; [ DW_TAG_file_type ]
!5 = metadata !{i32 786453, null, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !6, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!6 = metadata !{metadata !7}
!7 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!8 = metadata !{i32 2, i32 0, metadata !3, null}
!9 = metadata !{metadata !"tmp.c", metadata !"/home/tim/llvm/build"}
!10 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
