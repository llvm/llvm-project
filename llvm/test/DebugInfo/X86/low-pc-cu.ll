; RUN: llc -mtriple=x86_64-apple-darwin -filetype=obj < %s \
; RUN:     | llvm-dwarfdump -debug-dump=info - | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-V4
; RUN: llc -mtriple=x86_64-apple-darwin -filetype=obj -dwarf-version=3 < %s \
; RUN:     | llvm-dwarfdump -debug-dump=info - | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-V3


; Check that we use DW_AT_low_pc and that it has the right encoding depending
; on dwarf version.

; CHECK: DW_TAG_compile_unit [1]
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_low_pc [DW_FORM_addr]       (0x0000000000000000)
; CHECK-NOT: DW_TAG
; CHECK-V3: DW_AT_high_pc [DW_FORM_addr]
; CHECK-V4: DW_AT_high_pc [DW_FORM_data4]
; CHECK: DW_TAG_subprogram [2]
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_low_pc [DW_FORM_addr]
; CHECK-NOT: DW_TAG
; CHECK-V3: DW_AT_high_pc [DW_FORM_addr]
; CHECK-V4: DW_AT_high_pc [DW_FORM_data4]

; Function Attrs: nounwind uwtable
define void @z() #0 {
entry:
  ret void, !dbg !11
}

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!8, !9}
!llvm.ident = !{!10}

!0 = !{!"0x11\0012\00clang version 3.5.0 (trunk 204164) (llvm/trunk 204183)\000\00\000\00\001", !1, !2, !2, !3, !2, !2} ; [ DW_TAG_compile_unit ] [/usr/local/google/home/echristo/z.c] [DW_LANG_C99]
!1 = !{!"z.c", !"/usr/local/google/home/echristo"}
!2 = !{}
!3 = !{!4}
!4 = !{!"0x2e\00z\00z\00\001\000\001\000\006\00256\000\001", !1, !5, !6, null, void ()* @z, null, null, !2} ; [ DW_TAG_subprogram ] [line 1] [def] [z]
!5 = !{!"0x29", !1}          ; [ DW_TAG_file_type ] [/usr/local/google/home/echristo/z.c]
!6 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !7, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = !{null}
!8 = !{i32 2, !"Dwarf Version", i32 4}
!9 = !{i32 1, !"Debug Info Version", i32 2}
!10 = !{!"clang version 3.5.0 (trunk 204164) (llvm/trunk 204183)"}
!11 = !MDLocation(line: 1, scope: !4)
