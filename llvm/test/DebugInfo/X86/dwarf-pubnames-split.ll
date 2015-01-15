; RUN: llc -mtriple=x86_64-pc-linux-gnu -split-dwarf=Enable %s -o - | FileCheck %s
; Derived from:

; int main (void) {
;    return 0;
; }

; Check that we get a symbol off of the debug_info section when using split dwarf and pubnames.

; CHECK: .LpubTypes_begin0:
; CHECK-NEXT: .short    2                       # DWARF Version
; CHECK-NEXT: .long     .L.debug_info_begin0    # Offset of Compilation Unit Info

; Function Attrs: nounwind uwtable
define i32 @main() #0 {
entry:
  %retval = alloca i32, align 4
  store i32 0, i32* %retval
  ret i32 0, !dbg !10
}

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!9, !11}

!0 = !{!"0x11\0012\00clang version 3.4 (trunk 189287) (llvm/trunk 189296)\000\00\000\00\000", !1, !2, !2, !3, !2, !2} ; [ DW_TAG_compile_unit ] [/usr/local/google/home/echristo/tmp/foo.c] [DW_LANG_C99]
!1 = !{!"foo.c", !"/usr/local/google/home/echristo/tmp"}
!2 = !{}
!3 = !{!4}
!4 = !{!"0x2e\00main\00main\00\001\000\001\000\006\00256\000\001", !1, !5, !6, null, i32 ()* @main, null, null, !2} ; [ DW_TAG_subprogram ] [line 1] [def] [main]
!5 = !{!"0x29", !1}          ; [ DW_TAG_file_type ] [/usr/local/google/home/echristo/tmp/foo.c]
!6 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !7, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = !{!8}
!8 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!9 = !{i32 2, !"Dwarf Version", i32 3}
!10 = !MDLocation(line: 2, scope: !4)
!11 = !{i32 1, !"Debug Info Version", i32 2}
