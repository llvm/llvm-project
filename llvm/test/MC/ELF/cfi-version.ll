; RUN: %llc_dwarf %s -o - -dwarf-version 2 -filetype=obj | llvm-dwarfdump - | FileCheck %s --check-prefix=DWARF2
; RUN: %llc_dwarf %s -o - -dwarf-version 3 -filetype=obj | llvm-dwarfdump - | FileCheck %s --check-prefix=DWARF34
; RUN: %llc_dwarf %s -o - -dwarf-version 4 -filetype=obj | llvm-dwarfdump - | FileCheck %s --check-prefix=DWARF34

; .debug_frame is not emitted for targeting Windows x64.
; REQUIRES: debug_frame

; Function Attrs: nounwind
define i32 @foo() #0 {
entry:
  %call = call i32 bitcast (i32 (...)* @bar to i32 ()*)(), !dbg !12
  %add = add nsw i32 %call, 1, !dbg !12
  ret i32 %add, !dbg !12
}

declare i32 @bar(...) #1

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!9, !10}
!llvm.ident = !{!11}

!0 = !{!"0x11\0012\00clang version 3.5.0 \000\00\000\00\001", !1, !2, !2, !3, !2, !2} ; [ DW_TAG_compile_unit ] [/tmp/test.c] [DW_LANG_C99]
!1 = !{!"test.c", !"/tmp"}
!2 = !{}
!3 = !{!4}
!4 = !{!"0x2e\00foo\00foo\00\002\000\001\000\006\00256\000\002", !1, !5, !6, null, i32 ()* @foo, null, null, !2} ; [ DW_TAG_subprogram ] [line 2] [def] [foo]
!5 = !{!"0x29", !1}          ; [ DW_TAG_file_type ] [/tmp/test.c]
!6 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !7, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = !{!8}
!8 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!9 = !{i32 2, !"Dwarf Version", i32 4}
!10 = !{i32 1, !"Debug Info Version", i32 2}
!11 = !{!"clang version 3.5.0 "}
!12 = !MDLocation(line: 2, scope: !4)

; DWARF2:      .debug_frame contents:
; DWARF2:        Version:               1
; DWARF2-NEXT:   Augmentation:

; DWARF34:      .debug_frame contents:
; DWARF34:        Version:               3
; DWARF34-NEXT:   Augmentation:
