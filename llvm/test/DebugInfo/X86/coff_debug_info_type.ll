; RUN: llc -mtriple=i686-pc-mingw32 -dwarf-accel-tables=Enable -filetype=asm -O0 < %s | FileCheck %s
; RUN: llc -mtriple=i686-pc-cygwin -dwarf-accel-tables=Enable -filetype=asm -O0 < %s | FileCheck %s
; RUN: llc -mtriple=i686-w64-mingw32 -dwarf-accel-tables=Enable -filetype=asm -O0 < %s | FileCheck %s
; CHECK:    .section  .debug_info
; CHECK:    .section  .apple_names
; CHECK:    .section  .apple_types

; RUN: llc -mtriple=i686-pc-win32 -filetype=asm -O0 < %s | FileCheck -check-prefix=WIN32 %s
; WIN32:    .section .debug$S,"dr"

; RUN: llc -mtriple=i686-pc-win32 -filetype=null -O0 < %s

; generated from:
; clang -g -S -emit-llvm test.c -o test.ll
; int main()
; {
; 	return 0;
; }

define i32 @main() #0 {
entry:
  %retval = alloca i32, align 4
  store i32 0, i32* %retval
  ret i32 0, !dbg !10
}

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf"="true" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!9, !11}

!0 = !{!"0x11\0012\00clang version 3.4 \000\00\000\00\000", !1, !2, !2, !3, !2, !2} ; [ DW_TAG_compile_unit ] [C:\Projects/test.c] [DW_LANG_C99]
!1 = !{!"test.c", !"C:\5CProjects"}
!2 = !{}
!3 = !{!4}
!4 = !{!"0x2e\00main\00main\00\001\000\001\000\006\000\000\002", !1, !5, !6, null, i32 ()* @main, null, null, !2} ; [ DW_TAG_subprogram ] [line 1] [def] [scope 2] [main]
!5 = !{!"0x29", !1}          ; [ DW_TAG_file_type ] [C:\Projects/test.c]
!6 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !7, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = !{!8}
!8 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!9 = !{i32 2, !"Dwarf Version", i32 3}
!10 = !MDLocation(line: 3, scope: !4)
!11 = !{i32 1, !"Debug Info Version", i32 2}
