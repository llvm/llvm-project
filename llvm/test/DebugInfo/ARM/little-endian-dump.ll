; RUN: llc -O0 -filetype=obj -mtriple=arm-none-linux < %s | llvm-dwarfdump - | FileCheck %s

; CHECK: file format elf32-littlearm

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-n32-S64"

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}
!llvm.ident = !{!7}

!0 = !{i32 786449, !1, i32 12, !"clang version 3.6.0 ", i1 false, !"", i32 0, !2, !2, !2, !2, !2, !"", i32 1} ; [ DW_TAG_compile_unit ] [/a/empty.c] [DW_LANG_C99]
!1 = !{!"empty.c", !"/a"}
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 1}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{i32 1, !"min_enum_size", i32 4}
!7 = !{!"clang version 3.6.0 "}
