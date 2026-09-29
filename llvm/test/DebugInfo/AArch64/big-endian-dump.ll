; RUN: llc -O0 -filetype=obj -mtriple=aarch64_be-none-linux < %s | llvm-dwarfdump - | FileCheck %s

; CHECK: file format elf64-bigaarch64

target datalayout = "E-m:e-i64:64-i128:128-n32:64-S128"

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = !{i32 786449, !1, i32 12, !"clang version 3.6.0 ", i1 false, !"", i32 0, !2, !2, !2, !2, !2, !"", i32 1} ; [ DW_TAG_compile_unit ] [/a/empty.c] [DW_LANG_C99]
!1 = !{!"empty.c", !"/a"}
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 1}
!5 = !{!"clang version 3.6.0 "}
