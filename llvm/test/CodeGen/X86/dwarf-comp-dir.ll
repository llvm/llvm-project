; RUN: llc %s -o %t -filetype=obj
; RUN: llvm-dwarfdump -debug-dump=line %t | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!5}

!0 = metadata !{i32 720913, metadata !4, i32 12, metadata !"clang version 3.1 (trunk 143523)", i1 true, metadata !"", i32 0, metadata !2, metadata !2, metadata !2, metadata !2, null, metadata !""} ; [ DW_TAG_compile_unit ]
!2 = metadata !{i32 0}
!3 = metadata !{i32 786473, metadata !4} ; [ DW_TAG_file_type ]
!4 = metadata !{metadata !"empty.c", metadata !"/home/nlewycky"}

; The important part of the following check is that dir = #0.
;                        Dir  Mod Time   File Len   File Name
;                        ---- ---------- ---------- ---------------------------
; CHECK: file_names[  1]    0 0x00000000 0x00000000 empty.c
!5 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
