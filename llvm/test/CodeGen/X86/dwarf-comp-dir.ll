; RUN: llc %s -o %t -filetype=obj
; RUN: llvm-dwarfdump %t | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

!llvm.dbg.cu = !{!0}

!0 = metadata !{i32 720913, i32 0, i32 12, metadata !"empty.c", metadata !"/home/nlewycky", metadata !"clang version 3.1 (trunk 143523)", i1 true, i1 true, metadata !"", i32 0, metadata !1, metadata !1, metadata !1, metadata !1} ; [ DW_TAG_compile_unit ]
!1 = metadata !{metadata !2}
!2 = metadata !{i32 0}

; The important part of the following check is that dir = #0.
;                        Dir  Mod Time   File Len   File Name
;                        ---- ---------- ---------- ---------------------------
; CHECK: file_names[  1]    0 0x00000000 0x00000000 empty.c
