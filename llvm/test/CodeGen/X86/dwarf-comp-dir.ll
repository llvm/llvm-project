; RUN: llc %s -o %t -filetype=obj
; RUN: llvm-dwarfdump -debug-line %t | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.1 (trunk 143523)", isOptimized: true, emissionKind: FullDebug, file: !4, enums: !2, retainedTypes: !7, globals: !2)
!2 = !{}
!3 = !DIFile(filename: "empty.c", directory: "/home/nlewycky")
!4 = !DIFile(filename: "empty.c", directory: "/home/nlewycky")
!6 = !DICompositeType(tag: DW_TAG_structure_type, name: "foo", line: 1, size: 8, align: 8, file: !4, elements: !2, identifier: "_ZTS3foo")
!7 = !{!6}

; The important part of the following check is that dir = #0.
; CHECK: file_names[  1]:
; CHECK-NEXT: name: "empty.c"
; CHECK-NEXT: dir_index: 0
!5 = !{i32 1, !"Debug Info Version", i32 3}
