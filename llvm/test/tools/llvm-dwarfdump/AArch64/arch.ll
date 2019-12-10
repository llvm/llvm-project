; RUN: llc -O0 %s -filetype=obj -o %t.o
; RUN: llvm-dwarfdump -arch arm64   %t.o | FileCheck %s
; RUN: llvm-dwarfdump -arch 0x0100000c %t.o | FileCheck %s
; CHECK: file format Mach-O arm64
;
; RUN: llvm-dwarfdump -arch i386 %t.o \
; RUN:    | FileCheck %s --allow-empty --check-prefix=NEGATIVE
; RUN: llvm-dwarfdump -arch 0 %t.o \
; RUN:    | FileCheck %s --allow-empty --check-prefix=NEGATIVE
; NEGATIVE-NOT: file format
source_filename = "/tmp/empty.c"
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-ios"

!llvm.module.flags = !{!1, !2, !3, !4}
!llvm.dbg.cu = !{!5}

!1 = !{i32 2, !"Dwarf Version", i32 4}
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i32 1, !"wchar_size", i32 4}
!4 = !{i32 7, !"PIC Level", i32 2}
!5 = distinct !DICompileUnit(language: DW_LANG_C99, file: !6, producer: "Apple clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!6 = !DIFile(filename: "/tmp/empty.c", directory: "/Volumes/Data/llvm-project")
