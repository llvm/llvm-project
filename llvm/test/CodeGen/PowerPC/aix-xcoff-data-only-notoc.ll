; RUN: llc -mtriple powerpc-ibm-aix-xcoff < %s | FileCheck %s
; RUN: llc -mtriple powerpc64-ibm-aix-xcoff < %s 2>&1 | FileCheck %s

; RUN: llc -mtriple powerpc-ibm-aix-xcoff -filetype=obj -o %t.o < %s
; RUN: llvm-readobj --syms %t.o | FileCheck --check-prefix=SYMS %s

; RUN: llc -mtriple powerpc64-ibm-aix-xcoff -filetype=obj -o %t64.o < %s
; RUN: llvm-readobj --syms %t64.o | FileCheck --check-prefix=SYMS %s

@a = external global i32, align 4
@b = external global i64, align 8
@c = external global i16, align 2
@globa = common global i32 0, align 4

@ptr = internal global void (...)* null, align 4

; CHECK-NOT: .toc
; SYMS-NOT: Name: TOC
