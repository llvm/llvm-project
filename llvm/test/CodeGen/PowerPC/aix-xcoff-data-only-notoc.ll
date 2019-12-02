; RUN: llc -mtriple powerpc-ibm-aix-xcoff < %s | FileCheck %s
; RUN: llc -mtriple powerpc64-ibm-aix-xcoff < %s 2>&1 | FileCheck %s

@a = external global i32, align 4
@b = external global i64, align 8
@c = external global i16, align 2
@globa = common global i32 0, align 4

@ptr = internal global void (...)* null, align 4

; CHECK-NOT: .toc

