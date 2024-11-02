; RUN: not --crash llc -verify-machineinstrs -mtriple powerpc-ibm-aix-xcoff 2>&1 < %s | \
; RUN:   FileCheck %s
; RUN: not --crash llc -verify-machineinstrs -mtriple powerpc64-ibm-aix-xcoff 2>&1 < %s |\
; RUN:   FileCheck %s

; RUN: not --crash llc -verify-machineinstrs -mtriple powerpc-ibm-aix-xcoff \
; RUN:     -filetype=obj -o %t.o < %s 2>&1 | \
; RUN:   FileCheck %s

; RUN: not --crash llc -verify-machineinstrs -mtriple powerpc64-ibm-aix-xcoff \
; RUN:     -filetype=obj -o %t.o 2>&1 < %s 2>&1 | \
; RUN:   FileCheck %s

; CHECK: LLVM ERROR: Cannot not be both dllexport and non-default visibility
@b_e = hidden dllexport global i32 0, align 4
