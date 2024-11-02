; RUN: llc -mtriple powerpc-ibm-aix-xcoff  -verify-machineinstrs < %s | FileCheck %s
; RUN: not --crash llc -filetype=obj -mtriple powerpc-ibm-aix-xcoff  \
; RUN:                 -verify-machineinstrs < %s 2>&1 | \
; RUN:   FileCheck %s --check-prefix=OBJ

; RUN: llc -mtriple powerpc64-ibm-aix-xcoff -verify-machineinstrs < %s | FileCheck %s
; RUN: not --crash llc -filetype=obj -mtriple powerpc64-ibm-aix-xcoff  \
; RUN:                 -verify-machineinstrs < %s 2>&1 | \
; RUN:   FileCheck %s --check-prefix=OBJ

@i = global i32 55, align 4 #0

attributes #0 = { "toc-data" }
; CHECK:            .toc
; CHECK-NEXT:       .csect i[TD],2
; CHECK-NEXT:       .globl i[TD]
; CHECK-NEXT:       .align 2
; CHECK-NEXT:       .vbyte 4, 55

; OBJ: LLVM ERROR:  toc-data not yet supported when writing object files.
