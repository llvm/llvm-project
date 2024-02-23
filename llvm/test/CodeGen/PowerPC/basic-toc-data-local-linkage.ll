; RUN: llc  -mtriple powerpc-ibm-aix-xcoff  -verify-machineinstrs \
; RUN:     < %s 2>&1 | FileCheck %s
; RUN: llc  -mtriple powerpc64-ibm-aix-xcoff  -verify-machineinstrs \
; RUN:     < %s 2>&1 | FileCheck %s

@ilocal = internal global i32 0, align 4 #0

define dso_local i32 @read_i32_local_linkage() {
  entry:
    %0 = load i32, ptr @ilocal, align 4
    ret i32 %0
}

attributes #0 = { "toc-data" }

; CHECK:      .toc
; CHECK-NEXT: .csect ilocal[TD],2
; CHECK-NEXT: .space  4
