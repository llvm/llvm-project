; RUN: not --crash llc  -mtriple powerpc-ibm-aix-xcoff  -verify-machineinstrs \
; RUN:     < %s 2>&1 | FileCheck %s
; RUN: not --crash llc  -mtriple powerpc64-ibm-aix-xcoff  -verify-machineinstrs \
; RUN:     < %s 2>&1 | FileCheck %s

@iprivate = private global i32 55 #0

define nonnull ptr @get() local_unnamed_addr {
entry:
  ret ptr @iprivate
}

attributes #0 = { "toc-data" }

; CHECK: LLVM ERROR: A GlobalVariable with private linkage is not currently supported by the toc data transformation.
