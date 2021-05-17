; RUN: not --crash llc  -mtriple powerpc-ibm-aix-xcoff  -verify-machineinstrs \
; RUN:     < %s 2>&1 | FileCheck %s

@ilocal = internal global i32 0, align 4 #0

define dso_local i32 @read_i32_local_linkage() {
  entry:
    %0 = load i32, i32* @ilocal, align 4
    ret i32 %0
}

; CHECK: LLVM ERROR: A GlobalVariable with private or local linkage is not currently supported by the toc data transformation.

attributes #0 = { "toc-data" }
