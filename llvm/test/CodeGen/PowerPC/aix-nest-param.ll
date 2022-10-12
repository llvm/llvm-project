; RUN: not --crash llc -mtriple powerpc-ibm-aix-xcoff < %s 2>&1 | FileCheck %s
; RUN: not --crash llc -mtriple powerpc64-ibm-aix-xcoff < %s 2>&1 | FileCheck %s

define ptr @nest_receiver(ptr nest %arg) nounwind {
  ret ptr %arg
}

define ptr @nest_caller(ptr %arg) nounwind {
  %result = call ptr @nest_receiver(ptr nest %arg)
  ret ptr %result
}

; CHECK: LLVM ERROR: Nest arguments are unimplemented.
