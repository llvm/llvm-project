; RUN: not --crash llc -mtriple powerpc-ibm-aix-xcoff < %s 2>&1 | FileCheck %s
; RUN: not --crash llc -mtriple powerpc64-ibm-aix-xcoff < %s 2>&1 | FileCheck %s

define i8* @nest_receiver(i8* nest %arg) nounwind {
  ret i8* %arg
}

define i8* @nest_caller(i8* %arg) nounwind {
  %result = call i8* @nest_receiver(i8* nest %arg)
  ret i8* %result
}

; CHECK: LLVM ERROR: Nest arguments are unimplemented.
