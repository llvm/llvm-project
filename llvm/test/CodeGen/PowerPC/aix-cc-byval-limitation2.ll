; RUN: not --crash llc -mtriple powerpc-ibm-aix-xcoff < %s 2>&1 | FileCheck %s
; RUN: not --crash llc -mtriple powerpc64-ibm-aix-xcoff < %s 2>&1 | FileCheck %s

%struct.S = type { i8 }

define void @foo(i32 %i1, i32 %i2, i32 %i3, i32 %i4, i32 %i5, i32 %i6, i32 %i7, i32 %i8, %struct.S* byval(%struct.S) align 1 %s) {
entry:
  ret void
}

; CHECK: LLVM ERROR: ByVal arguments passed on stack not implemented yet
