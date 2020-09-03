; RUN: not --crash llc -mtriple powerpc-ibm-aix-xcoff < %s 2>&1 | FileCheck %s
; RUN: not --crash llc -mtriple powerpc64-ibm-aix-xcoff < %s 2>&1 | FileCheck %s

%struct.S = type { [1 x i8] }

define void @bar() {
entry:
  %s1 = alloca %struct.S, align 32
  %agg.tmp = alloca %struct.S, align 32
  call void @foo(i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, %struct.S* byval(%struct.S) align 32 %agg.tmp)
  ret void
}

declare void @foo(i32, i32, i32, i32, i32, i32, i32, i32, %struct.S* byval(%struct.S) align 32)

; CHECK: LLVM ERROR: Pass-by-value arguments with alignment greater than register width are not supported.
