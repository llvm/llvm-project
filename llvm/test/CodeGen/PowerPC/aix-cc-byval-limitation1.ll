; RUN: not --crash llc -mtriple powerpc-ibm-aix-xcoff < %s 2>&1 | FileCheck %s
; RUN: not --crash llc -mtriple powerpc64-ibm-aix-xcoff < %s 2>&1 | FileCheck %s

%struct.S = type { [9 x i8] }

define void @bar() {
entry:
  %s1 = alloca %struct.S, align 1
  %agg.tmp = alloca %struct.S, align 1
  call void @foo(%struct.S* byval(%struct.S) align 1 %agg.tmp)
  ret void
}

declare void @foo(%struct.S* byval(%struct.S) align 1)

; CHECK: LLVM ERROR: Pass-by-value arguments are only supported in a single register.
