; RUN: not llc -mtriple powerpc-ibm-aix-xcoff < %s 2>&1 | FileCheck %s
; RUN: not llc -mtriple powerpc64-ibm-aix-xcoff < %s 2>&1 | FileCheck %s

%struct.S = type { i32, i32 }

define void @bar() {
entry:
  %s1 = alloca %struct.S, align 4
  %agg.tmp = alloca %struct.S, align 4
  call void @foo(%struct.S* byval(%struct.S) align 4 %agg.tmp)
  ret void
}

declare void @foo(%struct.S* byval(%struct.S) align 4)

; CHECK: LLVM ERROR: Passing structure by value is unimplemented.
