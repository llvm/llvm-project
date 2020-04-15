; RUN: not --crash llc -mtriple powerpc-ibm-aix-xcoff < %s 2>&1 | FileCheck %s
; RUN: not --crash llc -mtriple powerpc64-ibm-aix-xcoff < %s 2>&1 | FileCheck %s

%struct.S = type { [65 x i8] }

define void @foo(%struct.S* byval(%struct.S) align 1 %s) {
entry:
  ret void
}

; CHECK: LLVM ERROR: Passing ByVals split between registers and stack not yet implemented.
