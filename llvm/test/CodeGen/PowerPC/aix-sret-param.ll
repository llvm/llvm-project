; RUN: not llc -mtriple powerpc-ibm-aix-xcoff < %s 2>&1 | FileCheck %s
; RUN: not llc -mtriple powerpc64-ibm-aix-xcoff < %s 2>&1 | FileCheck %s

%struct.S = type { i32 }

define void @barv() {
entry:
  %tmp = alloca %struct.S, align 4
  call void @foo(%struct.S* sret %tmp)
  ret void
}

declare void @foo(%struct.S* sret)

; CHECK: LLVM ERROR: Struct return arguments are unimplemented.
