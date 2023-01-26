; Check that the GHC calling convention works (s390x)
; Variable-sized stack allocations are not supported in GHC calling convention
;
; RUN: not --crash llc -mtriple=s390x-ibm-linux < %s 2>&1 | FileCheck %s

define ghccc void @foo() nounwind {
entry:
  %0 = call ptr @llvm.stacksave()
  call void @llvm.stackrestore(ptr %0)
  ret void
}

declare ptr @llvm.stacksave()
declare void @llvm.stackrestore(ptr)

; CHECK: LLVM ERROR: Variable-sized stack allocations are not supported in GHC calling convention
