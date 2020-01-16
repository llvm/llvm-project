; Check that the GHC calling convention works (s390x)
; Variable-sized stack allocations are not supported in GHC calling convention
;
; RUN: not llc -mtriple=s390x-ibm-linux < %s 2>&1 | FileCheck %s

define ghccc void @foo() nounwind {
entry:
  %0 = call i8* @llvm.stacksave()
  call void @llvm.stackrestore(i8* %0)
  ret void
}

declare i8* @llvm.stacksave()
declare void @llvm.stackrestore(i8*)

; CHECK: LLVM ERROR: Variable-sized stack allocations are not supported in GHC calling convention
