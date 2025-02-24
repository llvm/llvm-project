; Check that the GHC calling convention works (s390x)
; At most 2048*sizeof(long)=16384 bytes of stack space may be used
;
; RUN: not llc -mtriple=s390x-ibm-linux < %s 2>&1 | FileCheck %s

define ghccc void @foo() nounwind {
entry:
  alloca [16385 x i8], align 1
  ret void
}

; CHECK: LLVM ERROR: Pre allocated stack space for GHC function is too small
