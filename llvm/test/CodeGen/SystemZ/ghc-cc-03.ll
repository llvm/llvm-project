; Check that the GHC calling convention works (s390x)
; In GHC calling convention the only allowed return type is void
;
; RUN: not --crash llc -mtriple=s390x-ibm-linux < %s 2>&1 | FileCheck %s

define ghccc i64 @foo() nounwind {
entry:
  ret i64 42
}

; CHECK: LLVM ERROR: GHC functions return void only
