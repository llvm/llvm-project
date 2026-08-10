; Check that the GHC calling convention works (s390x)
; Check that no more than 12 integer arguments are passed
;
; RUN: not --crash llc -mtriple=s390x-ibm-linux < %s 2>&1 | FileCheck %s

define ghccc void @foo() nounwind {
entry:
  tail call ghccc void (...) @bar(i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7, i64 8, i64 9, i64 10, i64 11, i64 12, i64 13);
  ret void
}

declare ghccc void @bar(...)

; CHECK: LLVM ERROR: No registers left in GHC calling convention
