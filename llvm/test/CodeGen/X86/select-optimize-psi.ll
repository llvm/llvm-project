; RUN: not opt -mtriple=x86_64-unknown-unknown -passes='select-optimize' -disable-output < %s 2>&1 | FileCheck %s

;; Check that if we try to run select-optimize without requiring PSI,
;; we get an appropriate usage error rather than an assertion or crash.

; CHECK: LLVM ERROR: this pass requires the profile-summary module analysis to be available

define void @test() {
  ret void
}
