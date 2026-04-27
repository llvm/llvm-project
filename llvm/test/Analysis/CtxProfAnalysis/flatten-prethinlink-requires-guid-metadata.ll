; RUN: not opt -passes='ctx-prof-flatten-prethinlink' %s -disable-output 2>&1 | FileCheck %s

; CHECK: LLVM ERROR: this pass requires the GUID metadata to be available.

define void @test() {
  ret void
}
