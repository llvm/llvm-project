; RUN: not llvm-as < %s 2>&1 | FileCheck %s

define void @test(ptr %ptr) {
entry:
  ; This one is valid
  load i32, ptr %ptr, !noundef !{}
  ; CHECK: noundef metadata must be empty
  load i32, ptr %ptr, !noundef !{i32 0}
  ; CHECK: noundef applies only to load instructions
  store i32 0, ptr %ptr, !noundef !{}
  ret void
}
