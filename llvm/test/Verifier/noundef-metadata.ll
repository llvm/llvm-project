; RUN: not llvm-as < %s 2>&1 | FileCheck %s

define void @test(ptr %ptr, i32 %i) {
entry:
  ; This one is valid
  load i32, ptr %ptr, !noundef !{}

  ; CHECK: noundef metadata must be empty
  load i32, ptr %ptr, !noundef !{i32 0}

  ; CHECK: noundef applies only to load instructions
  store i32 0, ptr %ptr, !noundef !{}

  ret void

bb:
  ; This one is valid
  phi i32 [%i, %entry], !noundef !{}

  ; CHECK: noundef metadata must be empty
  phi i32 [%i, %entry], !noundef !{i32 0}

  ret void
}
