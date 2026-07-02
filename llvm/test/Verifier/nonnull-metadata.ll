; RUN: not llvm-as < %s 2>&1 | FileCheck %s

declare ptr @dummy()

define void @test(ptr %p) {
  ; CHECK: nonnull applies only to pointer types
  load i32, ptr %p, !nonnull !{}

  ; CHECK: nonnull applies only to load instructions, use attributes for calls or invokes
  call ptr @dummy(), !nonnull !{}

  ; CHECK: nonnull metadata must be empty
  load ptr, ptr %p, !nonnull !{i32 0}

  ret void
}
