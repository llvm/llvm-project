; RUN: not llvm-as < %s 2>&1 | FileCheck %s

declare ptr @dummy()

define void @test_not_pointer(ptr %p, i32 %i) {
entry:
  ; CHECK: nonnull applies only to pointer types
  load i32, ptr %p, !nonnull !{}
  ; CHECK: nonnull applies only to load instructions or phi nodes, use attributes for calls or invokes
  call ptr @dummy(), !nonnull !{}
  ; CHECK: nonnull metadata must be empty
  load ptr, ptr %p, !nonnull !{i32 0}
  ; This one is valid
  load ptr, ptr %p, !nonnull !{}
  ret void

bb:
  ; This one is valid
  phi ptr [%p, %entry], !nonnull !{}
  ; CHECK: nonnull metadata must be empty
  phi ptr [%p, %entry], !nonnull !{i32 0}
  ; CHECK: nonnull applies only to pointer types
  phi i32 [%i, %entry], !nonnull !{}
  ret void
}
