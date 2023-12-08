; RUN: not llvm-as < %s 2>&1 | FileCheck %s

declare ptr @dummy()

; CHECK: nonnull applies only to pointer types
define void @test_not_pointer(ptr %p) {
  load i32, ptr %p, !nonnull !{}
  ret void
}

; CHECK: nonnull applies only to load instructions, use attributes for calls or invokes
define void @test_not_load() {
  call ptr @dummy(), !nonnull !{}
  ret void
}

; CHECK: nonnull metadata must be empty
define void @test_invalid_arg(ptr %p) {
  load ptr, ptr %p, !nonnull !{i32 0}
  ret void
}
