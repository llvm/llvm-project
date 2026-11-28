; RUN: not llvm-as < %s 2>&1 | FileCheck %s

declare ptr @dummy()

; CHECK: nofreeobj applies only to load and inttoptr instructions
define void @test_not_inttoptr() {
  call ptr @dummy(), !nofreeobj !{}
  ret void
}

; CHECK: nofreeobj metadata must be empty
define void @test_invalid_arg(i32 %p) {
  inttoptr i32 %p to ptr, !nofreeobj !{i32 0}
  ret void
}

; CHECK: nofreeobj applies only to pointer types
define void @test_not_pointer(ptr %p) {
  load i32, ptr %p, !nofreeobj !{}
  ret void
}
