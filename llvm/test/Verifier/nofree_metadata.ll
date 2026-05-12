; RUN: not llvm-as < %s 2>&1 | FileCheck %s

declare ptr @dummy()

; CHECK: nofree applies only to inttoptr instruction
define void @test_not_inttoptr() {
  call ptr @dummy(), !nofree !{}
  ret void
}

; CHECK: nofree metadata must be empty
define void @test_invalid_arg(i32 %p) {
  inttoptr i32 %p to ptr, !nofree !{i32 0}
  ret void
}
