; RUN: not llvm-as < %s -o /dev/null 2>&1 | FileCheck %s

; CHECK: Target type should have no parameters
; CHECK-NEXT: target("aarch64.svcount", i32, 0)
define void @test_alloca_svcount_ptr_int() {
  %ptr = alloca target("aarch64.svcount", i32, 0)
  ret void
}
