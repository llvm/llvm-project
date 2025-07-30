; RUN: not llvm-as < %s -o /dev/null 2>&1 | FileCheck %s

declare void @llvm.test.immarg.range.intrinsic.i32(i32 immarg range(i32 -3, 4))

define void @test_int_immarg_with_range() {
  ; CHECK: immarg value -4 out of range [-3, 4)
  call void @llvm.test.immarg.range.intrinsic.i32(i32 -4)
  ret void
}
