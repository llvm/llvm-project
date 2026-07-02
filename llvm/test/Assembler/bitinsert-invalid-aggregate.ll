; RUN: not llvm-as < %s 2>&1 | FileCheck %s

; CHECK: error: invalid bitinsert operands

define void @test_insert_array(b32 %base, [2 x i8] %val) {
  %res = bitinsert b32 %base, [2 x i8] %val, i32 0
  ret void
}
