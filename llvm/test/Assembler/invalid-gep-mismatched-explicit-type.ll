; RUN: not llvm-as < %s 2>&1 | FileCheck %s
; CHECK: <stdin>:4:22: error: explicit pointee type doesn't match operand's pointee type (i16 vs i32)
define void @test(i32* %t) {
  %x = getelementptr i16, i32* %t, i32 0
  ret void
}
