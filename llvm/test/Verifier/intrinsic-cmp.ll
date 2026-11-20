; RUN: not opt -S -passes=verify 2>&1 < %s | FileCheck %s

define void @result_len_is_at_least_2bits_wide(i32 %arg1, i32 %arg2) {
  ; CHECK: result type must be at least 2 bits wide
  %res2 = call i1 @llvm.scmp.i1.i32(i32 %arg1, i32 %arg2)
  ret void
}
