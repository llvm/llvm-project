; RUN: opt -S -mtriple aarch64-linux-gnu -passes=instnamer < %s | FileCheck %s

define i32 @f_0(i32) {
  %2 = add i32 %0, 2
  br label %3

; <label>:3:
  ret i32 %2
}
