; RUN: llc --global-isel=false -fast-isel -O0 -mtriple=aarch64-none-none < %s | FileCheck %s

; Check that the shl instruction did not get folded in together with 
; the cmp instruction. It would create a miscompilation 

@A = dso_local global [5 x i8] c"\C8\AA\C8\AA\AA"
@.str = private unnamed_addr constant [13 x i8] c"TRUE BRANCH\0A\00"
@.str.1 = private unnamed_addr constant [14 x i8] c"FALSE BRANCH\0A\00"

define dso_local i32 @main() {

  %tmp = load i8, ptr getelementptr inbounds ([5 x i8], ptr @A, i64 0, i64 1)
  %tmp2 = load i8, ptr getelementptr inbounds ([5 x i8], ptr @A, i64 0, i64 2)
  %op1 = xor i8 %tmp, -49
  %op2 = mul i8 %op1, %op1
; CHECK-NOT: cmp [[REGS:.*]] #[[SHIFT_VAL:[0-9]+]]
  %op3 = shl i8 %op2, 3
  %tmp3 = icmp eq i8 %tmp2, %op3
  br i1 %tmp3, label %_true, label %_false

_true:
  %res = call i32 (ptr, ...) @printf(ptr noundef @.str)
  br label %_ret

_false:
  %res2 = call i32 (ptr, ...) @printf(ptr noundef @.str.1)
  br label %_ret

_ret:
  ret i32 0
}

declare i32 @printf(ptr noundef, ...) 
