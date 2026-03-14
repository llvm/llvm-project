; RUN: llc -mtriple=arm64-eabi -mcpu=cyclone < %s | FileCheck %s

; CHECK: foo
; CHECK-DAG: str w[[REG0:[0-9]+]], [x29, #24]
; CHECK-DAG: str w[[REG0]], [x29, #28]
define i32 @foo(i32 %a) nounwind {
  %retval = alloca i32, align 4
  %a.addr = alloca i32, align 4
  %arr = alloca [32 x i32], align 4
  %i = alloca i32, align 4
  %arr2 = alloca [32 x i32], align 4
  %j = alloca i32, align 4
  store i32 %a, ptr %a.addr, align 4
  %tmp = load i32, ptr %a.addr, align 4
  %tmp1 = zext i32 %tmp to i64
  %v = mul i64 4, %tmp1
  %vla = alloca i8, i64 %v, align 4
  %tmp3 = load i32, ptr %a.addr, align 4
  store i32 %tmp3, ptr %i, align 4
  %tmp4 = load i32, ptr %a.addr, align 4
  store i32 %tmp4, ptr %j, align 4
  %tmp5 = load i32, ptr %j, align 4
  store i32 %tmp5, ptr %retval
  %x = load i32, ptr %retval
  ret i32 %x
}
