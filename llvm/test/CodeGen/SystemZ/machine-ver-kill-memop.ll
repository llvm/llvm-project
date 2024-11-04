; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z16 -disable-machine-dce \
; RUN:   -verify-machineinstrs -O3
;
; Test that the MemoryOperand of the produced KILL instruction is removed
; and as a result the machine verifier succeeds.

define void @fun(ptr %Src, ptr %Dst) {
  br label %2

2:
  %3 = load i32, ptr %Src, align 4
  %4 = freeze i32 %3
  %5 = icmp eq i32 %4, 0
  %6 = load i32, ptr poison, align 4
  %7 = select i1 %5, i32 0, i32 %6
  %8 = load i32, ptr %Src, align 4
  %9 = freeze i32 %8
  %10 = icmp eq i32 %9, 0
  %11 = load i32, ptr poison, align 4
  %12 = select i1 %10, i32 0, i32 %11
  br label %13

13:
  %14 = phi i32 [ %12, %13 ], [ %7, %2 ]
  %15 = icmp slt i32 %14, 5
  %16 = zext i1 %15 to i64
  %17 = xor i64 poison, %16
  store i64 %17, ptr %Dst, align 8
  br label %13
}
