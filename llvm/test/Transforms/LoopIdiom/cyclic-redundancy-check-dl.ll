; REQUIRES: asserts
; RUN: not --crash opt -passes=loop-idiom -S %s

target datalayout = "p:16:16"

define void @test_with_dl() {
entry:
  br label %ph

ph:
  %crc.use = phi i32 [ 0, %entry ], [ %crc.next, %loop ]
  br label %loop

loop:
  %iv = phi i16 [ 0, %ph ], [ %iv.next, %loop ]
  %crc = phi i32 [ 0, %ph ], [ %crc.next, %loop ]
  %lshr.crc.1 = lshr i32 %crc, 1
  %crc.and.1 = and i32 %crc, 1
  %sb.check = icmp eq i32 %crc.and.1, 0
  %xor = xor i32 %lshr.crc.1, 0
  %crc.next = select i1 %sb.check, i32 %lshr.crc.1, i32 %xor
  %iv.next = add i16 %iv, 1
  %exit.cond = icmp ult i16 %iv, 7
  br i1 %exit.cond, label %loop, label %ph
}
