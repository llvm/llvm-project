; PR28852: Check machine code sinking is not stopped by SUBREG_TO_REG.
; RUN: llc < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: foo
; CHECK-NOT: imull
; CHECK: retq
; CHECK: imull

define void @foo(i64 %value, i32 %kLengthBits, i32* nocapture %bits, i64* nocapture %bit_buffer_64, i32 %x) local_unnamed_addr {
entry:
  %mul = mul i32 %x, %kLengthBits
  %add = add i32 %mul, 3
  %conv = zext i32 %add to i64
  %mul2 = mul nuw nsw i64 %conv, 5
  %sub = sub i64 64, %value
  %conv4 = trunc i64 %sub to i32
  %tmp0 = load i32, i32* %bits, align 4
  %cmp = icmp ult i32 %tmp0, %conv4
  br i1 %cmp, label %if.then, label %if.end, !prof !0

if.then:                                          ; preds = %entry
  %add7 = add i64 %mul2, %value
  %tmp1 = load i64, i64* %bit_buffer_64, align 8
  %add8 = add i64 %add7, %tmp1
  store i64 %add8, i64* %bit_buffer_64, align 8
  %conv9 = trunc i64 %mul2 to i32
  store i32 %conv9, i32* %bits, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

!0 = !{!"branch_weights", i32 1, i32 2000}
