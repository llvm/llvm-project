; RUN: opt < %s -loop-reduce -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128-ni:1"

; Check this completes sensibly. The sequence here is quick for scev to
; calculate, but creates a very large tree if recursed into, the same node
; being processed many times. I will also naturally have a setupcost of
; 0xffffffff, which LSR will treat as invalid.
; CHECK-LABEL: func
; CHECK: load i32, i32* %gep

define i32 @func(i32* %in) {
entry:
  %load = load i32, i32* %in, align 4
  %a1 = add i32 %load, 1
  %m1 = mul i32 %a1, %load
  %a2 = add i32 %m1, 1
  %m2 = mul i32 %a2, %m1
  %a3 = add i32 %m2, 1
  %m3 = mul i32 %a3, %m2
  %a4 = add i32 %m3, 1
  %m4 = mul i32 %a4, %m3
  %a5 = add i32 %m4, 1
  %m5 = mul i32 %a5, %m4
  %a6 = add i32 %m5, 1
  %m6 = mul i32 %a6, %m5
  %a7 = add i32 %m6, 1
  %m7 = mul i32 %a7, %m6
  %a8 = add i32 %m7, 1
  %m8 = mul i32 %a8, %m7
  %a9 = add i32 %m8, 1
  %m9 = mul i32 %a9, %m8
  %a10 = add i32 %m9, 1
  %m10 = mul i32 %a10, %m9
  %a11 = add i32 %m10, 1
  %m11 = mul i32 %a11, %m10
  %a12 = add i32 %m11, 1
  %m12 = mul i32 %a12, %m11
  %a13 = add i32 %m12, 1
  %m13 = mul i32 %a13, %m12
  %a14 = add i32 %m13, 1
  %m14 = mul i32 %a14, %m13
  %a15 = add i32 %m14, 1
  %m15 = mul i32 %a15, %m14
  %a16 = add i32 %m15, 1
  %m16 = mul i32 %a16, %m15
  %a17 = add i32 %m16, 1
  %m17 = mul i32 %a17, %m16
  %a18 = add i32 %m17, 1
  %m18 = mul i32 %a18, %m17
  %a19 = add i32 %m18, 1
  %m19 = mul i32 %a19, %m18
  %a20 = add i32 %m19, 1
  %m20 = mul i32 %a20, %m19
  %a21 = add i32 %m20, 1
  %m21 = mul i32 %a21, %m20
  %a22 = add i32 %m21, 1
  %m22 = mul i32 %a22, %m21
  %a23 = add i32 %m22, 1
  %m23 = mul i32 %a23, %m22
  %a24 = add i32 %m23, 1
  %m24 = mul i32 %a24, %m23
  %a25 = add i32 %m24, 1
  %m25 = mul i32 %a25, %m24
  %a26 = add i32 %m25, 1
  %m26 = mul i32 %a26, %m25
  %a27 = add i32 %m26, 1
  %m27 = mul i32 %a27, %m26
  %a28 = add i32 %m27, 1
  %m28 = mul i32 %a28, %m27
  %a29 = add i32 %m28, 1
  %m29 = mul i32 %a29, %m28
  %a30 = add i32 %m29, 1
  %m30 = mul i32 %a30, %m29
  %a31 = add i32 %m30, 1
  %m31 = mul i32 %a31, %m30
  br label %loop

loop:
  %lp = phi i32 [ %m31, %entry ], [ %linc, %loop ]
  %0 = sext i32 %lp to i64
  %gep = getelementptr inbounds i32, i32* %in, i64 %0
  %loopload = load i32, i32* %gep, align 4
  store i32 0, i32* %gep, align 4
  %linc = add i32 %lp, 1
  %lcmp = icmp eq i32 %linc, 100
  br i1 %lcmp, label %exit, label %loop

exit:
  %ll = phi i32 [ %loopload, %loop ]
  ret i32 %ll
}

