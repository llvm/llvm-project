; RUN: llc < %s | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin10.0.0"

%0 = type <{ %1, %1 }>
%1 = type { i8, i8, i8, i8 }

@g_2 = global %0 zeroinitializer
@g_4 = global %1 zeroinitializer, align 4


; PR8642
define i16 @test1(i1 zeroext %C, i8** nocapture %argv) nounwind ssp {
entry:
  br i1 %C, label %cond.end.i, label %cond.false.i

cond.false.i:                                     ; preds = %entry
  br label %cond.end.i

cond.end.i:                                       ; preds = %entry
  %call1 = phi i16 [ trunc (i32 srem (i32 1, i32 zext (i1 icmp eq (%1* bitcast (i8* getelementptr inbounds (%0* @g_2, i64 0, i32 1, i32 0) to %1*), %1* @g_4) to i32)) to i16), %cond.false.i ], [ 1, %entry ]
  ret i16 %call1
}

; CHECK: test1:
; CHECK: testb %dil, %dil
; CHECK: jne LBB0_2
; CHECK: divl
; CHECK: LBB0_2:
