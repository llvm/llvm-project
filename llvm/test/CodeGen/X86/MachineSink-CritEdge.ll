; RUN: llc < %s | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin10.0.0"

define i32 @f(i32 %x) nounwind ssp {
entry:
  %shl.i = shl i32 %x, 12
  %neg.i = xor i32 %shl.i, -1
  %add.i = add nsw i32 %neg.i, %x
  %shr.i = ashr i32 %add.i, 22
  %xor.i = xor i32 %shr.i, %add.i
  %shl5.i = shl i32 %xor.i, 13
  %neg6.i = xor i32 %shl5.i, -1
  %add8.i = add nsw i32 %xor.i, %neg6.i
  %shr10.i = ashr i32 %add8.i, 8
  %xor12.i = xor i32 %shr10.i, %add8.i
  %add16.i = mul i32 %xor12.i, 9
  %shr18.i = ashr i32 %add16.i, 15
  %xor20.i = xor i32 %shr18.i, %add16.i
  %shl22.i = shl i32 %xor20.i, 27
  %neg23.i = xor i32 %shl22.i, -1
  %add25.i = add nsw i32 %xor20.i, %neg23.i
  %shr27.i = ashr i32 %add25.i, 31
  %rem = srem i32 %x, 7
  %cmp = icmp eq i32 %rem, 3
  br i1 %cmp, label %land.lhs.true, label %do.body.preheader

land.lhs.true:
  %call3 = tail call i32 @g(i32 %x) nounwind
  %cmp4 = icmp eq i32 %call3, 10
  br i1 %cmp4, label %do.body.preheader, label %if.then

; %shl.i should be sinked all the way down to do.body.preheader, but not into the loop.
; CHECK: do.body.preheader
; CHECK-NOT: do.body
; CHECK: shll	$12

do.body.preheader:
  %xor29.i = xor i32 %shr27.i, %add25.i
  br label %do.body

if.then:
  %add = add nsw i32 %x, 11
  ret i32 %add

do.body:
  %x.addr.1 = phi i32 [ %add9, %do.body ], [ %x, %do.body.preheader ]
  %xor = xor i32 %xor29.i, %x.addr.1
  %add9 = add nsw i32 %xor, %x.addr.1
  %and = and i32 %add9, 13
  %tobool = icmp eq i32 %and, 0
  br i1 %tobool, label %if.end, label %do.body

if.end:
  ret i32 %add9
}

declare i32 @g(i32)
