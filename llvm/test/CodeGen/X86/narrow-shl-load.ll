; RUN: llc -march=x86-64 < %s | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-pc-linux-gnu"

; DAGCombiner should fold this code in finite time.
; rdar://8606584

define void @test1() nounwind readnone {
bb.nph:
  br label %while.cond

while.cond:                                       ; preds = %while.cond, %bb.nph
  %tmp6 = load i32* undef, align 4
  %and = or i64 undef, undef
  %conv11 = zext i32 undef to i64
  %conv14 = zext i32 %tmp6 to i64
  %shl15 = shl i64 %conv14, 1
  %shl15.masked = and i64 %shl15, 4294967294
  %and17 = or i64 %shl15.masked, %conv11
  %add = add i64 %and17, 1
  %xor = xor i64 %add, %and
  %tmp20 = load i64* undef, align 8
  %add21 = add i64 %xor, %tmp20
  %conv22 = trunc i64 %add21 to i32
  store i32 %conv22, i32* undef, align 4
  br i1 false, label %while.end, label %while.cond

while.end:                                        ; preds = %while.cond
  ret void
}


; DAGCombiner shouldn't fold the sdiv (ashr) away.
; rdar://8636812
; CHECK: test2:
; CHECK:   sarl

define i32 @test2() nounwind {
entry:
  %i = alloca i32, align 4
  %j = alloca i8, align 1
  store i32 127, i32* %i, align 4
  store i8 0, i8* %j, align 1
  %tmp3 = load i32* %i, align 4
  %mul = mul nsw i32 %tmp3, 2
  %conv4 = trunc i32 %mul to i8
  %conv5 = sext i8 %conv4 to i32
  %div6 = sdiv i32 %conv5, 2
  %conv7 = trunc i32 %div6 to i8
  %conv9 = sext i8 %conv7 to i32
  %cmp = icmp eq i32 %conv9, -1
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  ret i32 0

if.end:                                           ; preds = %entry
  call void @abort() noreturn
  unreachable
}

declare void @abort() noreturn

declare void @exit(i32) noreturn

; DAG Combiner can't fold this into a load of the 1'th byte.
; PR8757
define i32 @test3(i32 *%P) nounwind ssp {
  store volatile i32 128, i32* %P
  %tmp4.pre = load i32* %P
  %phitmp = trunc i32 %tmp4.pre to i16
  %phitmp13 = shl i16 %phitmp, 8
  %phitmp14 = ashr i16 %phitmp13, 8
  %phitmp15 = lshr i16 %phitmp14, 8
  %phitmp16 = zext i16 %phitmp15 to i32
  ret i32 %phitmp16
  
; CHECK: movl	$128, (%rdi)
; CHECK-NEXT: movsbl	(%rdi), %eax
; CHECK-NEXT: movzbl	%ah, %eax
; CHECK-NEXT: ret
}
