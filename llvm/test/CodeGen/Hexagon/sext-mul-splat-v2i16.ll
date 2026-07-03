; RUN: llc -O2 -mtriple=hexagon < %s | FileCheck %s

; Test that sext v2i16 multiplied by a splatted scalar sext i16 (where the
; splat is hoisted and arrives as assertsext v2i32) is optimized to vmpyh.

; CHECK-LABEL: test_sext_mul_splat:
; CHECK-NOT: vsxthw
; CHECK-NOT: mpyi
; CHECK: vmpyh
define void @test_sext_mul_splat(ptr %src, ptr %dst, i16 signext %scale, i32 %n) {
entry:
  %scale32 = sext i16 %scale to i32
  %ins0 = insertelement <2 x i32> poison, i32 %scale32, i32 0
  %splat = insertelement <2 x i32> %ins0, i32 %scale32, i32 1
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %ptr = getelementptr <2 x i16>, ptr %src, i32 %i
  %v = load <2 x i16>, ptr %ptr, align 4
  %ext = sext <2 x i16> %v to <2 x i32>
  %mul = mul nsw <2 x i32> %ext, %splat
  %dptr = getelementptr <2 x i32>, ptr %dst, i32 %i
  store <2 x i32> %mul, ptr %dptr, align 8
  %i.next = add i32 %i, 1
  %done = icmp eq i32 %i.next, %n
  br i1 %done, label %exit, label %loop

exit:
  ret void
}
