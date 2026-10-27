; LLVM intrinsics become AIR functions named after the call's own overload
; suffix (AIRIntrinsicMappings.td): precise unless the call allows approximate
; functions (afn), with a signedness infix for integer operations. Intrinsics
; Metal accepts as they are, like llvm.copysign, are left alone. __mulhi is
; inlined.
; RUN: llc -mtriple=air %s -o - | FileCheck %s

; CHECK-LABEL: define void @kernel
; CHECK: call float @air.sin.f32(float
; CHECK: call float @air.sqrt.f32(float
; CHECK: call afn float @air.fast_sqrt.f32(float
; CHECK: call afn half @air.sqrt.f16(half
; CHECK: call <4 x float> @air.tan.v4f32(<4 x float>
; CHECK: call float @air.fma.f32(float
; CHECK: call float @llvm.copysign.f32(float
; CHECK: call i32 @air.min.s.i32(i32
; CHECK: call i32 @air.max.u.i32(i32
; CHECK: call i32 @air.abs.s.i32(i32 %{{[^,)]*}})
; CHECK: call i32 @air.popcount.i32(i32
; CHECK: call i32 @air.clz.i32(i32 %{{[^,]*}}, i1 false)
; CHECK: zext i32 {{.*}} to i64
; CHECK: mul i64
; CHECK: lshr i64 {{.*}}, 32
; CHECK: trunc i64 {{.*}} to i32
; CHECK: ret void
; CHECK-NOT: declare {{.*}} @llvm.sin
; CHECK-NOT: declare {{.*}} @llvm.sqrt
; CHECK-NOT: __mulhi

declare float @llvm.sin.f32(float)
declare float @llvm.sqrt.f32(float)
declare half @llvm.sqrt.f16(half)
declare <4 x float> @llvm.tan.v4f32(<4 x float>)
declare float @llvm.fma.f32(float, float, float)
declare float @llvm.copysign.f32(float, float)
declare i32 @llvm.smin.i32(i32, i32)
declare i32 @llvm.umax.i32(i32, i32)
declare i32 @llvm.abs.i32(i32, i1)
declare i32 @llvm.ctpop.i32(i32)
declare i32 @llvm.ctlz.i32(i32, i1)
declare i32 @__mulhi(i32, i32)

define void @kernel(float %x, half %hx, <4 x float> %v, i32 %a, i32 %b,
                    ptr %fout, ptr %hout, ptr %vout, ptr %iout) {
entry:
  %s = call float @llvm.sin.f32(float %x)
  %r = call float @llvm.sqrt.f32(float %s)
  %rf = call afn float @llvm.sqrt.f32(float %r)
  %h = call afn half @llvm.sqrt.f16(half %hx)
  %t = call <4 x float> @llvm.tan.v4f32(<4 x float> %v)
  %m = call float @llvm.fma.f32(float %rf, float %rf, float %x)
  %c = call float @llvm.copysign.f32(float %m, float %x)
  store float %c, ptr %fout
  store half %h, ptr %hout
  store <4 x float> %t, ptr %vout
  %i0 = call i32 @llvm.smin.i32(i32 %a, i32 %b)
  %i1 = call i32 @llvm.umax.i32(i32 %i0, i32 %b)
  %i2 = call i32 @llvm.abs.i32(i32 %i1, i1 true)
  %i3 = call i32 @llvm.ctpop.i32(i32 %i2)
  %i4 = call i32 @llvm.ctlz.i32(i32 %i3, i1 false)
  %i5 = call i32 @__mulhi(i32 %i4, i32 %b)
  store i32 %i5, ptr %iout
  ret void
}
