; AIR's air.fmin/fmax drop NaN; wrap renamed calls in a NaN-propagation select.
; On Metal they also do not order signed zeros: of two zeros, minimum is the
; OR of the encodings (-0 if either is) and maximum their AND. The portable
; passes (AGX) leave that out.
; RUN: llc -mtriple=air %s -o - | FileCheck %s
; RUN: llc -mtriple=air -air-portable %s -o - | FileCheck %s --check-prefix=PORTABLE

; CHECK-LABEL: define void @kernel
; CHECK-NOT: llvm.minimum
; CHECK-NOT: llvm.maximum
; CHECK: %mn = call float @air.fmin.f32(float %a, float %b)
; CHECK: %[[IA:.*]] = bitcast float %a to i32
; CHECK: %[[IB:.*]] = bitcast float %b to i32
; CHECK: %[[OR:.*]] = or i32 %[[IA]], %[[IB]]
; CHECK: %[[SH:.*]] = shl i32 %[[OR]], 1
; CHECK: %[[Z:.*]] = icmp eq i32 %[[SH]], 0
; CHECK: %[[ZF:.*]] = bitcast i32 %[[OR]] to float
; CHECK: %[[O:.*]] = select i1 %[[Z]], float %[[ZF]], float %mn
; CHECK: fcmp uno float %a, %b
; CHECK: select i1 %{{.*}}, float +qnan, float %[[O]]
; CHECK: %mx = call float @air.fmax.f32(float %a, float %b)
; CHECK: or i32
; CHECK: icmp eq i32
; CHECK: %[[AND:.*]] = and i32
; CHECK: bitcast i32 %[[AND]] to float
; CHECK: fcmp uno
; CHECK: select i1
; CHECK: ret void

; PORTABLE-LABEL: define void @kernel
; PORTABLE: %mn = call float @air.fmin.f32(
; PORTABLE-NEXT: %nan_check = fcmp uno float
; PORTABLE-NEXT: %mn.nan = select i1 %nan_check, float +qnan, float %mn
; PORTABLE: %mx = call float @air.fmax.f32(
; PORTABLE-NEXT: %nan_check1 = fcmp uno float
; PORTABLE-NEXT: %mx.nan = select i1 %nan_check1, float +qnan, float %mx

declare float @llvm.minimum.f32(float, float)
declare float @llvm.maximum.f32(float, float)

define void @kernel(float %a, float %b, ptr %out) {
entry:
  %mn = call float @llvm.minimum.f32(float %a, float %b)
  %mx = call float @llvm.maximum.f32(float %a, float %b)
  %s = fadd float %mn, %mx
  store float %s, ptr %out
  ret void
}
