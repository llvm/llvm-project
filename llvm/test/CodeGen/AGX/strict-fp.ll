; RUN: llc -mtriple=agx -mcpu=agx-g13g -mattr=+ieee-fp32,+canonical-nan -stop-after=agx-strict-fp < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g13g -mattr=+ieee-fp32,+canonical-nan -verify-machineinstrs < %s -o /dev/null
; RUN: llc -mtriple=agx -mcpu=agx-g13g -stop-after=agx-strict-fp < %s | FileCheck %s --check-prefix=DEFAULT

; ieee-fp32: additions keep the hardware instruction and recompute the result
; in a conditional block when both operands are below 2^-101; division is
; computed with integers, comparisons order denormals.
; canonical-nan: NaN results become 0x7fffffff.

; CHECK-LABEL: define agx_kernel void @ops(
; CHECK:         %add = fadd float %a, %b
; CHECK:         fcmp olt float {{.*}}, f0x0D000000
; CHECK:         br i1 {{.*}}, label %[[SLOW:[0-9]+]], label %[[JOIN:[0-9]+]]
; CHECK:       [[JOIN]]:
; CHECK-NEXT:    %[[SUM:[0-9]+]] = phi float [ %add, %0 ], [ %{{[0-9]+}}, %[[SLOW]] ]
; CHECK-NEXT:    %[[NAN:[0-9]+]] = fcmp uno float %[[SUM]], %[[SUM]]
; CHECK-NEXT:    select i1 %[[NAN]], float +nan(0x3FFFFF), float %[[SUM]]
; CHECK-NOT:     fdiv
; CHECK:         call float @llvm.agx.rcp.f32(
; CHECK:         select i1 {{.*}}, float +nan(0x3FFFFF), float %div
; CHECK:         %cmp = fcmp olt float
; CHECK:         icmp slt i32
; CHECK:         select i1 {{.*}}, i1 {{.*}}, i1 %cmp

; DEFAULT-LABEL: define agx_kernel void @ops(
; DEFAULT-NEXT:    %add = fadd float %a, %b
; DEFAULT-NEXT:    %div = fdiv float %add, %c
; DEFAULT-NEXT:    %cmp = fcmp olt float %div, %a
define agx_kernel void @ops(ptr addrspace(1) %out, float %a, float %b, float %c) {
  %add = fadd float %a, %b
  %div = fdiv float %add, %c
  %cmp = fcmp olt float %div, %a
  %sel = select i1 %cmp, float %div, float %b
  store float %sel, ptr addrspace(1) %out
  ret void
}

; Functions that flush float denormals use the hardware operations; NaNs are
; still canonicalized.
; CHECK-LABEL: define agx_kernel void @ftz(
; CHECK-NEXT:    %add = fadd float %a, %b
; CHECK-NEXT:    %[[N:[0-9]+]] = fcmp uno float %add, %add
; CHECK-NEXT:    %[[R:[0-9]+]] = select i1 %[[N]], float +nan(0x3FFFFF), float %add
; CHECK-NEXT:    store float %[[R]]
define agx_kernel void @ftz(ptr addrspace(1) %out, float %a, float %b) #0 {
  %add = fadd float %a, %b
  store float %add, ptr addrspace(1) %out
  ret void
}

; Half NaNs become 0x7fff.
; CHECK-LABEL: define agx_kernel void @half(
; CHECK:         select i1 {{.*}}, half +nan(0x1FF), half %m
define agx_kernel void @half(ptr addrspace(1) %out, half %a, half %b) {
  %m = fmul half %a, %b
  store half %m, ptr addrspace(1) %out
  ret void
}

attributes #0 = { denormal_fpenv(float: preservesign) }
