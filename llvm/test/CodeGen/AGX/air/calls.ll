; AIR keeps calls to its own functions, which may return values; AGX has no
; calls, so agx-lower-air inlines them into the kernels.
; RUN: llc -mtriple=air -air-portable %s -o %t.air.ll
; RUN: FileCheck %s --check-prefix=PORTABLE < %t.air.ll
; RUN: llc -mtriple=agx -mcpu=agx-g14c %t.air.ll -o - | FileCheck %s

; PORTABLE: call float @helper(float

; CHECK-LABEL: k:
; CHECK: fmul32 r{{[0-9]+}}, r{{[0-9]+}}, 3.0
; CHECK-NOT: call
; CHECK: stop
define internal float @helper(float %x) noinline {
  %y = fmul float %x, 3.0
  ret float %y
}

define void @k(ptr addrspace(1) %p) "air-kernel" {
  %v = load float, ptr addrspace(1) %p
  %r = call float @helper(float %v)
  store float %r, ptr addrspace(1) %p
  ret void
}
