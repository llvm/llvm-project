; RUN: not llc -mtriple=agx -mcpu=agx-g13g < %s 2>&1 | FileCheck %s

; Double precision arithmetic is computed in software (see soft-f64.ll),
; except for the remainder and vector operations.
; CHECK: error: {{.*}}in function doubles {{.*}}: double precision floating point arithmetic (AGX has no double precision hardware)
; CHECK-NOT: error

define agx_kernel void @doubles(ptr addrspace(1) %out, double %a, double %b) {
  %r = frem double %a, %b
  store double %r, ptr addrspace(1) %out
  ret void
}
