; RUN: not llc -mtriple=agx -mcpu=agx-g13g < %s 2>&1 | FileCheck %s

; CHECK: error: {{.*}}in function doubles {{.*}}: double precision floating point arithmetic (AGX has no double precision hardware)
; CHECK-NOT: error

define agx_kernel void @doubles(ptr addrspace(1) %out, double %a, float %b) {
  %e = fpext float %b to double
  %s = fadd double %a, %e
  %m = fmul double %s, %a
  store double %m, ptr addrspace(1) %out
  ret void
}
