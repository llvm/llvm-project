; RUN: llc -mtriple=agx -mcpu=agx-g14c < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -mattr=+agx-checked < %s | FileCheck %s --check-prefix=CHECKED

; Only a kernel that uses a global of the global data buffer takes the
; buffer's address in uniforms; the module's other kernels do not, and checked
; mode's site table (a global no kernel reads) adds none.

@counter = addrspace(1) global i32 0, align 4

; CHECK-LABEL: uses:
; CHECK: // global data buffer at u4_u5
; CHECKED-LABEL: uses:
; CHECKED: // global data buffer at u{{[0-9]+}}_u{{[0-9]+}}
define agx_kernel void @uses(i32 %v) {
  store i32 %v, ptr addrspace(1) @counter, align 4
  ret void
}

; CHECK-LABEL: plain:
; CHECK: // uniform registers: 12 x 16-bit (threadgroups_per_grid at u3)
; CHECK-NOT: global data buffer at
; CHECK: stop
; CHECKED-LABEL: plain:
; CHECKED: // uniform registers: {{[0-9]+}} x 16-bit (threadgroups_per_grid at u{{[0-9]+}})
; CHECKED-NOT: global data buffer at
; CHECKED: stop
define agx_kernel void @plain(ptr addrspace(1) %p, i32 %v) {
  store i32 %v, ptr addrspace(1) %p, align 4
  ret void
}
