; A 64-bit atomic emulated in a function the kernels call: the lock table
; comes from the kernels' parameter (described in a pre-baked !air.kernel at
; location index 30), passed down the calls.
; RUN: llc -mtriple=air -air-keep-function-cost=1 -air-keep-function-min-size=2 %s -o - | FileCheck %s

; CHECK-LABEL: define internal i64 @add64(i64 %a, i64 %v, ptr addrspace(1) %air_atomic_locks)
; CHECK: atomiclock.held:
; CHECK-LABEL: define internal i64 @twice(i64 %a, i64 %v, ptr addrspace(1) %air_atomic_locks)
; CHECK: call i64 @add64(i64 %a, i64 %v, ptr addrspace(1) %air_atomic_locks)
; CHECK: call i64 @add64(i64 %a, i64 %v, ptr addrspace(1) %air_atomic_locks)
define i64 @add64(i64 %a, i64 %v) {
  %p = inttoptr i64 %a to ptr addrspace(1)
  %o = atomicrmw add ptr addrspace(1) %p, i64 %v monotonic
  %r = add i64 %o, 1
  ret i64 %r
}

define i64 @twice(i64 %a, i64 %v) {
  %x = call i64 @add64(i64 %a, i64 %v)
  %y = call i64 @add64(i64 %a, i64 %v)
  %r = add i64 %x, %y
  ret i64 %r
}

; CHECK-LABEL: define void @k(ptr addrspace(1) {{.*}}%p, ptr addrspace(1) {{.*}}%air_atomic_locks)
; CHECK: call i64 @twice(i64 %a, i64 3, ptr addrspace(1) %air_atomic_locks)
; CHECK: call i64 @twice(i64 %a, i64 5, ptr addrspace(1) %air_atomic_locks)
define void @k(ptr addrspace(1) %p) "air-kernel" {
  %a = ptrtoint ptr addrspace(1) %p to i64
  %x = call i64 @twice(i64 %a, i64 3)
  %y = call i64 @twice(i64 %a, i64 5)
  %r = add i64 %x, %y
  store i64 %r, ptr addrspace(1) %p
  ret void
}

; CHECK: !{ptr @k, !{{[0-9]+}}, [[ARGS:![0-9]+]]}
; CHECK-DAG: [[ARGS]] = !{!{{[0-9]+}}, [[LOCKS:![0-9]+]]}
; CHECK-DAG: [[LOCKS]] = !{i32 1, !"air.buffer", !"air.location_index", i32 30, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"air_atomic_locks"}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"long", !"air.arg_name", !"p"}
