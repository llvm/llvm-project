; A constant-address-space parameter is packed into _scalar_buf as a scalar
; passed by reference only when it is only ever loaded directly. A constant
; buffer indexed through a GEP stays a buffer argument; it used to be packed
; too, and its uses replaced with undef, so the kernel read garbage.
; RUN: llc -mtriple=air %s -o - | FileCheck %s

; CHECK-LABEL: define void @buffer(ptr addrspace(2) %in, ptr addrspace(1) {{.*}}%out
; CHECK-NOT: undef
; CHECK: getelementptr i32, ptr addrspace(2) %in
; CHECK-LABEL: define void @byref(ptr addrspace(1) {{.*}}%out, ptr addrspace(1) {{.*}}%_scalar_buf
; CHECK: !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, {{.*}}!"air.arg_name", !"in"}

declare { i32, i32, i32 } @air.thread_position_in_grid.v3i32()

define void @buffer(ptr addrspace(2) %in, ptr addrspace(1) %out) {
entry:
  %t = call { i32, i32, i32 } @air.thread_position_in_grid.v3i32()
  %tid = extractvalue { i32, i32, i32 } %t, 0
  %pa = getelementptr i32, ptr addrspace(2) %in, i32 %tid
  %a = load i32, ptr addrspace(2) %pa
  %po = getelementptr i32, ptr addrspace(1) %out, i32 %tid
  store i32 %a, ptr addrspace(1) %po
  ret void
}

define void @byref(ptr addrspace(1) %out, ptr addrspace(2) %n) {
entry:
  %t = call { i32, i32, i32 } @air.thread_position_in_grid.v3i32()
  %tid = extractvalue { i32, i32, i32 } %t, 0
  %v = load i32, ptr addrspace(2) %n
  %po = getelementptr i32, ptr addrspace(1) %out, i32 %tid
  store i32 %v, ptr addrspace(1) %po
  ret void
}
