; RUN: llc -mtriple=agx -mcpu=agx-g14c -mattr=+agx-checked -stop-after=agx-checked-accesses < %s | FileCheck %s

; A pointer read back from an aggregate built with insertvalue (MLIR's memref
; descriptors) is checked against the bounds of the argument stored in it.

; CHECK: define agx_kernel void @k(ptr addrspace(1) %in, i64 %i, {{.*}}"agx-checked"="record"{{.*}}, i64 "agx-checked"="lo0" [[LO:%[0-9]+]], i64 "agx-checked"="hi0" [[HI:%[0-9]+]])
; CHECK: icmp ult i64 %{{.*}}, [[LO]]
; CHECK: icmp ugt i64 %{{.*}}, [[HI]]
; CHECK: attributes {{.*}}"agx-checked-slots"="0:-1"

define agx_kernel void @k(ptr addrspace(1) %in, i64 %i) {
  %d0 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64 } poison, ptr addrspace(1) %in, 0
  %d1 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64 } %d0, ptr addrspace(1) %in, 1
  %d2 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64 } %d1, i64 0, 2
  %p = extractvalue { ptr addrspace(1), ptr addrspace(1), i64 } %d2, 1
  %q = getelementptr i32, ptr addrspace(1) %p, i64 %i
  store i32 1, ptr addrspace(1) %q, align 4
  ret void
}
