; agx-lower-air accepts only what AIR has, and kernels that !air.kernel fully
; describes.
; RUN: split-file %s %t
; RUN: not opt -mtriple=agx -passes=agx-lower-air -disable-output %t/double.ll 2>&1 | FileCheck %s --check-prefix=DOUBLE
; RUN: not opt -mtriple=agx -passes=agx-lower-air -disable-output %t/cast.ll 2>&1 | FileCheck %s --check-prefix=CAST
; RUN: not opt -mtriple=agx -passes=agx-lower-air -disable-output %t/i1.ll 2>&1 | FileCheck %s --check-prefix=I1
; RUN: not opt -mtriple=agx -passes=agx-lower-air -disable-output %t/undescribed.ll 2>&1 | FileCheck %s --check-prefix=UNDESCRIBED
; RUN: not opt -mtriple=agx -passes=agx-lower-air -disable-output %t/bare.ll 2>&1 | FileCheck %s --check-prefix=BARE
; RUN: not opt -mtriple=agx -passes=agx-lower-air -disable-output %t/two-tg.ll 2>&1 | FileCheck %s --check-prefix=TWOTG
; RUN: not opt -mtriple=agx -passes=agx-lower-air -disable-output %t/argbuf.ll 2>&1 | FileCheck %s --check-prefix=ARGBUF
; RUN: not opt -mtriple=agx -passes=agx-lower-air -disable-output %t/order.ll 2>&1 | FileCheck %s --check-prefix=ORDER

; DOUBLE: error: {{.*}}a value of a type AIR does not have in k (AIR does not have it)
; CAST: error: {{.*}}an address space cast in k (AIR does not have it)
; I1: error: {{.*}}an i1 in memory in k (AIR does not have it)
; UNDESCRIBED: error: {{.*}}parameter 1 is not described in !air.kernel
; BARE: error: {{.*}}the !air.kernel entry of this kernel describes no parameters
; TWOTG: error: {{.*}}more than one threadgroup buffer parameter
; ARGBUF: error: {{.*}}argument buffers (air.indirect_buffer) are not supported
; ORDER: error: {{.*}}buffer locations are not in parameter order

;--- double.ll
define void @k(ptr addrspace(1) %p) {
  %v = load double, ptr addrspace(1) %p
  ret void
}
!air.kernel = !{!0}
!0 = !{ptr @k, !{}, !{!1}}
!1 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1}

;--- cast.ll
define void @k(ptr addrspace(1) %p) {
  %q = addrspacecast ptr addrspace(1) %p to ptr
  store i32 0, ptr %q
  ret void
}
!air.kernel = !{!0}
!0 = !{ptr @k, !{}, !{!1}}
!1 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1}

;--- i1.ll
define void @k(ptr addrspace(1) %p) {
  store i1 true, ptr addrspace(1) %p
  ret void
}
!air.kernel = !{!0}
!0 = !{ptr @k, !{}, !{!1}}
!1 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1}

;--- undescribed.ll
define void @k(ptr addrspace(1) %p, i32 %n) {
  store i32 %n, ptr addrspace(1) %p
  ret void
}
!air.kernel = !{!0}
!0 = !{ptr @k, !{}, !{!1}}
!1 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1}

;--- bare.ll
define void @k(ptr addrspace(1) %p) {
  ret void
}
!air.kernel = !{!0}
!0 = !{ptr @k}

;--- two-tg.ll
define void @k(ptr addrspace(3) %a, ptr addrspace(3) %b) {
  ret void
}
!air.kernel = !{!0}
!0 = !{ptr @k, !{}, !{!1, !2}}
!1 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 3}
!2 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 3}

;--- argbuf.ll
define void @k(ptr addrspace(2) %a) {
  ret void
}
!air.kernel = !{!0}
!0 = !{ptr @k, !{}, !{!1}}
!1 = !{i32 0, !"air.indirect_buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2}

;--- order.ll
define void @k(ptr addrspace(1) %a, ptr addrspace(1) %b) {
  ret void
}
!air.kernel = !{!0}
!0 = !{ptr @k, !{}, !{!1, !2}}
!1 = !{i32 0, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1}
!2 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1}
