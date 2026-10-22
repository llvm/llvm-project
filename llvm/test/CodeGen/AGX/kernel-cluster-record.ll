; RUN: llc -mtriple=agx -mcpu=agx-g14c -filetype=obj < %s | llvm-readelf -x .agx.kernels - | FileCheck %s

; A kernel with agx-cluster-dims ends its .agx.kernels record with the
; cluster extension: tag 1, then x, y and z. The record is 4 words longer.
; CHECK: Hex dump of section '.agx.kernels':
; CHECK-NEXT: 0x00000000 48000000 01000000 01000000 6b000000
; CHECK-NEXT: 0x00000010 {{[0-9a-f]+}} {{[0-9a-f]+}} ffffffff {{[0-9a-f]+}}
; CHECK-NEXT: 0x00000020 00000000 00000000 01000000 00000000
; CHECK-NEXT: 0x00000030 08000000 00000000 01000000 02000000
; CHECK-NEXT: 0x00000040 01000000 01000000
define agx_kernel void @k(ptr addrspace(1) %out) "agx-cluster-dims"="2,1,1" {
  store i32 1, ptr addrspace(1) %out
  ret void
}
