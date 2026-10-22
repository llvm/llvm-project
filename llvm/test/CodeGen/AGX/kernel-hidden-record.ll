; RUN: llc -mtriple=agx -mcpu=agx-g14c -filetype=obj < %s | llvm-readelf -x .agx.kernels - | FileCheck %s

; An argument with "agx-hidden"="threadgroup-bytes" is not a source argument:
; the record lists one argument and ends with extension 4 and the argument's
; first half.
; CHECK: Hex dump of section '.agx.kernels':
; CHECK-NEXT: 0x00000000 40000000 01000000 01000000 6b000000
; CHECK-NEXT: 0x00000010 {{[0-9a-f]+}} {{[0-9a-f]+}} ffffffff {{[0-9a-f]+}}
; CHECK-NEXT: 0x00000020 00000000 00000000 01000000 00000000
; CHECK-NEXT: 0x00000030 08000000 00000000 04000000 04000000
define agx_kernel void @k(ptr addrspace(1) %out, i32 "agx-hidden"="threadgroup-bytes" %bytes) {
  store i32 %bytes, ptr addrspace(1) %out
  ret void
}
