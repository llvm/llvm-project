; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -filetype=obj < %s | llvm-readelf -x .agx.kernels - | FileCheck %s --check-prefix=SECTION

; By-reference arguments whose uses are loads at constant offsets are passed
; in uniform registers: no device load reads them.
; CHECK-LABEL: byref:
; CHECK:         // uniform registers: 18 x 16-bit (threadgroups_per_grid at u6)
; CHECK-NOT:     device_load
; CHECK:         device_store
define agx_kernel void @byref(ptr addrspace(4) byref(i64) %out, ptr addrspace(4) byref([12 x i8]) %s, ptr addrspace(4) byref(i32) %n) {
  %p = load ptr addrspace(1), ptr addrspace(4) %out
  %a = load i32, ptr addrspace(4) %s
  %b.p = getelementptr i8, ptr addrspace(4) %s, i64 6
  %b = load i16, ptr addrspace(4) %b.p
  %c.p = getelementptr i8, ptr addrspace(4) %s, i64 8
  %c = load float, ptr addrspace(4) %c.p
  %nn = load i32, ptr addrspace(4) %n
  %b32 = zext i16 %b to i32
  %ci = fptosi float %c to i32
  %s1 = add i32 %a, %b32
  %s2 = add i32 %s1, %ci
  %s3 = add i32 %s2, %nn
  store i32 %s3, ptr addrspace(1) %p
  ret void
}

; A by-reference argument indexed dynamically keeps its pointer.
; CHECK-LABEL: dynamic:
; CHECK:         device_load
define agx_kernel void @dynamic(ptr addrspace(1) %out, ptr addrspace(4) byref([4 x i32]) %t, i32 %i) {
  %e = getelementptr [4 x i32], ptr addrspace(4) %t, i32 0, i32 %i
  %v = load i32, ptr addrspace(4) %e
  store i32 %v, ptr addrspace(1) %out
  ret void
}

; The records: size, version 1, name length, name, uniform halves,
; threadgroups_per_grid half, global data half (none), GPRs, threadgroup
; bytes, stack bytes, argument count, then (kind, bytes, half) per argument.
; In byref, the values of out (8 bytes), s (12 bytes) and n (4 bytes) start
; at halves 0, 4 and 10, and threadgroups_per_grid at half 12. In dynamic, t
; is a pointer at half 4.
; SECTION: Hex dump of section '.agx.kernels':
; SECTION-NEXT: 0x00000000 54000000 01000000 05000000 62797265
; SECTION-NEXT: 0x00000010 66000000 12000000 0c000000 ffffffff
; SECTION-NEXT: 0x00000020 03000000 00000000 00000000 03000000
; SECTION-NEXT: 0x00000030 00000000 08000000 00000000 00000000
; SECTION-NEXT: 0x00000040 0c000000 04000000 00000000 04000000
; SECTION-NEXT: 0x00000050 0a000000 54000000 01000000 07000000
; SECTION-NEXT: 0x00000060 64796e61 6d696300 10000000 0a000000
; SECTION-NEXT: 0x00000070 ffffffff 02000000 00000000 00000000
; SECTION-NEXT: 0x00000080 03000000 00000000 08000000 00000000
; SECTION-NEXT: 0x00000090 01000000 10000000 04000000 00000000
; SECTION-NEXT: 0x000000a0 04000000 08000000
