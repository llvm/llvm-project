; RUN: llc < %s -mtriple=bpfel -mcpu=v3 -verify-machineinstrs | FileCheck %s
;
; Mismatched alignment (dst 8, src 1): expect plain byte copies, no shifts/ORs
; and no wide (u64) access.
define void @memcpy_dst8_src1(ptr align 8 %a, ptr align 1 %b) {
; CHECK-LABEL: memcpy_dst8_src1:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    w3 = *(u8 *)(r2 + 7)
; CHECK-NEXT:    *(u8 *)(r1 + 7) = w3
; CHECK-NEXT:    w3 = *(u8 *)(r2 + 6)
; CHECK-NEXT:    *(u8 *)(r1 + 6) = w3
; CHECK-NEXT:    w3 = *(u8 *)(r2 + 5)
; CHECK-NEXT:    *(u8 *)(r1 + 5) = w3
; CHECK-NEXT:    w3 = *(u8 *)(r2 + 4)
; CHECK-NEXT:    *(u8 *)(r1 + 4) = w3
; CHECK-NEXT:    w3 = *(u8 *)(r2 + 3)
; CHECK-NEXT:    *(u8 *)(r1 + 3) = w3
; CHECK-NEXT:    w3 = *(u8 *)(r2 + 2)
; CHECK-NEXT:    *(u8 *)(r1 + 2) = w3
; CHECK-NEXT:    w3 = *(u8 *)(r2 + 1)
; CHECK-NEXT:    *(u8 *)(r1 + 1) = w3
; CHECK-NEXT:    w2 = *(u8 *)(r2 + 0)
; CHECK-NEXT:    *(u8 *)(r1 + 0) = w2
; CHECK-NEXT:    exit
entry:
  tail call void @llvm.memcpy.p0.p0.i64(ptr align 8 %a, ptr align 1 %b, i64 8, i1 false)
  ret void
}

define void @memmove_dst8_src1(ptr align 8 %a, ptr align 1 %b) {
; CHECK-LABEL: memmove_dst8_src1:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    w3 = *(u8 *)(r2 + 0)
; CHECK-NEXT:    w4 = *(u8 *)(r2 + 1)
; CHECK-NEXT:    w5 = *(u8 *)(r2 + 2)
; CHECK-NEXT:    w0 = *(u8 *)(r2 + 3)
; CHECK-NEXT:    w6 = *(u8 *)(r2 + 4)
; CHECK-NEXT:    w7 = *(u8 *)(r2 + 5)
; CHECK-NEXT:    w8 = *(u8 *)(r2 + 6)
; CHECK-NEXT:    w2 = *(u8 *)(r2 + 7)
; CHECK-NEXT:    *(u8 *)(r1 + 7) = w2
; CHECK-NEXT:    *(u8 *)(r1 + 6) = w8
; CHECK-NEXT:    *(u8 *)(r1 + 5) = w7
; CHECK-NEXT:    *(u8 *)(r1 + 4) = w6
; CHECK-NEXT:    *(u8 *)(r1 + 3) = w0
; CHECK-NEXT:    *(u8 *)(r1 + 2) = w5
; CHECK-NEXT:    *(u8 *)(r1 + 1) = w4
; CHECK-NEXT:    *(u8 *)(r1 + 0) = w3
; CHECK-NEXT:    exit
entry:
  tail call void @llvm.memmove.p0.p0.i64(ptr align 8 %a, ptr align 1 %b, i64 8, i1 false)
  ret void
}

declare void @llvm.memcpy.p0.p0.i64(ptr nocapture writeonly, ptr nocapture readonly, i64, i1)
declare void @llvm.memmove.p0.p0.i64(ptr nocapture writeonly, ptr nocapture readonly, i64, i1)
