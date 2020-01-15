; RUN: llc -march=amdgcn -mtriple=amdgcn-- -verify-machineinstrs -o - %s | FileCheck %s

; CHECK-LABEL: foo
; CHECK: buffer_load_ushort
; CHECK: buffer_load_ushort
; CHECK: buffer_load_ushort
; CHECK: buffer_load_ushort
; CHECK: buffer_load_ushort
; CHECK: buffer_load_ushort
; CHECK: buffer_load_ushort
; CHECK: buffer_load_ushort
; CHECK: buffer_load_ushort
; CHECK: buffer_load_ushort
; CHECK: buffer_load_ushort
; CHECK: buffer_load_ushort
; CHECK: buffer_load_ushort
; CHECK: buffer_load_ushort
; CHECK: buffer_load_ushort
; CHECK: buffer_load_ushort
; CHECK: v_bfe_i32
; CHECK: v_bfe_i32

define <2 x i16> @foo(<8 x i16> addrspace(1) * %p0, <8 x i16> addrspace(1) * %p1) {
  br i1 undef, label %T, label %F

T:
  %t = load volatile <8 x i16>, <8 x i16> addrspace(1) * %p0
  br label %exit

F:
  %f = load volatile <8 x i16>, <8 x i16> addrspace(1) * %p1
  br label %exit

exit:
  %m = phi <8 x i16> [ %t, %T ], [ %f, %F ]
  %v2 = shufflevector <8 x i16> %m, <8 x i16> undef, <2 x i32> <i32 0, i32 1>
  %b2 = icmp sgt <2 x i16> %v2, <i16 -1, i16 -1>
  %r2 = select <2 x i1> %b2, <2 x i16> <i16 -32768, i16 -32768>, <2 x i16> <i16 -1, i16 -1>
  ret <2 x i16> %r2
}
