; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs < %s | FileCheck %s

; An address computed in one block and used by an access in another is copied
; next to the access, where its base and scaled 32-bit index fold into the
; access: no 64-bit addition remains.

; CHECK-LABEL: later_block:
; CHECK-NOT:     iadd r{{[0-9]+}}_r{{[0-9]+}}
; CHECK:         device_load 0, i32, x, r{{[0-9]+}}, u0_u1, r{{[0-9]+}}, {{(un)?}}signed
; CHECK:         device_store 0, i32, x, r{{[0-9]+}}, u2_u3, r{{[0-9]+}}, {{(un)?}}signed
define agx_kernel void @later_block(i64 %a, i64 %b, i32 %n) {
entry:
  %i = call i32 @llvm.agx.thread.position.in.threadgroup.x()
  %e = sext i32 %i to i64
  %o = shl nsw i64 %e, 2
  %pa = add i64 %a, %o
  %pb = add i64 %b, %o
  %c = icmp slt i32 %i, %n
  br i1 %c, label %then, label %exit
then:
  %la = inttoptr i64 %pa to ptr addrspace(1)
  %v = load i32, ptr addrspace(1) %la, align 4
  %sb = inttoptr i64 %pb to ptr addrspace(1)
  store i32 %v, ptr addrspace(1) %sb, align 4
  br label %exit
exit:
  ret void
}

declare i32 @llvm.agx.thread.position.in.threadgroup.x()

; Threadgroup addresses computed in 64-bit integers, whose scaled index also
; feeds a device access, are computed in 32 bits.

@s = internal addrspace(3) global [64 x i32] poison

; CHECK-LABEL: narrow_threadgroup:
; CHECK-NOT:     iadd r{{[0-9]+}}_r{{[0-9]+}}
; CHECK:         device_load 0, i32, x, r{{[0-9]+}}, u0_u1, r{{[0-9]+}}, {{(un)?}}signed
; CHECK:         threadgroup_store i32, x, r{{[0-9]+}}, r{{[0-9]+}}l, 0
define agx_kernel void @narrow_threadgroup(i64 %a) {
  %i = call i32 @llvm.agx.thread.position.in.threadgroup.x()
  %e = sext i32 %i to i64
  %o = shl nsw i64 %e, 2
  %pa = add i64 %o, %a
  %la = inttoptr i64 %pa to ptr addrspace(1)
  %v = load i32, ptr addrspace(1) %la, align 4
  %ps = add i64 %o, ptrtoint (ptr addrspace(3) @s to i64)
  %ls = inttoptr i64 %ps to ptr addrspace(3)
  store i32 %v, ptr addrspace(3) %ls, align 4
  ret void
}
