; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs < %s | FileCheck %s

; An address computed in one block and used by an access in another is copied
; next to the access, where its base and scaled 32-bit index fold into the
; access: no 64-bit addition remains.

; CHECK-LABEL: later_block:
; CHECK-NOT:     iadd r{{[0-9]+}}_r{{[0-9]+}}
; CHECK:         device_load 0, i32, x, r{{[0-9]+}}, u0_u1, r{{[0-9]+}}, signed
; CHECK:         device_store 0, i32, x, r{{[0-9]+}}, u2_u3, r{{[0-9]+}}, signed
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
