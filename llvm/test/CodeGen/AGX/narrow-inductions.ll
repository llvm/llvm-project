; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -agx-narrow-ivs=0 -verify-machineinstrs < %s \
; RUN:   | FileCheck %s --check-prefix=WIDE

; PTX's 64-bit byte offset over a 256-element threadgroup tile (0 to 4096)
; counts in one 32-bit register: one add and the loop test, no 64-bit add
; and no compare of both halves.

@tile = internal addrspace(3) global [256 x <4 x float>] undef, align 16

; CHECK-LABEL: tile_sum:
; CHECK:       [[LOOP:\.LBB[0-9_]+]]:
; CHECK:       threadgroup_load i32, xyzw, {{.*}}, [[OFF:r[0-9]+]]l, 0
; CHECK:       iadd [[OFF]], [[OFF]], 16
; CHECK-NOT:   icmpsel
; CHECK:       while_icmp r0l, nueq, [[OFF]],
; CHECK-NEXT:  jmp_exec_any [[LOOP]]
; WIDE-LABEL:  tile_sum:
; WIDE:        iadd r{{[0-9]+}}_r{{[0-9]+}}, r{{[0-9]+}}_r{{[0-9]+}}, 16
define agx_kernel void @tile_sum(ptr addrspace(1) %out) {
entry:
  %base = ptrtoint ptr addrspace(3) @tile to i64
  br label %loop

loop:
  %off = phi i64 [ 0, %entry ], [ %off.next, %loop ]
  %acc = phi float [ 0.0, %entry ], [ %sum, %loop ]
  %a = add i64 %base, %off
  %p = inttoptr i64 %a to ptr addrspace(3)
  %v = load <4 x float>, ptr addrspace(3) %p, align 16
  %x = extractelement <4 x float> %v, i64 0
  %w = extractelement <4 x float> %v, i64 3
  %m = fmul float %x, %w
  %sum = fadd float %acc, %m
  %off.next = add nuw nsw i64 %off, 16
  %done = icmp eq i64 %off.next, 4096
  br i1 %done, label %exit, label %loop

exit:
  store float %sum, ptr addrspace(1) %out
  ret void
}
