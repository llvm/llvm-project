; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs < %s | FileCheck %s

; A branch on booleans combined with and / or / xor (bitonic sort's
; `(a > b) == up`) computes them in 32 bits, where the compares produce them,
; not in 16 bits followed by a move to 32 bits for the branch; the branch
; on (b ^ 1) compares b with zero itself, and on a == b compares the two
; booleans, not their xor with zero.

; CHECK-LABEL: compare_exchange:
; CHECK:       icmpsel ugt, [[GT:r[0-9]+]],
; CHECK-NOT:   xor
; CHECK:       if_icmp r0l, ueq, [[GT]], {{r[0-9]+}}, 1
define agx_kernel void @compare_exchange(ptr addrspace(3) %t, i32 %p, i32 %s) {
  %a = load i32, ptr addrspace(3) %t
  %q = getelementptr i32, ptr addrspace(3) %t, i32 1
  %b = load i32, ptr addrspace(3) %q
  %m = and i32 %p, %s
  %up = icmp eq i32 %m, 0
  %gt = icmp ugt i32 %a, %b
  %c = icmp eq i1 %gt, %up
  br i1 %c, label %swap, label %end
swap:
  store i32 %b, ptr addrspace(3) %t
  store i32 %a, ptr addrspace(3) %q
  br label %end
end:
  ret void
}
