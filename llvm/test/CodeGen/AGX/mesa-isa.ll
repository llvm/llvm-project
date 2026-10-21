; strict mode: only the instruction forms Mesa's AGX compiler emits.
; RUN: llc -mtriple=agx -mcpu=agx-g13g -mattr=+agx-strict -show-mc-encoding < %s | FileCheck %s --check-prefixes=CHECK,BIT45-0
; RUN: llc -mtriple=agx -mcpu=agx-g14c -mattr=+agx-strict -show-mc-encoding < %s | FileCheck %s --check-prefixes=CHECK,BIT45-0
; RUN: llc -mtriple=agx -mcpu=agx-g13c -mattr=+agx-strict -show-mc-encoding < %s | FileCheck %s --check-prefixes=CHECK,BIT45-1
; RUN: llc -mtriple=agx -mcpu=agx-g14d -mattr=+agx-strict -show-mc-encoding < %s | FileCheck %s --check-prefixes=CHECK,BIT45-1
; The module flag selects the same mode.
; RUN: sed 's/^;FLAG //' %s | llc -mtriple=agx -mcpu=agx-g14c -show-mc-encoding | FileCheck %s --check-prefixes=CHECK,BIT45-0
; Without it, G14 has the native float add and u64 min/max.
; RUN: llc -mtriple=agx -mcpu=agx-g14c -show-mc-encoding < %s | FileCheck %s --check-prefix=NATIVE
; Every object instruction is checked as well.
; RUN: llc -mtriple=agx -mcpu=agx-g13g -mattr=+agx-strict -filetype=obj < %s -o /dev/null

;FLAG !llvm.module.flags = !{!0}
;FLAG !0 = !{i32 1, !"agx-strict", i32 1}

; Mesa has no float atomics: atomicrmw fadd is a compare-and-swap loop. Bit 45
; (encoding byte 5, 0x20) is set only on multi-cluster G13 and two-die parts.
; CHECK-LABEL: fadd:
; CHECK-NOT: atomic sub
; CHECK: atomic cmpxchg
; BIT45-0-SAME: encoding: [0xd5,0x08,0x00,0x8d,0x00,0xd0,0x04,0x00]
; BIT45-1-SAME: encoding: [0xd5,0x08,0x00,0x8d,0x00,0xf0,0x04,0x00]
; NATIVE-LABEL: fadd:
; NATIVE: atomic sub, 0, r1, u0_u1, 0, signed, r1, 1
define void @fadd(ptr addrspace(1) %p, float %v) {
  %r = atomicrmw fadd ptr addrspace(1) %p, float %v monotonic
  ret void
}

; A 64-bit unsigned min takes the lock table instead of the G14 form.
; CHECK-LABEL: umin64:
; CHECK: %wide.atomic
; NATIVE-LABEL: umin64:
; NATIVE: atomic cmpxchg, 0, _, u0_u1, 0, signed, r2_r3, 3
define void @umin64(ptr addrspace(1) %p, i64 %v) {
  %r = atomicrmw umin ptr addrspace(1) %p, i64 %v monotonic
  ret void
}

; Threadgroup accesses take the long (8-byte) form that Mesa always packs.
; CHECK-LABEL: local:
; The default is the short (6-byte) form.
; CHECK: threadgroup_store i32, x, r2, r1l, 0 {{.*}} encoding: [0x29,0x89,0x02,0x02,0x14,0x00,0x00,0x00]
; CHECK: threadgroup_load i32, x, r1, 0, 0 {{.*}} encoding: [0x69,0x85,0x80,0x02,0x14,0x00,0x00,0x00]
; NATIVE-LABEL: local:
; NATIVE: threadgroup_store i32, x, r2, r1l, 0 {{.*}} encoding: [0x29,0x09,0x02,0x02,0x14,0x00]
@buf = internal addrspace(3) global [64 x i32] poison
define void @local(ptr addrspace(1) %out, i32 %i) {
  %p = getelementptr i32, ptr addrspace(3) @buf, i32 %i
  store i32 7, ptr addrspace(3) %p
  %v = load i32, ptr addrspace(3) @buf
  store i32 %v, ptr addrspace(1) %out
  ret void
}
