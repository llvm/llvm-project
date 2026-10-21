; Volatile and system-scope accesses of device memory reach memory shared with
; the host while the dispatch runs (ptx-re-blockers B7): a load is an f59d
; (memory_barrier 3, 1, 9) and a coherent device_load (bits 44-46 = 7, byte 5
; 0xf0), a store a coherent device_store and an f533 (memory_barrier 0, 3, 3).
; Device atomics see no host writes, so system-scope 32-bit atomic loads and
; stores are such loads and stores; other scopes keep the compare-and-swap.
; RUN: llc -mtriple=agx -mcpu=agx-g14c -show-mc-encoding < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g13g -mattr=+mesa-isa -filetype=obj < %s -o /dev/null

; CHECK-LABEL: poll:
; ld.volatile
; CHECK:      memory_barrier 3, 1, 9
; CHECK-NEXT: device_load 0, i32, x, r1, u0_u1, 0, signed, 3 // encoding: [0x05,0x09,0x00,0x0d,0x00,0xf0,0x12,0x00]
; ld.acquire.sys: the load, then the acquire fence
; CHECK:      memory_barrier 3, 1, 9
; CHECK:      device_load 0, i32, x, r4, u0_u1, 1, signed, 3
; CHECK-NEXT: wait 0
; CHECK:      memory_barrier 3, 1, 9
; ld.acquire.gpu stays an atomic
; CHECK:      atomic cmpxchg, 0, r2, r2_r3, 0, signed, r6_r7, 0
; st.release.sys: the release fence, the coherent store and its f533
; CHECK:      memory_barrier 0, 3, 3
; CHECK-NEXT: device_store 0, i32, x, r1, u2_u3, 0, signed, 3 // encoding: [0x45,0x09,0x04,0x0d,0x00,0xf0,0x12,0x00]
; CHECK-NEXT: memory_barrier 0, 3, 3
; st.volatile
; CHECK-NEXT: device_store 0, i32, x, r1, u2_u3, 1, signed, 3
; CHECK-NEXT: memory_barrier 0, 3, 3
; a plain store
; CHECK-NEXT: device_store 0, i32, x, r1, u2_u3, 2, signed // encoding: [0x45,0x09,0x24,0x0d,0x00,0xc0,0x12,0x00]
define void @poll(ptr addrspace(1) %flag, ptr addrspace(1) %out) {
  %v = load volatile i32, ptr addrspace(1) %flag
  %p1 = getelementptr i32, ptr addrspace(1) %flag, i32 1
  %a = load atomic i32, ptr addrspace(1) %p1 syncscope("one-as") acquire, align 4
  %p2 = getelementptr i32, ptr addrspace(1) %flag, i32 2
  %g = load atomic i32, ptr addrspace(1) %p2 syncscope("agent-one-as") acquire, align 4
  %s = add i32 %v, %a
  %t = add i32 %s, %g
  store atomic i32 %t, ptr addrspace(1) %out syncscope("one-as") release, align 4
  %o1 = getelementptr i32, ptr addrspace(1) %out, i32 1
  store volatile i32 %t, ptr addrspace(1) %o1
  %o2 = getelementptr i32, ptr addrspace(1) %out, i32 2
  store i32 %t, ptr addrspace(1) %o2
  ret void
}
