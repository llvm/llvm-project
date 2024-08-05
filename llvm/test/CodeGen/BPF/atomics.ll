; RUN: llc < %s -march=bpfel -verify-machineinstrs -show-mc-encoding | FileCheck --check-prefixes=CHECK,CHECK-V2 %s
; RUN: llc < %s -march=bpfel -verify-machineinstrs -show-mc-encoding -mcpu=v3 | FileCheck --check-prefixes=CHECK,CHECK-V3 %s

; CHECK-LABEL: test_load_add_32
; CHECK-V2: r2 = atomic_fetch_add((u32 *)(r1 + 0), r2)
; CHECK-V3: w2 = atomic_fetch_add((u32 *)(r1 + 0), w2)
; CHECK: encoding: [0xc3,0x21,0x00,0x00,0x01,0x00,0x00,0x00]
define void @test_load_add_32(ptr %p, i32 zeroext %v) {
entry:
  atomicrmw add ptr %p, i32 %v seq_cst
  ret void
}

; CHECK-LABEL: test_load_add_64
; CHECK: r2 = atomic_fetch_add((u64 *)(r1 + 0), r2)
; CHECK: encoding: [0xdb,0x21,0x00,0x00,0x01,0x00,0x00,0x00]
define void @test_load_add_64(ptr %p, i64 zeroext %v) {
entry:
  atomicrmw add ptr %p, i64 %v seq_cst
  ret void
}
