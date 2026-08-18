; RUN: llc -mtriple=bpfel -mcpu=v3 -verify-machineinstrs -show-mc-encoding < %s | FileCheck %s
; RUN: llc -mtriple=bpfel -mcpu=v4 -verify-machineinstrs -show-mc-encoding < %s | FileCheck %s
;
; A relaxed (monotonic) atomicrmw whose result is used must select the
; BPF_FETCH form, so the destination register receives the value that was in
; memory before the operation. Selecting the no-fetch "lock" form here leaves
; the operand in the register and silently returns the wrong value.
;
; The same operation with an unused result must keep selecting the compact
; no-fetch form.

; CHECK-LABEL: test_add_64_ret
; CHECK: r0 = atomic_fetch_add((u64 *)(r1 + 0), r0)
; CHECK: encoding: [0xdb,0x01,0x00,0x00,0x01,0x00,0x00,0x00]
define dso_local i64 @test_add_64_ret(ptr %p, i64 %v) {
entry:
  %0 = atomicrmw add ptr %p, i64 %v monotonic
  ret i64 %0
}

; CHECK-LABEL: test_add_64_noret
; CHECK: lock *(u64 *)(r1 + 0) += r2
; CHECK: encoding: [0xdb,0x21,0x00,0x00,0x00,0x00,0x00,0x00]
define dso_local void @test_add_64_noret(ptr %p, i64 %v) {
entry:
  %0 = atomicrmw add ptr %p, i64 %v monotonic
  ret void
}

; CHECK-LABEL: test_add_32_ret
; CHECK: w0 = atomic_fetch_add((u32 *)(r1 + 0), w0)
; CHECK: encoding: [0xc3,0x01,0x00,0x00,0x01,0x00,0x00,0x00]
define dso_local i32 @test_add_32_ret(ptr %p, i32 %v) {
entry:
  %0 = atomicrmw add ptr %p, i32 %v monotonic
  ret i32 %0
}

; CHECK-LABEL: test_add_32_noret
; CHECK: lock *(u32 *)(r1 + 0) += w2
; CHECK: encoding: [0xc3,0x21,0x00,0x00,0x00,0x00,0x00,0x00]
define dso_local void @test_add_32_noret(ptr %p, i32 %v) {
entry:
  %0 = atomicrmw add ptr %p, i32 %v monotonic
  ret void
}

; CHECK-LABEL: test_sub_64_ret
; CHECK: r0 = -r0
; CHECK: r0 = atomic_fetch_add((u64 *)(r1 + 0), r0)
define dso_local i64 @test_sub_64_ret(ptr %p, i64 %v) {
entry:
  %0 = atomicrmw sub ptr %p, i64 %v monotonic
  ret i64 %0
}

; CHECK-LABEL: test_sub_64_noret
; CHECK: r2 = -r2
; CHECK: lock *(u64 *)(r1 + 0) += r2
define dso_local void @test_sub_64_noret(ptr %p, i64 %v) {
entry:
  %0 = atomicrmw sub ptr %p, i64 %v monotonic
  ret void
}

; CHECK-LABEL: test_sub_32_ret
; CHECK: w0 = -w0
; CHECK: w0 = atomic_fetch_add((u32 *)(r1 + 0), w0)
define dso_local i32 @test_sub_32_ret(ptr %p, i32 %v) {
entry:
  %0 = atomicrmw sub ptr %p, i32 %v monotonic
  ret i32 %0
}

; CHECK-LABEL: test_and_32_ret
; CHECK: w0 = atomic_fetch_and((u32 *)(r1 + 0), w0)
define dso_local i32 @test_and_32_ret(ptr %p, i32 %v) {
entry:
  %0 = atomicrmw and ptr %p, i32 %v monotonic
  ret i32 %0
}

; CHECK-LABEL: test_and_32_noret
; CHECK: lock *(u32 *)(r1 + 0) &= w2
define dso_local void @test_and_32_noret(ptr %p, i32 %v) {
entry:
  %0 = atomicrmw and ptr %p, i32 %v monotonic
  ret void
}

; CHECK-LABEL: test_or_32_ret
; CHECK: w0 = atomic_fetch_or((u32 *)(r1 + 0), w0)
define dso_local i32 @test_or_32_ret(ptr %p, i32 %v) {
entry:
  %0 = atomicrmw or ptr %p, i32 %v monotonic
  ret i32 %0
}

; CHECK-LABEL: test_xor_32_ret
; CHECK: w0 = atomic_fetch_xor((u32 *)(r1 + 0), w0)
define dso_local i32 @test_xor_32_ret(ptr %p, i32 %v) {
entry:
  %0 = atomicrmw xor ptr %p, i32 %v monotonic
  ret i32 %0
}

; CHECK-LABEL: test_and_64_ret
; CHECK: r0 = atomic_fetch_and((u64 *)(r1 + 0), r0)
define dso_local i64 @test_and_64_ret(ptr %p, i64 %v) {
entry:
  %0 = atomicrmw and ptr %p, i64 %v monotonic
  ret i64 %0
}

; CHECK-LABEL: test_and_64_noret
; CHECK: lock *(u64 *)(r1 + 0) &= r2
define dso_local void @test_and_64_noret(ptr %p, i64 %v) {
entry:
  %0 = atomicrmw and ptr %p, i64 %v monotonic
  ret void
}
