; RUN: llc -march=bpfel -filetype=obj -o - %s | llvm-objdump -d - | FileCheck %s

; CHECK-LABEL: test_load_add_32
; CHECK: c3 21
; CHECK: lock *(u32 *)(r1 + 0) += r2
define void @test_load_add_32(i32* %p, i32 zeroext %v) {
entry:
  atomicrmw add i32* %p, i32 %v seq_cst
  ret void
}

; CHECK-LABEL: test_load_add_64
; CHECK: db 21
; CHECK: lock *(u64 *)(r1 + 0) += r2
define void @test_load_add_64(i64* %p, i64 zeroext %v) {
entry:
  atomicrmw add i64* %p, i64 %v seq_cst
  ret void
}
