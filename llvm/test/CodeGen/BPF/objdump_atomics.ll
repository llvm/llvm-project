; RUN: llc -march=bpfel -filetype=obj -o - %s | llvm-objdump --no-print-imm-hex -d - | FileCheck %s

; CHECK-LABEL: test_load_add_32
; CHECK: c3 21
; CHECK: lock *(u32 *)(r1 + 0) += r2
define void @test_load_add_32(ptr %p, i32 zeroext %v) {
entry:
  atomicrmw add ptr %p, i32 %v seq_cst
  ret void
}

; CHECK-LABEL: test_load_add_64
; CHECK: db 21
; CHECK: lock *(u64 *)(r1 + 0) += r2
define void @test_load_add_64(ptr %p, i64 zeroext %v) {
entry:
  atomicrmw add ptr %p, i64 %v seq_cst
  ret void
}
