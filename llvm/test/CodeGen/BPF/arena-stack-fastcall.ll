; RUN: llc -mtriple=bpfel -mcpu=v4 -mattr=+arena-stack < %s | FileCheck %s
;
; The slots for registers that are saved around a bpf_fastcall function are in
; the BPF stack. A local that is in the stack in arena takes no room there.

declare void @fast() #0

define internal void @use(ptr %p, i64 %v) noinline {
  store i64 %v, ptr %p
  ret void
}

define void @f(i64 %a) {
; CHECK-LABEL: f:
; CHECK:       r12 += 64
; CHECK:       *(u64 *)(r10 - 8) = r[[R:[0-9]]]
; CHECK-NEXT:  call fast
; CHECK-NEXT:  r[[R]] = *(u64 *)(r10 - 8)
  %buf = alloca [64 x i8], align 8
  call void @fast() #0
  call void @use(ptr %buf, i64 %a)
  ret void
}

attributes #0 = { "bpf_fastcall" }
