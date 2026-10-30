; 64- and 128-bit (and 8- and 16-bit) atomics are emulated in device memory
; only (the threadgroup form hung an M2 Max).
; RUN: not llc -mtriple=air %s -o /dev/null 2>&1 | FileCheck %s
; CHECK: Metal has no 64-bit atomic operations outside device memory (in tg)
@g = internal addrspace(3) global i64 undef
define void @tg(ptr addrspace(1) %out, i64 %v) "air-kernel" {
  %o = atomicrmw add ptr addrspace(3) @g, i64 %v monotonic
  store i64 %o, ptr addrspace(1) %out
  ret void
}
