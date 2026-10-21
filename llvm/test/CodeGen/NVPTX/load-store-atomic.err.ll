; RUN: not llc < %s -march=nvptx64 -mcpu=sm_100 -mattr=+ptx88 2>&1 | FileCheck %s

; CHECK: error: unsupported atomic store: target supports atomics up to 16 bytes, but this atomic accesses 32 bytes
; CHECK: error: unsupported atomic load: target supports atomics up to 16 bytes, but this atomic accesses 32 bytes

define void @test_i256_global_atomic(ptr addrspace(1) %a, ptr addrspace(1) %b) {
  %a.load = load atomic i256, ptr addrspace(1) %a seq_cst, align 32
  store atomic i256 %a.load, ptr addrspace(1) %b seq_cst, align 32
  ret void
}
