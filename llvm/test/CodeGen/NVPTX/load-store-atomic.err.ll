; RUN: not llc < %s -march=nvptx64 -mcpu=sm_100 -mattr=+ptx88 2>&1 | FileCheck %s

; CHECK: error: unsupported atomic store
; CHECK: error: unsupported atomic load
; CHECK: error: unsupported atomic store
; CHECK: error: unsupported atomic load

;; TODO: we could actually support this but we don't currently support b128
;;       load lowering.
define void @test_i128_generic_atomic(ptr %a, ptr %b) {
  %a.load = load atomic i128, ptr %a seq_cst, align 16
  store atomic i128 %a.load, ptr %b seq_cst, align 16
  ret void
}

define void @test_i256_global_atomic(ptr addrspace(1) %a, ptr addrspace(1) %b) {
  %a.load = load atomic i256, ptr addrspace(1) %a seq_cst, align 32
  store atomic i256 %a.load, ptr addrspace(1) %b seq_cst, align 32
  ret void
}
