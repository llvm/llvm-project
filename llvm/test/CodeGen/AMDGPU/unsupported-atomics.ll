; RUN: not llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -filetype=null %s 2>&1 | FileCheck %s

; CHECK: error: unsupported atomic load
define i128 @test_load_i128(ptr %p) nounwind {
  %ld = load atomic i128, ptr %p seq_cst, align 16
  ret i128 %ld
}

; CHECK: error: unsupported atomic store
define void @test_store_i128(ptr %p, i128 %val) nounwind {
  store atomic i128 %val, ptr %p seq_cst, align 16
  ret void
}

; CHECK: error: unsupported cmpxchg
define { i128, i1 } @cmpxchg_i128(ptr %p, i128 %cmp, i128 %val) nounwind {
  %ret = cmpxchg ptr %p, i128 %cmp, i128 %val seq_cst monotonic
  ret { i128, i1 } %ret
}

; CHECK: error: unsupported cmpxchg
define i128 @atomicrmw_xchg_i128(ptr %p, i128 %val) nounwind {
  %ret = atomicrmw xchg ptr %p, i128 %val seq_cst
  ret i128 %ret
}

; CHECK: error: unsupported cmpxchg
define i64 @atomicrmw_xchg_i64_align4(ptr %p, i64 %val) nounwind {
  %ret = atomicrmw xchg ptr %p, i64 %val seq_cst, align 4
  ret i64 %ret
}

; CHECK: error: unsupported cmpxchg
define double @atomicrmw_fadd_f64_align4(ptr %p, double %val) nounwind {
  %ret = atomicrmw fadd ptr %p, double %val seq_cst, align 4
  ret double %ret
}

; CHECK: error: unsupported cmpxchg
define fp128 @atomicrmw_fadd_f128_align4(ptr %p, fp128 %val) nounwind {
  %ret = atomicrmw fadd ptr %p, fp128 %val seq_cst, align 4
  ret fp128 %ret
}

; CHECK: error: unsupported cmpxchg
define fp128 @atomicrmw_fadd_f128(ptr %p, fp128 %val) nounwind {
  %ret = atomicrmw fadd ptr %p, fp128 %val seq_cst, align 16
  ret fp128 %ret
}

; CHECK: error: unsupported cmpxchg
define <2 x half> @test_atomicrmw_fadd_v2f16_global_agent_align2(ptr addrspace(1) %ptr, <2 x half> %value) {
  %res = atomicrmw fadd ptr addrspace(1) %ptr, <2 x half> %value syncscope("agent") seq_cst, align 2
  ret <2 x half> %res
}
