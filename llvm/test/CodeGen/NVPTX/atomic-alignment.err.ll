; RUN: not llc -mtriple=nvptx64 -mcpu=sm_80 -filetype=null %s 2>&1 | FileCheck %s

; CHECK: error: unsupported atomic load: instruction alignment 1 is smaller than the required 4-byte alignment for this atomic operation
define i32 @load_i32_align1(ptr %p) {
  %ret = load atomic i32, ptr %p seq_cst, align 1
  ret i32 %ret
}

; CHECK: error: unsupported atomic store: instruction alignment 1 is smaller than the required 4-byte alignment for this atomic operation
define void @store_i32_align1(ptr %p, i32 %v) {
  store atomic i32 %v, ptr %p seq_cst, align 1
  ret void
}

; CHECK: error: unsupported cmpxchg: instruction alignment 4 is smaller than the required 8-byte alignment for this atomic operation
define { i64, i1 } @cmpxchg_i64_align4(ptr %p, i64 %cmp, i64 %new) {
  %ret = cmpxchg ptr %p, i64 %cmp, i64 %new seq_cst monotonic, align 4
  ret { i64, i1 } %ret
}

; CHECK: error: unsupported atomicrmw xchg: instruction alignment 4 is smaller than the required 8-byte alignment for this atomic operation
define i64 @atomicrmw_xchg_i64_align4(ptr %p, i64 %v) {
  %ret = atomicrmw xchg ptr %p, i64 %v seq_cst, align 4
  ret i64 %ret
}
