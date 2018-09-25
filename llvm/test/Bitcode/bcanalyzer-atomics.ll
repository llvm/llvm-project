; RUN: llvm-as < %s | llvm-bcanalyzer -dump | FileCheck %s
; Make sure the names of atomics are known

; CHECK: INST_CMPXCHG
; CHECK: INST_STOREATOMIC
; CHECK: INST_LOADATOMIC
; CHECK: INST_FENCE
define void @atomics(i32* %ptr) {
  store atomic i32 0, i32* %ptr monotonic, align 4
  %load = load atomic i32, i32* %ptr monotonic, align 4
  %xchg = cmpxchg i32* %ptr, i32 0, i32 5 acquire monotonic
  fence seq_cst
  ret void
}
