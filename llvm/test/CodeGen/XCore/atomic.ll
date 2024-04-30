; RUN: llc < %s -march=xcore | FileCheck %s

; CHECK-LABEL: atomic_fence
; CHECK: #MEMBARRIER
; CHECK: #MEMBARRIER
; CHECK: #MEMBARRIER
; CHECK: #MEMBARRIER
; CHECK: retsp 0
define void @atomic_fence() nounwind {
entry:
  fence acquire
  fence release
  fence acq_rel
  fence seq_cst
  ret void
}

@pool = external global i64

define void @atomicloadstore() nounwind {
entry:
; CHECK-LABEL: atomicloadstore

; CHECK: __atomic_load_4
  %0 = load atomic i32, ptr @pool seq_cst, align 4

; CHECK: __atomic_load_2
  %1 = load atomic i16, ptr @pool seq_cst, align 2

; CHECK: __atomic_load_1
  %2 = load atomic i8, ptr @pool seq_cst, align 1

; CHECK: __atomic_store_4
  store atomic i32 %0, ptr @pool seq_cst, align 4

; CHECK: __atomic_store_2
  store atomic i16 %1, ptr @pool seq_cst, align 2

; CHECK: __atomic_store_1
  store atomic i8 %2, ptr @pool seq_cst, align 1

  ret void
}
