; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=atomic-ordering --test FileCheck --test-arg --check-prefixes=INTERESTING,CHECK --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck -check-prefixes=RESULT,CHECK %s < %t

; CHECK-LABEL: @load_atomic_keep(
; INTERESTING: seq_cst
; RESULT: %op = load atomic i32, ptr %ptr syncscope("agent") seq_cst, align 4
define i32 @load_atomic_keep(ptr %ptr) {
  %op = load atomic i32, ptr %ptr syncscope("agent") seq_cst, align 4
  ret i32 %op
}

; CHECK-LABEL: @load_atomic_drop(
; INTERESTING: load
; RESULT: %op = load i32, ptr %ptr, align 4
define i32 @load_atomic_drop(ptr %ptr) {
  %op = load atomic i32, ptr %ptr syncscope("agent") seq_cst, align 4
  ret i32 %op
}

; CHECK-LABEL: @store_atomic_keep(
; INTERESTING: syncscope("agent")
; RESULT: store atomic i32 0, ptr %ptr syncscope("agent") seq_cst, align 4
define void @store_atomic_keep(ptr %ptr) {
  store atomic i32 0, ptr %ptr syncscope("agent") seq_cst, align 4
  ret void
}

; CHECK-LABEL: @store_atomic_drop(
; INTERESTING: store
; RESULT: store i32 0, ptr %ptr, align 4
define void @store_atomic_drop(ptr %ptr) {
  store atomic i32 0, ptr %ptr syncscope("agent") seq_cst, align 4
  ret void
}

; CHECK-LABEL: @atomicrmw_atomic_seq_cst_keep(
; INTERESTING: seq_cst
; RESULT: %val = atomicrmw add ptr %ptr, i32 3 syncscope("agent") seq_cst, align 4
define i32 @atomicrmw_atomic_seq_cst_keep(ptr %ptr) {
  %val = atomicrmw add ptr %ptr, i32 3 syncscope("agent") seq_cst, align 4
  ret i32 %val
}

; CHECK-LABEL: @atomicrmw_atomic_seq_cst_drop(
; INTERESTING: atomicrmw
; RESULT: %val = atomicrmw add ptr %ptr, i32 3 monotonic, align 4
define i32 @atomicrmw_atomic_seq_cst_drop(ptr %ptr) {
  %val = atomicrmw add ptr %ptr, i32 3 seq_cst
  ret i32 %val
}

; CHECK-LABEL: @cmpxchg_atomic_seq_cst_seq_cst_keep(
; INTERESTING: seq_cst seq_cst
; RESULT: %val = cmpxchg ptr %ptr, i32 %old, i32 %in syncscope("agent") seq_cst seq_cst, align 4
define { i32, i1 } @cmpxchg_atomic_seq_cst_seq_cst_keep(ptr %ptr, i32 %old, i32 %in) {
  %val = cmpxchg ptr %ptr, i32 %old, i32 %in syncscope("agent") seq_cst seq_cst
  ret { i32, i1 } %val
}

; CHECK-LABEL: @cmpxchg_seq_cst_seq_cst_keep_left(
; INTERESTING: syncscope("agent") seq_cst
; RESULT: %val = cmpxchg ptr %ptr, i32 %old, i32 %in syncscope("agent") seq_cst monotonic, align 4
define { i32, i1 } @cmpxchg_seq_cst_seq_cst_keep_left(ptr %ptr, i32 %old, i32 %in) {
  %val = cmpxchg ptr %ptr, i32 %old, i32 %in syncscope("agent") seq_cst seq_cst
  ret { i32, i1 } %val
}

; CHECK-LABEL: @cmpxchg_seq_cst_seq_cst_keep_right(
; INTERESTING: seq_cst, align 4
; RESULT: %val = cmpxchg ptr %ptr, i32 %old, i32 %in syncscope("agent") monotonic seq_cst, align 4
define { i32, i1 } @cmpxchg_seq_cst_seq_cst_keep_right(ptr %ptr, i32 %old, i32 %in) {
  %val = cmpxchg ptr %ptr, i32 %old, i32 %in syncscope("agent") seq_cst seq_cst, align 4
  ret { i32, i1 } %val
}

; CHECK-LABEL: @cmpxchg_seq_cst_seq_cst_drop(
; INTERESTING: = cmpxchg ptr
; RESULT: %val = cmpxchg ptr %ptr, i32 %old, i32 %in syncscope("agent") monotonic monotonic, align 4
define { i32, i1 } @cmpxchg_seq_cst_seq_cst_drop(ptr %ptr, i32 %old, i32 %in) {
  %val = cmpxchg ptr %ptr, i32 %old, i32 %in syncscope("agent") seq_cst seq_cst
  ret { i32, i1 } %val
}
