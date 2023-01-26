; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=syncscopes --test FileCheck --test-arg --check-prefixes=INTERESTING,CHECK --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck -check-prefixes=RESULT,CHECK %s < %t

; CHECK-LABEL: @load_syncscope_keep(
; INTERESTING: syncscope("agent")
; RESULT: %op = load atomic i32, ptr %ptr syncscope("agent") seq_cst, align 4
define i32 @load_syncscope_keep(ptr %ptr) {
  %op = load atomic i32, ptr %ptr syncscope("agent") seq_cst, align 4
  ret i32 %op
}

; CHECK-LABEL: @load_syncscope_drop(
; INTERESTING: load atomic
; RESULT: %op = load atomic i32, ptr %ptr seq_cst, align 4
define i32 @load_syncscope_drop(ptr %ptr) {
  %op = load atomic i32, ptr %ptr syncscope("agent") seq_cst, align 4
  ret i32 %op
}

; CHECK-LABEL: @store_syncscope_keep(
; INTERESTING: syncscope("agent")
; RESULT: store atomic i32 0, ptr %ptr syncscope("agent") seq_cst, align 4
define void @store_syncscope_keep(ptr %ptr) {
  store atomic i32 0, ptr %ptr syncscope("agent") seq_cst, align 4
  ret void
}

; CHECK-LABEL: @store_syncscope_drop(
; INTERESTING: store
; RESULT: store atomic i32 0, ptr %ptr seq_cst, align 4
define void @store_syncscope_drop(ptr %ptr) {
  store atomic i32 0, ptr %ptr syncscope("agent") seq_cst, align 4
  ret void
}

; CHECK-LABEL: @atomicrmw_syncscope_keep(
; INTERESTING: syncscope("agent")
; RESULT: %val = atomicrmw add ptr %ptr, i32 3 syncscope("agent") seq_cst, align 4
define i32 @atomicrmw_syncscope_keep(ptr %ptr) {
  %val = atomicrmw add ptr %ptr, i32 3 syncscope("agent") seq_cst, align 4
  ret i32 %val
}

; CHECK-LABEL: @atomicrmw_syncscope_drop(
; INTERESTING: atomicrmw
; RESULT: %val = atomicrmw add ptr %ptr, i32 3 seq_cst, align 4
define i32 @atomicrmw_syncscope_drop(ptr %ptr) {
  %val = atomicrmw add ptr %ptr, i32 3 seq_cst
  ret i32 %val
}

; CHECK-LABEL: @cmpxchg_syncscope_keep(
; INTERESTING: syncscope("agent")
; RESULT: %val = cmpxchg ptr %ptr, i32 %old, i32 %in syncscope("agent") seq_cst seq_cst, align 4
define { i32, i1 } @cmpxchg_syncscope_keep(ptr %ptr, i32 %old, i32 %in) {
  %val = cmpxchg ptr %ptr, i32 %old, i32 %in syncscope("agent") seq_cst seq_cst
  ret { i32, i1 } %val
}

; CHECK-LABEL: @cmpxchg_syncscope_drop(
; INTERESTING: = cmpxchg ptr
; RESULT: %val = cmpxchg ptr %ptr, i32 %old, i32 %in seq_cst seq_cst, align 4
define { i32, i1 } @cmpxchg_syncscope_drop(ptr %ptr, i32 %old, i32 %in) {
  %val = cmpxchg ptr %ptr, i32 %old, i32 %in syncscope("agent") seq_cst seq_cst
  ret { i32, i1 } %val
}

; CHECK-LABEL: @fence_syncscope_keep(
; INTERESTING: syncscope("agent")
; RESULT: fence syncscope("agent") acquire
define void @fence_syncscope_keep() {
  fence syncscope("agent") acquire
  ret void
}

; CHECK-LABEL: @fence_syncscope_drop(
; INTERESTING: fence
; RESULT: fence acquire
define void @fence_syncscope_drop() {
  fence syncscope("agent") acquire
  ret void
}
