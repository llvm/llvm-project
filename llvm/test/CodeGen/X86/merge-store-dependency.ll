; RUN: llc < %s -mtriple=i686-unknown-linux-android24 -verify-machineinstrs \
; RUN:   -debug-only=machine-scheduler -o - 2>&1 | FileCheck %s
; REQUIRES: asserts

;; MOVUPSmr is a merged store from stack objects %ir.arg1, %ir.arg2, %ir.arg3,
;; %ir.arg4.
;; Check that the merged store has dependency with %ir.arg4.

; CHECK:       ********** MI Scheduling **********
; CHECK-LABEL: f:%bb.0 bb
; CHECK:       SU([[ARG4:[0-9]+]]):{{.*}}MOV32rm{{.*}}load (s32) from %ir.arg4
; CHECK:       SU([[#WIDEN:]]):{{.*}}MOVUPSmr{{.*}}store (s128) into
; CHECK:         Predecessors:
; CHECK:           SU([[ARG4]]):{{.*}}Memory
; CHECK:       SU([[#WIDEN+1]])
;

define void @f(ptr %arg, ptr byval(ptr) %arg1, ptr byval(ptr) %arg2, ptr byval(ptr) %arg3, ptr byval(ptr) %arg4) #0 {
bb:
  %inst = alloca ptr, align 4
  %inst5 = alloca ptr, align 4
  %inst6 = alloca ptr, align 4
  %inst7 = alloca ptr, align 4
  %inst9 = load ptr, ptr %arg1, align 4
  store ptr null, ptr %arg1, align 4
  store ptr %inst9, ptr %inst, align 4
  %inst10 = load ptr, ptr %arg2, align 4
  store ptr null, ptr %arg2, align 4
  store ptr %inst10, ptr %inst5, align 4
  %inst11 = load ptr, ptr %arg3, align 4
  store ptr null, ptr %arg3, align 4
  store ptr %inst11, ptr %inst6, align 4
  %inst12 = load ptr, ptr %arg4, align 4
  store ptr null, ptr %arg4, align 4
  store ptr %inst12, ptr %inst7, align 4
  call void @g(ptr %arg, ptr byval(ptr) %inst, ptr byval(ptr) %inst5, ptr byval(ptr) %inst6, ptr byval(ptr) %inst7)
  call void @h(ptr %arg4)
  call void @h(ptr %arg3)
  call void @h(ptr %arg2)
  call void @h(ptr %arg1)
  ret void
}

declare void @g(ptr, ptr, ptr, ptr, ptr)

declare void @h(ptr)

attributes #0 = { optsize "frame-pointer"="non-leaf" "target-cpu"="i686" "target-features"="+sse,+sse2" "tune-cpu"="generic" }
