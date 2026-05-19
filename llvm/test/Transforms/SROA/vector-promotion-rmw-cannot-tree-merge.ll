; REQUIRES: asserts
; RUN: opt < %s -passes='sroa<preserve-cfg>' -disable-output -debug-only=sroa 2>&1 | FileCheck %s
; RUN: opt < %s -passes='sroa<modify-cfg>' -disable-output -debug-only=sroa 2>&1 | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-n8:16:32:64"

;
; Negative tests: patterns that should NOT trigger the init + RMW tree-merge
; rewrite. Each function violates one precondition; the check below is
; global, so none of them may emit the debug line "Tree structured merge
; rewrite (RMW):" that the rewrite prints when it fires.
;

; CHECK-NOT: Tree structured merge rewrite (RMW):

; Coverage gap: partial loads/stores don't tile the alloca — element 2
; is neither loaded nor stored, so the partition-tile check fails.
define <4 x float> @coverage_gap(<4 x float> %init, <2 x float> %a, <1 x float> %b) {
entry:
  %alloca = alloca [4 x float], align 16
  store <4 x float> %init, ptr %alloca, align 16

  %p0 = getelementptr inbounds [4 x float], ptr %alloca, i32 0, i32 0
  %v0 = load <2 x float>, ptr %p0, align 8
  %r0 = fadd <2 x float> %v0, %a
  store <2 x float> %r0, ptr %p0, align 8

  ; (nothing touches element 2)

  %p1 = getelementptr inbounds [4 x float], ptr %alloca, i32 0, i32 3
  %v1 = load <1 x float>, ptr %p1, align 4
  %r1 = fadd <1 x float> %v1, %b
  store <1 x float> %r1, ptr %p1, align 4

  %result = load <4 x float>, ptr %alloca, align 16
  ret <4 x float> %result
}

; Missing init store: the alloca has partial loads/stores but no
; full-width init, so the RMW path (which needs an init seed) refuses.
define <4 x float> @missing_init_store(<2 x float> %a, <2 x float> %b) {
entry:
  %alloca = alloca [4 x float], align 16

  %p0 = getelementptr inbounds [4 x float], ptr %alloca, i32 0, i32 0
  %v0 = load <2 x float>, ptr %p0, align 8
  %r0 = fadd <2 x float> %v0, %a
  store <2 x float> %r0, ptr %p0, align 8

  %p1 = getelementptr inbounds [4 x float], ptr %alloca, i32 0, i32 2
  %v1 = load <2 x float>, ptr %p1, align 8
  %r1 = fadd <2 x float> %v1, %b
  store <2 x float> %r1, ptr %p1, align 8

  %result = load <4 x float>, ptr %alloca, align 16
  ret <4 x float> %result
}

; Atomic RMW: the partial load/store at slice 0 carry memory-ordering
; (monotonic). The rewrite would drop the atomic stores and elide the
; atomic loads, losing that ordering, so the RMW path must reject it.
define <4 x float> @atomic_rmw(<4 x float> %init, <2 x float> %a, <2 x float> %b) {
entry:
  %alloca = alloca [4 x float], align 16
  store <4 x float> %init, ptr %alloca, align 16

  %p0 = getelementptr inbounds [4 x float], ptr %alloca, i32 0, i32 0
  %v0 = load atomic <2 x float>, ptr %p0 monotonic, align 8
  %r0 = fadd <2 x float> %v0, %a
  store atomic <2 x float> %r0, ptr %p0 monotonic, align 8

  %p1 = getelementptr inbounds [4 x float], ptr %alloca, i32 0, i32 2
  %v1 = load <2 x float>, ptr %p1, align 8
  %r1 = fadd <2 x float> %v1, %b
  store <2 x float> %r1, ptr %p1, align 8

  %result = load <4 x float>, ptr %alloca, align 16
  ret <4 x float> %result
}

; Ordering constraint: the init store must precede every partial access.
; Here a partial load/store pair at slice 0 appears BEFORE the full-width
; init store, so the rewrite must refuse — otherwise the per-slice SSA
; seed would be wrong (the pre-init partial load actually reads
; undef/garbage, not the init value).
define <4 x float> @init_store_not_first(<4 x float> %init, <2 x float> %a, <2 x float> %b) {
entry:
  %alloca = alloca [4 x float], align 16

  ; Partial load+store BEFORE the init store.
  %p_pre = getelementptr inbounds [4 x float], ptr %alloca, i32 0, i32 0
  %v_pre = load <2 x float>, ptr %p_pre, align 8
  %r_pre = fadd <2 x float> %v_pre, %a
  store <2 x float> %r_pre, ptr %p_pre, align 8

  ; Full-width init store comes after.
  store <4 x float> %init, ptr %alloca, align 16

  %p1 = getelementptr inbounds [4 x float], ptr %alloca, i32 0, i32 2
  %v1 = load <2 x float>, ptr %p1, align 8
  %r1 = fadd <2 x float> %v1, %b
  store <2 x float> %r1, ptr %p1, align 8

  %result = load <4 x float>, ptr %alloca, align 16
  ret <4 x float> %result
}

; Ordering constraint: when the final full-width load shares the partial
; accesses' basic block, it must come after every partial access. Here it
; sits between two partial accesses, so the rewrite must refuse —
; otherwise the load would observe a stale slice value.
define <4 x float> @full_load_in_middle(<4 x float> %init, <2 x float> %a, <2 x float> %b) {
entry:
  %alloca = alloca [4 x float], align 16
  store <4 x float> %init, ptr %alloca, align 16

  %p0 = getelementptr inbounds [4 x float], ptr %alloca, i32 0, i32 0
  %v0 = load <2 x float>, ptr %p0, align 8
  %r0 = fadd <2 x float> %v0, %a
  store <2 x float> %r0, ptr %p0, align 8

  ; Full-width load is between the two partial pairs.
  %intermediate = load <4 x float>, ptr %alloca, align 16

  %p1 = getelementptr inbounds [4 x float], ptr %alloca, i32 0, i32 2
  %v1 = load <2 x float>, ptr %p1, align 8
  %r1 = fadd <2 x float> %v1, %b
  store <2 x float> %r1, ptr %p1, align 8

  ret <4 x float> %intermediate
}
