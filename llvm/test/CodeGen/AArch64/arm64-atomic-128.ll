; RUN: llc < %s -mtriple=arm64-linux-gnu -verify-machineinstrs -mcpu=cyclone | FileCheck %s
; RUN: llc < %s -mtriple=arm64-linux-gnu -verify-machineinstrs -mcpu=cyclone -mattr=+outline-atomics | FileCheck %s -check-prefix=OUTLINE-ATOMICS

@var = global i128 0

define i128 @val_compare_and_swap(i128* %p, i128 %oldval, i128 %newval) {
; OUTLINE-ATOMICS: bl __aarch64_cas16_acq
; CHECK-LABEL: val_compare_and_swap:
; CHECK: [[LABEL:.?LBB[0-9]+_[0-9]+]]:
; CHECK: ldaxp   [[RESULTLO:x[0-9]+]], [[RESULTHI:x[0-9]+]], [x[[ADDR:[0-9]+]]]
; CHECK-DAG: eor     [[MISMATCH_LO:x[0-9]+]], [[RESULTLO]], x2
; CHECK-DAG: eor     [[MISMATCH_HI:x[0-9]+]], [[RESULTHI]], x3
; CHECK: orr [[MISMATCH:x[0-9]+]], [[MISMATCH_LO]], [[MISMATCH_HI]]
; CHECK: cbnz    [[MISMATCH]], [[DONE:.LBB[0-9]+_[0-9]+]]
; CHECK: stxp   [[SCRATCH_RES:w[0-9]+]], x4, x5, [x[[ADDR]]]
; CHECK: cbnz   [[SCRATCH_RES]], [[LABEL]]
; CHECK: [[DONE]]:
  %pair = cmpxchg i128* %p, i128 %oldval, i128 %newval acquire acquire
  %val = extractvalue { i128, i1 } %pair, 0
  ret i128 %val
}

define void @fetch_and_nand(i128* %p, i128 %bits) {
; CHECK-LABEL: fetch_and_nand:
; CHECK: [[LABEL:.?LBB[0-9]+_[0-9]+]]:
; CHECK: ldxp  [[DEST_REGLO:x[0-9]+]], [[DEST_REGHI:x[0-9]+]], [x0]
; CHECK-DAG: and    [[TMP_REGLO:x[0-9]+]], [[DEST_REGLO]], x2
; CHECK-DAG: and    [[TMP_REGHI:x[0-9]+]], [[DEST_REGHI]], x3
; CHECK-DAG: mvn    [[SCRATCH_REGLO:x[0-9]+]], [[TMP_REGLO]]
; CHECK-DAG: mvn    [[SCRATCH_REGHI:x[0-9]+]], [[TMP_REGHI]]
; CHECK: stlxp  [[SCRATCH_RES:w[0-9]+]], [[SCRATCH_REGLO]], [[SCRATCH_REGHI]], [x0]
; CHECK: cbnz   [[SCRATCH_RES]], [[LABEL]]

; CHECK-DAG: stp    [[DEST_REGLO]], [[DEST_REGHI]]
  %val = atomicrmw nand i128* %p, i128 %bits release
  store i128 %val, i128* @var, align 16
  ret void
}

define void @fetch_and_or(i128* %p, i128 %bits) {
; CHECK-LABEL: fetch_and_or:
; CHECK: [[LABEL:.?LBB[0-9]+_[0-9]+]]:
; CHECK: ldaxp  [[DEST_REGLO:x[0-9]+]], [[DEST_REGHI:x[0-9]+]], [x0]
; CHECK-DAG: orr    [[SCRATCH_REGLO:x[0-9]+]], [[DEST_REGLO]], x2
; CHECK-DAG: orr    [[SCRATCH_REGHI:x[0-9]+]], [[DEST_REGHI]], x3
; CHECK: stlxp  [[SCRATCH_RES:w[0-9]+]], [[SCRATCH_REGLO]], [[SCRATCH_REGHI]], [x0]
; CHECK: cbnz   [[SCRATCH_RES]], [[LABEL]]

; CHECK-DAG: stp    [[DEST_REGLO]], [[DEST_REGHI]]
  %val = atomicrmw or i128* %p, i128 %bits seq_cst
  store i128 %val, i128* @var, align 16
  ret void
}

define void @fetch_and_add(i128* %p, i128 %bits) {
; CHECK-LABEL: fetch_and_add:
; CHECK: [[LABEL:.?LBB[0-9]+_[0-9]+]]:
; CHECK: ldaxp  [[DEST_REGLO:x[0-9]+]], [[DEST_REGHI:x[0-9]+]], [x0]
; CHECK: adds   [[SCRATCH_REGLO:x[0-9]+]], [[DEST_REGLO]], x2
; CHECK: adcs   [[SCRATCH_REGHI:x[0-9]+]], [[DEST_REGHI]], x3
; CHECK: stlxp  [[SCRATCH_RES:w[0-9]+]], [[SCRATCH_REGLO]], [[SCRATCH_REGHI]], [x0]
; CHECK: cbnz   [[SCRATCH_RES]], [[LABEL]]

; CHECK-DAG: stp    [[DEST_REGLO]], [[DEST_REGHI]]
  %val = atomicrmw add i128* %p, i128 %bits seq_cst
  store i128 %val, i128* @var, align 16
  ret void
}

define void @fetch_and_sub(i128* %p, i128 %bits) {
; CHECK-LABEL: fetch_and_sub:
; CHECK: [[LABEL:.?LBB[0-9]+_[0-9]+]]:
; CHECK: ldaxp  [[DEST_REGLO:x[0-9]+]], [[DEST_REGHI:x[0-9]+]], [x0]
; CHECK: subs   [[SCRATCH_REGLO:x[0-9]+]], [[DEST_REGLO]], x2
; CHECK: sbcs    [[SCRATCH_REGHI:x[0-9]+]], [[DEST_REGHI]], x3
; CHECK: stlxp  [[SCRATCH_RES:w[0-9]+]], [[SCRATCH_REGLO]], [[SCRATCH_REGHI]], [x0]
; CHECK: cbnz   [[SCRATCH_RES]], [[LABEL]]

; CHECK-DAG: stp    [[DEST_REGLO]], [[DEST_REGHI]]
  %val = atomicrmw sub i128* %p, i128 %bits seq_cst
  store i128 %val, i128* @var, align 16
  ret void
}

define void @fetch_and_min(i128* %p, i128 %bits) {
; CHECK-LABEL: fetch_and_min:
; CHECK: [[LABEL:.?LBB[0-9]+_[0-9]+]]:
; CHECK: ldaxp   [[DEST_REGLO:x[0-9]+]], [[DEST_REGHI:x[0-9]+]], [x0]
; CHECK: cmp     [[DEST_REGLO]], x2
; CHECK: cset    [[LOCMP:w[0-9]+]], ls
; CHECK: cmp     [[DEST_REGHI:x[0-9]+]], x3
; CHECK: cset    [[HICMP:w[0-9]+]], le
; CHECK: csel    [[CMP:w[0-9]+]], [[LOCMP]], [[HICMP]], eq
; CHECK: cmp     [[CMP]], #0
; CHECK-DAG: csel    [[SCRATCH_REGHI:x[0-9]+]], [[DEST_REGHI]], x3, ne
; CHECK-DAG: csel    [[SCRATCH_REGLO:x[0-9]+]], [[DEST_REGLO]], x2, ne
; CHECK: stlxp  [[SCRATCH_RES:w[0-9]+]], [[SCRATCH_REGLO]], [[SCRATCH_REGHI]], [x0]
; CHECK: cbnz   [[SCRATCH_RES]], [[LABEL]]

; CHECK-DAG: stp    [[DEST_REGLO]], [[DEST_REGHI]]
  %val = atomicrmw min i128* %p, i128 %bits seq_cst
  store i128 %val, i128* @var, align 16
  ret void
}

define void @fetch_and_max(i128* %p, i128 %bits) {
; CHECK-LABEL: fetch_and_max:
; CHECK: [[LABEL:.?LBB[0-9]+_[0-9]+]]:
; CHECK: ldaxp  [[DEST_REGLO:x[0-9]+]], [[DEST_REGHI:x[0-9]+]], [x0]
; CHECK: cmp     [[DEST_REGLO]], x2
; CHECK: cset    [[LOCMP:w[0-9]+]], hi
; CHECK: cmp     [[DEST_REGHI:x[0-9]+]], x3
; CHECK: cset    [[HICMP:w[0-9]+]], gt
; CHECK: csel    [[CMP:w[0-9]+]], [[LOCMP]], [[HICMP]], eq
; CHECK: cmp     [[CMP]], #0
; CHECK-DAG: csel    [[SCRATCH_REGHI:x[0-9]+]], [[DEST_REGHI]], x3, ne
; CHECK-DAG: csel    [[SCRATCH_REGLO:x[0-9]+]], [[DEST_REGLO]], x2, ne
; CHECK: stlxp  [[SCRATCH_RES:w[0-9]+]], [[SCRATCH_REGLO]], [[SCRATCH_REGHI]], [x0]
; CHECK: cbnz   [[SCRATCH_RES]], [[LABEL]]

; CHECK-DAG: stp    [[DEST_REGLO]], [[DEST_REGHI]]
  %val = atomicrmw max i128* %p, i128 %bits seq_cst
  store i128 %val, i128* @var, align 16
  ret void
}

define void @fetch_and_umin(i128* %p, i128 %bits) {
; CHECK-LABEL: fetch_and_umin:
; CHECK: [[LABEL:.?LBB[0-9]+_[0-9]+]]:
; CHECK: ldaxp  [[DEST_REGLO:x[0-9]+]], [[DEST_REGHI:x[0-9]+]], [x0]
; CHECK: cmp     [[DEST_REGLO]], x2
; CHECK: cset    [[LOCMP:w[0-9]+]], ls
; CHECK: cmp     [[DEST_REGHI:x[0-9]+]], x3
; CHECK: cset    [[HICMP:w[0-9]+]], ls
; CHECK: csel    [[CMP:w[0-9]+]], [[LOCMP]], [[HICMP]], eq
; CHECK: cmp     [[CMP]], #0
; CHECK-DAG: csel    [[SCRATCH_REGHI:x[0-9]+]], [[DEST_REGHI]], x3, ne
; CHECK-DAG: csel    [[SCRATCH_REGLO:x[0-9]+]], [[DEST_REGLO]], x2, ne
; CHECK: stlxp  [[SCRATCH_RES:w[0-9]+]], [[SCRATCH_REGLO]], [[SCRATCH_REGHI]], [x0]
; CHECK: cbnz   [[SCRATCH_RES]], [[LABEL]]

; CHECK-DAG: stp    [[DEST_REGLO]], [[DEST_REGHI]]
  %val = atomicrmw umin i128* %p, i128 %bits seq_cst
  store i128 %val, i128* @var, align 16
  ret void
}

define void @fetch_and_umax(i128* %p, i128 %bits) {
; CHECK-LABEL: fetch_and_umax:
; CHECK: [[LABEL:.?LBB[0-9]+_[0-9]+]]:
; CHECK: ldaxp  [[DEST_REGLO:x[0-9]+]], [[DEST_REGHI:x[0-9]+]], [x0]
; CHECK: cmp     [[DEST_REGLO]], x2
; CHECK: cset    [[LOCMP:w[0-9]+]], hi
; CHECK: cmp     [[DEST_REGHI:x[0-9]+]], x3
; CHECK: cset    [[HICMP:w[0-9]+]], hi
; CHECK: csel    [[CMP:w[0-9]+]], [[LOCMP]], [[HICMP]], eq
; CHECK: cmp     [[CMP]], #0
; CHECK-DAG: csel    [[SCRATCH_REGHI:x[0-9]+]], [[DEST_REGHI]], x3, ne
; CHECK-DAG: csel    [[SCRATCH_REGLO:x[0-9]+]], [[DEST_REGLO]], x2, ne
; CHECK: stlxp  [[SCRATCH_RES:w[0-9]+]], [[SCRATCH_REGLO]], [[SCRATCH_REGHI]], [x0]
; CHECK: cbnz   [[SCRATCH_RES]], [[LABEL]]

; CHECK-DAG: stp    [[DEST_REGLO]], [[DEST_REGHI]]
  %val = atomicrmw umax i128* %p, i128 %bits seq_cst
  store i128 %val, i128* @var, align 16
  ret void
}

define i128 @atomic_load_seq_cst(i128* %p) {
; CHECK-LABEL: atomic_load_seq_cst:
; CHECK-NOT: dmb
; CHECK-LABEL: ldaxp
; CHECK-NOT: dmb
   %r = load atomic i128, i128* %p seq_cst, align 16
   ret i128 %r
}

define i128 @atomic_load_relaxed(i64, i64, i128* %p) {
; CHECK-LABEL: atomic_load_relaxed:
; CHECK-NOT: dmb
; CHECK: [[LABEL:.?LBB[0-9]+_[0-9]+]]:
; CHECK: ldxp [[LO:x[0-9]+]], [[HI:x[0-9]+]], [x2]
; CHECK-NEXT: stxp [[SUCCESS:w[0-9]+]], [[LO]], [[HI]], [x2]
; CHECK: cbnz [[SUCCESS]], [[LABEL]]
; CHECK-NOT: dmb
   %r = load atomic i128, i128* %p monotonic, align 16
   ret i128 %r
}


define void @atomic_store_seq_cst(i128 %in, i128* %p) {
; CHECK-LABEL: atomic_store_seq_cst:
; CHECK-NOT: dmb
; CHECK: [[LABEL:.?LBB[0-9]+_[0-9]+]]:
; CHECK: ldaxp xzr, [[IGNORED:x[0-9]+]], [x2]
; CHECK: stlxp [[SUCCESS:w[0-9]+]], x0, x1, [x2]
; CHECK: cbnz [[SUCCESS]], [[LABEL]]
; CHECK-NOT: dmb
   store atomic i128 %in, i128* %p seq_cst, align 16
   ret void
}

define void @atomic_store_release(i128 %in, i128* %p) {
; CHECK-LABEL: atomic_store_release:
; CHECK-NOT: dmb
; CHECK: [[LABEL:.?LBB[0-9]+_[0-9]+]]:
; CHECK: ldxp xzr, [[IGNORED:x[0-9]+]], [x2]
; CHECK: stlxp [[SUCCESS:w[0-9]+]], x0, x1, [x2]
; CHECK: cbnz [[SUCCESS]], [[LABEL]]
; CHECK-NOT: dmb
   store atomic i128 %in, i128* %p release, align 16
   ret void
}

define void @atomic_store_relaxed(i128 %in, i128* %p) {
; CHECK-LABEL: atomic_store_relaxed:
; CHECK-NOT: dmb
; CHECK: [[LABEL:.?LBB[0-9]+_[0-9]+]]:
; CHECK: ldxp xzr, [[IGNORED:x[0-9]+]], [x2]
; CHECK: stxp [[SUCCESS:w[0-9]+]], x0, x1, [x2]
; CHECK: cbnz [[SUCCESS]], [[LABEL]]
; CHECK-NOT: dmb
   store atomic i128 %in, i128* %p unordered, align 16
   ret void
}
