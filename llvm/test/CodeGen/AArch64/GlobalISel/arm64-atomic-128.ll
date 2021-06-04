; RUN: llc < %s -mtriple=arm64-linux-gnu -verify-machineinstrs -global-isel -global-isel-abort=1 | FileCheck %s --check-prefix=CHECK-LLSC-O1
; RUN: llc < %s -mtriple=arm64-linux-gnu -verify-machineinstrs -mcpu=apple-a13 -global-isel -global-isel-abort=1 | FileCheck %s --check-prefix=CHECK-CAS-O1
; RUN: llc < %s -mtriple=arm64-linux-gnu -verify-machineinstrs -O0 -global-isel -global-isel-abort=1 | FileCheck %s --check-prefix=CHECK-LLSC-O0
; RUN: llc < %s -mtriple=arm64-linux-gnu -verify-machineinstrs -O0 -mcpu=apple-a13 -global-isel -global-isel-abort=1 | FileCheck %s --check-prefix=CHECK-CAS-O0
@var = global i128 0

define void @val_compare_and_swap(i128* %p, i128 %oldval, i128 %newval) {
; CHECK-LLSC-O1-LABEL: val_compare_and_swap:
; CHECK-LLSC-O1:    ldaxp {{x[0-9]+}}, {{x[0-9]+}}, [x0]
; [... LOTS of stuff that is generic IR unrelated to atomic operations ...]
; CHECK-LLSC-O1:    stxp {{w[0-9]+}}, {{x[0-9]+}}, {{x[0-9]+}}, [x0]
;
; CHECK-CAS-O1-LABEL: val_compare_and_swap:
; CHECK-CAS-O1:    caspa x2, x3, x4, x5, [x0]
; CHECK-CAS-O1:    mov v[[OLD:[0-9]+]].d[0], x2
; CHECK-CAS-O1:    mov v[[OLD]].d[1], x3
; CHECK-CAS-O1:    str q[[OLD]], [x0]

; CHECK-LLSC-O0-LABEL: val_compare_and_swap:
; CHECK-LLSC-O0:  .LBB0_1:
; CHECK-LLSC-O0:    ldaxp [[OLD_LO:x[0-9]+]], [[OLD_HI:x[0-9]+]], [x0]
; CHECK-LLSC-O0:    cmp [[OLD_LO]], x2
; CHECK-LLSC-O0:    cset [[EQUAL_TMP:w[0-9]+]], ne
; CHECK-LLSC-O0:    cmp [[OLD_HI]], x3
; CHECK-LLSC-O0:    cinc [[EQUAL:w[0-9]+]], [[EQUAL_TMP]], ne
; CHECK-LLSC-O0:    cbnz [[EQUAL]], .LBB0_3
; CHECK-LLSC-O0:    stlxp [[STATUS:w[0-9]+]], x4, x5, [x0]
; CHECK-LLSC-O0:    cbnz [[STATUS]], .LBB0_1
; CHECK-LLSC-O0:  .LBB0_3:
; CHECK-LLSC-O0:    mov v[[OLD:[0-9]+]].d[0], [[OLD_LO]]
; CHECK-LLSC-O0:    mov v[[OLD]].d[1], [[OLD_HI]]
; CHECK-LLSC-O0:    str q[[OLD]], [x0]


; CHECK-CAS-O0-LABEL: val_compare_and_swap:
; CHECK-CAS-O0:    str x3, [sp, #[[SLOT:[0-9]+]]]
; CHECK-CAS-O0:    mov [[NEW_HI_TMP:x[0-9]+]], x5
; CHECK-CAS-O0:    ldr [[DESIRED_HI_TMP:x[0-9]+]], [sp, #[[SLOT]]]
; CHECK-CAS-O0:    mov [[DESIRED_HI:x[0-9]+]], [[DESIRED_HI_TMP]]
; CHECK-CAS-O0:    mov [[NEW_HI:x[0-9]+]], [[NEW_HI_TMP]]
; CHECK-CAS-O0:    caspa x2, [[DESIRED_HI]], x4, [[NEW_HI]], [x0]
; CHECK-CAS-O0:    mov [[OLD_LO:x[0-9]+]], x2
; CHECK-CAS-O0:    mov [[OLD_HI:x[0-9]+]], x3
; CHECK-CAS-O0:    mov v[[OLD:[0-9]+]].d[0], [[OLD_LO]]
; CHECK-CAS-O0:    mov v[[OLD]].d[1], [[OLD_HI]]
; CHECK-CAS-O0:    str q[[OLD]], [x0]

%pair = cmpxchg i128* %p, i128 %oldval, i128 %newval acquire acquire
  %val = extractvalue { i128, i1 } %pair, 0
  store i128 %val, i128* %p
  ret void
}

define void @val_compare_and_swap_monotonic_seqcst(i128* %p, i128 %oldval, i128 %newval) {
; CHECK-LLSC-O1-LABEL: val_compare_and_swap_monotonic_seqcst:
; CHECK-LLSC-O1:    ldaxp {{x[0-9]+}}, {{x[0-9]+}}, [x0]
; [... LOTS of stuff that is generic IR unrelated to atomic operations ...]
; CHECK-LLSC-O1:    stlxp {{w[0-9]+}}, {{x[0-9]+}}, {{x[0-9]+}}, [x0]
;
; CHECK-CAS-O1-LABEL: val_compare_and_swap_monotonic_seqcst:
; CHECK-CAS-O1:    caspal x2, x3, x4, x5, [x0]

; CHECK-LLSC-O0-LABEL: val_compare_and_swap_monotonic_seqcst:
; CHECK-LLSC-O0:  .LBB1_1:
; CHECK-LLSC-O0:    ldaxp
; CHECK-LLSC-O0:    stlxp

; CHECK-CAS-O0-LABEL: val_compare_and_swap_monotonic_seqcst:
; CHECK-CAS-O0:    caspal

  %pair = cmpxchg i128* %p, i128 %oldval, i128 %newval monotonic seq_cst
  %val = extractvalue { i128, i1 } %pair, 0
  store i128 %val, i128* %p
  ret void
}

define void @val_compare_and_swap_release_acquire(i128* %p, i128 %oldval, i128 %newval) {
; CHECK-LLSC-O1-LABEL: val_compare_and_swap_release_acquire:
; CHECK-LLSC-O1:    ldaxp {{x[0-9]+}}, {{x[0-9]+}}, [x0]
; [... LOTS of stuff that is generic IR unrelated to atomic operations ...]
; CHECK-LLSC-O1:    stlxp {{w[0-9]+}}, {{x[0-9]+}}, {{x[0-9]+}}, [x0]
;
; CHECK-CAS-O1-LABEL: val_compare_and_swap_release_acquire:
; CHECK-CAS-O1:    caspal x2, x3, x4, x5, [x0]

; CHECK-LLSC-O0-LABEL: val_compare_and_swap_release_acquire:
; CHECK-LLSC-O0:  .LBB2_1:
; CHECK-LLSC-O0:    ldaxp
; CHECK-LLSC-O0:    stlxp

; CHECK-CAS-O0-LABEL: val_compare_and_swap_release_acquire:
; CHECK-CAS-O0:    caspal

  %pair = cmpxchg i128* %p, i128 %oldval, i128 %newval release acquire
  %val = extractvalue { i128, i1 } %pair, 0
  store i128 %val, i128* %p
  ret void
}
