; RUN: llc < %s -mtriple=arm64-apple-ios -global-isel -global-isel-abort=1 -verify-machineinstrs | FileCheck %s --check-prefixes=CHECK-NOLSE,CHECK-NOLSE-O1
; RUN: llc < %s -mtriple=arm64-apple-ios -global-isel -global-isel-abort=1 -O0 -verify-machineinstrs | FileCheck %s --check-prefixes=CHECK-NOLSE,CHECK-NOLSE-O0
; RUN: llc < %s -mtriple=arm64-apple-ios -global-isel -global-isel-abort=1 -mcpu=apple-a13 -verify-machineinstrs | FileCheck %s --check-prefixes=CHECK-LSE,CHECK-LSE-O1
; RUN: llc < %s -mtriple=arm64-apple-ios -global-isel -global-isel-abort=1 -mcpu=apple-a13 -O0 -verify-machineinstrs | FileCheck %s --check-prefixes=CHECK-LSE,CHECK-LSE-O0

define i32 @val_compare_and_swap(i32* %p, i32 %cmp, i32 %new) #0 {
; CHECK-NOLSE-LABEL: val_compare_and_swap:
; CHECK-NOLSE-O1: LBB0_1:
; CHECK-NOLSE-O1:     ldaxr   [[VAL:w[0-9]+]], [x0]
; CHECK-NOLSE-O1:     cmp     [[VAL]], w1
; CHECK-NOLSE-O1:     b.ne    LBB0_4
; CHECK-NOLSE-O1:     stxr    [[STATUS:w[0-9]+]], w2, [x0]
; CHECK-NOLSE-O1:     cbnz    [[STATUS]], LBB0_1
; CHECK-NOLSE-O1:     mov     w0, [[VAL]]
; CHECK-NOLSE-O1:     ret
; CHECK-NOLSE-O1: LBB0_4:
; CHECK-NOLSE-O1:     clrex
; CHECK-NOLSE-O1:     mov     w0, [[VAL]]
; CHECK-NOLSE-O1:     ret

; CHECK-NOLSE-O0:     mov x[[ADDR:[0-9]+]], x0
; CHECK-NOLSE-O0: LBB0_1:
; CHECK-NOLSE-O0:     ldaxr   w0, [x[[ADDR]]]
; CHECK-NOLSE-O0:     cmp     w0, w1
; CHECK-NOLSE-O0:     b.ne    LBB0_3
; CHECK-NOLSE-O0:     stlxr    [[STATUS:w[0-9]+]], w2, [x[[ADDR]]]
; CHECK-NOLSE-O0:     cbnz    [[STATUS]], LBB0_1
; CHECK-NOLSE-O0: LBB0_3:
; CHECK-NOLSE-O0:     ret

; CHECK-LSE-LABEL: val_compare_and_swap:
; CHECK-LSE-O1: casa w1, w2, [x0]
; CHECK-LSE-O1: mov x0, x1

; CHECK-LSE-O0: mov     x[[ADDR:[0-9]+]], x0
; CHECK-LSE-O0: mov     x0, x1
; CHECK-LSE-O0: casa    w0, w2, [x[[ADDR]]]

  %pair = cmpxchg i32* %p, i32 %cmp, i32 %new acquire acquire
  %val = extractvalue { i32, i1 } %pair, 0
  ret i32 %val
}

define i32 @val_compare_and_swap_from_load(i32* %p, i32 %cmp, i32* %pnew) #0 {
; CHECK-NOLSE-LABEL: val_compare_and_swap_from_load:
; CHECK-NOLSE-O1:     ldr [[NEW:w[0-9]+]], [x2]
; CHECK-NOLSE-O1: LBB1_1:
; CHECK-NOLSE-O1:     ldaxr   [[VAL:w[0-9]+]], [x0]
; CHECK-NOLSE-O1:     cmp     [[VAL]], w1
; CHECK-NOLSE-O1:     b.ne    LBB1_4
; CHECK-NOLSE-O1:     stxr    [[STATUS:w[0-9]+]], [[NEW]], [x0]
; CHECK-NOLSE-O1:     cbnz    [[STATUS]], LBB1_1
; CHECK-NOLSE-O1:     mov     w0, [[VAL]]
; CHECK-NOLSE-O1:     ret
; CHECK-NOLSE-O1: LBB1_4:
; CHECK-NOLSE-O1:     clrex
; CHECK-NOLSE-O1:     mov     w0, [[VAL]]
; CHECK-NOLSE-O1:     ret

; CHECK-NOLSE-O0:     mov x[[ADDR:[0-9]+]], x0
; CHECK-NOLSE-O0:     ldr [[NEW:w[0-9]+]], [x2]
; CHECK-NOLSE-O0: LBB1_1:
; CHECK-NOLSE-O0:     ldaxr   w0, [x[[ADDR]]]
; CHECK-NOLSE-O0:     cmp     w0, w1
; CHECK-NOLSE-O0:     b.ne    LBB1_3
; CHECK-NOLSE-O0:     stlxr    [[STATUS:w[0-9]+]], [[NEW]], [x[[ADDR]]]
; CHECK-NOLSE-O0:     cbnz    [[STATUS]], LBB1_1
; CHECK-NOLSE-O0: LBB1_3:
; CHECK-NOLSE-O0:     ret

; CHECK-LSE-LABEL: val_compare_and_swap_from_load:
; CHECK-LSE-O1: ldr [[NEW:w[0-9]+]], [x2]
; CHECK-LSE-O1: casa w1, [[NEW]], [x0]
; CHECK-LSE-O1: mov x0, x1

; CHECK-LSE-O0: mov     x[[ADDR:[0-9]+]], x0
; CHECK-LSE-O0: mov     x0, x1
; CHECK-LSE-O0: ldr [[NEW:w[0-9]+]], [x2]
; CHECK-LSE-O0: casa    w0, [[NEW]], [x[[ADDR]]]

  %new = load i32, i32* %pnew
  %pair = cmpxchg i32* %p, i32 %cmp, i32 %new acquire acquire
  %val = extractvalue { i32, i1 } %pair, 0
  ret i32 %val
}

define i32 @val_compare_and_swap_rel(i32* %p, i32 %cmp, i32 %new) #0 {
; CHECK-NOLSE-LABEL: val_compare_and_swap_rel:
; CHECK-NOLSE-O1: LBB2_1:
; CHECK-NOLSE-O1:     ldaxr   [[VAL:w[0-9]+]], [x0]
; CHECK-NOLSE-O1:     cmp     [[VAL]], w1
; CHECK-NOLSE-O1:     b.ne    LBB2_4
; CHECK-NOLSE-O1:     stlxr    [[STATUS:w[0-9]+]], w2, [x0]
; CHECK-NOLSE-O1:     cbnz    [[STATUS]], LBB2_1
; CHECK-NOLSE-O1:     mov     w0, [[VAL]]
; CHECK-NOLSE-O1:     ret
; CHECK-NOLSE-O1: LBB2_4:
; CHECK-NOLSE-O1:     clrex
; CHECK-NOLSE-O1:     mov     w0, [[VAL]]
; CHECK-NOLSE-O1:     ret

; CHECK-NOLSE-O0:     mov x[[ADDR:[0-9]+]], x0
; CHECK-NOLSE-O0: LBB2_1:
; CHECK-NOLSE-O0:     ldaxr   w0, [x[[ADDR]]]
; CHECK-NOLSE-O0:     cmp     w0, w1
; CHECK-NOLSE-O0:     b.ne    LBB2_3
; CHECK-NOLSE-O0:     stlxr    [[STATUS:w[0-9]+]], w2, [x[[ADDR]]]
; CHECK-NOLSE-O0:     cbnz    [[STATUS]], LBB2_1
; CHECK-NOLSE-O0: LBB2_3:
; CHECK-NOLSE-O0:     ret

; CHECK-LSE-LABEL: val_compare_and_swap_rel:
; CHECK-LSE-O1: casal w1, w2, [x0]
; CHECK-LSE-O1: mov x0, x1

; CHECK-LSE-O0: mov     x[[ADDR:[0-9]+]], x0
; CHECK-LSE-O0: mov     x0, x1
; CHECK-LSE-O0: casal   w0, w2, [x[[ADDR]]]

  %pair = cmpxchg i32* %p, i32 %cmp, i32 %new acq_rel monotonic
  %val = extractvalue { i32, i1 } %pair, 0
  ret i32 %val
}

define i64 @val_compare_and_swap_64(i64* %p, i64 %cmp, i64 %new) #0 {
; CHECK-NOLSE-LABEL: val_compare_and_swap_64:
; CHECK-NOLSE-O1: LBB3_1:
; CHECK-NOLSE-O1:     ldxr   [[VAL:x[0-9]+]], [x0]
; CHECK-NOLSE-O1:     cmp     [[VAL]], x1
; CHECK-NOLSE-O1:     b.ne    LBB3_4
; CHECK-NOLSE-O1:     stxr    [[STATUS:w[0-9]+]], x2, [x0]
; CHECK-NOLSE-O1:     cbnz    [[STATUS]], LBB3_1
; CHECK-NOLSE-O1:     mov     x0, [[VAL]]
; CHECK-NOLSE-O1:     ret
; CHECK-NOLSE-O1: LBB3_4:
; CHECK-NOLSE-O1:     clrex
; CHECK-NOLSE-O1:     mov     x0, [[VAL]]
; CHECK-NOLSE-O1:     ret

; CHECK-NOLSE-O0:     mov x[[ADDR:[0-9]+]], x0
; CHECK-NOLSE-O0: LBB3_1:
; CHECK-NOLSE-O0:     ldaxr   x0, [x[[ADDR]]]
; CHECK-NOLSE-O0:     cmp     x0, x1
; CHECK-NOLSE-O0:     b.ne    LBB3_3
; CHECK-NOLSE-O0:     stlxr    [[STATUS:w[0-9]+]], x2, [x[[ADDR]]]
; CHECK-NOLSE-O0:     cbnz    [[STATUS]], LBB3_1
; CHECK-NOLSE-O0: LBB3_3:
; CHECK-NOLSE-O0:     ret

; CHECK-LSE-LABEL: val_compare_and_swap_64:
; CHECK-LSE-O1: cas x1, x2, [x0]
; CHECK-LSE-O1: mov x0, x1

; CHECK-LSE-O0: mov     x[[ADDR:[0-9]+]], x0
; CHECK-LSE-O0: mov     x0, x1
; CHECK-LSE-O0: cas     x0, x2, [x[[ADDR]]]

  %pair = cmpxchg i64* %p, i64 %cmp, i64 %new monotonic monotonic
  %val = extractvalue { i64, i1 } %pair, 0
  ret i64 %val
}

define i32 @fetch_and_nand(i32* %p) #0 {
; CHECK-NOLSE-LABEL: fetch_and_nand:
; CHECK-NOLSE-O1: LBB4_1:
; CHECK-NOLSE-O1:     ldxr    [[VAL:w[0-9]+]], [x0]
; CHECK-NOLSE-O1:     and     [[NEWTMP:w[0-9]+]], [[VAL]], #0x7
; CHECK-NOLSE-O1:     mvn     [[NEW:w[0-9]+]], [[NEWTMP]]
; CHECK-NOLSE-O1:     stlxr   [[STATUS:w[0-9]+]], [[NEW]], [x0]
; CHECK-NOLSE-O1:     cbnz    [[STATUS]], LBB4_1
; CHECK-NOLSE-O1:     mov     w0, [[VAL]]
; CHECK-NOLSE-O1:     ret

; CHECK-NOLSE-O0: ldxr
; CHECK-NOLSE-O0: stlxr

; CHECK-LSE-LABEL: fetch_and_nand:
; CHECK-LSE-O1: LBB4_1:
; CHECK-LSE-O1:     ldxr    w[[VAL:[0-9]+]], [x0]
; CHECK-LSE-O1:     and     [[NEWTMP:w[0-9]+]], w[[VAL]], #0x7
; CHECK-LSE-O1:     mvn     [[NEW:w[0-9]+]], [[NEWTMP]]
; CHECK-LSE-O1:     stlxr   [[STATUS:w[0-9]+]], [[NEW]], [x0]
; CHECK-LSE-O1:     cbnz    [[STATUS]], LBB4_1
; CHECK-LSE-O1:     mov     x0, x[[VAL]]

  %val = atomicrmw nand i32* %p, i32 7 release
  ret i32 %val
}

define i64 @fetch_and_nand_64(i64* %p) #0 {
; CHECK-NOLSE-LABEL: fetch_and_nand_64
; CHECK-NOLSE-O1: LBB5_1:
; CHECK-NOLSE-O1:     ldaxr    [[VAL:x[0-9]+]], [x0]
; CHECK-NOLSE-O1:     and     [[NEWTMP:x[0-9]+]], [[VAL]], #0x7
; CHECK-NOLSE-O1:     mvn     [[NEW:x[0-9]+]], [[NEWTMP]]
; CHECK-NOLSE-O1:     stlxr   [[STATUS:w[0-9]+]], [[NEW]], [x0]
; CHECK-NOLSE-O1:     cbnz    [[STATUS]], LBB5_1
; CHECK-NOLSE-O1:     mov     x0, [[VAL]]
; CHECK-NOLSE-O1:     ret

; CHECK-NOLSE-O0: ldaxr
; CHECK-NOLSE-O0: stlxr

; CHECK-LSE-LABEL: fetch_and_nand_64:
; CHECK-LSE-O1: LBB5_1:
; CHECK-LSE-O1:     ldaxr    [[VAL:x[0-9]+]], [x0]
; CHECK-LSE-O1:     and     [[NEWTMP:x[0-9]+]], [[VAL]], #0x7
; CHECK-LSE-O1:     mvn     [[NEW:x[0-9]+]], [[NEWTMP]]
; CHECK-LSE-O1:     stlxr   [[STATUS:w[0-9]+]], [[NEW]], [x0]
; CHECK-LSE-O1:     cbnz    [[STATUS]], LBB5_1
; CHECK-LSE-O1:     mov     x0, [[VAL]]

  %val = atomicrmw nand i64* %p, i64 7 acq_rel
  ret i64 %val
}

define i32 @fetch_and_or(i32* %p) #0 {
; CHECK-NOLSE-LABEL: fetch_and_or:
; CHECK-NOLSE-O1:     mov [[FIVE:w[0-9]+]], #5
; CHECK-NOLSE-O1: LBB6_1:
; CHECK-NOLSE-O1:     ldaxr   [[VAL:w[0-9]+]], [x0]
; CHECK-NOLSE-O1:     orr     [[NEW:w[0-9]+]], [[VAL]], [[FIVE]]
; CHECK-NOLSE-O1:     stlxr   [[STATUS:w[0-9]+]], [[NEW]], [x0]
; CHECK-NOLSE-O1:     cbnz    [[STATUS]], LBB6_1
; CHECK-NOLSE-O1:     mov     w0, [[VAL]]
; CHECK-NOLSE-O1:     ret

; CHECK-NOLSE-O0: ldaxr
; CHECK-NOLSE-O0: stlxr

; CHECK-LSE-LABEL: fetch_and_or:
; CHECK-LSE:       ; %bb.0:
; CHECK-LSE:    mov w8, #5
; CHECK-LSE:    ldsetal w8, w0, [x0]
; CHECK-LSE:    ret
  %val = atomicrmw or i32* %p, i32 5 seq_cst
  ret i32 %val
}

define i64 @fetch_and_or_64(i64* %p) #0 {
; CHECK-NOLSE-LABEL: fetch_and_or_64:
; CHECK-NOLSE-O1: LBB7_1:
; CHECK-NOLSE-O1:     ldxr   [[VAL:x[0-9]+]], [x0]
; CHECK-NOLSE-O1:     orr     [[NEW:x[0-9]+]], [[VAL]], #0x7
; CHECK-NOLSE-O1:     stxr   [[STATUS:w[0-9]+]], [[NEW]], [x0]
; CHECK-NOLSE-O1:     cbnz    [[STATUS]], LBB7_1
; CHECK-NOLSE-O1:     mov     x0, [[VAL]]
; CHECK-NOLSE-O1:     ret

; CHECK-NOLSE-O0: ldxr
; CHECK-NOLSE-O0: stxr

; CHECK-LSE-LABEL: fetch_and_or_64:
; CHECK-LSE:    mov w[[SEVEN:[0-9]+]], #7
; CHECK-LSE:    ldset x[[SEVEN]], x0, [x0]
; CHECK-LSE:    ret
  %val = atomicrmw or i64* %p, i64 7 monotonic
  ret i64 %val
}

define void @acquire_fence() #0 {
; CHECK-NOLSE-LABEL: acquire_fence:
; CHECK-NOLSE:    dmb ish
; CHECK-NOLSE:    ret
;
; CHECK-LSE-LABEL: acquire_fence:
; CHECK-LSE:    dmb ish
; CHECK-LSE:    ret
   fence acquire
   ret void
}

define void @release_fence() #0 {
; CHECK-NOLSE-LABEL: release_fence:
; CHECK-NOLSE:    dmb ish
; CHECK-NOLSE:    ret
;
; CHECK-LSE-LABEL: release_fence:
; CHECK-LSE:       ; %bb.0:
; CHECK-LSE:    dmb ish
; CHECK-LSE:    ret
   fence release
   ret void
}

define void @seq_cst_fence() #0 {
; CHECK-LABEL: seq_cst_fence:
; CHECK-NOLSE:    dmb ish
; CHECK-NOLSE:    ret
;
; CHECK-LSE-LABEL: seq_cst_fence:
; CHECK-LSE:       ; %bb.0:
; CHECK-LSE:    dmb ish
; CHECK-LSE:    ret
   fence seq_cst
   ret void
}

define i32 @atomic_load(i32* %p) #0 {
; CHECK-LABEL: atomic_load:
; CHECK-NOLSE:    ldar w0, [x0]
; CHECK-NOLSE:    ret
;
; CHECK-LSE-LABEL: atomic_load:
; CHECK-LSE:    ldar w0, [x0]
; CHECK-LSE:    ret
   %r = load atomic i32, i32* %p seq_cst, align 4
   ret i32 %r
}

define i8 @atomic_load_relaxed_8(i8* %p, i32 %off32) #0 {
; CHECK-NOLSE-LABEL: atomic_load_relaxed_8:
; CHECK-NOLSE-O1: ldrb    {{w[0-9]+}}, [x0, #4095]
; CHECK-NOLSE-O1: ldrb    {{w[0-9]+}}, [x0, w1, sxtw]
; CHECK-NOLSE-O1: ldurb   {{w[0-9]+}}, [x0, #-256]
; CHECK-NOLSE-O1: add     x[[ADDR:[0-9]+]], x0, #291, lsl #12
; CHECK-NOLSE-O1: ldrb    {{w[0-9]+}}, [x[[ADDR]]]

; CHECK-LSE: ldrb
; CHECK-LSE: ldrb
; CHECK-LSE: ld{{u?}}rb
; CHECK-LSE: ldrb

  %ptr_unsigned = getelementptr i8, i8* %p, i32 4095
  %val_unsigned = load atomic i8, i8* %ptr_unsigned monotonic, align 1

  %ptr_regoff = getelementptr i8, i8* %p, i32 %off32
  %val_regoff = load atomic i8, i8* %ptr_regoff unordered, align 1
  %tot1 = add i8 %val_unsigned, %val_regoff

  %ptr_unscaled = getelementptr i8, i8* %p, i32 -256
  %val_unscaled = load atomic i8, i8* %ptr_unscaled monotonic, align 1
  %tot2 = add i8 %tot1, %val_unscaled

  %ptr_random = getelementptr i8, i8* %p, i32 1191936 ; 0x123000 (i.e. ADD imm)
  %val_random = load atomic i8, i8* %ptr_random unordered, align 1
  %tot3 = add i8 %tot2, %val_random

  ret i8 %tot3
}

define i16 @atomic_load_relaxed_16(i16* %p, i32 %off32) #0 {
; CHECK-NOLSE-LABEL: atomic_load_relaxed_16:
; CHECK-NOLSE-O1: ldrh    {{w[0-9]+}}, [x0, #8190]
; CHECK-NOLSE-O1: ldrh    {{w[0-9]+}}, [x0, w1, sxtw #1]
; CHECK-NOLSE-O1: ldurh   {{w[0-9]+}}, [x0, #-256]
; CHECK-NOLSE-O1: add     x[[ADDR:[0-9]+]], x0, #291, lsl #12          ; =1191936
; CHECK-NOLSE-O1: ldrh    {{w[0-9]+}}, [x[[ADDR]]]

; CHECK-LSE: ldrh
; CHECK-LSE: ldrh
; CHECK-LSE: ld{{u?}}rh
; CHECK-LSE: ldrh

  %ptr_unsigned = getelementptr i16, i16* %p, i32 4095
  %val_unsigned = load atomic i16, i16* %ptr_unsigned monotonic, align 2

  %ptr_regoff = getelementptr i16, i16* %p, i32 %off32
  %val_regoff = load atomic i16, i16* %ptr_regoff unordered, align 2
  %tot1 = add i16 %val_unsigned, %val_regoff

  %ptr_unscaled = getelementptr i16, i16* %p, i32 -128
  %val_unscaled = load atomic i16, i16* %ptr_unscaled monotonic, align 2
  %tot2 = add i16 %tot1, %val_unscaled

  %ptr_random = getelementptr i16, i16* %p, i32 595968 ; 0x123000/2 (i.e. ADD imm)
  %val_random = load atomic i16, i16* %ptr_random unordered, align 2
  %tot3 = add i16 %tot2, %val_random

  ret i16 %tot3
}

define i32 @atomic_load_relaxed_32(i32* %p, i32 %off32) #0 {
; CHECK-NOLSE-LABEL: atomic_load_relaxed_32:
; CHECK-NOLSE-O1: ldr    {{w[0-9]+}}, [x0, #16380]
; CHECK-NOLSE-O1: ldr    {{w[0-9]+}}, [x0, w1, sxtw #2]
; CHECK-NOLSE-O1: ldur   {{w[0-9]+}}, [x0, #-256]
; CHECK-NOLSE-O1: add     x[[ADDR:[0-9]+]], x0, #291, lsl #12          ; =1191936
; CHECK-NOLSE-O1: ldr    {{w[0-9]+}}, [x[[ADDR]]]

; CHECK-LSE-LABEL: atomic_load_relaxed_32:
; CHECK-LSE:    ldr {{w[0-9]+}}, [x0, #16380]
; CHECK-LSE:    ldr {{w[0-9]+}}, [x0, w1, sxtw #2]
; CHECK-LSE:    ldur {{w[0-9]+}}, [x0, #-256]
; CHECK-LSE:    add x[[ADDR:[0-9]+]], x0, #291, lsl #12 ; =1191936
; CHECK-LSE:    ldr {{w[0-9]+}}, [x[[ADDR]]]

  %ptr_unsigned = getelementptr i32, i32* %p, i32 4095
  %val_unsigned = load atomic i32, i32* %ptr_unsigned monotonic, align 4

  %ptr_regoff = getelementptr i32, i32* %p, i32 %off32
  %val_regoff = load atomic i32, i32* %ptr_regoff unordered, align 4
  %tot1 = add i32 %val_unsigned, %val_regoff

  %ptr_unscaled = getelementptr i32, i32* %p, i32 -64
  %val_unscaled = load atomic i32, i32* %ptr_unscaled monotonic, align 4
  %tot2 = add i32 %tot1, %val_unscaled

  %ptr_random = getelementptr i32, i32* %p, i32 297984 ; 0x123000/4 (i.e. ADD imm)
  %val_random = load atomic i32, i32* %ptr_random unordered, align 4
  %tot3 = add i32 %tot2, %val_random

  ret i32 %tot3
}

define i64 @atomic_load_relaxed_64(i64* %p, i32 %off32) #0 {
; CHECK-NOLSE-LABEL: atomic_load_relaxed_64:
; CHECK-NOLSE-O1:    ldr {{x[0-9]+}}, [x0, #32760]
; CHECK-NOLSE-O1:    ldr {{x[0-9]+}}, [x0, w1, sxtw #3]
; CHECK-NOLSE-O1:    ldur {{x[0-9]+}}, [x0, #-256]
; CHECK-NOLSE-O1:    add x[[ADDR:[0-9]+]], x0, #291, lsl #12
; CHECK-NOLSE-O1:    ldr {{x[0-9]+}}, [x[[ADDR]]]

; CHECK-LSE-LABEL: atomic_load_relaxed_64:
; CHECK-LSE:    ldr {{x[0-9]+}}, [x0, #32760]
; CHECK-LSE:    ldr {{x[0-9]+}}, [x0, w1, sxtw #3]
; CHECK-LSE:    ldur {{x[0-9]+}}, [x0, #-256]
; CHECK-LSE:    add x[[ADDR:[0-9]+]], x0, #291, lsl #12
; CHECK-LSE:    ldr {{x[0-9]+}}, [x[[ADDR]]]

  %ptr_unsigned = getelementptr i64, i64* %p, i32 4095
  %val_unsigned = load atomic i64, i64* %ptr_unsigned monotonic, align 8

  %ptr_regoff = getelementptr i64, i64* %p, i32 %off32
  %val_regoff = load atomic i64, i64* %ptr_regoff unordered, align 8
  %tot1 = add i64 %val_unsigned, %val_regoff

  %ptr_unscaled = getelementptr i64, i64* %p, i32 -32
  %val_unscaled = load atomic i64, i64* %ptr_unscaled monotonic, align 8
  %tot2 = add i64 %tot1, %val_unscaled

  %ptr_random = getelementptr i64, i64* %p, i32 148992 ; 0x123000/8 (i.e. ADD imm)
  %val_random = load atomic i64, i64* %ptr_random unordered, align 8
  %tot3 = add i64 %tot2, %val_random

  ret i64 %tot3
}


define void @atomc_store(i32* %p) #0 {
; CHECK-NOLSE-LABEL: atomc_store:
; CHECK-NOLSE:    mov w8, #4
; CHECK-NOLSE:    stlr w8, [x0]
; CHECK-NOLSE:    ret
;
; CHECK-LSE-LABEL: atomc_store:
; CHECK-LSE:    mov [[FOUR:w[0-9]+]], #4
; CHECK-LSE:    stlr [[FOUR]], [x0]
; CHECK-LSE:    ret
   store atomic i32 4, i32* %p seq_cst, align 4
   ret void
}

define void @atomic_store_relaxed_8(i8* %p, i32 %off32, i8 %val) #0 {
; CHECK-NOLSE-LABEL: atomic_store_relaxed_8:
; CHECK-NOLSE: strb    w2
; CHECK-NOLSE: strb    w2
; CHECK-NOLSE: strb    w2
; CHECK-NOLSE: strb    w2

  %ptr_unsigned = getelementptr i8, i8* %p, i32 4095
  store atomic i8 %val, i8* %ptr_unsigned monotonic, align 1

  %ptr_regoff = getelementptr i8, i8* %p, i32 %off32
  store atomic i8 %val, i8* %ptr_regoff unordered, align 1

  %ptr_unscaled = getelementptr i8, i8* %p, i32 -256
  store atomic i8 %val, i8* %ptr_unscaled monotonic, align 1

  %ptr_random = getelementptr i8, i8* %p, i32 1191936 ; 0x123000 (i.e. ADD imm)
  store atomic i8 %val, i8* %ptr_random unordered, align 1

  ret void
}

define void @atomic_store_relaxed_16(i16* %p, i32 %off32, i16 %val) #0 {
; CHECK-NOLSE-LABEL: atomic_store_relaxed_16:
; CHECK-NOLSE: strh    w2
; CHECK-NOLSE: strh    w2
; CHECK-NOLSE: strh    w2
; CHECK-NOLSE: strh    w2

; CHECK-LSE: strh w2
; CHECK-LSE: strh w2
; CHECK-LSE: strh w2
; CHECK-LSE: strh w2

  %ptr_unsigned = getelementptr i16, i16* %p, i32 4095
  store atomic i16 %val, i16* %ptr_unsigned monotonic, align 2

  %ptr_regoff = getelementptr i16, i16* %p, i32 %off32
  store atomic i16 %val, i16* %ptr_regoff unordered, align 2

  %ptr_unscaled = getelementptr i16, i16* %p, i32 -128
  store atomic i16 %val, i16* %ptr_unscaled monotonic, align 2

  %ptr_random = getelementptr i16, i16* %p, i32 595968 ; 0x123000/2 (i.e. ADD imm)
  store atomic i16 %val, i16* %ptr_random unordered, align 2

  ret void
}

define void @atomic_store_relaxed_32(i32* %p, i32 %off32, i32 %val) #0 {
; CHECK-NOLSE-LABEL: atomic_store_relaxed_32:
; CHECK-NOLSE:    str w2
; CHECK-NOLSE:    str w2
; CHECK-NOLSE:    stur w2
; CHECK-NOLSE:    str w2

; CHECK-LSE-LABEL: atomic_store_relaxed_32:
; CHECK-LSE:    str w2
; CHECK-LSE:    str w2
; CHECK-LSE:    stur w2
; CHECK-LSE:    str w2
  %ptr_unsigned = getelementptr i32, i32* %p, i32 4095
  store atomic i32 %val, i32* %ptr_unsigned monotonic, align 4

  %ptr_regoff = getelementptr i32, i32* %p, i32 %off32
  store atomic i32 %val, i32* %ptr_regoff unordered, align 4

  %ptr_unscaled = getelementptr i32, i32* %p, i32 -64
  store atomic i32 %val, i32* %ptr_unscaled monotonic, align 4

  %ptr_random = getelementptr i32, i32* %p, i32 297984 ; 0x123000/4 (i.e. ADD imm)
  store atomic i32 %val, i32* %ptr_random unordered, align 4

  ret void
}

define void @atomic_store_relaxed_64(i64* %p, i32 %off32, i64 %val) #0 {
; CHECK-NOLSE-LABEL: atomic_store_relaxed_64:
; CHECK-NOLSE:    str x2
; CHECK-NOLSE:    str x2
; CHECK-NOLSE:    stur x2
; CHECK-NOLSE:    str x2

; CHECK-LSE-LABEL: atomic_store_relaxed_64:
; CHECK-LSE:    str x2
; CHECK-LSE:    str x2
; CHECK-LSE:    stur x2
; CHECK-LSE:    str x2
  %ptr_unsigned = getelementptr i64, i64* %p, i32 4095
  store atomic i64 %val, i64* %ptr_unsigned monotonic, align 8

  %ptr_regoff = getelementptr i64, i64* %p, i32 %off32
  store atomic i64 %val, i64* %ptr_regoff unordered, align 8

  %ptr_unscaled = getelementptr i64, i64* %p, i32 -32
  store atomic i64 %val, i64* %ptr_unscaled monotonic, align 8

  %ptr_random = getelementptr i64, i64* %p, i32 148992 ; 0x123000/8 (i.e. ADD imm)
  store atomic i64 %val, i64* %ptr_random unordered, align 8

  ret void
}

attributes #0 = { nounwind }
