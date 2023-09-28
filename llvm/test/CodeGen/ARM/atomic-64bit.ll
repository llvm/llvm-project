; RUN: llc < %s -mtriple=armv7-apple-ios | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-LE
; RUN: llc < %s -mtriple=thumbv7-none-linux-gnueabihf | FileCheck %s --check-prefix=CHECK-THUMB --check-prefix=CHECK-THUMB-LE
; RUN: llc < %s -mtriple=armebv7 -target-abi apcs | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-BE
; RUN: llc < %s -mtriple=thumbebv7-none-linux-gnueabihf | FileCheck %s --check-prefix=CHECK-THUMB --check-prefix=CHECK-THUMB-BE
; RUN: llc < %s -mtriple=armv7m--none-eabi | FileCheck %s --check-prefix=CHECK-M
; RUN: llc < %s -mtriple=armv8m--none-eabi | FileCheck %s --check-prefix=CHECK-M

define i64 @test1(ptr %ptr, i64 %val) {
; CHECK-LABEL: test1:
; CHECK: dmb {{ish$}}
; CHECK: ldrexd [[REG1:(r[0-9]?[02468])]], [[REG2:(r[0-9]?[13579])]]
; CHECK-LE: adds [[REG3:(r[0-9]?[02468])]], [[REG1]]
; CHECK-LE: adc [[REG4:(r[0-9]?[13579])]], [[REG2]]
; CHECK-BE: adds [[REG4:(r[0-9]?[13579])]], [[REG2]]
; CHECK-BE: adc [[REG3:(r[0-9]?[02468])]], [[REG1]]
; CHECK: strexd {{[a-z0-9]+}}, [[REG3]], [[REG4]]
; CHECK: cmp
; CHECK: bne
; CHECK: dmb {{ish$}}

; CHECK-THUMB-LABEL: test1:
; CHECK-THUMB: dmb {{ish$}}
; CHECK-THUMB: ldrexd [[REG1:[a-z0-9]+]], [[REG2:[a-z0-9]+]]
; CHECK-THUMB-LE: adds.w [[REG3:[a-z0-9]+]], [[REG1]]
; CHECK-THUMB-LE: adc.w [[REG4:[a-z0-9]+]], [[REG2]]
; CHECK-THUMB-BE: adds.w [[REG4:[a-z0-9]+]], [[REG2]]
; CHECK-THUMB-BE: adc.w [[REG3:[a-z0-9]+]], [[REG1]]
; CHECK-THUMB: strexd {{[a-z0-9]+}}, [[REG3]], [[REG4]]
; CHECK-THUMB: cmp
; CHECK-THUMB: bne
; CHECK-THUMB: dmb {{ish$}}

; CHECK-M: __atomic_fetch_add_8

  %r = atomicrmw add ptr %ptr, i64 %val seq_cst
  ret i64 %r
}

define i64 @test2(ptr %ptr, i64 %val) {
; CHECK-LABEL: test2:
; CHECK: dmb {{ish$}}
; CHECK: ldrexd [[REG1:(r[0-9]?[02468])]], [[REG2:(r[0-9]?[13579])]]
; CHECK-LE: subs [[REG3:(r[0-9]?[02468])]], [[REG1]]
; CHECK-LE: sbc [[REG4:(r[0-9]?[13579])]], [[REG2]]
; CHECK-BE: subs [[REG4:(r[0-9]?[13579])]], [[REG2]]
; CHECK-BE: sbc [[REG3:(r[0-9]?[02468])]], [[REG1]]
; CHECK: strexd {{[a-z0-9]+}}, [[REG3]], [[REG4]]
; CHECK: cmp
; CHECK: bne
; CHECK: dmb {{ish$}}

; CHECK-THUMB-LABEL: test2:
; CHECK-THUMB: dmb {{ish$}}
; CHECK-THUMB: ldrexd [[REG1:[a-z0-9]+]], [[REG2:[a-z0-9]+]]
; CHECK-THUMB-LE: subs.w [[REG3:[a-z0-9]+]], [[REG1]]
; CHECK-THUMB-LE: sbc.w [[REG4:[a-z0-9]+]], [[REG2]]
; CHECK-THUMB-BE: subs.w [[REG4:[a-z0-9]+]], [[REG2]]
; CHECK-THUMB-BE: sbc.w [[REG3:[a-z0-9]+]], [[REG1]]
; CHECK-THUMB: strexd {{[a-z0-9]+}}, [[REG3]], [[REG4]]
; CHECK-THUMB: cmp
; CHECK-THUMB: bne
; CHECK-THUMB: dmb {{ish$}}

; CHECK-M: __atomic_fetch_sub_8

  %r = atomicrmw sub ptr %ptr, i64 %val seq_cst
  ret i64 %r
}

define i64 @test3(ptr %ptr, i64 %val) {
; CHECK-LABEL: test3:
; CHECK: dmb {{ish$}}
; CHECK: ldrexd [[REG1:(r[0-9]?[02468])]], [[REG2:(r[0-9]?[13579])]]
; CHECK-LE-DAG: and [[REG3:(r[0-9]?[02468])]], [[REG1]],
; CHECK-LE-DAG: and [[REG4:(r[0-9]?[13579])]], [[REG2]],
; CHECK-BE-DAG: and [[REG4:(r[0-9]?[13579])]], [[REG2]],
; CHECK-BE-DAG: and [[REG3:(r[0-9]?[02468])]], [[REG1]],
; CHECK: strexd {{[a-z0-9]+}}, [[REG3]], [[REG4]]
; CHECK: cmp
; CHECK: bne
; CHECK: dmb {{ish$}}

; CHECK-THUMB-LABEL: test3:
; CHECK-THUMB: dmb {{ish$}}
; CHECK-THUMB: ldrexd [[REG1:[a-z0-9]+]], [[REG2:[a-z0-9]+]]
; CHECK-THUMB-LE-DAG: and.w [[REG3:[a-z0-9]+]], [[REG1]],
; CHECK-THUMB-LE-DAG: and.w [[REG4:[a-z0-9]+]], [[REG2]],
; CHECK-THUMB-BE-DAG: and.w [[REG4:[a-z0-9]+]], [[REG2]],
; CHECK-THUMB-BE-DAG: and.w [[REG3:[a-z0-9]+]], [[REG1]],
; CHECK-THUMB: strexd {{[a-z0-9]+}}, [[REG3]], [[REG4]]
; CHECK-THUMB: cmp
; CHECK-THUMB: bne
; CHECK-THUMB: dmb {{ish$}}

; CHECK-M: _atomic_fetch_and_8

  %r = atomicrmw and ptr %ptr, i64 %val seq_cst
  ret i64 %r
}

define i64 @test4(ptr %ptr, i64 %val) {
; CHECK-LABEL: test4:
; CHECK: dmb {{ish$}}
; CHECK: ldrexd [[REG1:(r[0-9]?[02468])]], [[REG2:(r[0-9]?[13579])]]
; CHECK-LE-DAG: orr [[REG3:(r[0-9]?[02468])]], [[REG1]],
; CHECK-LE-DAG: orr [[REG4:(r[0-9]?[13579])]], [[REG2]],
; CHECK-BE-DAG: orr [[REG4:(r[0-9]?[13579])]], [[REG2]],
; CHECK-BE-DAG: orr [[REG3:(r[0-9]?[02468])]], [[REG1]],
; CHECK: strexd {{[a-z0-9]+}}, [[REG3]], [[REG4]]
; CHECK: cmp
; CHECK: bne
; CHECK: dmb {{ish$}}

; CHECK-THUMB-LABEL: test4:
; CHECK-THUMB: dmb {{ish$}}
; CHECK-THUMB: ldrexd [[REG1:[a-z0-9]+]], [[REG2:[a-z0-9]+]]
; CHECK-THUMB-LE-DAG: orr.w [[REG3:[a-z0-9]+]], [[REG1]],
; CHECK-THUMB-LE-DAG: orr.w [[REG4:[a-z0-9]+]], [[REG2]],
; CHECK-THUMB-BE-DAG: orr.w [[REG4:[a-z0-9]+]], [[REG2]],
; CHECK-THUMB-BE-DAG: orr.w [[REG3:[a-z0-9]+]], [[REG1]],
; CHECK-THUMB: strexd {{[a-z0-9]+}}, [[REG3]], [[REG4]]
; CHECK-THUMB: cmp
; CHECK-THUMB: bne
; CHECK-THUMB: dmb {{ish$}}

; CHECK-M: __atomic_fetch_or_8

  %r = atomicrmw or ptr %ptr, i64 %val seq_cst
  ret i64 %r
}

define i64 @test5(ptr %ptr, i64 %val) {
; CHECK-LABEL: test5:
; CHECK: dmb {{ish$}}
; CHECK: ldrexd [[REG1:(r[0-9]?[02468])]], [[REG2:(r[0-9]?[13579])]]
; CHECK-LE-DAG: eor [[REG3:(r[0-9]?[02468])]], [[REG1]],
; CHECK-LE-DAG: eor [[REG4:(r[0-9]?[13579])]], [[REG2]],
; CHECK-BE-DAG: eor [[REG4:(r[0-9]?[13579])]], [[REG2]],
; CHECK-BE-DAG: eor [[REG3:(r[0-9]?[02468])]], [[REG1]],
; CHECK: strexd {{[a-z0-9]+}}, [[REG3]], [[REG4]]
; CHECK: cmp
; CHECK: bne
; CHECK: dmb {{ish$}}

; CHECK-THUMB-LABEL: test5:
; CHECK-THUMB: dmb {{ish$}}
; CHECK-THUMB: ldrexd [[REG1:[a-z0-9]+]], [[REG2:[a-z0-9]+]]
; CHECK-THUMB-LE-DAG: eor.w [[REG3:[a-z0-9]+]], [[REG1]],
; CHECK-THUMB-LE-DAG: eor.w [[REG4:[a-z0-9]+]], [[REG2]],
; CHECK-THUMB-BE-DAG: eor.w [[REG4:[a-z0-9]+]], [[REG2]],
; CHECK-THUMB-BE-DAG: eor.w [[REG3:[a-z0-9]+]], [[REG1]],
; CHECK-THUMB: strexd {{[a-z0-9]+}}, [[REG3]], [[REG4]]
; CHECK-THUMB: cmp
; CHECK-THUMB: bne
; CHECK-THUMB: dmb {{ish$}}

; CHECK-M: __atomic_fetch_xor_8

  %r = atomicrmw xor ptr %ptr, i64 %val seq_cst
  ret i64 %r
}

define i64 @test6(ptr %ptr, i64 %val) {
; CHECK-LABEL: test6:
; CHECK: dmb {{ish$}}
; CHECK: ldrexd [[REG1:(r[0-9]?[02468])]], [[REG2:(r[0-9]?[13579])]]
; CHECK: strexd {{[a-z0-9]+}}, {{r[0-9]?[02468]}}, {{r[0-9]?[13579]}}
; CHECK: cmp
; CHECK: bne
; CHECK: dmb {{ish$}}

; CHECK-THUMB-LABEL: test6:
; CHECK-THUMB: dmb {{ish$}}
; CHECK-THUMB: ldrexd [[REG1:[a-z0-9]+]], [[REG2:[a-z0-9]+]]
; CHECK-THUMB: strexd {{[a-z0-9]+}}, {{[a-z0-9]+}}, {{[a-z0-9]+}}
; CHECK-THUMB: cmp
; CHECK-THUMB: bne
; CHECK-THUMB: dmb {{ish$}}

; CHECK-M: __atomic_exchange_8

  %r = atomicrmw xchg ptr %ptr, i64 %val seq_cst
  ret i64 %r
}

define i64 @test7(ptr %ptr, i64 %val1, i64 %val2) {
; CHECK-LABEL: test7:
; CHECK-DAG: mov [[VAL1LO:r[0-9]+]], r1
; CHECK: ldrexd [[REG1:(r[0-9]?[02468])]], [[REG2:(r[0-9]?[13579])]]
; CHECK-LE-DAG: eor     [[MISMATCH_LO:.*]], [[REG1]], [[VAL1LO]]
; CHECK-LE-DAG: eor     [[MISMATCH_HI:.*]], [[REG2]], r2
; CHECK-BE-DAG: eor     [[MISMATCH_LO:.*]], [[REG2]], r2
; CHECK-BE-DAG: eor     [[MISMATCH_HI:.*]], [[REG1]], r1
; CHECK: orrs    {{r[0-9]+}}, [[MISMATCH_LO]], [[MISMATCH_HI]]
; CHECK: bne
; CHECK-DAG: dmb {{ish$}}
; CHECK: strexd {{[a-z0-9]+}}, {{r[0-9]?[02468]}}, {{r[0-9]?[13579]}}
; CHECK: cmp
; CHECK: beq
; CHECK: dmb {{ish$}}

; CHECK-THUMB-LABEL: test7:
; CHECK-THUMB: ldrexd [[REG1:[a-z0-9]+]], [[REG2:[a-z0-9]+]]
; CHECK-THUMB-LE-DAG: eor.w     [[MISMATCH_LO:[a-z0-9]+]], [[REG1]], r2
; CHECK-THUMB-LE-DAG: eor.w     [[MISMATCH_HI:[a-z0-9]+]], [[REG2]], r3
; CHECK-THUMB-BE-DAG: eor.w     [[MISMATCH_HI:[a-z0-9]+]], [[REG1]], r2
; CHECK-THUMB-BE-DAG: eor.w     [[MISMATCH_LO:[a-z0-9]+]], [[REG2]], r3
; CHECK-THUMB-LE: orrs.w    {{.*}}, [[MISMATCH_LO]], [[MISMATCH_HI]]
; CHECK-THUMB: bne
; CHECK-THUMB: dmb {{ish$}}
; CHECK-THUMB: strexd {{[a-z0-9]+}}, {{[a-z0-9]+}}, {{[a-z0-9]+}}
; CHECK-THUMB: cmp
; CHECK-THUMB: beq
; CHECK-THUMB: dmb {{ish$}}

; CHECK-M: __atomic_compare_exchange_8

  %pair = cmpxchg ptr %ptr, i64 %val1, i64 %val2 seq_cst seq_cst
  %r = extractvalue { i64, i1 } %pair, 0
  ret i64 %r
}

; Compiles down to a single ldrexd, except on M class devices where ldrexd
; isn't supported.
define i64 @test8(ptr %ptr) {
; CHECK-LABEL: test8:
; CHECK: ldrexd [[REG1:(r[0-9]?[02468])]], [[REG2:(r[0-9]?[13579])]]
; CHECK-NOT: strexd
; CHECK: clrex
; CHECK-NOT: strexd
; CHECK: dmb {{ish$}}

; CHECK-THUMB-LABEL: test8:
; CHECK-THUMB: ldrexd [[REG1:[a-z0-9]+]], [[REG2:[a-z0-9]+]]
; CHECK-THUMB-NOT: strexd
; CHECK-THUMB: clrex
; CHECK-THUMB-NOT: strexd
; CHECK-THUMB: dmb {{ish$}}

; CHECK-M: __atomic_load_8

  %r = load atomic i64, ptr %ptr seq_cst, align 8
  ret i64 %r
}

; Compiles down to atomicrmw xchg; there really isn't any more efficient
; way to write it. Except on M class devices, where ldrexd/strexd aren't
; supported.
define void @test9(ptr %ptr, i64 %val) {
; CHECK-LABEL: test9:
; CHECK: dmb {{ish$}}
; CHECK: ldrexd [[REG1:(r[0-9]?[02468])]], [[REG2:(r[0-9]?[13579])]]
; CHECK: strexd {{[a-z0-9]+}}, {{r[0-9]?[02468]}}, {{r[0-9]?[13579]}}
; CHECK: cmp
; CHECK: bne
; CHECK: dmb {{ish$}}

; CHECK-THUMB-LABEL: test9:
; CHECK-THUMB: dmb {{ish$}}
; CHECK-THUMB: ldrexd [[REG1:[a-z0-9]+]], [[REG2:[a-z0-9]+]]
; CHECK-THUMB: strexd {{[a-z0-9]+}}, {{[a-z0-9]+}}, {{[a-z0-9]+}}
; CHECK-THUMB: cmp
; CHECK-THUMB: bne
; CHECK-THUMB: dmb {{ish$}}

; CHECK-M: __atomic_store_8

  store atomic i64 %val, ptr %ptr seq_cst, align 8
  ret void
}

define i64 @test10(ptr %ptr, i64 %val) {
; CHECK-LABEL: test10:
; CHECK: dmb {{ish$}}
; CHECK: ldrexd [[REG1:(r[0-9]?[02468])]], [[REG2:(r[0-9]?[13579])]]
; CHECK: mov     [[OUT_HI:[a-z0-9]+]], r2
; CHECK-LE: subs {{[^,]+}}, r1, [[REG1]]
; CHECK-BE: subs {{[^,]+}}, r2, [[REG2]]
; CHECK-LE: sbcs {{[^,]+}}, r2, [[REG2]]
; CHECK-BE: sbcs {{[^,]+}}, r1, [[REG1]]
; CHECK: mov     [[CMP:[a-z0-9]+]], #0
; CHECK: movwge  [[CMP]], #1
; CHECK: cmp     [[CMP]], #0
; CHECK: movne   [[OUT_HI]], [[REG2]]
; CHECK: mov     [[OUT_LO:[a-z0-9]+]], r1
; CHECK: movne   [[OUT_LO]], [[REG1]]
; CHECK: strexd {{[a-z0-9]+}}, [[OUT_LO]], [[OUT_HI]]
; CHECK: cmp
; CHECK: bne
; CHECK: dmb {{ish$}}

; CHECK-THUMB-LABEL: test10:
; CHECK-THUMB: dmb {{ish$}}
; CHECK-THUMB: ldrexd [[REG1:[a-z0-9]+]], [[REG2:[a-z0-9]+]]
; CHECK-THUMB: mov      [[OUT_LO:[a-z0-9]+]], r2
; CHECK-THUMB-LE: subs.w {{[^,]+}}, r2, [[REG1]]
; CHECK-THUMB-BE: subs.w {{[^,]+}}, r3, [[REG2]]
; CHECK-THUMB-LE: sbcs.w {{[^,]+}}, r3, [[REG2]]
; CHECK-THUMB-BE: sbcs.w {{[^,]+}}, r2, [[REG1]]
; CHECK-THUMB: mov.w     [[CMP:[a-z0-9]+]], #0
; CHECK-THUMB: movge.w   [[CMP]], #1
; CHECK-THUMB: cmp.w     [[CMP]], #0
; CHECK-THUMB: mov       [[OUT_HI:[a-z0-9]+]], r3
; CHECK-THUMB: movne   [[OUT_HI]], [[REG2]]
; CHECK-THUMB: movne   [[OUT_LO]], [[REG1]]
; CHECK-THUMB: strexd {{[a-z0-9]+}}, [[OUT_LO]], [[OUT_HI]]
; CHECK-THUMB: cmp
; CHECK-THUMB: bne
; CHECK-THUMB: dmb {{ish$}}

; CHECK-M: __atomic_compare_exchange_8

  %r = atomicrmw min ptr %ptr, i64 %val seq_cst
  ret i64 %r
}

define i64 @test11(ptr %ptr, i64 %val) {
; CHECK-LABEL: test11:
; CHECK: dmb {{ish$}}
; CHECK: ldrexd [[REG1:(r[0-9]?[02468])]], [[REG2:(r[0-9]?[13579])]]
; CHECK: mov     [[OUT_HI:[a-z0-9]+]], r2
; CHECK-LE: subs    {{[^,]+}}, r1, [[REG1]]
; CHECK-BE: subs    {{[^,]+}}, r2, [[REG2]]
; CHECK-LE: sbcs    {{[^,]+}}, r2, [[REG2]]
; CHECK-BE: sbcs    {{[^,]+}}, r1, [[REG1]]
; CHECK: mov     [[CMP:[a-z0-9]+]], #0
; CHECK: movwhs  [[CMP]], #1
; CHECK: cmp     [[CMP]], #0
; CHECK: movne   [[OUT_HI]], [[REG2]]
; CHECK: mov     [[OUT_LO:[a-z0-9]+]], r1
; CHECK: movne   [[OUT_LO]], [[REG1]]
; CHECK: strexd {{[a-z0-9]+}}, [[OUT_LO]], [[OUT_HI]]
; CHECK: cmp
; CHECK: bne
; CHECK: dmb {{ish$}}

; CHECK-THUMB-LABEL: test11:
; CHECK-THUMB: dmb {{ish$}}
; CHECK-THUMB: ldrexd [[REG1:[a-z0-9]+]], [[REG2:[a-z0-9]+]]
; CHECK-THUMB: mov      [[OUT_LO:[a-z0-9]+]], r2
; CHECK-THUMB-LE: subs.w {{[^,]+}}, r2, [[REG1]]
; CHECK-THUMB-BE: subs.w {{[^,]+}}, r3, [[REG2]]
; CHECK-THUMB-LE: sbcs.w {{[^,]+}}, r3, [[REG2]]
; CHECK-THUMB-BE: sbcs.w {{[^,]+}}, r2, [[REG1]]
; CHECK-THUMB: mov.w     [[CMP:[a-z0-9]+]], #0
; CHECK-THUMB: movhs.w   [[CMP]], #1
; CHECK-THUMB: cmp.w     [[CMP]], #0
; CHECK-THUMB: mov       [[OUT_HI:[a-z0-9]+]], r3
; CHECK-THUMB: movne   [[OUT_HI]], [[REG2]]
; CHECK-THUMB: movne   [[OUT_LO]], [[REG1]]
; CHECK-THUMB: strexd {{[a-z0-9]+}}, [[OUT_LO]], [[OUT_HI]]
; CHECK-THUMB: cmp
; CHECK-THUMB: bne
; CHECK-THUMB: dmb {{ish$}}

; CHECK-M: __atomic_compare_exchange_8

  %r = atomicrmw umin ptr %ptr, i64 %val seq_cst
  ret i64 %r
}

define i64 @test12(ptr %ptr, i64 %val) {
; CHECK-LABEL: test12:
; CHECK: dmb {{ish$}}
; CHECK: ldrexd [[REG1:(r[0-9]?[02468])]], [[REG2:(r[0-9]?[13579])]]
; CHECK: mov     [[OUT_HI:[a-z0-9]+]], r2
; CHECK-LE: subs    {{[^,]+}}, r1, [[REG1]]
; CHECK-BE: subs    {{[^,]+}}, r2, [[REG2]]
; CHECK-LE: sbcs    {{[^,]+}}, r2, [[REG2]]
; CHECK-BE: sbcs    {{[^,]+}}, r1, [[REG1]]
; CHECK: mov     [[CMP:[a-z0-9]+]], #0
; CHECK: movwlt  [[CMP]], #1
; CHECK: cmp     [[CMP]], #0
; CHECK: movne   [[OUT_HI]], [[REG2]]
; CHECK: mov     [[OUT_LO:[a-z0-9]+]], r1
; CHECK: movne   [[OUT_LO]], [[REG1]]
; CHECK: strexd {{[a-z0-9]+}}, [[OUT_LO]], [[OUT_HI]]
; CHECK: cmp
; CHECK: bne
; CHECK: dmb {{ish$}}

; CHECK-THUMB-LABEL: test12:
; CHECK-THUMB: dmb {{ish$}}
; CHECK-THUMB: ldrexd [[REG1:[a-z0-9]+]], [[REG2:[a-z0-9]+]]
; CHECK-THUMB: mov      [[OUT_LO:[a-z0-9]+]], r2
; CHECK-THUMB-LE: subs.w {{[^,]+}}, r2, [[REG1]]
; CHECK-THUMB-BE: subs.w {{[^,]+}}, r3, [[REG2]]
; CHECK-THUMB-LE: sbcs.w {{[^,]+}}, r3, [[REG2]]
; CHECK-THUMB-BE: sbcs.w {{[^,]+}}, r2, [[REG1]]
; CHECK-THUMB: mov.w     [[CMP:[a-z0-9]+]], #0
; CHECK-THUMB: movlt.w   [[CMP]], #1
; CHECK-THUMB: cmp.w     [[CMP]], #0
; CHECK-THUMB: mov       [[OUT_HI:[a-z0-9]+]], r3
; CHECK-THUMB: movne   [[OUT_HI]], [[REG2]]
; CHECK-THUMB: movne   [[OUT_LO]], [[REG1]]
; CHECK-THUMB: strexd {{[a-z0-9]+}}, [[OUT_LO]], [[OUT_HI]]
; CHECK-THUMB: cmp
; CHECK-THUMB: bne
; CHECK-THUMB: dmb {{ish$}}

; CHECK-M: __atomic_compare_exchange_8

  %r = atomicrmw max ptr %ptr, i64 %val seq_cst
  ret i64 %r
}

define i64 @test13(ptr %ptr, i64 %val) {
; CHECK-LABEL: test13:
; CHECK: dmb {{ish$}}
; CHECK: ldrexd [[REG1:(r[0-9]?[02468])]], [[REG2:(r[0-9]?[13579])]]
; CHECK: mov     [[OUT_HI:[a-z0-9]+]], r2
; CHECK-LE: subs    {{[^,]+}}, r1, [[REG1]]
; CHECK-BE: subs    {{[^,]+}}, r2, [[REG2]]
; CHECK-LE: sbcs    {{[^,]+}}, r2, [[REG2]]
; CHECK-BE: sbcs    {{[^,]+}}, r1, [[REG1]]
; CHECK: mov     [[CMP:[a-z0-9]+]], #0
; CHECK: movwlo  [[CMP]], #1
; CHECK: cmp     [[CMP]], #0
; CHECK: movne   [[OUT_HI]], [[REG2]]
; CHECK: mov     [[OUT_LO:[a-z0-9]+]], r1
; CHECK: movne   [[OUT_LO]], [[REG1]]
; CHECK: strexd {{[a-z0-9]+}}, [[OUT_LO]], [[OUT_HI]]
; CHECK: cmp
; CHECK: bne
; CHECK: dmb {{ish$}}

; CHECK-THUMB-LABEL: test13:
; CHECK-THUMB: dmb {{ish$}}
; CHECK-THUMB: ldrexd [[REG1:[a-z0-9]+]], [[REG2:[a-z0-9]+]]
; CHECK-THUMB: mov      [[OUT_LO:[a-z0-9]+]], r2
; CHECK-THUMB-LE: subs.w {{[^,]+}}, r2, [[REG1]]
; CHECK-THUMB-BE: subs.w {{[^,]+}}, r3, [[REG2]]
; CHECK-THUMB-LE: sbcs.w {{[^,]+}}, r3, [[REG2]]
; CHECK-THUMB-BE: sbcs.w {{[^,]+}}, r2, [[REG1]]
; CHECK-THUMB: mov.w     [[CMP:[a-z0-9]+]], #0
; CHECK-THUMB: movlo.w   [[CMP]], #1
; CHECK-THUMB: cmp.w     [[CMP]], #0
; CHECK-THUMB: mov       [[OUT_HI:[a-z0-9]+]], r3
; CHECK-THUMB: movne   [[OUT_HI]], [[REG2]]
; CHECK-THUMB: movne   [[OUT_LO]], [[REG1]]
; CHECK-THUMB: strexd {{[a-z0-9]+}}, [[OUT_LO]], [[OUT_HI]]
; CHECK-THUMB: cmp
; CHECK-THUMB: bne
; CHECK-THUMB: dmb {{ish$}}

; CHECK-M: __atomic_compare_exchange_8

  %r = atomicrmw umax ptr %ptr, i64 %val seq_cst
  ret i64 %r
}

