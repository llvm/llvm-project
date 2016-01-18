; RUN: llc < %s -mtriple=armv7-apple-ios | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-LE
; RUN: llc < %s -mtriple=thumbv7-none-linux-gnueabihf | FileCheck %s --check-prefix=CHECK-THUMB --check-prefix=CHECK-THUMB-LE
; RUN: llc < %s -mtriple=armebv7 -target-abi apcs | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-BE
; RUN: llc < %s -mtriple=thumbebv7-none-linux-gnueabihf | FileCheck %s --check-prefix=CHECK-THUMB --check-prefix=CHECK-THUMB-BE

define i64 @test1(i64* %ptr, i64 %val) {
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

  %r = atomicrmw add i64* %ptr, i64 %val seq_cst
  ret i64 %r
}

define i64 @test2(i64* %ptr, i64 %val) {
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

  %r = atomicrmw sub i64* %ptr, i64 %val seq_cst
  ret i64 %r
}

define i64 @test3(i64* %ptr, i64 %val) {
; CHECK-LABEL: test3:
; CHECK: dmb {{ish$}}
; CHECK: ldrexd [[REG1:(r[0-9]?[02468])]], [[REG2:(r[0-9]?[13579])]]
; CHECK-LE-DAG: and [[REG3:(r[0-9]?[02468])]], [[REG1]]
; CHECK-LE-DAG: and [[REG4:(r[0-9]?[13579])]], [[REG2]]
; CHECK-BE-DAG: and [[REG4:(r[0-9]?[13579])]], [[REG2]]
; CHECK-BE-DAG: and [[REG3:(r[0-9]?[02468])]], [[REG1]]
; CHECK: strexd {{[a-z0-9]+}}, [[REG3]], [[REG4]]
; CHECK: cmp
; CHECK: bne
; CHECK: dmb {{ish$}}

; CHECK-THUMB-LABEL: test3:
; CHECK-THUMB: dmb {{ish$}}
; CHECK-THUMB: ldrexd [[REG1:[a-z0-9]+]], [[REG2:[a-z0-9]+]]
; CHECK-THUMB-LE-DAG: and.w [[REG3:[a-z0-9]+]], [[REG1]]
; CHECK-THUMB-LE-DAG: and.w [[REG4:[a-z0-9]+]], [[REG2]]
; CHECK-THUMB-BE-DAG: and.w [[REG4:[a-z0-9]+]], [[REG2]]
; CHECK-THUMB-BE-DAG: and.w [[REG3:[a-z0-9]+]], [[REG1]]
; CHECK-THUMB: strexd {{[a-z0-9]+}}, [[REG3]], [[REG4]]
; CHECK-THUMB: cmp
; CHECK-THUMB: bne
; CHECK-THUMB: dmb {{ish$}}

  %r = atomicrmw and i64* %ptr, i64 %val seq_cst
  ret i64 %r
}

define i64 @test4(i64* %ptr, i64 %val) {
; CHECK-LABEL: test4:
; CHECK: dmb {{ish$}}
; CHECK: ldrexd [[REG1:(r[0-9]?[02468])]], [[REG2:(r[0-9]?[13579])]]
; CHECK-LE-DAG: orr [[REG3:(r[0-9]?[02468])]], [[REG1]]
; CHECK-LE-DAG: orr [[REG4:(r[0-9]?[13579])]], [[REG2]]
; CHECK-BE-DAG: orr [[REG4:(r[0-9]?[13579])]], [[REG2]]
; CHECK-BE-DAG: orr [[REG3:(r[0-9]?[02468])]], [[REG1]]
; CHECK: strexd {{[a-z0-9]+}}, [[REG3]], [[REG4]]
; CHECK: cmp
; CHECK: bne
; CHECK: dmb {{ish$}}

; CHECK-THUMB-LABEL: test4:
; CHECK-THUMB: dmb {{ish$}}
; CHECK-THUMB: ldrexd [[REG1:[a-z0-9]+]], [[REG2:[a-z0-9]+]]
; CHECK-THUMB-LE-DAG: orr.w [[REG3:[a-z0-9]+]], [[REG1]]
; CHECK-THUMB-LE-DAG: orr.w [[REG4:[a-z0-9]+]], [[REG2]]
; CHECK-THUMB-BE-DAG: orr.w [[REG4:[a-z0-9]+]], [[REG2]]
; CHECK-THUMB-BE-DAG: orr.w [[REG3:[a-z0-9]+]], [[REG1]]
; CHECK-THUMB: strexd {{[a-z0-9]+}}, [[REG3]], [[REG4]]
; CHECK-THUMB: cmp
; CHECK-THUMB: bne
; CHECK-THUMB: dmb {{ish$}}

  %r = atomicrmw or i64* %ptr, i64 %val seq_cst
  ret i64 %r
}

define i64 @test5(i64* %ptr, i64 %val) {
; CHECK-LABEL: test5:
; CHECK: dmb {{ish$}}
; CHECK: ldrexd [[REG1:(r[0-9]?[02468])]], [[REG2:(r[0-9]?[13579])]]
; CHECK-LE-DAG: eor [[REG3:(r[0-9]?[02468])]], [[REG1]]
; CHECK-LE-DAG: eor [[REG4:(r[0-9]?[13579])]], [[REG2]]
; CHECK-BE-DAG: eor [[REG4:(r[0-9]?[13579])]], [[REG2]]
; CHECK-BE-DAG: eor [[REG3:(r[0-9]?[02468])]], [[REG1]]
; CHECK: strexd {{[a-z0-9]+}}, [[REG3]], [[REG4]]
; CHECK: cmp
; CHECK: bne
; CHECK: dmb {{ish$}}

; CHECK-THUMB-LABEL: test5:
; CHECK-THUMB: dmb {{ish$}}
; CHECK-THUMB: ldrexd [[REG1:[a-z0-9]+]], [[REG2:[a-z0-9]+]]
; CHECK-THUMB-LE-DAG: eor.w [[REG3:[a-z0-9]+]], [[REG1]]
; CHECK-THUMB-LE-DAG: eor.w [[REG4:[a-z0-9]+]], [[REG2]]
; CHECK-THUMB-BE-DAG: eor.w [[REG4:[a-z0-9]+]], [[REG2]]
; CHECK-THUMB-BE-DAG: eor.w [[REG3:[a-z0-9]+]], [[REG1]]
; CHECK-THUMB: strexd {{[a-z0-9]+}}, [[REG3]], [[REG4]]
; CHECK-THUMB: cmp
; CHECK-THUMB: bne
; CHECK-THUMB: dmb {{ish$}}

  %r = atomicrmw xor i64* %ptr, i64 %val seq_cst
  ret i64 %r
}

define i64 @test6(i64* %ptr, i64 %val) {
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

  %r = atomicrmw xchg i64* %ptr, i64 %val seq_cst
  ret i64 %r
}

define i64 @test7(i64* %ptr, i64 %val1, i64 %val2) {
; CHECK-LABEL: test7:
; CHECK-DAG: mov [[VAL1LO:r[0-9]+]], r1
; CHECK-DAG: dmb {{ish$}}
; CHECK: ldrexd [[REG1:(r[0-9]?[02468])]], [[REG2:(r[0-9]?[13579])]]
; CHECK-LE-DAG: eor     [[MISMATCH_LO:r[0-9]+]], [[REG1]], [[VAL1LO]]
; CHECK-LE-DAG: eor     [[MISMATCH_HI:r[0-9]+]], [[REG2]], r2
; CHECK-BE-DAG: eor     [[MISMATCH_LO:r[0-9]+]], [[REG2]], r2
; CHECK-BE-DAG: eor     [[MISMATCH_HI:r[0-9]+]], [[REG1]], r1
; CHECK: orrs    {{r[0-9]+}}, [[MISMATCH_LO]], [[MISMATCH_HI]]
; CHECK: bne
; CHECK: strexd {{[a-z0-9]+}}, {{r[0-9]?[02468]}}, {{r[0-9]?[13579]}}
; CHECK: cmp
; CHECK: bne
; CHECK: dmb {{ish$}}

; CHECK-THUMB-LABEL: test7:
; CHECK-THUMB: dmb {{ish$}}
; CHECK-THUMB: ldrexd [[REG1:[a-z0-9]+]], [[REG2:[a-z0-9]+]]
; CHECK-THUMB-LE-DAG: eor.w     [[MISMATCH_LO:[a-z0-9]+]], [[REG1]], r2
; CHECK-THUMB-LE-DAG: eor.w     [[MISMATCH_HI:[a-z0-9]+]], [[REG2]], r3
; CHECK-THUMB-BE-DAG: eor.w     [[MISMATCH_HI:[a-z0-9]+]], [[REG1]], r2
; CHECK-THUMB-BE-DAG: eor.w     [[MISMATCH_LO:[a-z0-9]+]], [[REG2]], r3
; CHECK-THUMB-LE: orrs    [[MISMATCH_HI]], [[MISMATCH_LO]]
; CHECK-THUMB: bne
; CHECK-THUMB: strexd {{[a-z0-9]+}}, {{[a-z0-9]+}}, {{[a-z0-9]+}}
; CHECK-THUMB: cmp
; CHECK-THUMB: bne
; CHECK-THUMB: dmb {{ish$}}

  %pair = cmpxchg i64* %ptr, i64 %val1, i64 %val2 seq_cst seq_cst
  %r = extractvalue { i64, i1 } %pair, 0
  ret i64 %r
}

; Compiles down to a single ldrexd
define i64 @test8(i64* %ptr) {
; CHECK-LABEL: test8:
; CHECK: ldrexd [[REG1:(r[0-9]?[02468])]], [[REG2:(r[0-9]?[13579])]]
; CHECK: dmb {{ish$}}

; CHECK-THUMB-LABEL: test8:
; CHECK-THUMB: ldrexd [[REG1:[a-z0-9]+]], [[REG2:[a-z0-9]+]]
; CHECK-THUMB: dmb {{ish$}}

  %r = load atomic i64, i64* %ptr seq_cst, align 8
  ret i64 %r
}

; Compiles down to atomicrmw xchg; there really isn't any more efficient
; way to write it.
define void @test9(i64* %ptr, i64 %val) {
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

  store atomic i64 %val, i64* %ptr seq_cst, align 8
  ret void
}

define i64 @test10(i64* %ptr, i64 %val) {
; CHECK-LABEL: test10:
; CHECK: dmb {{ish$}}
; CHECK: ldrexd [[REG1:(r[0-9]?[02468])]], [[REG2:(r[0-9]?[13579])]]
; CHECK: mov     [[CARRY_LO:[a-z0-9]+]], #0
; CHECK: mov     [[CARRY_HI:[a-z0-9]+]], #0
; CHECK: mov     [[OUT_HI:[a-z0-9]+]], r2
; CHECK-LE: cmp     [[REG1]], r1
; CHECK-BE: cmp     [[REG2]], r2
; CHECK: movwls  [[CARRY_LO]], #1
; CHECK-LE: cmp     [[REG2]], r2
; CHECK-BE: cmp     [[REG1]], r1
; CHECK: movwle  [[CARRY_HI]], #1
; CHECK: moveq   [[CARRY_HI]], [[CARRY_LO]]
; CHECK: cmp     [[CARRY_HI]], #0
; CHECK: movne   [[OUT_HI]], [[REG2]]
; CHECK: mov     [[OUT_LO:[a-z0-9]+]], r1
; CHECK: movne   [[OUT_LO]], [[REG1]]
; CHECK: strexd {{[a-z0-9]+}}, [[REG3]], [[REG4]]
; CHECK: cmp
; CHECK: bne
; CHECK: dmb {{ish$}}

; CHECK-THUMB-LABEL: test10:
; CHECK-THUMB: dmb {{ish$}}
; CHECK-THUMB: ldrexd [[REG1:[a-z0-9]+]], [[REG2:[a-z0-9]+]]
; CHECK-THUMB: mov.w     [[CARRY_LO:[a-z0-9]+|lr]], #0
; CHECK-THUMB: movs     [[CARRY_HI:[a-z0-9]+|lr]], #0
; CHECK-THUMB-LE: cmp     [[REG1]], r2
; CHECK-THUMB-BE: cmp     [[REG2]], r3
; CHECK-THUMB: movls.w  [[CARRY_LO]], #1
; CHECK-THUMB-LE: cmp     [[REG2]], r3
; CHECK-THUMB-BE: cmp     [[REG1]], r2
; CHECK-THUMB: movle  [[CARRY_HI]], #1
; CHECK-THUMB: moveq   [[CARRY_HI]], [[CARRY_LO]]
; CHECK-THUMB: mov     [[OUT_HI:[a-z0-9]+]], r3
; CHECK-THUMB: cmp     [[CARRY_HI]], #0
; CHECK-THUMB: mov     [[OUT_LO:[a-z0-9]+]], r2
; CHECK-THUMB: movne   [[OUT_HI]], [[REG2]]
; CHECK-THUMB: movne   [[OUT_LO]], [[REG1]]
; CHECK-THUMB: strexd {{[a-z0-9]+}}, [[REG3]], [[REG4]]
; CHECK-THUMB: cmp
; CHECK-THUMB: bne
; CHECK-THUMB: dmb {{ish$}}

  %r = atomicrmw min i64* %ptr, i64 %val seq_cst
  ret i64 %r
}

define i64 @test11(i64* %ptr, i64 %val) {
; CHECK-LABEL: test11:
; CHECK: dmb {{ish$}}
; CHECK: ldrexd [[REG1:(r[0-9]?[02468])]], [[REG2:(r[0-9]?[13579])]]
; CHECK: mov     [[CARRY_LO:[a-z0-9]+]], #0
; CHECK: mov     [[CARRY_HI:[a-z0-9]+]], #0
; CHECK: mov     [[OUT_HI:[a-z0-9]+]], r2
; CHECK-LE: cmp     [[REG1]], r1
; CHECK-BE: cmp     [[REG2]], r2
; CHECK: movwls  [[CARRY_LO]], #1
; CHECK-LE: cmp     [[REG2]], r2
; CHECK-BE: cmp     [[REG1]], r1
; CHECK: movwls  [[CARRY_HI]], #1
; CHECK: moveq   [[CARRY_HI]], [[CARRY_LO]]
; CHECK: cmp     [[CARRY_HI]], #0
; CHECK: movne   [[OUT_HI]], [[REG2]]
; CHECK: mov     [[OUT_LO:[a-z0-9]+]], r1
; CHECK: movne   [[OUT_LO]], [[REG1]]
; CHECK: strexd {{[a-z0-9]+}}, [[REG3]], [[REG4]]
; CHECK: cmp
; CHECK: bne
; CHECK: dmb {{ish$}}

; CHECK-THUMB-LABEL: test11:
; CHECK-THUMB: dmb {{ish$}}
; CHECK-THUMB: ldrexd [[REG1:[a-z0-9]+]], [[REG2:[a-z0-9]+]]
; CHECK-THUMB: mov.w     [[CARRY_LO:[a-z0-9]+]], #0
; CHECK-THUMB: movs     [[CARRY_HI:[a-z0-9]+]], #0
; CHECK-THUMB-LE: cmp     [[REG1]], r2
; CHECK-THUMB-BE: cmp     [[REG2]], r3
; CHECK-THUMB: movls.w  [[CARRY_LO]], #1
; CHECK-THUMB-LE: cmp     [[REG2]], r3
; CHECK-THUMB-BE: cmp     [[REG1]], r2
; CHECK-THUMB: movls  [[CARRY_HI]], #1
; CHECK-THUMB: moveq   [[CARRY_HI]], [[CARRY_LO]]
; CHECK-THUMB: mov     [[OUT_HI:[a-z0-9]+]], r3
; CHECK-THUMB: cmp     [[CARRY_HI]], #0
; CHECK-THUMB: mov     [[OUT_LO:[a-z0-9]+]], r2
; CHECK-THUMB: movne   [[OUT_HI]], [[REG2]]
; CHECK-THUMB: movne   [[OUT_LO]], [[REG1]]
; CHECK-THUMB: strexd {{[a-z0-9]+}}, [[REG3]], [[REG4]]
; CHECK-THUMB: cmp
; CHECK-THUMB: bne
; CHECK-THUMB: dmb {{ish$}}

  %r = atomicrmw umin i64* %ptr, i64 %val seq_cst
  ret i64 %r
}

define i64 @test12(i64* %ptr, i64 %val) {
; CHECK-LABEL: test12:
; CHECK: dmb {{ish$}}
; CHECK: ldrexd [[REG1:(r[0-9]?[02468])]], [[REG2:(r[0-9]?[13579])]]
; CHECK: mov     [[CARRY_LO:[a-z0-9]+]], #0
; CHECK: mov     [[CARRY_HI:[a-z0-9]+]], #0
; CHECK: mov     [[OUT_HI:[a-z0-9]+]], r2
; CHECK-LE: cmp     [[REG1]], r1
; CHECK-BE: cmp     [[REG2]], r2
; CHECK: movwhi  [[CARRY_LO]], #1
; CHECK-LE: cmp     [[REG2]], r2
; CHECK-BE: cmp     [[REG1]], r1
; CHECK: movwgt  [[CARRY_HI]], #1
; CHECK: moveq   [[CARRY_HI]], [[CARRY_LO]]
; CHECK: cmp     [[CARRY_HI]], #0
; CHECK: movne   [[OUT_HI]], [[REG2]]
; CHECK: mov     [[OUT_LO:[a-z0-9]+]], r1
; CHECK: movne   [[OUT_LO]], [[REG1]]
; CHECK: strexd {{[a-z0-9]+}}, [[REG3]], [[REG4]]
; CHECK: cmp
; CHECK: bne
; CHECK: dmb {{ish$}}

; CHECK-THUMB-LABEL: test12:
; CHECK-THUMB: dmb {{ish$}}
; CHECK-THUMB: ldrexd [[REG1:[a-z0-9]+]], [[REG2:[a-z0-9]+]]
; CHECK-THUMB: mov.w     [[CARRY_LO:[a-z0-9]+]], #0
; CHECK-THUMB: movs     [[CARRY_HI:[a-z0-9]+]], #0
; CHECK-THUMB-LE: cmp     [[REG1]], r2
; CHECK-THUMB-BE: cmp     [[REG2]], r3
; CHECK-THUMB: movhi.w  [[CARRY_LO]], #1
; CHECK-THUMB-LE: cmp     [[REG2]], r3
; CHECK-THUMB-BE: cmp     [[REG1]], r2
; CHECK-THUMB: movgt  [[CARRY_HI]], #1
; CHECK-THUMB: moveq   [[CARRY_HI]], [[CARRY_LO]]
; CHECK-THUMB: mov     [[OUT_HI:[a-z0-9]+]], r3
; CHECK-THUMB: cmp     [[CARRY_HI]], #0
; CHECK-THUMB: mov     [[OUT_LO:[a-z0-9]+]], r2
; CHECK-THUMB: movne   [[OUT_HI]], [[REG2]]
; CHECK-THUMB: movne   [[OUT_LO]], [[REG1]]
; CHECK-THUMB: strexd {{[a-z0-9]+}}, [[REG3]], [[REG4]]
; CHECK-THUMB: cmp
; CHECK-THUMB: bne
; CHECK-THUMB: dmb {{ish$}}

  %r = atomicrmw max i64* %ptr, i64 %val seq_cst
  ret i64 %r
}

define i64 @test13(i64* %ptr, i64 %val) {
; CHECK-LABEL: test13:
; CHECK: dmb {{ish$}}
; CHECK: ldrexd [[REG1:(r[0-9]?[02468])]], [[REG2:(r[0-9]?[13579])]]
; CHECK: mov     [[CARRY_LO:[a-z0-9]+]], #0
; CHECK: mov     [[CARRY_HI:[a-z0-9]+]], #0
; CHECK: mov     [[OUT_HI:[a-z0-9]+]], r2
; CHECK-LE: cmp     [[REG1]], r1
; CHECK-BE: cmp     [[REG2]], r2
; CHECK: movwhi  [[CARRY_LO]], #1
; CHECK-LE: cmp     [[REG2]], r2
; CHECK-BE: cmp     [[REG1]], r1
; CHECK: movwhi  [[CARRY_HI]], #1
; CHECK: moveq   [[CARRY_HI]], [[CARRY_LO]]
; CHECK: cmp     [[CARRY_HI]], #0
; CHECK: movne   [[OUT_HI]], [[REG2]]
; CHECK: mov     [[OUT_LO:[a-z0-9]+]], r1
; CHECK: movne   [[OUT_LO]], [[REG1]]
; CHECK: strexd {{[a-z0-9]+}}, [[REG3]], [[REG4]]
; CHECK: cmp
; CHECK: bne
; CHECK: dmb {{ish$}}

; CHECK-THUMB-LABEL: test13:
; CHECK-THUMB: dmb {{ish$}}
; CHECK-THUMB: ldrexd [[REG1:[a-z0-9]+]], [[REG2:[a-z0-9]+]]
; CHECK-THUMB: mov.w     [[CARRY_LO:[a-z0-9]+]], #0
; CHECK-THUMB: movs     [[CARRY_HI:[a-z0-9]+]], #0
; CHECK-THUMB-LE: cmp     [[REG1]], r2
; CHECK-THUMB-BE: cmp     [[REG2]], r3
; CHECK-THUMB: movhi.w  [[CARRY_LO]], #1
; CHECK-THUMB-LE: cmp     [[REG2]], r3
; CHECK-THUMB-BE: cmp     [[REG1]], r2
; CHECK-THUMB: movhi  [[CARRY_HI]], #1
; CHECK-THUMB: moveq   [[CARRY_HI]], [[CARRY_LO]]
; CHECK-THUMB: mov     [[OUT_HI:[a-z0-9]+]], r3
; CHECK-THUMB: cmp     [[CARRY_HI]], #0
; CHECK-THUMB: mov     [[OUT_LO:[a-z0-9]+]], r2
; CHECK-THUMB: movne   [[OUT_HI]], [[REG2]]
; CHECK-THUMB: movne   [[OUT_LO]], [[REG1]]
; CHECK-THUMB: strexd {{[a-z0-9]+}}, [[REG3]], [[REG4]]
; CHECK-THUMB: cmp
; CHECK-THUMB: bne
; CHECK-THUMB: dmb {{ish$}}
  %r = atomicrmw umax i64* %ptr, i64 %val seq_cst
  ret i64 %r
}

