; NOTE: Do not autogenerate
; RUN: split-file %s %t
; RUN: llc -O3 -stop-after=finalize-isel %t/estimate.ll -o - | FileCheck %s --check-prefix=MIR --enable-var-scope
; RUN: llc -O3 %t/estimate.ll -o - | FileCheck %s --check-prefix=ASM --enable-var-scope --implicit-check-not='{{^[[:space:]]+sqrtps[[:space:]]}}' --implicit-check-not='{{^[[:space:]]+divps[[:space:]]}}'
; RUN: llc -O3 %t/fallback.ll -o - | FileCheck %s --check-prefix=FALLBACK

; On i686 with SSE1 but no SSE2 or x87, v4f32 is legal while scalar f32
; requires softening. The v4f32 cases create refinement constants before type
; legalization. The v8f32 cases split first and create them during the
; AfterLegalizeTypes combine. MIR checks bind constant-pool entries to their
; uses and follow complete refinement dataflow because post-RA assembly does
; not preserve that value identity. Assembly checks preserve working boundaries
; and exclude fallback. The fallback input positively establishes the excluded
; native operations.

;--- estimate.ll

target triple = "i686-unknown-linux-gnu"

; ASM-LABEL: rsqrt_v4_steps_0:
; ASM:       # %bb.0:
; ASM-NEXT:    rsqrtps %xmm1, %xmm1
; ASM-NEXT:    mulps %xmm1, %xmm0
; ASM-NEXT:    retl
define <4 x float> @rsqrt_v4_steps_0(
    <4 x float> %n, <4 x float> %x) #0 {
  %sqrt = call afn ninf <4 x float> @llvm.sqrt.v4f32(<4 x float> %x)
  %q = fdiv arcp ninf <4 x float> %n, %sqrt
  ret <4 x float> %q
}

; MIR-LABEL: name: rsqrt_v4_default
; MIR: constants:
; MIR-NEXT: - id: [[THREE:[0-9]+]]
; MIR-NEXT: value: '<4 x float> splat (float -3.000000e+00)'
; MIR: - id: [[HALF:[0-9]+]]
; MIR-NEXT: value: '<4 x float> splat (float -5.000000e-01)'
; MIR: body:
; MIR: %[[ARG:[0-9]+]]:vr128 = COPY $xmm1
; MIR: %[[NUM:[0-9]+]]:vr128 = COPY $xmm0
; MIR: %[[EST:[0-9]+]]:vr128 = {{.*}}RSQRTPSr %[[ARG]]
; MIR: %[[AE:[0-9]+]]:vr128 = {{.*}}MULPSrr %[[ARG]], %[[EST]],
; MIR: %[[AEE:[0-9]+]]:vr128 = {{.*}}MULPSrr %[[AE]], %[[EST]],
; MIR: %[[RHS:[0-9]+]]:vr128 = {{.*}}ADDPSrm %[[AEE]], $noreg, 1, $noreg, %const.[[THREE]], $noreg,
; MIR: %[[LHS:[0-9]+]]:vr128 = {{.*}}MULPSrm %[[EST]], $noreg, 1, $noreg, %const.[[HALF]], $noreg,
; MIR: %[[REFINED:[0-9]+]]:vr128 = {{.*}}MULPSrr %[[LHS]], {{(killed )?}}%[[RHS]],
; MIR: %[[RESULT:[0-9]+]]:vr128 = {{.*}}MULPSrr %[[NUM]], {{(killed )?}}%[[REFINED]],
; MIR-NEXT: $xmm0 = COPY %[[RESULT]]
; MIR-NEXT: RET 0, $xmm0
; ASM-LABEL: rsqrt_v4_default:
; ASM:       rsqrtps
; ASM-NOT:   {{^[[:space:]]+sqrtps}}
; ASM-NOT:   divps
; ASM:       retl
define <4 x float> @rsqrt_v4_default(
    <4 x float> %n, <4 x float> %x) #1 {
  %sqrt = call afn ninf <4 x float> @llvm.sqrt.v4f32(<4 x float> %x)
  %q = fdiv arcp ninf <4 x float> %n, %sqrt
  ret <4 x float> %q
}

; MIR-LABEL: name: rsqrt_v4_steps_2
; MIR: constants:
; MIR-NEXT: - id: [[HALF:[0-9]+]]
; MIR-NEXT: value: '<4 x float> splat (float -5.000000e-01)'
; MIR: - id: [[THREE:[0-9]+]]
; MIR-NEXT: value: '<4 x float> splat (float -3.000000e+00)'
; MIR: body:
; MIR: %[[ARG:[0-9]+]]:vr128 = COPY $xmm1
; MIR: %[[NUM:[0-9]+]]:vr128 = COPY $xmm0
; MIR: %[[EST:[0-9]+]]:vr128 = {{.*}}RSQRTPSr %[[ARG]]
; MIR: %[[HALF_LOAD:[0-9]+]]:vr128 = MOVAPSrm $noreg, 1, $noreg, %const.[[HALF]], $noreg
; MIR: %[[LHS1:[0-9]+]]:vr128 = {{.*}}MULPSrr %[[EST]], %[[HALF_LOAD]],
; MIR: %[[AE1:[0-9]+]]:vr128 = {{.*}}MULPSrr %[[ARG]], %[[EST]],
; MIR: %[[AEE1:[0-9]+]]:vr128 = {{.*}}MULPSrr %[[AE1]], %[[EST]],
; MIR: %[[THREE_LOAD:[0-9]+]]:vr128 = MOVAPSrm $noreg, 1, $noreg, %const.[[THREE]], $noreg
; MIR: %[[RHS1:[0-9]+]]:vr128 = {{.*}}ADDPSrr %[[AEE1]], %[[THREE_LOAD]],
; MIR: %[[REF1:[0-9]+]]:vr128 = {{.*}}MULPSrr %[[LHS1]], {{(killed )?}}%[[RHS1]],
; MIR: %[[AE2:[0-9]+]]:vr128 = {{.*}}MULPSrr %[[ARG]], %[[REF1]],
; MIR: %[[AEE2:[0-9]+]]:vr128 = {{.*}}MULPSrr %[[AE2]], %[[REF1]],
; MIR: %[[RHS2:[0-9]+]]:vr128 = {{.*}}ADDPSrr %[[AEE2]], %[[THREE_LOAD]],
; MIR: %[[LHS2:[0-9]+]]:vr128 = {{.*}}MULPSrr %[[REF1]], %[[HALF_LOAD]],
; MIR: %[[REF2:[0-9]+]]:vr128 = {{.*}}MULPSrr %[[LHS2]], {{(killed )?}}%[[RHS2]],
; MIR: %[[RESULT:[0-9]+]]:vr128 = {{.*}}MULPSrr %[[NUM]], {{(killed )?}}%[[REF2]],
; MIR-NEXT: $xmm0 = COPY %[[RESULT]]
; MIR-NEXT: RET 0, $xmm0
; ASM-LABEL: rsqrt_v4_steps_2:
; ASM:       rsqrtps
; ASM-NOT:   {{^[[:space:]]+sqrtps}}
; ASM-NOT:   divps
; ASM:       retl
define <4 x float> @rsqrt_v4_steps_2(
    <4 x float> %n, <4 x float> %x) #2 {
  %sqrt = call afn ninf <4 x float> @llvm.sqrt.v4f32(<4 x float> %x)
  %q = fdiv arcp ninf <4 x float> %n, %sqrt
  ret <4 x float> %q
}

; ASM-LABEL: rsqrt_v8_steps_0:
; ASM:         rsqrtps %xmm2, %xmm2
; ASM-NEXT:    rsqrtps 16(%esp), %xmm3
; ASM-NEXT:    mulps %xmm3, %xmm1
; ASM-NEXT:    mulps %xmm2, %xmm0
; ASM:         retl
define <8 x float> @rsqrt_v8_steps_0(
    <8 x float> %n, <8 x float> %x) #3 {
  %sqrt = call afn ninf <8 x float> @llvm.sqrt.v8f32(<8 x float> %x)
  %q = fdiv arcp ninf <8 x float> %n, %sqrt
  ret <8 x float> %q
}

; MIR-LABEL: name: rsqrt_v8_default
; MIR: constants:
; MIR-NEXT: - id: [[HALF:[0-9]+]]
; MIR-NEXT: value: '<4 x float> splat (float -5.000000e-01)'
; MIR: - id: [[THREE:[0-9]+]]
; MIR-NEXT: value: '<4 x float> splat (float -3.000000e+00)'
; MIR: body:
; MIR: %[[ARG_LO:[0-9]+]]:vr128 = COPY $xmm2
; MIR: %[[NUM_HI:[0-9]+]]:vr128 = COPY $xmm1
; MIR: %[[NUM_LO:[0-9]+]]:vr128 = COPY $xmm0
; MIR: %[[ARG_HI:[0-9]+]]:vr128 = MOVAPSrm %fixed-stack.{{[0-9]+}}, 1, $noreg, 0, $noreg
; MIR: %[[EST_HI:[0-9]+]]:vr128 = {{.*}}RSQRTPSr %[[ARG_HI]]
; MIR: %[[HALF_LOAD:[0-9]+]]:vr128 = MOVAPSrm $noreg, 1, $noreg, %const.[[HALF]], $noreg
; MIR: %[[LHS_HI:[0-9]+]]:vr128 = {{.*}}MULPSrr %[[EST_HI]], %[[HALF_LOAD]],
; MIR: %[[AE_HI:[0-9]+]]:vr128 = {{.*}}MULPSrr %[[ARG_HI]], %[[EST_HI]],
; MIR: %[[AEE_HI:[0-9]+]]:vr128 = {{.*}}MULPSrr %[[AE_HI]], %[[EST_HI]],
; MIR: %[[THREE_LOAD:[0-9]+]]:vr128 = MOVAPSrm $noreg, 1, $noreg, %const.[[THREE]], $noreg
; MIR: %[[RHS_HI:[0-9]+]]:vr128 = {{.*}}ADDPSrr %[[AEE_HI]], %[[THREE_LOAD]],
; MIR: %[[REF_HI:[0-9]+]]:vr128 = {{.*}}MULPSrr %[[LHS_HI]], {{(killed )?}}%[[RHS_HI]],
; MIR: %[[EST_LO:[0-9]+]]:vr128 = {{.*}}RSQRTPSr %[[ARG_LO]]
; MIR: %[[LHS_LO:[0-9]+]]:vr128 = {{.*}}MULPSrr %[[EST_LO]], %[[HALF_LOAD]],
; MIR: %[[AE_LO:[0-9]+]]:vr128 = {{.*}}MULPSrr %[[ARG_LO]], %[[EST_LO]],
; MIR: %[[AEE_LO:[0-9]+]]:vr128 = {{.*}}MULPSrr %[[AE_LO]], %[[EST_LO]],
; MIR: %[[RHS_LO:[0-9]+]]:vr128 = {{.*}}ADDPSrr %[[AEE_LO]], %[[THREE_LOAD]],
; MIR: %[[REF_LO:[0-9]+]]:vr128 = {{.*}}MULPSrr %[[LHS_LO]], {{(killed )?}}%[[RHS_LO]],
; MIR: %[[RESULT_LO:[0-9]+]]:vr128 = {{.*}}MULPSrr %[[NUM_LO]], {{(killed )?}}%[[REF_LO]],
; MIR-NEXT: %[[RESULT_HI:[0-9]+]]:vr128 = {{.*}}MULPSrr %[[NUM_HI]], {{(killed )?}}%[[REF_HI]],
; MIR-NEXT: $xmm0 = COPY %[[RESULT_LO]]
; MIR-NEXT: $xmm1 = COPY %[[RESULT_HI]]
; MIR-NEXT: RET 0, $xmm0, $xmm1
; ASM-LABEL: rsqrt_v8_default:
; ASM-COUNT-2: rsqrtps
; ASM-NOT:   {{^[[:space:]]+sqrtps}}
; ASM-NOT:   divps
; ASM:       retl
define <8 x float> @rsqrt_v8_default(
    <8 x float> %n, <8 x float> %x) #4 {
  %sqrt = call afn ninf <8 x float> @llvm.sqrt.v8f32(<8 x float> %x)
  %q = fdiv arcp ninf <8 x float> %n, %sqrt
  ret <8 x float> %q
}

; MIR-LABEL: name: rsqrt_v8_steps_2
; MIR: constants:
; MIR-NEXT: - id: [[HALF:[0-9]+]]
; MIR-NEXT: value: '<4 x float> splat (float -5.000000e-01)'
; MIR: - id: [[THREE:[0-9]+]]
; MIR-NEXT: value: '<4 x float> splat (float -3.000000e+00)'
; MIR: body:
; MIR: %[[ARG_LO:[0-9]+]]:vr128 = COPY $xmm2
; MIR: %[[NUM_HI:[0-9]+]]:vr128 = COPY $xmm1
; MIR: %[[NUM_LO:[0-9]+]]:vr128 = COPY $xmm0
; MIR: %[[ARG_HI:[0-9]+]]:vr128 = MOVAPSrm %fixed-stack.{{[0-9]+}}, 1, $noreg, 0, $noreg
; MIR: %[[EST_HI:[0-9]+]]:vr128 = {{.*}}RSQRTPSr %[[ARG_HI]]
; MIR: %[[HALF_LOAD:[0-9]+]]:vr128 = MOVAPSrm $noreg, 1, $noreg, %const.[[HALF]], $noreg
; MIR: %[[LHS_HI1:[0-9]+]]:vr128 = {{.*}}MULPSrr %[[EST_HI]], %[[HALF_LOAD]],
; MIR: %[[AE_HI1:[0-9]+]]:vr128 = {{.*}}MULPSrr %[[ARG_HI]], %[[EST_HI]],
; MIR: %[[AEE_HI1:[0-9]+]]:vr128 = {{.*}}MULPSrr %[[AE_HI1]], %[[EST_HI]],
; MIR: %[[THREE_LOAD:[0-9]+]]:vr128 = MOVAPSrm $noreg, 1, $noreg, %const.[[THREE]], $noreg
; MIR: %[[RHS_HI1:[0-9]+]]:vr128 = {{.*}}ADDPSrr %[[AEE_HI1]], %[[THREE_LOAD]],
; MIR: %[[REF_HI1:[0-9]+]]:vr128 = {{.*}}MULPSrr %[[LHS_HI1]], {{(killed )?}}%[[RHS_HI1]],
; MIR: %[[AE_HI2:[0-9]+]]:vr128 = {{.*}}MULPSrr %[[ARG_HI]], %[[REF_HI1]],
; MIR: %[[AEE_HI2:[0-9]+]]:vr128 = {{.*}}MULPSrr %[[AE_HI2]], %[[REF_HI1]],
; MIR: %[[RHS_HI2:[0-9]+]]:vr128 = {{.*}}ADDPSrr %[[AEE_HI2]], %[[THREE_LOAD]],
; MIR: %[[LHS_HI2:[0-9]+]]:vr128 = {{.*}}MULPSrr %[[REF_HI1]], %[[HALF_LOAD]],
; MIR: %[[REF_HI2:[0-9]+]]:vr128 = {{.*}}MULPSrr %[[LHS_HI2]], {{(killed )?}}%[[RHS_HI2]],
; MIR: %[[EST_LO:[0-9]+]]:vr128 = {{.*}}RSQRTPSr %[[ARG_LO]]
; MIR: %[[LHS_LO1:[0-9]+]]:vr128 = {{.*}}MULPSrr %[[EST_LO]], %[[HALF_LOAD]],
; MIR: %[[AE_LO1:[0-9]+]]:vr128 = {{.*}}MULPSrr %[[ARG_LO]], %[[EST_LO]],
; MIR: %[[AEE_LO1:[0-9]+]]:vr128 = {{.*}}MULPSrr %[[AE_LO1]], %[[EST_LO]],
; MIR: %[[RHS_LO1:[0-9]+]]:vr128 = {{.*}}ADDPSrr %[[AEE_LO1]], %[[THREE_LOAD]],
; MIR: %[[REF_LO1:[0-9]+]]:vr128 = {{.*}}MULPSrr %[[LHS_LO1]], {{(killed )?}}%[[RHS_LO1]],
; MIR: %[[AE_LO2:[0-9]+]]:vr128 = {{.*}}MULPSrr %[[ARG_LO]], %[[REF_LO1]],
; MIR: %[[AEE_LO2:[0-9]+]]:vr128 = {{.*}}MULPSrr %[[AE_LO2]], %[[REF_LO1]],
; MIR: %[[RHS_LO2:[0-9]+]]:vr128 = {{.*}}ADDPSrr %[[AEE_LO2]], %[[THREE_LOAD]],
; MIR: %[[LHS_LO2:[0-9]+]]:vr128 = {{.*}}MULPSrr %[[REF_LO1]], %[[HALF_LOAD]],
; MIR: %[[REF_LO2:[0-9]+]]:vr128 = {{.*}}MULPSrr %[[LHS_LO2]], {{(killed )?}}%[[RHS_LO2]],
; MIR: %[[RESULT_LO:[0-9]+]]:vr128 = {{.*}}MULPSrr %[[NUM_LO]], {{(killed )?}}%[[REF_LO2]],
; MIR-NEXT: %[[RESULT_HI:[0-9]+]]:vr128 = {{.*}}MULPSrr %[[NUM_HI]], {{(killed )?}}%[[REF_HI2]],
; MIR-NEXT: $xmm0 = COPY %[[RESULT_LO]]
; MIR-NEXT: $xmm1 = COPY %[[RESULT_HI]]
; MIR-NEXT: RET 0, $xmm0, $xmm1
; ASM-LABEL: rsqrt_v8_steps_2:
; ASM-COUNT-2: rsqrtps
; ASM-NOT:   {{^[[:space:]]+sqrtps}}
; ASM-NOT:   divps
; ASM:       retl
define <8 x float> @rsqrt_v8_steps_2(
    <8 x float> %n, <8 x float> %x) #5 {
  %sqrt = call afn ninf <8 x float> @llvm.sqrt.v8f32(<8 x float> %x)
  %q = fdiv arcp ninf <8 x float> %n, %sqrt
  ret <8 x float> %q
}

; ASM-LABEL: div_v4_steps_0:
; ASM:       # %bb.0:
; ASM-NEXT:    rcpps %xmm1, %xmm1
; ASM-NEXT:    mulps %xmm1, %xmm0
; ASM-NEXT:    retl
define <4 x float> @div_v4_steps_0(
    <4 x float> %n, <4 x float> %d) #6 {
  %q = fdiv arcp ninf <4 x float> %n, %d
  ret <4 x float> %q
}

; ASM-LABEL: div_v4_steps_1:
; ASM:       # %bb.0:
; ASM-NEXT:    rcpps %xmm1, %xmm2
; ASM-NEXT:    movaps %xmm0, %xmm3
; ASM-NEXT:    mulps %xmm2, %xmm3
; ASM-NEXT:    mulps %xmm3, %xmm1
; ASM-NEXT:    subps %xmm1, %xmm0
; ASM-NEXT:    mulps %xmm2, %xmm0
; ASM-NEXT:    addps %xmm3, %xmm0
; ASM-NEXT:    retl
define <4 x float> @div_v4_steps_1(
    <4 x float> %n, <4 x float> %d) #7 {
  %q = fdiv arcp ninf <4 x float> %n, %d
  ret <4 x float> %q
}

; MIR-LABEL: name: div_v4_steps_2
; MIR: constants:
; MIR-NEXT: - id: [[ONE:[0-9]+]]
; MIR-NEXT: value: '<4 x float> splat (float 1.000000e+00)'
; MIR: body:
; MIR: %[[DIVISOR:[0-9]+]]:vr128 = COPY $xmm1
; MIR: %[[NUM:[0-9]+]]:vr128 = COPY $xmm0
; MIR: %[[EST:[0-9]+]]:vr128 = {{.*}}RCPPSr %[[DIVISOR]]
; MIR: %[[ARG_EST1:[0-9]+]]:vr128 = {{.*}}MULPSrr %[[DIVISOR]], %[[EST]],
; MIR: %[[ONE_LOAD:[0-9]+]]:vr128 = MOVAPSrm $noreg, 1, $noreg, %const.[[ONE]], $noreg
; MIR: %[[ERR1:[0-9]+]]:vr128 = {{.*}}SUBPSrr %[[ONE_LOAD]], {{(killed )?}}%[[ARG_EST1]],
; MIR: %[[CORR1:[0-9]+]]:vr128 = {{.*}}MULPSrr %[[EST]], {{(killed )?}}%[[ERR1]],
; MIR: %[[REF1:[0-9]+]]:vr128 = {{.*}}ADDPSrr %[[EST]], {{(killed )?}}%[[CORR1]],
; MIR: %[[NUM_EST:[0-9]+]]:vr128 = {{.*}}MULPSrr %[[NUM]], %[[REF1]],
; MIR: %[[ARG_NUM_EST:[0-9]+]]:vr128 = {{.*}}MULPSrr %[[DIVISOR]], %[[NUM_EST]],
; MIR: %[[ERR2:[0-9]+]]:vr128 = {{.*}}SUBPSrr %[[NUM]], {{(killed )?}}%[[ARG_NUM_EST]],
; MIR: %[[CORR2:[0-9]+]]:vr128 = {{.*}}MULPSrr %[[REF1]], {{(killed )?}}%[[ERR2]],
; MIR: %[[RESULT:[0-9]+]]:vr128 = {{.*}}ADDPSrr %[[NUM_EST]], {{(killed )?}}%[[CORR2]],
; MIR-NEXT: $xmm0 = COPY %[[RESULT]]
; MIR-NEXT: RET 0, $xmm0
; ASM-LABEL: div_v4_steps_2:
; ASM:       rcpps
; ASM-NOT:   divps
; ASM:       retl
define <4 x float> @div_v4_steps_2(
    <4 x float> %n, <4 x float> %d) #8 {
  %q = fdiv arcp ninf <4 x float> %n, %d
  ret <4 x float> %q
}

; ASM-LABEL: div_v8_steps_0:
; ASM:         rcpps 16(%esp), %xmm3
; ASM-NEXT:    mulps %xmm3, %xmm1
; ASM-NEXT:    rcpps %xmm2, %xmm2
; ASM-NEXT:    mulps %xmm2, %xmm0
; ASM:         retl
define <8 x float> @div_v8_steps_0(
    <8 x float> %n, <8 x float> %d) #9 {
  %q = fdiv arcp ninf <8 x float> %n, %d
  ret <8 x float> %q
}

; ASM-LABEL: div_v8_steps_1:
; ASM:         rcpps %xmm2, %xmm3
; ASM-NEXT:    movaps %xmm0, %xmm4
; ASM-NEXT:    mulps %xmm3, %xmm4
; ASM-NEXT:    mulps %xmm4, %xmm2
; ASM-NEXT:    subps %xmm2, %xmm0
; ASM-NEXT:    mulps %xmm3, %xmm0
; ASM-NEXT:    addps %xmm4, %xmm0
; ASM-NEXT:    movaps 16(%esp), %xmm2
; ASM-NEXT:    rcpps %xmm2, %xmm3
; ASM-NEXT:    movaps %xmm1, %xmm4
; ASM-NEXT:    mulps %xmm3, %xmm4
; ASM-NEXT:    mulps %xmm4, %xmm2
; ASM-NEXT:    subps %xmm2, %xmm1
; ASM-NEXT:    mulps %xmm3, %xmm1
; ASM-NEXT:    addps %xmm4, %xmm1
; ASM:         retl
define <8 x float> @div_v8_steps_1(
    <8 x float> %n, <8 x float> %d) #10 {
  %q = fdiv arcp ninf <8 x float> %n, %d
  ret <8 x float> %q
}

; MIR-LABEL: name: div_v8_steps_2
; MIR: constants:
; MIR-NEXT: - id: [[ONE:[0-9]+]]
; MIR-NEXT: value: '<4 x float> splat (float 1.000000e+00)'
; MIR: body:
; MIR: %[[DIVISOR_LO:[0-9]+]]:vr128 = COPY $xmm2
; MIR: %[[NUM_HI:[0-9]+]]:vr128 = COPY $xmm1
; MIR: %[[NUM_LO:[0-9]+]]:vr128 = COPY $xmm0
; MIR: %[[EST_LO:[0-9]+]]:vr128 = {{.*}}RCPPSr %[[DIVISOR_LO]]
; MIR: %[[ARG_EST_LO1:[0-9]+]]:vr128 = {{.*}}MULPSrr %[[DIVISOR_LO]], %[[EST_LO]],
; MIR: %[[ONE_LOAD:[0-9]+]]:vr128 = MOVAPSrm $noreg, 1, $noreg, %const.[[ONE]], $noreg
; MIR: %[[ERR_LO1:[0-9]+]]:vr128 = {{.*}}SUBPSrr %[[ONE_LOAD]], {{(killed )?}}%[[ARG_EST_LO1]],
; MIR: %[[CORR_LO1:[0-9]+]]:vr128 = {{.*}}MULPSrr %[[EST_LO]], {{(killed )?}}%[[ERR_LO1]],
; MIR: %[[REF_LO1:[0-9]+]]:vr128 = {{.*}}ADDPSrr %[[EST_LO]], {{(killed )?}}%[[CORR_LO1]],
; MIR: %[[NUM_EST_LO:[0-9]+]]:vr128 = {{.*}}MULPSrr %[[NUM_LO]], %[[REF_LO1]],
; MIR: %[[ARG_NUM_EST_LO:[0-9]+]]:vr128 = {{.*}}MULPSrr %[[DIVISOR_LO]], %[[NUM_EST_LO]],
; MIR: %[[ERR_LO2:[0-9]+]]:vr128 = {{.*}}SUBPSrr %[[NUM_LO]], {{(killed )?}}%[[ARG_NUM_EST_LO]],
; MIR: %[[CORR_LO2:[0-9]+]]:vr128 = {{.*}}MULPSrr %[[REF_LO1]], {{(killed )?}}%[[ERR_LO2]],
; MIR: %[[RESULT_LO:[0-9]+]]:vr128 = {{.*}}ADDPSrr %[[NUM_EST_LO]], {{(killed )?}}%[[CORR_LO2]],
; MIR: %[[DIVISOR_HI:[0-9]+]]:vr128 = MOVAPSrm %fixed-stack.{{[0-9]+}}, 1, $noreg, 0, $noreg
; MIR: %[[EST_HI:[0-9]+]]:vr128 = {{.*}}RCPPSr %[[DIVISOR_HI]]
; MIR: %[[ARG_EST_HI1:[0-9]+]]:vr128 = {{.*}}MULPSrr %[[DIVISOR_HI]], %[[EST_HI]],
; MIR: %[[ERR_HI1:[0-9]+]]:vr128 = {{.*}}SUBPSrr %[[ONE_LOAD]], {{(killed )?}}%[[ARG_EST_HI1]],
; MIR: %[[CORR_HI1:[0-9]+]]:vr128 = {{.*}}MULPSrr %[[EST_HI]], {{(killed )?}}%[[ERR_HI1]],
; MIR: %[[REF_HI1:[0-9]+]]:vr128 = {{.*}}ADDPSrr %[[EST_HI]], {{(killed )?}}%[[CORR_HI1]],
; MIR: %[[NUM_EST_HI:[0-9]+]]:vr128 = {{.*}}MULPSrr %[[NUM_HI]], %[[REF_HI1]],
; MIR: %[[ARG_NUM_EST_HI:[0-9]+]]:vr128 = {{.*}}MULPSrr %[[DIVISOR_HI]], %[[NUM_EST_HI]],
; MIR: %[[ERR_HI2:[0-9]+]]:vr128 = {{.*}}SUBPSrr %[[NUM_HI]], {{(killed )?}}%[[ARG_NUM_EST_HI]],
; MIR: %[[CORR_HI2:[0-9]+]]:vr128 = {{.*}}MULPSrr %[[REF_HI1]], {{(killed )?}}%[[ERR_HI2]],
; MIR: %[[RESULT_HI:[0-9]+]]:vr128 = {{.*}}ADDPSrr %[[NUM_EST_HI]], {{(killed )?}}%[[CORR_HI2]],
; MIR-NEXT: $xmm0 = COPY %[[RESULT_LO]]
; MIR-NEXT: $xmm1 = COPY %[[RESULT_HI]]
; MIR-NEXT: RET 0, $xmm0, $xmm1
; ASM-LABEL: div_v8_steps_2:
; ASM-COUNT-2: rcpps
; ASM-NOT:   divps
; ASM:       retl
define <8 x float> @div_v8_steps_2(
    <8 x float> %n, <8 x float> %d) #11 {
  %q = fdiv arcp ninf <8 x float> %n, %d
  ret <8 x float> %q
}

attributes #0 = {
  "reciprocal-estimates"="vec-sqrtf:0"
  "target-features"="+sse,-sse2,-x87"
}
attributes #1 = {
  "reciprocal-estimates"="vec-sqrtf"
  "target-features"="+sse,-sse2,-x87"
}
attributes #2 = {
  "reciprocal-estimates"="vec-sqrtf:2"
  "target-features"="+sse,-sse2,-x87"
}
attributes #3 = {
  "reciprocal-estimates"="vec-sqrtf:0"
  "target-features"="+sse,-sse2,-x87"
}
attributes #4 = {
  "reciprocal-estimates"="vec-sqrtf"
  "target-features"="+sse,-sse2,-x87"
}
attributes #5 = {
  "reciprocal-estimates"="vec-sqrtf:2"
  "target-features"="+sse,-sse2,-x87"
}
attributes #6 = {
  "reciprocal-estimates"="vec-divf:0"
  "target-features"="+sse,-sse2,-x87"
}
attributes #7 = {
  "reciprocal-estimates"="vec-divf:1"
  "target-features"="+sse,-sse2,-x87"
}
attributes #8 = {
  "reciprocal-estimates"="vec-divf:2"
  "target-features"="+sse,-sse2,-x87"
}
attributes #9 = {
  "reciprocal-estimates"="vec-divf:0"
  "target-features"="+sse,-sse2,-x87"
}
attributes #10 = {
  "reciprocal-estimates"="vec-divf:1"
  "target-features"="+sse,-sse2,-x87"
}
attributes #11 = {
  "reciprocal-estimates"="vec-divf:2"
  "target-features"="+sse,-sse2,-x87"
}

;--- fallback.ll

target triple = "i686-unknown-linux-gnu"

; FALLBACK-LABEL: fallback_sqrt_v4:
; FALLBACK: {{^[[:space:]]+sqrtps[[:space:]]}}
; FALLBACK: {{^[[:space:]]+rcpps[[:space:]]}}
define <4 x float> @fallback_sqrt_v4(
    <4 x float> %n, <4 x float> %x) #0 {
  %sqrt = call afn ninf <4 x float> @llvm.sqrt.v4f32(<4 x float> %x)
  %q = fdiv arcp ninf <4 x float> %n, %sqrt
  ret <4 x float> %q
}

; FALLBACK-LABEL: fallback_div_v4:
; FALLBACK: {{^[[:space:]]+divps[[:space:]]}}
define <4 x float> @fallback_div_v4(
    <4 x float> %n, <4 x float> %d) #1 {
  %q = fdiv arcp ninf <4 x float> %n, %d
  ret <4 x float> %q
}

attributes #0 = {
  "reciprocal-estimates"="!vec-sqrtf"
  "target-features"="+sse,-sse2,-x87"
}
attributes #1 = {
  "reciprocal-estimates"="!vec-divf"
  "target-features"="+sse,-sse2,-x87"
}
