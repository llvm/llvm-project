; NOTE: Do not autogenerate
; RUN: split-file %s %t
; RUN: llc %t/estimate.ll -o - | FileCheck %s --check-prefix=ASM --enable-var-scope --implicit-check-not='{{^[[:space:]]+sqrtps[[:space:]]}}' --implicit-check-not='{{^[[:space:]]+divps[[:space:]]}}'
; RUN: llc %t/fallback.ll -o - | FileCheck %s --check-prefix=FALLBACK

; On i686 with SSE1 but no SSE2 or x87, v4f32 is legal while scalar f32
; requires softening. The v4f32 cases create refinement constants before type
; legalization. The v8f32 cases split first and create them during the
; AfterLegalizeTypes combine. Assembly checks preserve working boundaries and
; exclude fallback. The fallback input positively establishes the excluded
; native operations. The companion PIC test checks refinement constant values,
; relocation form, and local arithmetic use.

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
