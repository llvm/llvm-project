; NOTE: Do not autogenerate
; RUN: split-file %s %t
; RUN: llc -O3 -relocation-model=pic %t/estimate.ll -o - | FileCheck %s --check-prefix=PIC --enable-var-scope --implicit-check-not='{{^[[:space:]]+sqrtps[[:space:]]}}' --implicit-check-not='{{^[[:space:]]+divps[[:space:]]}}'
; RUN: llc -O3 -relocation-model=pic %t/fallback.ll -o - | FileCheck %s --check-prefix=FALLBACK

; This test owns emitted PIC form, not post-RA value continuity. It binds each
; pool definition to the canonical i386 call/pop/GOT expression and @GOTOFF
; spelling, with immediate arithmetic form where adjacent. The non-PIC test
; owns semantic dataflow. Whole-input exclusions reject native fallback. The
; fallback input establishes the excluded operations in the same configuration.

;--- estimate.ll

target triple = "i686-unknown-linux-gnu"

; PIC: [[$V4_THREE:\.LCPI[0-9]+_[0-9]+]]:
; PIC-NEXT: .long 0xc0400000
; PIC-NEXT: .long 0xc0400000
; PIC-NEXT: .long 0xc0400000
; PIC-NEXT: .long 0xc0400000
; PIC: [[$V4_HALF:\.LCPI[0-9]+_[0-9]+]]:
; PIC-NEXT: .long 0xbf000000
; PIC-NEXT: .long 0xbf000000
; PIC-NEXT: .long 0xbf000000
; PIC-NEXT: .long 0xbf000000
; PIC-LABEL: rsqrt_v4_default:
; PIC: calll [[PB:\.L[0-9]+\$pb]]
; PIC: [[PB]]:
; PIC-NEXT: popl [[GOT:%e[a-z0-9]+]]
; PIC: [[TMP:\.Ltmp[0-9]+]]:
; PIC-NEXT: addl $_GLOBAL_OFFSET_TABLE_+([[TMP]]-[[PB]]), [[GOT]]
; PIC: rsqrtps
; PIC: addps [[$V4_THREE]]@GOTOFF([[GOT]]), {{%xmm[0-9]+}}
; PIC: mulps [[$V4_HALF]]@GOTOFF([[GOT]]), {{%xmm[0-9]+}}
; PIC: retl
define <4 x float> @rsqrt_v4_default(
    <4 x float> %n, <4 x float> %x) #0 {
  %sqrt = call afn ninf <4 x float> @llvm.sqrt.v4f32(<4 x float> %x)
  %q = fdiv arcp ninf <4 x float> %n, %sqrt
  ret <4 x float> %q
}

; PIC: [[$V8_HALF:\.LCPI[0-9]+_[0-9]+]]:
; PIC-NEXT: .long 0xbf000000
; PIC-NEXT: .long 0xbf000000
; PIC-NEXT: .long 0xbf000000
; PIC-NEXT: .long 0xbf000000
; PIC: [[$V8_THREE:\.LCPI[0-9]+_[0-9]+]]:
; PIC-NEXT: .long 0xc0400000
; PIC-NEXT: .long 0xc0400000
; PIC-NEXT: .long 0xc0400000
; PIC-NEXT: .long 0xc0400000
; PIC-LABEL: rsqrt_v8_default:
; PIC: calll [[PB:\.L[0-9]+\$pb]]
; PIC: [[PB]]:
; PIC-NEXT: popl [[GOT:%e[a-z0-9]+]]
; PIC: [[TMP:\.Ltmp[0-9]+]]:
; PIC-NEXT: addl $_GLOBAL_OFFSET_TABLE_+([[TMP]]-[[PB]]), [[GOT]]
; PIC: rsqrtps
; PIC-NEXT: movaps [[$V8_HALF]]@GOTOFF([[GOT]]), {{%xmm[0-9]+}}
; PIC: mulps
; PIC: movaps [[$V8_THREE]]@GOTOFF([[GOT]]), [[V8_THREE_REG:%xmm[0-9]+]]
; PIC-NEXT: addps [[V8_THREE_REG]], {{%xmm[0-9]+}}
; PIC: rsqrtps
; PIC: retl
define <8 x float> @rsqrt_v8_default(
    <8 x float> %n, <8 x float> %x) #0 {
  %sqrt = call afn ninf <8 x float> @llvm.sqrt.v8f32(<8 x float> %x)
  %q = fdiv arcp ninf <8 x float> %n, %sqrt
  ret <8 x float> %q
}

; PIC: [[$V4_ONE:\.LCPI[0-9]+_[0-9]+]]:
; PIC-NEXT: .long 0x3f800000
; PIC-NEXT: .long 0x3f800000
; PIC-NEXT: .long 0x3f800000
; PIC-NEXT: .long 0x3f800000
; PIC-LABEL: div_v4_steps_2:
; PIC: calll [[PB:\.L[0-9]+\$pb]]
; PIC: [[PB]]:
; PIC-NEXT: popl [[GOT:%e[a-z0-9]+]]
; PIC: [[TMP:\.Ltmp[0-9]+]]:
; PIC-NEXT: addl $_GLOBAL_OFFSET_TABLE_+([[TMP]]-[[PB]]), [[GOT]]
; PIC: rcpps
; PIC: movaps [[$V4_ONE]]@GOTOFF([[GOT]]), [[V4_ONE_REG:%xmm[0-9]+]]
; PIC-NEXT: subps {{%xmm[0-9]+}}, [[V4_ONE_REG]]
; PIC: retl
define <4 x float> @div_v4_steps_2(
    <4 x float> %n, <4 x float> %d) #1 {
  %q = fdiv arcp ninf <4 x float> %n, %d
  ret <4 x float> %q
}

; PIC: [[$V8_ONE:\.LCPI[0-9]+_[0-9]+]]:
; PIC-NEXT: .long 0x3f800000
; PIC-NEXT: .long 0x3f800000
; PIC-NEXT: .long 0x3f800000
; PIC-NEXT: .long 0x3f800000
; PIC-LABEL: div_v8_steps_2:
; PIC: calll [[PB:\.L[0-9]+\$pb]]
; PIC: [[PB]]:
; PIC-NEXT: popl [[GOT:%e[a-z0-9]+]]
; PIC: [[TMP:\.Ltmp[0-9]+]]:
; PIC-NEXT: addl $_GLOBAL_OFFSET_TABLE_+([[TMP]]-[[PB]]), [[GOT]]
; PIC: rcpps
; PIC: movaps [[$V8_ONE]]@GOTOFF([[GOT]]), {{%xmm[0-9]+}}
; PIC: subps
; PIC: rcpps
; PIC: subps
; PIC: retl
define <8 x float> @div_v8_steps_2(
    <8 x float> %n, <8 x float> %d) #1 {
  %q = fdiv arcp ninf <8 x float> %n, %d
  ret <8 x float> %q
}

attributes #0 = {
  "reciprocal-estimates"="vec-sqrtf"
  "target-features"="+sse,-sse2,-x87"
}
attributes #1 = {
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
