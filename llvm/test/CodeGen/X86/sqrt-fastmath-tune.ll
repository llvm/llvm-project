; RUN: llc < %s -mtriple=x86_64-unknown-unknown -O2 -mcpu=nehalem     | FileCheck %s --check-prefix=SCALAR-EST --check-prefix=VECTOR-EST
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -O2 -mcpu=sandybridge | FileCheck %s --check-prefix=SCALAR-ACC --check-prefix=VECTOR-EST
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -O2 -mcpu=broadwell   | FileCheck %s --check-prefix=SCALAR-ACC --check-prefix=VECTOR-EST
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -O2 -mcpu=skylake     | FileCheck %s --check-prefix=SCALAR-ACC --check-prefix=VECTOR-ACC

; RUN: llc < %s -mtriple=x86_64-unknown-unknown -O2 -mattr=+fast-scalar-fsqrt,-fast-vector-fsqrt | FileCheck %s --check-prefix=SCALAR-ACC --check-prefix=VECTOR-EST
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -O2 -mattr=-fast-scalar-fsqrt,+fast-vector-fsqrt | FileCheck %s --check-prefix=SCALAR-EST --check-prefix=VECTOR-ACC

declare float @llvm.sqrt.f32(float) #0
declare <4 x float> @llvm.sqrt.v4f32(<4 x float>) #0
declare <8 x float> @llvm.sqrt.v8f32(<8 x float>) #0

define float @foo_x1(float %f) #0 {
; SCALAR-EST-LABEL: foo_x1:
; SCALAR-EST:       # %bb.0:
; SCALAR-EST-NEXT:    rsqrtss %xmm0
; SCALAR-EST:         retq
;
; SCALAR-ACC-LABEL: foo_x1:
; SCALAR-ACC:       # %bb.0:
; SCALAR-ACC-NEXT:    {{^ *v?sqrtss %xmm0}}
; SCALAR-ACC-NEXT:    retq
  %call = tail call float @llvm.sqrt.f32(float %f) #1
  ret float %call
}

define <4 x float> @foo_x4(<4 x float> %f) #0 {
; VECTOR-EST-LABEL: foo_x4:
; VECTOR-EST:       # %bb.0:
; VECTOR-EST-NEXT:    rsqrtps %xmm0
; VECTOR-EST:         retq
;
; VECTOR-ACC-LABEL: foo_x4:
; VECTOR-ACC:       # %bb.0:
; VECTOR-ACC-NEXT:    {{^ *v?sqrtps %xmm0}}
; VECTOR-ACC-NEXT:    retq
  %call = tail call <4 x float> @llvm.sqrt.v4f32(<4 x float> %f) #1
  ret <4 x float> %call
}

define <8 x float> @foo_x8(<8 x float> %f) #0 {
; VECTOR-EST-LABEL: foo_x8:
; VECTOR-EST:       # %bb.0:
; VECTOR-EST-NEXT:    rsqrtps
; VECTOR-EST:         retq
;
; VECTOR-ACC-LABEL: foo_x8:
; VECTOR-ACC:       # %bb.0:
; VECTOR-ACC-NEXT:    {{^ *v?sqrtps %[xy]mm0}}
; VECTOR-ACC-NOT:     rsqrt
; VECTOR-ACC:         retq
  %call = tail call <8 x float> @llvm.sqrt.v8f32(<8 x float> %f) #1
  ret <8 x float> %call
}

attributes #0 = { "unsafe-fp-math"="true" }
attributes #1 = { nounwind readnone }
