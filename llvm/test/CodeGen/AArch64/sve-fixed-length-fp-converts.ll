; RUN: llc -aarch64-sve-vector-bits-min=128  -asm-verbose=0 < %s | FileCheck %s -check-prefix=NO_SVE
; RUN: llc -aarch64-sve-vector-bits-min=256  -asm-verbose=0 < %s | FileCheck %s
; RUN: llc -aarch64-sve-vector-bits-min=384  -asm-verbose=0 < %s | FileCheck %s
; RUN: llc -aarch64-sve-vector-bits-min=512  -asm-verbose=0 < %s | FileCheck %s
; RUN: llc -aarch64-sve-vector-bits-min=640  -asm-verbose=0 < %s | FileCheck %s
; RUN: llc -aarch64-sve-vector-bits-min=768  -asm-verbose=0 < %s | FileCheck %s
; RUN: llc -aarch64-sve-vector-bits-min=896  -asm-verbose=0 < %s | FileCheck %s
; RUN: llc -aarch64-sve-vector-bits-min=1024 -asm-verbose=0 < %s | FileCheck %s
; RUN: llc -aarch64-sve-vector-bits-min=1152 -asm-verbose=0 < %s | FileCheck %s
; RUN: llc -aarch64-sve-vector-bits-min=1280 -asm-verbose=0 < %s | FileCheck %s
; RUN: llc -aarch64-sve-vector-bits-min=1408 -asm-verbose=0 < %s | FileCheck %s
; RUN: llc -aarch64-sve-vector-bits-min=1536 -asm-verbose=0 < %s | FileCheck %s
; RUN: llc -aarch64-sve-vector-bits-min=1664 -asm-verbose=0 < %s | FileCheck %s
; RUN: llc -aarch64-sve-vector-bits-min=1792 -asm-verbose=0 < %s | FileCheck %s
; RUN: llc -aarch64-sve-vector-bits-min=1920 -asm-verbose=0 < %s | FileCheck %s
; RUN: llc -aarch64-sve-vector-bits-min=2048 -asm-verbose=0 < %s | FileCheck %s

target triple = "aarch64-unknown-linux-gnu"

; Don't use SVE when its registers are no bigger than NEON.
; NO_SVE-NOT: z{0-9}

; NOTE: fptrunc operations bigger than NEON are expanded. These tests just
; ensure we've correctly set the operation action for fixed length vector types
; that require SVE. They'll be updated to protect their expected code generation
; when lowering it implemented.

;
; fptrunc f32 -> f16
;

define <8 x half> @fptrunc_v8f32_v8f16(<8 x float>* %in) #0 {
; CHECK-LABEL: fptrunc_v8f32_v8f16:
; CHECK-COUNT-8: fcvt h{{[0-9]+}}, s{{[0-9]+}}
; CHECK-NOT: fcvt
; CHECK: ret
  %a = load <8 x float>, <8 x float>* %in
  %b = fptrunc <8 x float> %a to <8 x half>
  ret <8 x half> %b
}

define void @fptrunc_v16f32_v16f16(<16 x float>* %in, <16 x half>* %out) #0 {
; CHECK-LABEL: fptrunc_v16f32_v16f16:
; CHECK-COUNT-16: fcvt h{{[0-9]+}}, s{{[0-9]+}}
; CHECK-NOT: fcvt
; CHECK: ret
  %a = load <16 x float>, <16 x float>* %in
  %b = fptrunc <16 x float> %a to <16 x half>
  store <16 x half> %b, <16 x half>* %out
  ret void
}

define void @fptrunc_v32f32_v32f16(<32 x float>* %in, <32 x half>* %out) #0 {
; CHECK-LABEL: fptrunc_v32f32_v32f16:
; CHECK-COUNT-32: fcvt h{{[0-9]+}}, s{{[0-9]+}}
; CHECK-NOT: fcvt
; CHECK: ret
  %a = load <32 x float>, <32 x float>* %in
  %b = fptrunc <32 x float> %a to <32 x half>
  store <32 x half> %b, <32 x half>* %out
  ret void
}

define void @fptrunc_v64f32_v64f16(<64 x float>* %in, <64 x half>* %out) #0 {
; CHECK-LABEL: fptrunc_v64f32_v64f16:
; CHECK-COUNT-64: fcvt h{{[0-9]+}}, s{{[0-9]+}}
; CHECK-NOT: fcvt
; CHECK: ret
  %a = load <64 x float>, <64 x float>* %in
  %b = fptrunc <64 x float> %a to <64 x half>
  store <64 x half> %b, <64 x half>* %out
  ret void
}

;
; fptrunc f64 -> f16
;

define <4 x half> @fptrunc_v4f64_v4f16(<4 x double>* %in) #0 {
; CHECK-LABEL: fptrunc_v4f64_v4f16:
; CHECK-COUNT-4: fcvt h{{[0-9]+}}, d{{[0-9]+}}
; CHECK-NOT: fcvt
; CHECK: ret
  %a = load <4 x double>, <4 x double>* %in
  %b = fptrunc <4 x double> %a to <4 x half>
  ret <4 x half> %b
}

define <8 x half> @fptrunc_v8f64_v8f16(<8 x double>* %in) #0 {
; CHECK-LABEL: fptrunc_v8f64_v8f16:
; CHECK-COUNT-8: fcvt h{{[0-9]+}}, d{{[0-9]+}}
; CHECK-NOT: fcvt
; CHECK: ret
  %a = load <8 x double>, <8 x double>* %in
  %b = fptrunc <8 x double> %a to <8 x half>
  ret <8 x half> %b
}

define void @fptrunc_v16f64_v16f16(<16 x double>* %in, <16 x half>* %out) #0 {
; CHECK-LABEL: fptrunc_v16f64_v16f16:
; CHECK-COUNT-16: fcvt h{{[0-9]+}}, d{{[0-9]+}}
; CHECK-NOT: fcvt
; CHECK: ret
  %a = load <16 x double>, <16 x double>* %in
  %b = fptrunc <16 x double> %a to <16 x half>
  store <16 x half> %b, <16 x half>* %out
  ret void
}

define void @fptrunc_v32f64_v32f16(<32 x double>* %in, <32 x half>* %out) #0 {
; CHECK-LABEL: fptrunc_v32f64_v32f16:
; CHECK-COUNT-32: fcvt h{{[0-9]+}}, d{{[0-9]+}}
; CHECK-NOT: fcvt
; CHECK: ret
  %a = load <32 x double>, <32 x double>* %in
  %b = fptrunc <32 x double> %a to <32 x half>
  store <32 x half> %b, <32 x half>* %out
  ret void
}

;
; fptrunc f64 -> f32
;

define <4 x float> @fptrunc_v4f64_v4f32(<4 x double>* %in) #0 {
; CHECK-LABEL: fptrunc_v4f64_v4f32:
; CHECK-COUNT-4: fcvt s{{[0-9]+}}, d{{[0-9]+}}
; CHECK-NOT: fcvt
; CHECK: ret
  %a = load <4 x double>, <4 x double>* %in
  %b = fptrunc <4 x double> %a to <4 x float>
  ret <4 x float> %b
}

define void @fptrunc_v8f64_v8f32(<8 x double>* %in, <8 x float>* %out) #0 {
; CHECK-LABEL: fptrunc_v8f64_v8f32:
; CHECK-COUNT-8: fcvt s{{[0-9]+}}, d{{[0-9]+}}
; CHECK-NOT: fcvt
; CHECK: ret
  %a = load <8 x double>, <8 x double>* %in
  %b = fptrunc <8 x double> %a to <8 x float>
  store <8 x float> %b, <8 x float>* %out
  ret void
}

define void @fptrunc_v16f64_v16f32(<16 x double>* %in, <16 x float>* %out) #0 {
; CHECK-LABEL: fptrunc_v16f64_v16f32:
; CHECK-COUNT-16: fcvt s{{[0-9]+}}, d{{[0-9]+}}
; CHECK-NOT: fcvt
; CHECK: ret
  %a = load <16 x double>, <16 x double>* %in
  %b = fptrunc <16 x double> %a to <16 x float>
  store <16 x float> %b, <16 x float>* %out
  ret void
}

define void @fptrunc_v32f64_v32f32(<32 x double>* %in, <32 x float>* %out) #0 {
; CHECK-LABEL: fptrunc_v32f64_v32f32:
; CHECK-COUNT-32: fcvt s{{[0-9]+}}, d{{[0-9]+}}
; CHECK-NOT: fcvt
; CHECK: ret
  %a = load <32 x double>, <32 x double>* %in
  %b = fptrunc <32 x double> %a to <32 x float>
  store <32 x float> %b, <32 x float>* %out
  ret void
}

attributes #0 = { nounwind "target-features"="+sve" }
