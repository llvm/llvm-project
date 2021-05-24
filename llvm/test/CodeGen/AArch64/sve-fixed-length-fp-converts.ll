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
; vector uint_to_fp i8 -> f32
; AArch64 doesn't have a direct vector->f32 conversion instructions for
; elements smaller than i32, so make sure inputs are promoted to i32 first.
;

define void @uitofp_v4i8_v4f32(<4 x i8>* %in, <4 x float>* %out) #0 {
; CHECK-LABEL: uitofp_v4i8_v4f32:
; CHECK-COUNT-1: ucvt
  %vec = load <4 x i8>, <4 x i8>* %in
  %conv = uitofp <4 x i8> %vec to <4 x float>
  store <4 x float> %conv, <4 x float>* %out
  ret void
}

define void @uitofp_v8i8_v8f32(<8 x i8>* %in, <8 x float>* %out) #0 {
; CHECK-LABEL: uitofp_v8i8_v8f32:
; CHECK-COUNT-8: ucvt
  %vec = load <8 x i8>, <8 x i8>* %in
  %conv = uitofp <8 x i8> %vec to <8 x float>
  store <8 x float> %conv, <8 x float>* %out
  ret void
}

define void @uitofp_v16i8_v16f32(<16 x i8>* %in, <16 x float>* %out) #0 {
; CHECK-LABEL: uitofp_v16i8_v16f32:
; CHECK-COUNT-16: ucvt
  %vec = load <16 x i8>, <16 x i8>* %in
  %conv = uitofp <16 x i8> %vec to <16 x float>
  store <16 x float> %conv, <16 x float>* %out
  ret void
}

define void @uitofp_v32i8_v32f32(<32 x i8>* %in, <32 x float>* %out) #0 {
; CHECK-LABEL: uitofp_v32i8_v32f32:
; CHECK-COUNT-32: ucvt
  %vec = load <32 x i8>, <32 x i8>* %in
  %conv = uitofp <32 x i8> %vec to <32 x float>
  store <32 x float> %conv, <32 x float>* %out
  ret void
}

attributes #0 = { nounwind "target-features"="+sve" }
