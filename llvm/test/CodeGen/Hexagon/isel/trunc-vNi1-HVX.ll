; RUN: llc --mtriple=hexagon -mattr=+hvxv79,+hvx-length128b < %s | FileCheck %s

;
; Exhaustive tests for truncation of HVX vectors to boolean (i1) vectors.
; In 128-byte HVX mode:
;   Single register (HvxVR): v128i8, v64i16, v32i32 (1024 bits)
;   Pair register   (HvxWR): v256i8, v128i16, v64i32 (2048 bits)
;   Predicates      (HvxQR): v128i1, v64i1, v32i1
;
; Tests cover:
;   - Full HVX width (single register): tablegen V6_vandvrt patterns
;   - HVX pairs: tablegen Combineq patterns
;   - Sub-HVX width: C++ WidenHvxTruncateToBool (widen + trunc + extract)
;
; Very small vectors (v8i16, v8i32) are scalarized by the legalizer
; and do not exercise HVX paths, so they are not included here.
;

; ===========================================================================
; Category 1: Full HVX single-register width -> HVX predicate (tablegen)
;   These match tablegen patterns VecQ{8,16,32} (trunc HVI{8,16,32})
;   directly via V6_vandvrt.
; ===========================================================================

; CHECK-LABEL: trunc_v128i8_to_v128i1:
; CHECK: q{{[0-9]+}} = vand(v{{[0-9]+}},r{{[0-9]+}})
define void @trunc_v128i8_to_v128i1(<128 x i8> %v, ptr %p) {
  %1 = trunc <128 x i8> %v to <128 x i1>
  store <128 x i1> %1, ptr %p, align 16
  ret void
}

; CHECK-LABEL: trunc_v64i16_to_v64i1:
; CHECK: q{{[0-9]+}} = vand(v{{[0-9]+}},r{{[0-9]+}})
define i64 @trunc_v64i16_to_v64i1(<64 x i16> %v) {
  %1 = trunc <64 x i16> %v to <64 x i1>
  %2 = bitcast <64 x i1> %1 to i64
  ret i64 %2
}

; CHECK-LABEL: trunc_v32i32_to_v32i1:
; CHECK: q{{[0-9]+}} = vand(v{{[0-9]+}},r{{[0-9]+}})
define i32 @trunc_v32i32_to_v32i1(<32 x i32> %v) {
  %1 = trunc <32 x i32> %v to <32 x i1>
  %2 = bitcast <32 x i1> %1 to i32
  ret i32 %2
}

; ===========================================================================
; Category 2: HVX pair -> HVX predicate (tablegen, via Combineq)
;   These split the pair, apply V6_vandvrt to each half, and combine
;   the predicate halves.
; ===========================================================================

; CHECK-LABEL: trunc_v64i32_to_v64i1:
; CHECK-DAG: q{{[0-9]+}} = vand(v{{[0-9]+}},r{{[0-9]+}})
; CHECK-DAG: q{{[0-9]+}} = vand(v{{[0-9]+}},r{{[0-9]+}})
define void @trunc_v64i32_to_v64i1(<64 x i32> %v, ptr %p) {
  %1 = trunc <64 x i32> %v to <64 x i1>
  store <64 x i1> %1, ptr %p, align 8
  ret void
}

; CHECK-LABEL: trunc_v128i16_to_v128i1:
; CHECK-DAG: q{{[0-9]+}} = vand(v{{[0-9]+}},r{{[0-9]+}})
; CHECK-DAG: q{{[0-9]+}} = vand(v{{[0-9]+}},r{{[0-9]+}})
define void @trunc_v128i16_to_v128i1(<128 x i16> %v, ptr %p) {
  %1 = trunc <128 x i16> %v to <128 x i1>
  store <128 x i1> %1, ptr %p, align 16
  ret void
}

; ===========================================================================
; Category 3: Sub-HVX width -> predicate (C++ WidenHvxTruncateToBool)
;   Input is narrower than a single HVX register. The input is widened
;   to full HVX width, truncated to widened bool, then the result
;   subvector is extracted.
; ===========================================================================

; --- Half HVX width (512-bit input) ---

; CHECK-LABEL: trunc_v64i8_to_v64i1:
; CHECK: q{{[0-9]+}} = vand(v{{[0-9]+}},r{{[0-9]+}})
define i64 @trunc_v64i8_to_v64i1(<64 x i8> %v) {
  %1 = trunc <64 x i8> %v to <64 x i1>
  %2 = bitcast <64 x i1> %1 to i64
  ret i64 %2
}

; CHECK-LABEL: trunc_v32i16_to_v32i1:
; CHECK: q{{[0-9]+}} = vand(v{{[0-9]+}},r{{[0-9]+}})
define i32 @trunc_v32i16_to_v32i1(<32 x i16> %v) {
  %1 = trunc <32 x i16> %v to <32 x i1>
  %2 = bitcast <32 x i1> %1 to i32
  ret i32 %2
}

; CHECK-LABEL: trunc_v16i32_to_v16i1:
; CHECK: q{{[0-9]+}} = vand(v{{[0-9]+}},r{{[0-9]+}})
define i16 @trunc_v16i32_to_v16i1(<16 x i32> %v) {
  %1 = trunc <16 x i32> %v to <16 x i1>
  %2 = bitcast <16 x i1> %1 to i16
  ret i16 %2
}

; --- Quarter HVX width (256-bit input) ---

; CHECK-LABEL: trunc_v32i8_to_v32i1:
; CHECK: q{{[0-9]+}} = vand(v{{[0-9]+}},r{{[0-9]+}})
define i32 @trunc_v32i8_to_v32i1(<32 x i8> %v) {
  %1 = trunc <32 x i8> %v to <32 x i1>
  %2 = bitcast <32 x i1> %1 to i32
  ret i32 %2
}

; CHECK-LABEL: trunc_v16i16_to_v16i1:
; CHECK: q{{[0-9]+}} = vand(v{{[0-9]+}},r{{[0-9]+}})
define i16 @trunc_v16i16_to_v16i1(<16 x i16> %v) {
  %1 = trunc <16 x i16> %v to <16 x i1>
  %2 = bitcast <16 x i1> %1 to i16
  ret i16 %2
}

; --- Eighth HVX width and smaller (128-bit input) ---

; CHECK-LABEL: trunc_v16i8_to_v16i1:
; CHECK: q{{[0-9]+}} = vand(v{{[0-9]+}},r{{[0-9]+}})
define i16 @trunc_v16i8_to_v16i1(<16 x i8> %v) {
  %1 = trunc <16 x i8> %v to <16 x i1>
  %2 = bitcast <16 x i1> %1 to i16
  ret i16 %2
}

; CHECK-LABEL: trunc_v4i32_to_v4i1:
; CHECK: q{{[0-9]+}} = vand(v{{[0-9]+}},r{{[0-9]+}})
define i4 @trunc_v4i32_to_v4i1(<4 x i32> %v) {
  %1 = trunc <4 x i32> %v to <4 x i1>
  %2 = bitcast <4 x i1> %1 to i4
  ret i4 %2
}
