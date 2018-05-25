; RUN: opt < %s -instcombine -S | FileCheck %s

; (V * C1) * C2 => V * (C1 * C2)
; Verify this doesn't fold when no fast-math-flags are specified
define <4 x float> @test_fmul(<4 x float> %V) {
; CHECK-LABEL: @test_fmul(
; CHECK-NEXT:     [[TMP1:%.*]] = fmul <4 x float> [[V:%.*]], <float 1.000000e+00, float 2.000000e+00, float 3.000000e+00, float 4.000000e+00>
; CHECK-NEXT:     [[TMP2:%.*]] = fmul <4 x float> [[TMP1]], <float 1.000000e+00, float 2.000000e+05, float -3.000000e+00, float 4.000000e+00>
; CHECK-NEXT:     ret <4 x float> [[TMP2]]
        %Y = fmul <4 x float> %V, < float 1.000000e+00, float 2.000000e+00, float 3.000000e+00, float 4.000000e+00 >
        %Z = fmul <4 x float> %Y, < float 1.000000e+00, float 2.000000e+05, float -3.000000e+00, float 4.000000e+00 >
        ret <4 x float> %Z
}

; (V * C1) * C2 => V * (C1 * C2)
; Verify this folds with 'fast'
define <4 x float> @test_fmul_fast(<4 x float> %V) {
; CHECK-LABEL: @test_fmul_fast(
; CHECK-NEXT:     [[TMP1:%.*]] = fmul fast <4 x float> [[V:%.*]], <float 1.000000e+00, float 4.000000e+05, float -9.000000e+00, float 1.600000e+01>
; CHECK-NEXT:     ret <4 x float> [[TMP1]]
        %Y = fmul fast <4 x float> %V, < float 1.000000e+00, float 2.000000e+00, float 3.000000e+00, float 4.000000e+00 >
        %Z = fmul fast <4 x float> %Y, < float 1.000000e+00, float 2.000000e+05, float -3.000000e+00, float 4.000000e+00 >
        ret <4 x float> %Z
}

; (V * C1) * C2 => V * (C1 * C2)
; Verify this folds with 'reassoc' and 'nsz' ('nsz' not technically required)
define <4 x float> @test_fmul_reassoc_nsz(<4 x float> %V) {
; CHECK-LABEL: @test_fmul_reassoc_nsz(
; CHECK-NEXT:     [[TMP1:%.*]] = fmul reassoc nsz <4 x float> [[V:%.*]], <float 1.000000e+00, float 4.000000e+05, float -9.000000e+00, float 1.600000e+01>
; CHECK-NEXT:     ret <4 x float> [[TMP1]]
        %Y = fmul reassoc nsz <4 x float> %V, < float 1.000000e+00, float 2.000000e+00, float 3.000000e+00, float 4.000000e+00 >
        %Z = fmul reassoc nsz <4 x float> %Y, < float 1.000000e+00, float 2.000000e+05, float -3.000000e+00, float 4.000000e+00 >
        ret <4 x float> %Z
}

; (V * C1) * C2 => V * (C1 * C2)
; TODO: This doesn't require 'nsz'.  It should fold to V * { 1.0, 4.0e+05, -9.0, 16.0 }
define <4 x float> @test_fmul_reassoc(<4 x float> %V) {
; CHECK-LABEL: @test_fmul_reassoc(
; CHECK-NEXT:     [[TMP1:%.*]] = fmul reassoc <4 x float> [[V:%.*]], <float 1.000000e+00, float 2.000000e+00, float 3.000000e+00, float 4.000000e+00>
; CHECK-NEXT:     [[TMP2:%.*]] = fmul reassoc <4 x float> [[TMP1]], <float 1.000000e+00, float 2.000000e+05, float -3.000000e+00, float 4.000000e+00>
; CHECK-NEXT:     ret <4 x float> [[TMP2]]
        %Y = fmul reassoc <4 x float> %V, < float 1.000000e+00, float 2.000000e+00, float 3.000000e+00, float 4.000000e+00 >
        %Z = fmul reassoc <4 x float> %Y, < float 1.000000e+00, float 2.000000e+05, float -3.000000e+00, float 4.000000e+00 >
        ret <4 x float> %Z
}

; (V + C1) + C2 => V + (C1 + C2)
; Verify this doesn't fold when no fast-math-flags are specified
define <4 x float> @test_fadd(<4 x float> %V) {
; CHECK-LABEL: @test_fadd(
; CHECK-NEXT:     [[TMP1:%.*]] = fadd <4 x float> [[V:%.*]], <float 1.000000e+00, float 2.000000e+00, float 3.000000e+00, float 4.000000e+00>
; CHECK-NEXT:     [[TMP2:%.*]] = fadd <4 x float> [[TMP1]], <float 1.000000e+00, float 2.000000e+00, float -3.000000e+00, float 4.000000e+00>
; CHECK-NEXT:     ret <4 x float> [[TMP2]]
        %Y = fadd <4 x float> %V, < float 1.000000e+00, float 2.000000e+00, float 3.000000e+00, float 4.000000e+00 >
        %Z = fadd <4 x float> %Y, < float 1.000000e+00, float 2.000000e+00, float -3.000000e+00, float 4.000000e+00 >
        ret <4 x float> %Z
}

; (V + C1) + C2 => V + (C1 + C2)
; Verify this folds with 'fast'
define <4 x float> @test_fadd_fast(<4 x float> %V) {
; CHECK-LABEL: @test_fadd_fast(
; CHECK-NEXT:     [[TMP1:%.*]] = fadd fast <4 x float> [[V:%.*]], <float 2.000000e+00, float 4.000000e+00, float 0.000000e+00, float 8.000000e+00>
; CHECK-NEXT:     ret <4 x float> [[TMP1]]
        %Y = fadd fast <4 x float> %V, < float 1.000000e+00, float 2.000000e+00, float 3.000000e+00, float 4.000000e+00 >
        %Z = fadd fast <4 x float> %Y, < float 1.000000e+00, float 2.000000e+00, float -3.000000e+00, float 4.000000e+00 >
        ret <4 x float> %Z
}

; (V + C1) + C2 => V + (C1 + C2)
; Verify this folds with 'reassoc' and 'nsz' ('nsz' not technically required)
define <4 x float> @test_fadd_reassoc_nsz(<4 x float> %V) {
; CHECK-LABEL: @test_fadd_reassoc_nsz(
; CHECK-NEXT:     [[TMP1:%.*]] = fadd reassoc nsz <4 x float> [[V:%.*]], <float 2.000000e+00, float 4.000000e+00, float 0.000000e+00, float 8.000000e+00>
; CHECK-NEXT:     ret <4 x float> [[TMP1]]
        %Y = fadd reassoc nsz <4 x float> %V, < float 1.000000e+00, float 2.000000e+00, float 3.000000e+00, float 4.000000e+00 >
        %Z = fadd reassoc nsz <4 x float> %Y, < float 1.000000e+00, float 2.000000e+00, float -3.000000e+00, float 4.000000e+00 >
        ret <4 x float> %Z
}

; (V + C1) + C2 => V + (C1 + C2)
; TODO: This doesn't require 'nsz'.  It should fold to V + { 2.0, 4.0, 0.0, 8.0 }
define <4 x float> @test_fadd_reassoc(<4 x float> %V) {
; CHECK-LABEL: @test_fadd_reassoc(
; CHECK-NEXT:     [[TMP1:%.*]] = fadd reassoc <4 x float> [[V:%.*]], <float 1.000000e+00, float 2.000000e+00, float 3.000000e+00, float 4.000000e+00>
; CHECK-NEXT:     [[TMP2:%.*]] = fadd reassoc <4 x float> [[TMP1]], <float 1.000000e+00, float 2.000000e+00, float -3.000000e+00, float 4.000000e+00>
; CHECK-NEXT:     ret <4 x float> [[TMP2]]
        %Y = fadd reassoc <4 x float> %V, < float 1.000000e+00, float 2.000000e+00, float 3.000000e+00, float 4.000000e+00 >
        %Z = fadd reassoc <4 x float> %Y, < float 1.000000e+00, float 2.000000e+00, float -3.000000e+00, float 4.000000e+00 >
        ret <4 x float> %Z
}

; ( A + C1 ) + ( B + -C1 )
; Verify this doesn't fold when no fast-math-flags are specified
define <4 x float> @test_fadds_cancel_(<4 x float> %A, <4 x float> %B) {
; CHECK-LABEL: @test_fadds_cancel_(
; CHECK-NEXT:     [[TMP1:%.*]] = fadd <4 x float> [[A:%.*]], <float 1.000000e+00, float 2.000000e+00, float 3.000000e+00, float 4.000000e+00>
; CHECK-NEXT:     [[TMP2:%.*]] = fadd <4 x float> [[B:%.*]], <float -1.000000e+00, float -2.000000e+00, float -3.000000e+00, float -4.000000e+00>
; CHECK-NEXT:     [[TMP3:%.*]] = fadd <4 x float> [[TMP1]], [[TMP2]]
; CHECK-NEXT:     ret <4 x float> [[TMP3]]
        %X = fadd <4 x float> %A, < float 1.000000e+00, float 2.000000e+00, float 3.000000e+00, float 4.000000e+00 >
        %Y = fadd <4 x float> %B, < float -1.000000e+00, float -2.000000e+00, float -3.000000e+00, float -4.000000e+00 >
        %Z = fadd <4 x float> %X, %Y
        ret <4 x float> %Z
}

; ( A + C1 ) + ( B + -C1 )
; Verify this folds to 'A + B' with 'fast'
define <4 x float> @test_fadds_cancel_fast(<4 x float> %A, <4 x float> %B) {
; CHECK-LABEL: @test_fadds_cancel_fast(
; CHECK-NEXT:     [[TMP1:%.*]] = fadd fast <4 x float> [[A:%.*]], [[B:%.*]]
; CHECK-NEXT:     ret <4 x float> [[TMP1]]
        %X = fadd fast <4 x float> %A, < float 1.000000e+00, float 2.000000e+00, float 3.000000e+00, float 4.000000e+00 >
        %Y = fadd fast <4 x float> %B, < float -1.000000e+00, float -2.000000e+00, float -3.000000e+00, float -4.000000e+00 >
        %Z = fadd fast <4 x float> %X, %Y
        ret <4 x float> %Z
}

; ( A + C1 ) + ( B + -C1 )
; Verify this folds to 'A + B' with 'reassoc' and 'nsz' ('nsz' is required)
define <4 x float> @test_fadds_cancel_reassoc_nsz(<4 x float> %A, <4 x float> %B) {
; CHECK-LABEL: @test_fadds_cancel_reassoc_nsz(
; CHECK-NEXT:     [[TMP1:%.*]] = fadd reassoc nsz <4 x float> [[A:%.*]], [[B:%.*]]
; CHECK-NEXT:     ret <4 x float> [[TMP1]]
        %X = fadd reassoc nsz <4 x float> %A, < float 1.000000e+00, float 2.000000e+00, float 3.000000e+00, float 4.000000e+00 >
        %Y = fadd reassoc nsz <4 x float> %B, < float -1.000000e+00, float -2.000000e+00, float -3.000000e+00, float -4.000000e+00 >
        %Z = fadd reassoc nsz <4 x float> %X, %Y
        ret <4 x float> %Z
}

; ( A + C1 ) + ( B + -C1 )
; Verify the fold is not done with only 'reassoc' ('nsz' is required).
define <4 x float> @test_fadds_cancel_reassoc(<4 x float> %A, <4 x float> %B) {
; CHECK-LABEL: @test_fadds_cancel_reassoc(
; CHECK-NEXT:     [[TMP1:%.*]] = fadd reassoc <4 x float> [[A:%.*]], <float 1.000000e+00, float 2.000000e+00, float 3.000000e+00, float 4.000000e+00>
; CHECK-NEXT:     [[TMP2:%.*]] = fadd reassoc <4 x float> [[B:%.*]], <float -1.000000e+00, float -2.000000e+00, float -3.000000e+00, float -4.000000e+00>
; CHECK-NEXT:     [[TMP3:%.*]] = fadd reassoc <4 x float> [[TMP1]], [[TMP2]]
; CHECK-NEXT:     ret <4 x float> [[TMP3]]
        %X = fadd reassoc <4 x float> %A, < float 1.000000e+00, float 2.000000e+00, float 3.000000e+00, float 4.000000e+00 >
        %Y = fadd reassoc <4 x float> %B, < float -1.000000e+00, float -2.000000e+00, float -3.000000e+00, float -4.000000e+00 >
        %Z = fadd reassoc <4 x float> %X, %Y
        ret <4 x float> %Z
}
