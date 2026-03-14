; RUN: llc -asm-verbose=0 -mtriple aarch64 < %s | FileCheck %s

; The following code previously broke in the DAGCombiner. Specifically, trying to combine:
; extract_vector_elt (concat_vectors v4i16:a, v4i16:b), x
;   -> extract_vector_elt a, x

define half @test_combine_extract_concat_vectors(<4 x i16> %a) nounwind {
entry:
  %0 = shufflevector <4 x i16> %a, <4 x i16> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1 = bitcast <8 x i16> %0 to <8 x half>
  %2 = extractelement <8 x half> %1, i32 3
  ret half %2
}

; CHECK-LABEL: test_combine_extract_concat_vectors:
; CHECK-NEXT: mov h0, v0.h[3]
; CHECK-NEXT: ret
