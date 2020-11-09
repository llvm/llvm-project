; RUN: opt -S -denormal-fp-math=ieee %s | FileCheck -check-prefixes=IEEE,ALL %s
; RUN: opt -S -denormal-fp-math=preserve-sign %s | FileCheck -check-prefixes=PRESERVESIGN,ALL %s
; RUN: opt -S -denormal-fp-math=positive-zero %s | FileCheck -check-prefixes=POSITIVEZERO,ALL %s

; ALL: @no_denormal_fp_math_attr() [[NOATTR:#[0-9]+]] {
define i32 @no_denormal_fp_math_attr() #0 {
entry:
  ret i32 0
}

; ALL: denormal_fp_math_attr_preserve_sign_ieee() [[ATTR:#[0-9]+]] {
define i32 @denormal_fp_math_attr_preserve_sign_ieee() #1 {
entry:
  ret i32 0
}

; ALL-DAG: attributes [[ATTR]] = { nounwind "denormal-fp-math"="preserve-sign,ieee" }
; IEEE-DAG: attributes [[NOATTR]] = { nounwind "denormal-fp-math"="ieee,ieee" }
; PRESERVESIGN-DAG: attributes [[NOATTR]] = { nounwind "denormal-fp-math"="preserve-sign,preserve-sign" }
; POSITIVEZERO-DAG: attributes [[NOATTR]] = { nounwind "denormal-fp-math"="positive-zero,positive-zero" }

attributes #0 = { nounwind }
attributes #1 = { nounwind "denormal-fp-math"="preserve-sign,ieee" }
