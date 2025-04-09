; RUN: opt -S -denormal-fp-math-bf16=ieee %s | FileCheck -check-prefixes=IEEE,ALL %s
; RUN: opt -S -denormal-fp-math-bf16=preserve-sign %s | FileCheck -check-prefixes=PRESERVESIGN,ALL %s
; RUN: opt -S -denormal-fp-math-bf16=positive-zero %s | FileCheck -check-prefixes=POSITIVEZERO,ALL %s

; ALL: @no_denormal_fp_math_f32_attr() [[NOATTR:#[0-9]+]] {
define i32 @no_denormal_fp_math_f32_attr() #0 {
entry:
  ret i32 0
}

; ALL: denormal_fp_math_attr_preserve_sign_ieee() [[ATTR:#[0-9]+]] {
define i32 @denormal_fp_math_attr_preserve_sign_ieee() #1 {
entry:
  ret i32 0
}

; ALL-DAG: attributes [[ATTR]] = { nounwind "denormal-fp-math-bf16"="preserve-sign,ieee" }
; IEEE-DAG: attributes [[NOATTR]] = { nounwind "denormal-fp-math-bf16"="ieee,ieee" }
; PRESERVESIGN-DAG: attributes [[NOATTR]] = { nounwind "denormal-fp-math-bf16"="preserve-sign,preserve-sign" }
; POSITIVEZERO-DAG: attributes [[NOATTR]] = { nounwind "denormal-fp-math-bf16"="positive-zero,positive-zero" }

attributes #0 = { nounwind }
attributes #1 = { nounwind "denormal-fp-math-bf16"="preserve-sign,ieee" }
