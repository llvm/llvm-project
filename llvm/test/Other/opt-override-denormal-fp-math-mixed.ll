; RUN: opt -S -denormal-fp-math=ieee %s | FileCheck -check-prefixes=IEEE,ALL %s
; RUN: opt -S -denormal-fp-math=preserve-sign %s | FileCheck -check-prefixes=PRESERVESIGN,ALL %s
; RUN: opt -S -denormal-fp-math=positive-zero %s | FileCheck -check-prefixes=POSITIVEZERO,ALL %s

; RUN: opt -S -denormal-fp-math-f32=ieee %s | FileCheck -check-prefixes=IEEEF32,ALL %s
; RUN: opt -S -denormal-fp-math-f32=preserve-sign %s | FileCheck -check-prefixes=PRESERVESIGNF32,ALL %s
; RUN: opt -S -denormal-fp-math-f32=positive-zero %s | FileCheck -check-prefixes=POSITIVEZEROF32,ALL %s

; RUN: opt -S -denormal-fp-math=ieee -denormal-fp-math-f32=ieee %s | FileCheck -check-prefixes=IEEE-BOTH,ALL %s
; RUN: opt -S -denormal-fp-math=preserve-sign -denormal-fp-math-f32=preserve-sign %s | FileCheck -check-prefixes=PRESERVESIGN-BOTH,ALL %s
; RUN: opt -S -denormal-fp-math=positive-zero -denormal-fp-math-f32=positive-zero %s | FileCheck -check-prefixes=POSITIVEZERO-BOTH,ALL %s



; ALL: @no_denormal_fp_math_attrs() [[NOATTR:#[0-9]+]] {
define i32 @no_denormal_fp_math_attrs() #0 {
entry:
  ret i32 0
}

; ALL: both_denormal_fp_math_attrs_preserve_sign_ieee() [[ATTR:#[0-9]+]] {
define i32 @both_denormal_fp_math_attrs_preserve_sign_ieee() #1 {
entry:
  ret i32 0
}

; ALL-DAG: attributes [[ATTR]] = { nounwind denormal_fpenv(preservesign|ieee) }

; IEEE-DAG: attributes [[NOATTR]] = { nounwind denormal_fpenv(ieee) }
; PRESERVESIGN-DAG: attributes [[NOATTR]] = { nounwind denormal_fpenv(preservesign) }
; POSITIVEZERO-DAG: attributes [[NOATTR]] = { nounwind denormal_fpenv(positivezero) }

; IEEEF32-DAG: attributes [[NOATTR]] = { nounwind denormal_fpenv(ieee) }
; PRESERVESIGNF32-DAG: attributes [[NOATTR]] = { nounwind denormal_fpenv(float: preservesign) }
; POSITIVEZEROF32-DAG: attributes [[NOATTR]] = { nounwind denormal_fpenv(float: positivezero) }

; IEEE-BOTH-DAG: attributes [[NOATTR]] = { nounwind denormal_fpenv(ieee) }
; PRESERVESIGN-BOTH-DAG: attributes [[NOATTR]] = { nounwind denormal_fpenv(preservesign) }
; POSITIVEZERO-BOTH-DAG: attributes [[NOATTR]] = { nounwind denormal_fpenv(positivezero) }

attributes #0 = { nounwind }
attributes #1 = { nounwind denormal_fpenv(preservesign|ieee, float: preservesign|ieee) }
