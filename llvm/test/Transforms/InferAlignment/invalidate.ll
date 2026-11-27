; RUN: opt -passes='require<demanded-bits>,require<scalar-evolution>,require<access-info>,infer-alignment' \
; RUN:     -debug-pass-manager -disable-output < %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=PM \
; RUN:     --implicit-check-not='Invalidating analysis:'
; RUN: opt -passes='instcombine,infer-alignment,instcombine' -S < %s \
; RUN:     | FileCheck %s --check-prefix=IC

; PM: Running pass: InferAlignmentPass on changed_rauw
; PM-DAG: Invalidating analysis: DemandedBitsAnalysis on changed_rauw
; PM-DAG: Invalidating analysis: LoopAccessAnalysis on changed_rauw

; PM: Running pass: InferAlignmentPass on changed_align
; PM: Invalidating analysis: DemandedBitsAnalysis on changed_align

; PM: Running pass: InferAlignmentPass on unchanged

define i64 @changed_rauw(ptr %p) {
; IC-LABEL: define i64 @changed_rauw(ptr %p) {
; IC-NEXT:    store i32 1, ptr %p, align 16
; IC-NEXT:    ret i64 5
;
  store i32 1, ptr %p, align 16
  %int = ptrtoint ptr %p to i64
  %and = and i64 %int, 7
  %add = add i64 %and, 5
  ret i64 %add
}

define i32 @changed_align(ptr %p) {
  store i32 1, ptr %p, align 16
  %v = load i32, ptr %p, align 4
  ret i32 %v
}

define i32 @unchanged(ptr %p) {
  %v = load i32, ptr %p, align 4
  ret i32 %v
}
