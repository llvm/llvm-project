; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=instruction-flags --test FileCheck --test-arg --check-prefixes=INTERESTING,CHECK --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck -check-prefixes=RESULT,CHECK %s < %t

; CHECK-LABEL: @add_nuw_nsw_none(
; INTERESTING: = add
; RESULT: add i32
define i32 @add_nuw_nsw_none(i32 %a, i32 %b) {
  %op = add nuw nsw i32 %a, %b
  ret i32 %op
}

; CHECK-LABEL: @add_nuw_nsw_keep_nuw(
; INTERESTING: nuw
; RESULT: add nuw i32
define i32 @add_nuw_nsw_keep_nuw(i32 %a, i32 %b) {
  %op = add nuw nsw i32 %a, %b
  ret i32 %op
}

; CHECK-LABEL: @add_nuw_nsw_keep_nsw(
; INTERESTING: nuw
; RESULT: add nuw i32
define i32 @add_nuw_nsw_keep_nsw(i32 %a, i32 %b) {
  %op = add nuw nsw i32 %a, %b
  ret i32 %op
}

; CHECK-LABEL: @add_nuw_keep_nuw(
; INTERESTING: nuw
; RESULT: add nuw i32
define i32 @add_nuw_keep_nuw(i32 %a, i32 %b) {
  %op = add nuw i32 %a, %b
  ret i32 %op
}

; CHECK-LABEL: @add_nsw_keep_nsw(
; INTERESTING: nsw
; RESULT: add nsw i32
define i32 @add_nsw_keep_nsw(i32 %a, i32 %b) {
  %op = add nsw i32 %a, %b
  ret i32 %op
}

; CHECK-LABEL: @ashr_exact_drop(
; INTERESTING: = ashr
; RESULT: ashr i32
define i32 @ashr_exact_drop(i32 %a, i32 %b) {
  %op = ashr exact i32 %a, %b
  ret i32 %op
}

; CHECK-LABEL: @ashr_exact_keep(
; INTERESTING: exact
; RESULT: ashr exact i32
define i32 @ashr_exact_keep(i32 %a, i32 %b) {
  %op = ashr exact i32 %a, %b
  ret i32 %op
}

; CHECK-LABEL: @getelementptr_inbounds_drop(
; INTERESTING: getelementptr
; RESULT: getelementptr i32, ptr %a, i64 %b
define ptr @getelementptr_inbounds_drop(ptr %a, i64 %b) {
  %op = getelementptr inbounds i32, ptr %a, i64 %b
  ret ptr %op
}

; CHECK-LABEL: @getelementptr_inbounds_keep(
; INTERESTING: inbounds
; RESULT: getelementptr inbounds i32, ptr %a, i64 %b
define ptr @getelementptr_inbounds_keep(ptr %a, i64 %b) {
  %op = getelementptr inbounds i32, ptr %a, i64 %b
  ret ptr %op
}

; CHECK-LABEL: @fadd_reassoc_none(
; INTERESTING: = fadd
; RESULT: fadd float
define float @fadd_reassoc_none(float %a, float %b) {
  %op = fadd reassoc float %a, %b
  ret float %op
}

; CHECK-LABEL: @fadd_reassoc_keep(
; INTERESTING: fadd reassoc
; RESULT: fadd reassoc float
define float @fadd_reassoc_keep(float %a, float %b) {
  %op = fadd reassoc float %a, %b
  ret float %op
}

; CHECK-LABEL: @fadd_nnan_none(
; INTERESTING: = fadd
; RESULT: fadd float
define float @fadd_nnan_none(float %a, float %b) {
  %op = fadd nnan float %a, %b
  ret float %op
}

; CHECK-LABEL: @fadd_nnan_keep(
; INTERESTING: fadd nnan
; RESULT: fadd nnan float
define float @fadd_nnan_keep(float %a, float %b) {
  %op = fadd nnan float %a, %b
  ret float %op
}

; CHECK-LABEL: @fadd_ninf_none(
; INTERESTING: = fadd
; RESULT: fadd float
define float @fadd_ninf_none(float %a, float %b) {
  %op = fadd ninf float %a, %b
  ret float %op
}

; CHECK-LABEL: @fadd_ninf_keep(
; INTERESTING: fadd ninf
; RESULT: fadd ninf float
define float @fadd_ninf_keep(float %a, float %b) {
  %op = fadd ninf float %a, %b
  ret float %op
}

; CHECK-LABEL: @fadd_nsz_none(
; INTERESTING: = fadd
; RESULT: fadd float
define float @fadd_nsz_none(float %a, float %b) {
  %op = fadd nsz float %a, %b
  ret float %op
}

; CHECK-LABEL: @fadd_nsz_keep(
; INTERESTING: fadd nsz
; RESULT: fadd nsz float
define float @fadd_nsz_keep(float %a, float %b) {
  %op = fadd nsz float %a, %b
  ret float %op
}

; CHECK-LABEL: @fadd_arcp_none(
; INTERESTING: = fadd
; RESULT: fadd float
define float @fadd_arcp_none(float %a, float %b) {
  %op = fadd arcp float %a, %b
  ret float %op
}

; CHECK-LABEL: @fadd_arcp_keep(
; INTERESTING: fadd arcp
; RESULT: fadd arcp float
define float @fadd_arcp_keep(float %a, float %b) {
  %op = fadd arcp float %a, %b
  ret float %op
}

; CHECK-LABEL: @fadd_contract_none(
; INTERESTING: = fadd
; RESULT: fadd float
define float @fadd_contract_none(float %a, float %b) {
  %op = fadd contract float %a, %b
  ret float %op
}

; CHECK-LABEL: @fadd_contract_keep(
; INTERESTING: fadd contract
; RESULT: fadd contract float
define float @fadd_contract_keep(float %a, float %b) {
  %op = fadd contract float %a, %b
  ret float %op
}

; CHECK-LABEL: @fadd_afn_none(
; INTERESTING: = fadd
; RESULT: fadd float
define float @fadd_afn_none(float %a, float %b) {
  %op = fadd afn float %a, %b
  ret float %op
}

; CHECK-LABEL: @fadd_afn_keep(
; INTERESTING: fadd afn
; RESULT: fadd afn float
define float @fadd_afn_keep(float %a, float %b) {
  %op = fadd afn float %a, %b
  ret float %op
}

; CHECK-LABEL: @fadd_fast_none(
; INTERESTING: = fadd
; RESULT: fadd float
define float @fadd_fast_none(float %a, float %b) {
  %op = fadd fast float %a, %b
  ret float %op
}

; CHECK-LABEL: @fadd_nnan_ninf_keep_nnan(
; INTERESTING: nnan
; RESULT: fadd nnan float
define float @fadd_nnan_ninf_keep_nnan(float %a, float %b) {
  %op = fadd nnan ninf float %a, %b
  ret float %op
}
