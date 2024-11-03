; RUN: opt %loadPolly -polly-codegen-ppcg -polly-acc-dump-code \
; RUN: -disable-output < %s | \
; RUN: FileCheck -check-prefix=CODE %s

; REQUIRES: pollyacc

; CODE: Code
; CODE: ====
; CODE: No code generated

source_filename = "bugpoint-output-83bcdeb.bc"
target datalayout = "e-p:64:64:64-S128-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f16:16:16-f32:32:32-f64:64:64-f128:128:128-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

@__data_radiation_MOD_cobi = external global [168 x double], align 32

; Function Attrs: nounwind uwtable
define void @__radiation_rg_MOD_coe_so() #0 {
entry:
  %polly.access.kspec.load = load i32, ptr undef, align 4
  %0 = or i1 undef, undef
  br label %polly.preload.cond29

polly.preload.cond29:                             ; preds = %entry
  br i1 %0, label %polly.preload.exec31, label %polly.preload.merge30

polly.preload.merge30:                            ; preds = %polly.preload.exec31, %polly.preload.cond29
  %polly.preload..merge32 = phi double [ %polly.access.__data_radiation_MOD_cobi.load, %polly.preload.exec31 ], [ 0.000000e+00, %polly.preload.cond29 ]
  ret void

polly.preload.exec31:                             ; preds = %polly.preload.cond29
  %1 = sext i32 %polly.access.kspec.load to i64
  %2 = mul nsw i64 7, %1
  %3 = add nsw i64 0, %2
  %4 = add nsw i64 %3, 48
  %polly.access.__data_radiation_MOD_cobi = getelementptr double, ptr @__data_radiation_MOD_cobi, i64 %4
  %polly.access.__data_radiation_MOD_cobi.load = load double, ptr %polly.access.__data_radiation_MOD_cobi, align 8
  br label %polly.preload.merge30
}

attributes #0 = { nounwind uwtable }
