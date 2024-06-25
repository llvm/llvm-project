; RUN: opt %loadNPMPolly '-passes=print<polly-function-scops>' -disable-output -polly-isl-arg=-V < %s | FileCheck %s -match-full-lines --check-prefix=VERSION
; RUN: opt %loadNPMPolly '-passes=print<polly-function-scops>' -disable-output -polly-isl-arg=-h < %s | FileCheck %s -match-full-lines --check-prefix=HELP
; RUN: not opt %loadNPMPolly '-passes=print<polly-function-scops>' -disable-output -polly-isl-arg=-asdf < %s 2>&1| FileCheck %s -match-full-lines --check-prefix=UNKNOWN
; RUN: opt %loadNPMPolly '-passes=print<polly-function-scops>' -disable-output -polly-isl-arg=--schedule-algorithm=feautrier < %s

; VERSION: isl-{{.*}}-IMath-32
; HELP: Usage: -polly-isl-arg [OPTION...]
; UNKNOWN: -polly-isl-arg: unrecognized option: -asdf

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; Any valid SCoP causing the creation of a ScopInfo object.
define void @foo_1d(ptr %A) {
bb:
  br label %bb1

bb1:                                              ; preds = %bb6, %bb
  %i.0 = phi i64 [ 0, %bb ], [ %tmp7, %bb6 ]
  %exitcond = icmp ne i64 %i.0, 1024
  br i1 %exitcond, label %bb2, label %bb8

bb2:                                              ; preds = %bb1
  %tmp = sitofp i64 %i.0 to float
  %tmp3 = getelementptr inbounds float, ptr %A, i64 %i.0
  %tmp4 = load float, ptr %tmp3, align 4
  %tmp5 = fadd float %tmp4, %tmp
  store float %tmp5, ptr %tmp3, align 4
  br label %bb6

bb6:                                              ; preds = %bb2
  %tmp7 = add nuw nsw i64 %i.0, 1
  br label %bb1

bb8:                                              ; preds = %bb1
  ret void
}
