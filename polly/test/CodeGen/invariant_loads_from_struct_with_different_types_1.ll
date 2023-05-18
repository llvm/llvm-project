; RUN: opt %loadPolly -polly-codegen -polly-invariant-load-hoisting=true < %s
;
; Check we do not crash even though we pre-load values with different types
; from the same base pointer.
;
target datalayout = "e-m:o-p:32:32-f64:32:64-f80:128-n8:16:32-S128"

%struct.FFIIRFilterCoeffs.0.3.6.12.15.27.36.54.57.84.87.90 = type { i32, float, ptr, ptr }

; Function Attrs: nounwind ssp
define void @ff_iir_filter(ptr %c, ptr %dst, i32 %dstep) #0 {
entry:
  br i1 undef, label %if.end.325, label %for.body.38

for.body.38:                                      ; preds = %for.body.38, %entry
  %dst034.0180 = phi ptr [ undef, %for.body.38 ], [ %dst, %entry ]
  %gain42 = getelementptr inbounds %struct.FFIIRFilterCoeffs.0.3.6.12.15.27.36.54.57.84.87.90, ptr %c, i32 0, i32 1
  %cy44 = getelementptr inbounds %struct.FFIIRFilterCoeffs.0.3.6.12.15.27.36.54.57.84.87.90, ptr %c, i32 0, i32 3
  %add.ptr88 = getelementptr inbounds i16, ptr %dst034.0180, i32 %dstep
  store i16 undef, ptr %add.ptr88, align 2
  %0 = load float, ptr %gain42, align 4
  %1 = load ptr, ptr %cy44, align 4
  br i1 false, label %for.body.38, label %if.end.325

if.end.325:                                       ; preds = %for.body.38, %entry
  ret void
}
