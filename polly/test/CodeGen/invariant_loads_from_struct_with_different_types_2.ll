; RUN: opt %loadNPMPolly -passes=polly-codegen -polly-invariant-load-hoisting=true < %s
;
; Check we do not crash even though we pre-load values with different types
; from the same base pointer.
;
target datalayout = "e-m:o-p:32:32-f64:32:64-f80:128-n8:16:32-S128"

%struct.FFIIRFilterCoeffs.0.3.6.63.78.81.87.102.150.162.165.168.171 = type { i32, float, ptr, ptr }

; Function Attrs: nounwind ssp
define void @butterworth_init_coeffs(ptr %c) #0 {
entry:
  br i1 undef, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  unreachable

if.end:                                           ; preds = %entry
  br i1 undef, label %if.end.2, label %if.then.1

if.then.1:                                        ; preds = %if.end
  br label %return

if.end.2:                                         ; preds = %if.end
  br i1 undef, label %for.body.35, label %for.end.126

for.body.35:                                      ; preds = %if.end.2
  unreachable

for.end.126:                                      ; preds = %if.end.2
  %gain = getelementptr inbounds %struct.FFIIRFilterCoeffs.0.3.6.63.78.81.87.102.150.162.165.168.171, ptr %c, i32 0, i32 1
  br i1 undef, label %for.body.133, label %for.end.169

for.body.133:                                     ; preds = %for.body.133, %for.end.126
  store float undef, ptr %gain, align 4
  %cy = getelementptr inbounds %struct.FFIIRFilterCoeffs.0.3.6.63.78.81.87.102.150.162.165.168.171, ptr %c, i32 0, i32 3
  %0 = load ptr, ptr %cy, align 4
  br i1 false, label %for.body.133, label %for.end.169

for.end.169:                                      ; preds = %for.body.133, %for.end.126
  br label %return

return:                                           ; preds = %for.end.169, %if.then.1
  ret void
}
