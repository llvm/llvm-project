; RUN: llc < %s -march=amdgcn -mcpu=verde -verify-machineinstrs | FileCheck %s
; RUN: llc < %s -march=amdgcn -mcpu=tonga -verify-machineinstrs | FileCheck %s

target triple = "amdgcn--"

; CHECK-LABEL: {{^}}main:
;
; Test for compilation only. This generated an invalid machine instruction
; by trying to commute the operands of a V_CMP_EQ_i32_e32 instruction, both
; of which were in SGPRs.
define amdgpu_vs float @main(i32 %v) {
main_body:
  %d1 = call float @llvm.SI.load.const(<16 x i8> undef, i32 960)
  %d2 = call float @llvm.SI.load.const(<16 x i8> undef, i32 976)
  br i1 undef, label %ENDIF56, label %IF57

IF57:                                             ; preds = %ENDIF
  %v.1 = mul i32 %v, 2
  br label %ENDIF56

ENDIF56:                                          ; preds = %IF57, %ENDIF
  %v.2 = phi i32 [ %v, %main_body ], [ %v.1, %IF57 ]
  %d1.i = bitcast float %d1 to i32
  %cc1 = icmp eq i32 %d1.i, 0
  br i1 %cc1, label %ENDIF59, label %IF60

IF60:                                             ; preds = %ENDIF56
  %v.3 = mul i32 %v.2, 2
  br label %ENDIF59

ENDIF59:                                          ; preds = %IF60, %ENDIF56
  %v.4 = phi i32 [ %v.2, %ENDIF56 ], [ %v.3, %IF60 ]
  %d2.i = bitcast float %d2 to i32
  %cc2 = icmp eq i32 %d2.i, 0
  br i1 %cc2, label %ENDIF62, label %IF63

IF63:                                             ; preds = %ENDIF59
  unreachable

ENDIF62:                                          ; preds = %ENDIF59
  %r = bitcast i32 %v.4 to float
  ret float %r
}

; Function Attrs: nounwind readnone
declare float @llvm.SI.load.const(<16 x i8>, i32) #0

attributes #0 = { nounwind readnone }
attributes #1 = { readnone }
