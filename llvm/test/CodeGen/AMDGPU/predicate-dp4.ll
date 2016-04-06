;RUN: llc < %s -march=r600 -mcpu=cayman

; CHECK-LABEL: {{^}}main:
; CHECK: PRED_SETE_INT * Pred,
; CHECK: DOT4 T{{[0-9]+}}.X, T0.X, T0.X, Pred_sel_one
define amdgpu_ps void @main(<4 x float> inreg) {
main_body:
  %1 = extractelement <4 x float> %0, i32 0
  %2 = bitcast float %1 to i32
  %3 = icmp eq i32 %2, 0
  br i1 %3, label %IF, label %ENDIF

IF:                                             ; preds = %main_body
  %4 = call float @llvm.AMDGPU.dp4(<4 x float> %0, <4 x float> %0)
  br label %ENDIF

ENDIF:                                            ; preds = %IF, %main_body
  %5 = phi float [%4, %IF], [0.000000e+00, %main_body]
  %6 = insertelement <4 x float> undef, float %5, i32 0
  call void @llvm.R600.store.swizzle(<4 x float> %6, i32 0, i32 0)
  ret void
}

declare float @llvm.AMDGPU.dp4(<4 x float>, <4 x float>) #1
declare void @llvm.R600.store.swizzle(<4 x float>, i32, i32)
attributes #1 = { readnone }
