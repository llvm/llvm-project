; RUN: opt -S -dxil-intrinsic-expansion -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s --check-prefix=EXPCHECK
; RUN: opt -S -dxil-intrinsic-expansion -scalarizer -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s --check-prefix=DOPCHECK

; Make sure llvm.frexp is expanded into bit operations for DXIL. DXIL only
; allows extractvalue on DXIL struct types, so the results are forwarded to the
; extractvalue users and no aggregate is left behind.

define noundef float @frexp_float(float noundef %a, ptr %e) {
entry:
; EXPCHECK-LABEL: define noundef float @frexp_float(
; EXPCHECK:      [[BITS:%.*]] = bitcast float %a to i32
; EXPCHECK-NEXT: [[EXPBITS:%.*]] = and i32 [[BITS]], 2139095040
; EXPCHECK-NEXT: [[ISZERO:%.*]] = icmp eq i32 [[EXPBITS]], 0
; EXPCHECK-NEXT: [[ISINFNAN:%.*]] = icmp eq i32 [[EXPBITS]], 2139095040
; EXPCHECK-NEXT: [[SPECIAL:%.*]] = or i1 [[ISZERO]], [[ISINFNAN]]
; EXPCHECK-NEXT: [[SIGNMANT:%.*]] = and i32 [[BITS]], -2139095041
; EXPCHECK-NEXT: [[MANTBITS:%.*]] = or i32 [[SIGNMANT]], 1056964608
; EXPCHECK-NEXT: [[MANTF:%.*]] = bitcast i32 [[MANTBITS]] to float
; EXPCHECK-NEXT: [[MANT:%.*]] = select i1 [[SPECIAL]], float %a, float [[MANTF]]
; EXPCHECK-NEXT: [[SHR:%.*]] = lshr i32 [[EXPBITS]], 23
; EXPCHECK-NEXT: [[UNBIASED:%.*]] = sub i32 [[SHR]], 126
; EXPCHECK-NEXT: [[EXP:%.*]] = select i1 [[SPECIAL]], i32 0, i32 [[UNBIASED]]
; EXPCHECK-NEXT: [[XF:%.*]] = sitofp i32 [[EXP]] to float
; EXPCHECK-NEXT: store float [[XF]], ptr %e
; EXPCHECK-NEXT: ret float [[MANT]]
; DOPCHECK-LABEL: define noundef float @frexp_float(
; DOPCHECK-NOT:  @llvm.frexp
; DOPCHECK-NOT:  extractvalue
; DOPCHECK-NOT:  insertvalue
  %r = call { float, i32 } @llvm.frexp.f32.i32(float %a)
  %m = extractvalue { float, i32 } %r, 0
  %x = extractvalue { float, i32 } %r, 1
  %xf = sitofp i32 %x to float
  store float %xf, ptr %e
  ret float %m
}

define noundef <4 x float> @frexp_float4(<4 x float> noundef %a, ptr %e) {
entry:
; EXPCHECK-LABEL: define noundef <4 x float> @frexp_float4(
; EXPCHECK:      [[BITS:%.*]] = bitcast <4 x float> %a to <4 x i32>
; EXPCHECK-NEXT: [[EXPBITS:%.*]] = and <4 x i32> [[BITS]], splat (i32 2139095040)
; EXPCHECK:      [[MANT:%.*]] = select <4 x i1> [[SPECIAL:%.*]], <4 x float> %a, <4 x float> {{%.*}}
; EXPCHECK:      [[EXP:%.*]] = select <4 x i1> [[SPECIAL]], <4 x i32> zeroinitializer, <4 x i32> {{%.*}}
; EXPCHECK-NEXT: [[XF:%.*]] = sitofp <4 x i32> [[EXP]] to <4 x float>
; EXPCHECK-NEXT: store <4 x float> [[XF]], ptr %e
; EXPCHECK-NEXT: ret <4 x float> [[MANT]]
; DOPCHECK-LABEL: define noundef <4 x float> @frexp_float4(
; DOPCHECK-NOT:  @llvm.frexp
; DOPCHECK-NOT:  extractvalue
; DOPCHECK-NOT:  insertvalue
  %r = call { <4 x float>, <4 x i32> } @llvm.frexp.v4f32.v4i32(<4 x float> %a)
  %m = extractvalue { <4 x float>, <4 x i32> } %r, 0
  %x = extractvalue { <4 x float>, <4 x i32> } %r, 1
  %xf = sitofp <4 x i32> %x to <4 x float>
  store <4 x float> %xf, ptr %e
  ret <4 x float> %m
}

; Only the extractvalue users are forwarded; any other use of the result still
; gets an aggregate.
define { float, i32 } @frexp_aggregate(float noundef %a) {
entry:
; EXPCHECK-LABEL: define { float, i32 } @frexp_aggregate(
; EXPCHECK:      [[MANT:%.*]] = select i1 [[SPECIAL:%.*]], float %a, float {{%.*}}
; EXPCHECK:      [[EXP:%.*]] = select i1 [[SPECIAL]], i32 0, i32 {{%.*}}
; EXPCHECK-NEXT: [[AGG0:%.*]] = insertvalue { float, i32 } poison, float [[MANT]], 0
; EXPCHECK-NEXT: [[AGG1:%.*]] = insertvalue { float, i32 } [[AGG0]], i32 [[EXP]], 1
; EXPCHECK-NEXT: ret { float, i32 } [[AGG1]]
; DOPCHECK-LABEL: define { float, i32 } @frexp_aggregate(
; DOPCHECK-NOT:  @llvm.frexp
  %r = call { float, i32 } @llvm.frexp.f32.i32(float %a)
  ret { float, i32 } %r
}
