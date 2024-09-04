; RUN: llc -mtriple=dxil-pc-shadermodel6.3-library -debug-pass=Structure < %s -o /dev/null 2>&1 | \
; RUN:     grep -v "Verify generated machine code" | FileCheck %s
; RUN: llc %s -mtriple=dxil-pc-shadermodel6.3-library --filetype=asm -o - | FileCheck %s --check-prefixes=CHECKIR
; CHECK-LABEL: Pass Arguments:
; CHECK-NEXT: Target Library Information
; CHECK-NEXT: ModulePass Manager
; CHECK-NEXT:   DXIL Intrinsic Expansion
; CHECK-NEXT:   FunctionPass Manager
; CHECK-NEXT:     Dominator Tree Construction
; CHECK-NEXT:     Scalarize vector operations
; CHECK-NEXT:   DXIL Intrinsic Expansion
; CHECK-NEXT:   DXIL Resource analysis
; CHECK-NEXT:   DXIL Op Lowering
; CHECK-NEXT:   DXIL Finalize Linkage
; CHECK-NEXT:   DXIL Resource analysis
; CHECK-NEXT:   DXIL resource Information
; CHECK-NEXT:   DXIL Shader Flag Analysis
; CHECK-NEXT:   DXIL Translate Metadata
; CHECK-NEXT:   DXIL Prepare Module
; CHECK-NEXT:   DXIL Resource analysis
; CHECK-NEXT:   DXIL Metadata Pretty Printer
; CHECK-NEXT:   Print Module IR
; CHECKIR: target triple = "dxilv1.3-pc-shadermodel6.3-library"
; CHECKIR-LABEL: cos_sin_float_test
define noundef <4 x float> @cos_sin_float_test(<4 x float> noundef %a) {
    ; CHECKIR: [[ee0:%.*]] = extractelement <4 x float> %a, i64 0
    ; CHECKIR: [[ie0:%.*]] = call float @dx.op.unary.f32(i32 13, float [[ee0]])
    ; CHECKIR: [[ee1:%.*]] = extractelement <4 x float> %a, i64 1
    ; CHECKIR: [[ie1:%.*]] = call float @dx.op.unary.f32(i32 13, float [[ee1]])
    ; CHECKIR: [[ee2:%.*]] = extractelement <4 x float> %a, i64 2
    ; CHECKIR: [[ie2:%.*]] = call float @dx.op.unary.f32(i32 13, float [[ee2]])
    ; CHECKIR: [[ee3:%.*]] = extractelement <4 x float> %a, i64 3
    ; CHECKIR: [[ie3:%.*]] = call float @dx.op.unary.f32(i32 13, float [[ee3]])
    ; CHECKIR: [[ie4:%.*]] = call float @dx.op.unary.f32(i32 12, float [[ie0]])
    ; CHECKIR: [[ie5:%.*]] = call float @dx.op.unary.f32(i32 12, float [[ie1]])
    ; CHECKIR: [[ie6:%.*]] = call float @dx.op.unary.f32(i32 12, float [[ie2]])
    ; CHECKIR: [[ie7:%.*]] = call float @dx.op.unary.f32(i32 12, float [[ie3]])
    ; CHECKIR: insertelement <4 x float> poison, float [[ie4]], i64 0
    ; CHECKIR: insertelement <4 x float> %{{.*}}, float [[ie5]], i64 1
    ; CHECKIR: insertelement <4 x float> %{{.*}}, float [[ie6]], i64 2
    ; CHECKIR: insertelement <4 x float> %{{.*}}, float [[ie7]], i64 3
    %2 = tail call <4 x float> @llvm.sin.v4f32(<4 x float> %a) 
    %3 = tail call <4 x float> @llvm.cos.v4f32(<4 x float> %2) 
    ret <4 x float> %3 
} 
