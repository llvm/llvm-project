; RUN: opt -passes='require<profile-summary>,function(codegenprepare)' -mtriple=thumbv7-apple-ios %s -o - -mattr=+neon -S | FileCheck --check-prefix=IR-BOTH --check-prefix=IR-NORMAL %s
; RUN: opt -passes='require<profile-summary>,function(codegenprepare)' -mtriple=thumbv7-apple-ios %s -o - -mattr=+neon -S -stress-cgp-store-extract | FileCheck --check-prefix=IR-BOTH --check-prefix=IR-STRESS %s
; RUN: llc -mtriple=thumbv7-apple-ios %s -o - -mattr=+neon | FileCheck --check-prefix=ASM %s

; IR-BOTH-LABEL: @simpleOneInstructionPromotion
; IR-BOTH: [[LOAD:%[a-zA-Z_0-9-]+]] = load <2 x i32>, ptr %addr1
; IR-BOTH-NEXT: [[VECTOR_OR:%[a-zA-Z_0-9-]+]] = or <2 x i32> [[LOAD]], <i32 poison, i32 1>
; IR-BOTH-NEXT: [[EXTRACT:%[a-zA-Z_0-9-]+]] = extractelement <2 x i32> [[VECTOR_OR]], i32 1
; IR-BOTH-NEXT: store i32 [[EXTRACT]], ptr %dest
; IR-BOTH-NEXT: ret
;
; Make sure we got rid of any expensive vmov.32 instructions.
; ASM-LABEL: simpleOneInstructionPromotion:
; ASM: vldr [[LOAD:d[0-9]+]], [r0]
; ASM-NEXT: vorr.i32 [[LOAD]], #0x1
; ASM-NEXT: vst1.32 {[[LOAD]][1]}, [r1:32]
; ASM-NEXT: bx
define void @simpleOneInstructionPromotion(ptr %addr1, ptr %dest) {
  %in1 = load <2 x i32>, ptr %addr1, align 8
  %extract = extractelement <2 x i32> %in1, i32 1
  %out = or i32 %extract, 1
  store i32 %out, ptr %dest, align 4
  ret void
}

; IR-BOTH-LABEL: @unsupportedInstructionForPromotion
; IR-BOTH: [[LOAD:%[a-zA-Z_0-9-]+]] = load <2 x i32>, ptr %addr1
; IR-BOTH-NEXT: [[EXTRACT:%[a-zA-Z_0-9-]+]] = extractelement <2 x i32> [[LOAD]], i32 0
; IR-BOTH-NEXT: [[CMP:%[a-zA-Z_0-9-]+]] = icmp eq i32 [[EXTRACT]], %in2
; IR-BOTH-NEXT: store i1 [[CMP]], ptr %dest
; IR-BOTH-NEXT: ret
;
; ASM-LABEL: unsupportedInstructionForPromotion:
; ASM: vldr [[LOAD:d[0-9]+]], [r0]
; ASM: vmov.32 {{r[0-9]+}}, [[LOAD]]
; ASM: bx
define void @unsupportedInstructionForPromotion(ptr %addr1, i32 %in2, ptr %dest) {
  %in1 = load <2 x i32>, ptr %addr1, align 8
  %extract = extractelement <2 x i32> %in1, i32 0
  %out = icmp eq i32 %extract, %in2
  store i1 %out, ptr %dest, align 4
  ret void
}


; IR-BOTH-LABEL: @unsupportedChainInDifferentBBs
; IR-BOTH: [[LOAD:%[a-zA-Z_0-9-]+]] = load <2 x i32>, ptr %addr1
; IR-BOTH-NEXT: [[EXTRACT:%[a-zA-Z_0-9-]+]] = extractelement <2 x i32> [[LOAD]], i32 0
; IR-BOTH-NEXT: br i1 %bool, label %bb2, label %end
; BB2
; IR-BOTH: [[OR:%[a-zA-Z_0-9-]+]] = or i32 [[EXTRACT]], 1
; IR-BOTH-NEXT: store i32 [[OR]], ptr %dest, align 4
; IR-BOTH: ret
;
; ASM-LABEL: unsupportedChainInDifferentBBs:
; ASM: vldr [[LOAD:d[0-9]+]], [r0]
; ASM: vmov.32 {{r[0-9]+}}, [[LOAD]]
; ASM: bx
define void @unsupportedChainInDifferentBBs(ptr %addr1, ptr %dest, i1 %bool) {
bb1:
  %in1 = load <2 x i32>, ptr %addr1, align 8
  %extract = extractelement <2 x i32> %in1, i32 0
  br i1 %bool, label %bb2, label %end
bb2: 
  %out = or i32 %extract, 1
  store i32 %out, ptr %dest, align 4
  br label %end
end:
  ret void
}

; IR-LABEL: @chainOfInstructionsToPromote
; IR-BOTH: [[LOAD:%[a-zA-Z_0-9-]+]] = load <2 x i32>, ptr %addr1
; IR-BOTH-NEXT: [[VECTOR_OR1:%[a-zA-Z_0-9-]+]] = or <2 x i32> [[LOAD]], <i32 1, i32 poison>
; IR-BOTH-NEXT: [[VECTOR_OR2:%[a-zA-Z_0-9-]+]] = or <2 x i32> [[VECTOR_OR1]], <i32 1, i32 poison>
; IR-BOTH-NEXT: [[VECTOR_OR3:%[a-zA-Z_0-9-]+]] = or <2 x i32> [[VECTOR_OR2]], <i32 1, i32 poison>
; IR-BOTH-NEXT: [[VECTOR_OR4:%[a-zA-Z_0-9-]+]] = or <2 x i32> [[VECTOR_OR3]], <i32 1, i32 poison>
; IR-BOTH-NEXT: [[VECTOR_OR5:%[a-zA-Z_0-9-]+]] = or <2 x i32> [[VECTOR_OR4]], <i32 1, i32 poison>
; IR-BOTH-NEXT: [[VECTOR_OR6:%[a-zA-Z_0-9-]+]] = or <2 x i32> [[VECTOR_OR5]], <i32 1, i32 poison>
; IR-BOTH-NEXT: [[VECTOR_OR7:%[a-zA-Z_0-9-]+]] = or <2 x i32> [[VECTOR_OR6]], <i32 1, i32 poison>
; IR-BOTH-NEXT: [[EXTRACT:%[a-zA-Z_0-9-]+]] = extractelement <2 x i32> [[VECTOR_OR7]], i32 0
; IR-BOTH-NEXT: store i32 [[EXTRACT]], ptr %dest
; IR-BOTH-NEXT: ret
;
; ASM-LABEL: chainOfInstructionsToPromote:
; ASM: vldr [[LOAD:d[0-9]+]], [r0]
; ASM-NOT: vmov.32 {{r[0-9]+}}, [[LOAD]]
; ASM: bx
define void @chainOfInstructionsToPromote(ptr %addr1, ptr %dest) {
  %in1 = load <2 x i32>, ptr %addr1, align 8
  %extract = extractelement <2 x i32> %in1, i32 0
  %out1 = or i32 %extract, 1
  %out2 = or i32 %out1, 1
  %out3 = or i32 %out2, 1
  %out4 = or i32 %out3, 1
  %out5 = or i32 %out4, 1
  %out6 = or i32 %out5, 1
  %out7 = or i32 %out6, 1
  store i32 %out7, ptr %dest, align 4
  ret void
}

; IR-BOTH-LABEL: @unsupportedMultiUses
; IR-BOTH: [[LOAD:%[a-zA-Z_0-9-]+]] = load <2 x i32>, ptr %addr1
; IR-BOTH-NEXT: [[EXTRACT:%[a-zA-Z_0-9-]+]] = extractelement <2 x i32> [[LOAD]], i32 1
; IR-BOTH-NEXT: [[OR:%[a-zA-Z_0-9-]+]] = or i32 [[EXTRACT]], 1
; IR-BOTH-NEXT: store i32 [[OR]], ptr %dest
; IR-BOTH-NEXT: ret i32 [[OR]]
;
; ASM-LABEL: unsupportedMultiUses:
; ASM: vldr [[LOAD:d[0-9]+]], [r0]
; ASM: vmov.32 {{r[0-9]+}}, [[LOAD]]
; ASM: bx
define i32 @unsupportedMultiUses(ptr %addr1, ptr %dest) {
  %in1 = load <2 x i32>, ptr %addr1, align 8
  %extract = extractelement <2 x i32> %in1, i32 1
  %out = or i32 %extract, 1
  store i32 %out, ptr %dest, align 4
  ret i32 %out
}

; Check that we promote we a splat constant when this is a division.
; The NORMAL mode does not promote anything as divisions are not legal.
; IR-BOTH-LABEL: @udivCase
; IR-BOTH: [[LOAD:%[a-zA-Z_0-9-]+]] = load <2 x i32>, ptr %addr1
; Scalar version:
; IR-NORMAL-NEXT: [[EXTRACT:%[a-zA-Z_0-9-]+]] = extractelement <2 x i32> [[LOAD]], i32 1
; IR-NORMAL-NEXT: [[RES:%[a-zA-Z_0-9-]+]] = udiv i32 [[EXTRACT]], 7
; Vector version:
; IR-STRESS-NEXT: [[DIV:%[a-zA-Z_0-9-]+]] = udiv <2 x i32> [[LOAD]], splat (i32 7)
; IR-STRESS-NEXT: [[RES:%[a-zA-Z_0-9-]+]] = extractelement <2 x i32> [[DIV]], i32 1
;
; IR-BOTH-NEXT: store i32 [[RES]], ptr %dest
; IR-BOTH-NEXT: ret
define void @udivCase(ptr %addr1, ptr %dest) {
  %in1 = load <2 x i32>, ptr %addr1, align 8
  %extract = extractelement <2 x i32> %in1, i32 1
  %out = udiv i32 %extract, 7
  store i32 %out, ptr %dest, align 4
  ret void
}

; IR-BOTH-LABEL: @uremCase
; IR-BOTH: [[LOAD:%[a-zA-Z_0-9-]+]] = load <2 x i32>, ptr %addr1
; Scalar version:
; IR-NORMAL-NEXT: [[EXTRACT:%[a-zA-Z_0-9-]+]] = extractelement <2 x i32> [[LOAD]], i32 1
; IR-NORMAL-NEXT: [[RES:%[a-zA-Z_0-9-]+]] = urem i32 [[EXTRACT]], 7
; Vector version:
; IR-STRESS-NEXT: [[DIV:%[a-zA-Z_0-9-]+]] = urem <2 x i32> [[LOAD]], splat (i32 7)
; IR-STRESS-NEXT: [[RES:%[a-zA-Z_0-9-]+]] = extractelement <2 x i32> [[DIV]], i32 1
;
; IR-BOTH-NEXT: store i32 [[RES]], ptr %dest
; IR-BOTH-NEXT: ret 
define void @uremCase(ptr %addr1, ptr %dest) {
  %in1 = load <2 x i32>, ptr %addr1, align 8
  %extract = extractelement <2 x i32> %in1, i32 1
  %out = urem i32 %extract, 7
  store i32 %out, ptr %dest, align 4
  ret void
}

; IR-BOTH-LABEL: @sdivCase
; IR-BOTH: [[LOAD:%[a-zA-Z_0-9-]+]] = load <2 x i32>, ptr %addr1
; Scalar version:
; IR-NORMAL-NEXT: [[EXTRACT:%[a-zA-Z_0-9-]+]] = extractelement <2 x i32> [[LOAD]], i32 1
; IR-NORMAL-NEXT: [[RES:%[a-zA-Z_0-9-]+]] = sdiv i32 [[EXTRACT]], 7
; Vector version:
; IR-STRESS-NEXT: [[DIV:%[a-zA-Z_0-9-]+]] = sdiv <2 x i32> [[LOAD]], splat (i32 7)
; IR-STRESS-NEXT: [[RES:%[a-zA-Z_0-9-]+]] = extractelement <2 x i32> [[DIV]], i32 1
;
; IR-BOTH-NEXT: store i32 [[RES]], ptr %dest
; IR-BOTH-NEXT: ret 
define void @sdivCase(ptr %addr1, ptr %dest) {
  %in1 = load <2 x i32>, ptr %addr1, align 8
  %extract = extractelement <2 x i32> %in1, i32 1
  %out = sdiv i32 %extract, 7
  store i32 %out, ptr %dest, align 4
  ret void
}

; IR-BOTH-LABEL: @sremCase
; IR-BOTH: [[LOAD:%[a-zA-Z_0-9-]+]] = load <2 x i32>, ptr %addr1
; Scalar version:
; IR-NORMAL-NEXT: [[EXTRACT:%[a-zA-Z_0-9-]+]] = extractelement <2 x i32> [[LOAD]], i32 1
; IR-NORMAL-NEXT: [[RES:%[a-zA-Z_0-9-]+]] = srem i32 [[EXTRACT]], 7
; Vector version:
; IR-STRESS-NEXT: [[DIV:%[a-zA-Z_0-9-]+]] = srem <2 x i32> [[LOAD]], splat (i32 7)
; IR-STRESS-NEXT: [[RES:%[a-zA-Z_0-9-]+]] = extractelement <2 x i32> [[DIV]], i32 1
;
; IR-BOTH-NEXT: store i32 [[RES]], ptr %dest
; IR-BOTH-NEXT: ret 
define void @sremCase(ptr %addr1, ptr %dest) {
  %in1 = load <2 x i32>, ptr %addr1, align 8
  %extract = extractelement <2 x i32> %in1, i32 1
  %out = srem i32 %extract, 7
  store i32 %out, ptr %dest, align 4
  ret void
}

; IR-BOTH-LABEL: @fdivCase
; IR-BOTH: [[LOAD:%[a-zA-Z_0-9-]+]] = load <2 x float>, ptr %addr1
; Scalar version:  
; IR-NORMAL-NEXT: [[EXTRACT:%[a-zA-Z_0-9-]+]] = extractelement <2 x float> [[LOAD]], i32 1
; IR-NORMAL-NEXT: [[RES:%[a-zA-Z_0-9-]+]] = fdiv float [[EXTRACT]], 7.0
; Vector version:
; IR-STRESS-NEXT: [[DIV:%[a-zA-Z_0-9-]+]] = fdiv <2 x float> [[LOAD]], splat (float 7.000000e+00)
; IR-STRESS-NEXT: [[RES:%[a-zA-Z_0-9-]+]] = extractelement <2 x float> [[DIV]], i32 1
;
; IR-BOTH-NEXT: store float [[RES]], ptr %dest
; IR-BOTH-NEXT: ret
define void @fdivCase(ptr %addr1, ptr %dest) {
  %in1 = load <2 x float>, ptr %addr1, align 8   
  %extract = extractelement <2 x float> %in1, i32 1
  %out = fdiv float %extract, 7.0
  store float %out, ptr %dest, align 4
  ret void
}

; IR-BOTH-LABEL: @fremCase
; IR-BOTH: [[LOAD:%[a-zA-Z_0-9-]+]] = load <2 x float>, ptr %addr1
; Scalar version:  
; IR-NORMAL-NEXT: [[EXTRACT:%[a-zA-Z_0-9-]+]] = extractelement <2 x float> [[LOAD]], i32 1
; IR-NORMAL-NEXT: [[RES:%[a-zA-Z_0-9-]+]] = frem float [[EXTRACT]], 7.0
; Vector version:
; IR-STRESS-NEXT: [[DIV:%[a-zA-Z_0-9-]+]] = frem <2 x float> [[LOAD]], splat (float 7.000000e+00)
; IR-STRESS-NEXT: [[RES:%[a-zA-Z_0-9-]+]] = extractelement <2 x float> [[DIV]], i32 1
;
; IR-BOTH-NEXT: store float [[RES]], ptr %dest
; IR-BOTH-NEXT: ret
define void @fremCase(ptr %addr1, ptr %dest) {
  %in1 = load <2 x float>, ptr %addr1, align 8   
  %extract = extractelement <2 x float> %in1, i32 1
  %out = frem float %extract, 7.0
  store float %out, ptr %dest, align 4
  ret void
}

; Check that we do not promote when we may introduce undefined behavior
; like division by zero.
; IR-BOTH-LABEL: @undefDivCase
; IR-BOTH: [[LOAD:%[a-zA-Z_0-9-]+]] = load <2 x i32>, ptr %addr1
; IR-BOTH-NEXT: [[EXTRACT:%[a-zA-Z_0-9-]+]] = extractelement <2 x i32> [[LOAD]], i32 1
; IR-BOTH-NEXT: [[RES:%[a-zA-Z_0-9-]+]] = udiv i32 7, [[EXTRACT]]
; IR-BOTH-NEXT: store i32 [[RES]], ptr %dest
; IR-BOTH-NEXT: ret
define void @undefDivCase(ptr %addr1, ptr %dest) {
  %in1 = load <2 x i32>, ptr %addr1, align 8
  %extract = extractelement <2 x i32> %in1, i32 1
  %out = udiv i32 7, %extract
  store i32 %out, ptr %dest, align 4
  ret void
}


; Check that we do not promote when we may introduce undefined behavior
; like division by zero.
; IR-BOTH-LABEL: @undefRemCase
; IR-BOTH: [[LOAD:%[a-zA-Z_0-9-]+]] = load <2 x i32>, ptr %addr1
; IR-BOTH-NEXT: [[EXTRACT:%[a-zA-Z_0-9-]+]] = extractelement <2 x i32> [[LOAD]], i32 1
; IR-BOTH-NEXT: [[RES:%[a-zA-Z_0-9-]+]] = srem i32 7, [[EXTRACT]]
; IR-BOTH-NEXT: store i32 [[RES]], ptr %dest
; IR-BOTH-NEXT: ret
define void @undefRemCase(ptr %addr1, ptr %dest) {
  %in1 = load <2 x i32>, ptr %addr1, align 8
  %extract = extractelement <2 x i32> %in1, i32 1
  %out = srem i32 7, %extract
  store i32 %out, ptr %dest, align 4
  ret void
}

; Check that we use an undef mask for undefined behavior if the fast-math
; flag is set.
; IR-BOTH-LABEL: @undefConstantFRemCaseWithFastMath
; IR-BOTH: [[LOAD:%[a-zA-Z_0-9-]+]] = load <2 x float>, ptr %addr1
; Scalar version:  
; IR-NORMAL-NEXT: [[EXTRACT:%[a-zA-Z_0-9-]+]] = extractelement <2 x float> [[LOAD]], i32 1
; IR-NORMAL-NEXT: [[RES:%[a-zA-Z_0-9-]+]] = frem nnan float [[EXTRACT]], 7.0
; Vector version:
; IR-STRESS-NEXT: [[DIV:%[a-zA-Z_0-9-]+]] = frem nnan <2 x float> [[LOAD]], <float poison, float 7.000000e+00>
; IR-STRESS-NEXT: [[RES:%[a-zA-Z_0-9-]+]] = extractelement <2 x float> [[DIV]], i32 1
;
; IR-BOTH-NEXT: store float [[RES]], ptr %dest
; IR-BOTH-NEXT: ret
define void @undefConstantFRemCaseWithFastMath(ptr %addr1, ptr %dest) {
  %in1 = load <2 x float>, ptr %addr1, align 8   
  %extract = extractelement <2 x float> %in1, i32 1
  %out = frem nnan float %extract, 7.0
  store float %out, ptr %dest, align 4
  ret void
}

; Check that we use an undef mask for undefined behavior if the fast-math
; flag is set.
; IR-BOTH-LABEL: @undefVectorFRemCaseWithFastMath
; IR-BOTH: [[LOAD:%[a-zA-Z_0-9-]+]] = load <2 x float>, ptr %addr1
; Scalar version:  
; IR-NORMAL-NEXT: [[EXTRACT:%[a-zA-Z_0-9-]+]] = extractelement <2 x float> [[LOAD]], i32 1
; IR-NORMAL-NEXT: [[RES:%[a-zA-Z_0-9-]+]] = frem nnan float 7.000000e+00, [[EXTRACT]]
; Vector version:
; IR-STRESS-NEXT: [[DIV:%[a-zA-Z_0-9-]+]] = frem nnan <2 x float> <float poison, float 7.000000e+00>, [[LOAD]]
; IR-STRESS-NEXT: [[RES:%[a-zA-Z_0-9-]+]] = extractelement <2 x float> [[DIV]], i32 1
;
; IR-BOTH-NEXT: store float [[RES]], ptr %dest
; IR-BOTH-NEXT: ret
define void @undefVectorFRemCaseWithFastMath(ptr %addr1, ptr %dest) {
  %in1 = load <2 x float>, ptr %addr1, align 8   
  %extract = extractelement <2 x float> %in1, i32 1
  %out = frem nnan float 7.0, %extract
  store float %out, ptr %dest, align 4
  ret void
}

; Check that we are able to promote floating point value.
; This requires the STRESS mode, as floating point value are
; not promote on armv7.
; IR-BOTH-LABEL: @simpleOneInstructionPromotionFloat
; IR-BOTH: [[LOAD:%[a-zA-Z_0-9-]+]] = load <2 x float>, ptr %addr1
; Scalar version: 
; IR-NORMAL-NEXT: [[EXTRACT:%[a-zA-Z_0-9-]+]] = extractelement <2 x float> [[LOAD]], i32 1
; IR-NORMAL-NEXT: [[RES:%[a-zA-Z_0-9-]+]] = fadd float [[EXTRACT]], 1.0
; Vector version:
; IR-STRESS-NEXT: [[DIV:%[a-zA-Z_0-9-]+]] = fadd <2 x float> [[LOAD]], <float poison, float 1.000000e+00>
; IR-STRESS-NEXT: [[RES:%[a-zA-Z_0-9-]+]] = extractelement <2 x float> [[DIV]], i32 1
;
; IR-BOTH-NEXT: store float [[RES]], ptr %dest
; IR-BOTH-NEXT: ret
define void @simpleOneInstructionPromotionFloat(ptr %addr1, ptr %dest) {
  %in1 = load <2 x float>, ptr %addr1, align 8
  %extract = extractelement <2 x float> %in1, i32 1
  %out = fadd float %extract, 1.0
  store float %out, ptr %dest, align 4
  ret void
}

; Check that we correctly use a splat constant when we cannot
; determine at compile time the index of the extract.
; This requires the STRESS modes, as variable index are expensive
; to lower.
; IR-BOTH-LABEL: @simpleOneInstructionPromotionVariableIdx
; IR-BOTH: [[LOAD:%[a-zA-Z_0-9-]+]] = load <2 x i32>, ptr %addr1
; Scalar version:
; IR-NORMAL-NEXT: [[EXTRACT:%[a-zA-Z_0-9-]+]] = extractelement <2 x i32> [[LOAD]], i32 %idx
; IR-NORMAL-NEXT: [[RES:%[a-zA-Z_0-9-]+]] = or i32 [[EXTRACT]], 1
; Vector version:
; IR-STRESS-NEXT: [[OR:%[a-zA-Z_0-9-]+]] = or <2 x i32> [[LOAD]], splat (i32 1)
; IR-STRESS-NEXT: [[RES:%[a-zA-Z_0-9-]+]] = extractelement <2 x i32> [[OR]], i32 %idx
;
; IR-BOTH-NEXT: store i32 [[RES]], ptr %dest
; IR-BOTH-NEXT: ret
define void @simpleOneInstructionPromotionVariableIdx(ptr %addr1, ptr %dest, i32 %idx) {
  %in1 = load <2 x i32>, ptr %addr1, align 8
  %extract = extractelement <2 x i32> %in1, i32 %idx
  %out = or i32 %extract, 1
  store i32 %out, ptr %dest, align 4
  ret void
}

; Check a vector with more than 2 elements.
; IR-BOTH-LABEL: @simpleOneInstructionPromotion8x8
; IR-BOTH: [[LOAD:%[a-zA-Z_0-9-]+]] = load <8 x i8>, ptr %addr1
; IR-BOTH-NEXT: [[OR:%[a-zA-Z_0-9-]+]] = or <8 x i8> [[LOAD]], <i8 poison, i8 1, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison>
; IR-BOTH-NEXT: [[RES:%[a-zA-Z_0-9-]+]] = extractelement <8 x i8> [[OR]], i32 1
; IR-BOTH-NEXT: store i8 [[RES]], ptr %dest
; IR-BOTH-NEXT: ret
define void @simpleOneInstructionPromotion8x8(ptr %addr1, ptr %dest) {
  %in1 = load <8 x i8>, ptr %addr1, align 8
  %extract = extractelement <8 x i8> %in1, i32 1
  %out = or i8 %extract, 1
  store i8 %out, ptr %dest, align 4
  ret void
}

; Check that we optimized the sequence correctly when it can be
; lowered on a Q register.
; IR-BOTH-LABEL: @simpleOneInstructionPromotion
; IR-BOTH: [[LOAD:%[a-zA-Z_0-9-]+]] = load <4 x i32>, ptr %addr1
; IR-BOTH-NEXT: [[VECTOR_OR:%[a-zA-Z_0-9-]+]] = or <4 x i32> [[LOAD]], <i32 poison, i32 1, i32 poison, i32 poison>
; IR-BOTH-NEXT: [[EXTRACT:%[a-zA-Z_0-9-]+]] = extractelement <4 x i32> [[VECTOR_OR]], i32 1
; IR-BOTH-NEXT: store i32 [[EXTRACT]], ptr %dest
; IR-BOTH-NEXT: ret
;
; Make sure we got rid of any expensive vmov.32 instructions.
; ASM-LABEL: simpleOneInstructionPromotion4x32:
; ASM: vld1.64 {[[LOAD:d[0-9]+]], d{{[0-9]+}}}, [r0]
; The Q register used here must be [[LOAD]] / 2, but we cannot express that.
; ASM-NEXT: vorr.i32 q{{[[0-9]+}}, #0x1
; ASM-NEXT: vst1.32 {[[LOAD]][1]}, [r1]
; ASM-NEXT: bx
define void @simpleOneInstructionPromotion4x32(ptr %addr1, ptr %dest) {
  %in1 = load <4 x i32>, ptr %addr1, align 8
  %extract = extractelement <4 x i32> %in1, i32 1
  %out = or i32 %extract, 1
  store i32 %out, ptr %dest, align 1
  ret void
}
