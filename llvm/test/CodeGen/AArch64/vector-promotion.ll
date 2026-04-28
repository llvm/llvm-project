; RUN: opt -passes='require<profile-summary>,function(codegenprepare)' -mtriple=aarch64 %s -o - -S | FileCheck --check-prefix=IR-BOTH --check-prefix=IR-NORMAL %s
; RUN: opt -passes='require<profile-summary>,function(codegenprepare)' -mtriple=aarch64 %s -o - -S -stress-cgp-store-extract | FileCheck --check-prefix=IR-BOTH --check-prefix=IR-STRESS %s
; RUN: llc -mtriple=aarch64 %s -o - | FileCheck --check-prefix=ASM %s

; IR-BOTH-LABEL: @simpleOneInstructionPromotion
; IR-BOTH: [[LOAD:%[a-zA-Z_0-9-]+]] = load <2 x i32>, ptr %addr1
; IR-BOTH-NEXT: [[VECTOR_OR:%[a-zA-Z_0-9-]+]] = or <2 x i32> [[LOAD]], <i32 poison, i32 1>
; IR-BOTH-NEXT: [[EXTRACT:%[a-zA-Z_0-9-]+]] = extractelement <2 x i32> [[VECTOR_OR]], i32 1
; IR-BOTH-NEXT: store i32 [[EXTRACT]], ptr %dest
; IR-BOTH-NEXT: ret
; ASM-LABEL: simpleOneInstructionPromotion:
; ASM-NOT: umov
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
define void @unsupportedInstructionForPromotion(ptr %addr1, i32 %in2, ptr %dest) {
  %in1 = load <2 x i32>, ptr %addr1, align 8
  %extract = extractelement <2 x i32> %in1, i32 0
  %out = icmp eq i32 %extract, %in2
  store i1 %out, ptr %dest, align 4
  ret void
}

; IR-BOTH-LABEL: @chainOfInstructionsToPromote
; IR-BOTH: [[LOAD:%[a-zA-Z_0-9-]+]] = load <2 x i32>, ptr %addr1
; IR-BOTH-NEXT: [[VECTOR_OR1:%[a-zA-Z_0-9-]+]] = or <2 x i32> [[LOAD]], <i32 1, i32 poison>
; IR-BOTH-NEXT: [[VECTOR_OR2:%[a-zA-Z_0-9-]+]] = or <2 x i32> [[VECTOR_OR1]], <i32 1, i32 poison>
; IR-BOTH-NEXT: [[VECTOR_OR3:%[a-zA-Z_0-9-]+]] = or <2 x i32> [[VECTOR_OR2]], <i32 1, i32 poison>
; IR-BOTH-NEXT: [[EXTRACT:%[a-zA-Z_0-9-]+]] = extractelement <2 x i32> [[VECTOR_OR3]], i32 0
; IR-BOTH-NEXT: store i32 [[EXTRACT]], ptr %dest
; IR-BOTH-NEXT: ret
define void @chainOfInstructionsToPromote(ptr %addr1, ptr %dest) {
  %in1 = load <2 x i32>, ptr %addr1, align 8
  %extract = extractelement <2 x i32> %in1, i32 0
  %out1 = or i32 %extract, 1
  %out2 = or i32 %out1, 1
  %out3 = or i32 %out2, 1
  store i32 %out3, ptr %dest, align 4
  ret void
}

; IR-BOTH-LABEL: @simpleOneInstructionPromotionVariableIdx
; IR-BOTH: [[LOAD:%[a-zA-Z_0-9-]+]] = load <2 x i32>, ptr %addr1
; Scalar version:
; IR-NORMAL-NEXT: [[EXTRACT:%[a-zA-Z_0-9-]+]] = extractelement <2 x i32> [[LOAD]], i32 %idx
; IR-NORMAL-NEXT: [[RES:%[a-zA-Z_0-9-]+]] = or i32 [[EXTRACT]], 1
; Vector version:
; IR-STRESS-NEXT: [[OR:%[a-zA-Z_0-9-]+]] = or <2 x i32> [[LOAD]], splat (i32 1)
; IR-STRESS-NEXT: [[RES:%[a-zA-Z_0-9-]+]] = extractelement <2 x i32> [[OR]], i32 %idx
; IR-BOTH-NEXT: store i32 [[RES]], ptr %dest
; IR-BOTH-NEXT: ret
define void @simpleOneInstructionPromotionVariableIdx(ptr %addr1, ptr %dest, i32 %idx) {
  %in1 = load <2 x i32>, ptr %addr1, align 8
  %extract = extractelement <2 x i32> %in1, i32 %idx
  %out = or i32 %extract, 1
  store i32 %out, ptr %dest, align 4
  ret void
}
