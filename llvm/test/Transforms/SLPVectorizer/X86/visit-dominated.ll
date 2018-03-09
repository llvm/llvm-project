; RUN: opt -slp-vectorizer < %s -S | FileCheck %s
; Ensure each dominator block comes first in advance of its users.
; VEC_VALUE_QUALTYPE should dominate others.
; QUAL1_*(s) may be inducted by VEC_VALUE_QUALTYPE, since their pred is "entry".

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%AtomicInfo = type { %"class.clang::CodeGen::LValue" }
%"class.clang::QualType" = type { %"class.llvm::PointerIntPair.25" }
%"class.llvm::PointerIntPair.25" = type { i64 }
%"class.clang::CodeGen::LValue" = type { i32, i64*, %union.anon.1473, %"class.clang::QualType", %"class.clang::Qualifiers", i64, i8, [3 x i8], i64*, %"struct.clang::CodeGen::TBAAAccessInfo" }
%union.anon.1473 = type { %"class.llvm::Value"* }
%"class.llvm::Value" = type { i64*, i64*, i8, i8, i16, i32 }
%"class.clang::Qualifiers" = type { i32 }
%"struct.clang::CodeGen::TBAAAccessInfo" = type { %"class.clang::QualType", %"class.llvm::MDNode"*, i64 }
%"class.llvm::MDNode" = type { i64*, i32, i32, i64* }
%ExtQualsTypeCommonBase = type { %"class.clang::Type"*, %"class.clang::QualType" }
%"class.clang::Type" = type { %ExtQualsTypeCommonBase, %union.anon.26 }
%union.anon.26 = type { %"class.clang::Type::AttributedTypeBitfields", [4 x i8] }
%"class.clang::Type::AttributedTypeBitfields" = type { i32 }
%ExtQuals = type <{ %ExtQualsTypeCommonBase, %"class.llvm::FoldingSetBase::Node", %"class.clang::Qualifiers", [4 x i8] }>
%"class.llvm::FoldingSetBase::Node" = type { i8* }

define hidden fastcc void @_ZL21EmitAtomicUpdateValueRN5clang7CodeGen15CodeGenFunctionERN12_GLOBAL__N_110AtomicInfoENS0_6RValueENS0_7AddressE(%AtomicInfo* nocapture readonly dereferenceable(192) %Atomics) unnamed_addr {
entry:
  %agg = alloca %"class.clang::CodeGen::LValue", align 8
  %AtomicLValP00        = getelementptr inbounds %AtomicInfo, %AtomicInfo* %Atomics, i64 0, i32 0, i32 0

  %AtomicLValP02        = getelementptr inbounds %AtomicInfo, %AtomicInfo* %Atomics, i64 0, i32 0, i32 2, i32 0
; CHECK: [[VALUE0:%.+]] = getelementptr inbounds %AtomicInfo, %AtomicInfo* %Atomics, i64 0, i32 0, i32 2, i32 0
  %AtomicLValP03        = getelementptr inbounds %AtomicInfo, %AtomicInfo* %Atomics, i64 0, i32 0, i32 3, i32 0, i32 0

  %AtomicLVal = load i32, i32* %AtomicLValP00, align 8
  %tmp = bitcast %"class.llvm::Value"** %AtomicLValP02 to i64*
; CHECK: [[TMP:%.+]] = bitcast %"class.llvm::Value"** [[VALUE0]] to i64*

  %AtomicLVal.LValue = load i64, i64* %tmp, align 8
  %AtomicLVal.QualType = load i64, i64* %AtomicLValP03, align 8
; CHECK: [[VECP:%.+]] = bitcast i64* [[TMP]] to <2 x i64>*
; CHECK: [[VEC_VALUE_QUALTYPE:%.+]] = load <2 x i64>, <2 x i64>* [[VECP]], align 8

  switch i32 %AtomicLVal, label %if.else23 [
    i32 2, label %if.then
    i32 1, label %if.then11
  ]

; CHECK-LABEL: if.then11:
if.then11:                                        ; preds = %entry
; CHECK: [[QUAL1_11:%.+]] = extractelement <2 x i64> [[VEC_VALUE_QUALTYPE]], i32 1
  %and.57 = and i64 %AtomicLVal.QualType, -16
; CHECK:  = and i64 [[QUAL1_11]], -16

  %tmp5 = inttoptr i64 %and.57 to %ExtQualsTypeCommonBase*
  %Value.58 = getelementptr inbounds %ExtQualsTypeCommonBase, %ExtQualsTypeCommonBase* %tmp5, i64 0, i32 1, i32 0, i32 0
  %tmp6 = load i64, i64* %Value.58, align 8
  %tmp7 = and i64 %tmp6, 8
  %tobool.59 = icmp eq i64 %tmp7, 0
  br i1 %tobool.59, label %MakeVectorElt.exit, label %if.then.63

; CHECK-LABEL: if.then:
if.then:                                          ; preds = %entry
; CHECK: [[QUAL1:%.+]] = extractelement <2 x i64> [[VEC_VALUE_QUALTYPE]], i32 1
  %and.96 = and i64 %AtomicLVal.QualType, -16
; CHECK:  = and i64 [[QUAL1]], -16

  %tmp1 = inttoptr i64 %and.96 to %ExtQualsTypeCommonBase*
  %Value.97 = getelementptr inbounds %ExtQualsTypeCommonBase, %ExtQualsTypeCommonBase* %tmp1, i64 0, i32 1, i32 0, i32 0
  %tmp2 = load i64, i64* %Value.97, align 8
  %tmp3 = and i64 %tmp2, 8
  %tobool.98 = icmp eq i64 %tmp3, 0
  br i1 %tobool.98, label %MakeBitfield.exit, label %if.then.102

if.then.102:                                 ; preds = %if.then
  %and.99 = and i64 %tmp2, -16
  %tmp4 = inttoptr i64 %and.99 to %ExtQuals*
  %retval.100 = getelementptr inbounds %ExtQuals, %ExtQuals* %tmp4, i64 0, i32 2, i32 0
  %retval.101 = load i32, i32* %retval.100, align 8
  br label %MakeBitfield.exit

; CHECK_LABEL: MakeBitfield.exit:
MakeBitfield.exit: ; preds = %if.then.102, %if.then
  %retval.103 = phi i32 [ %retval.101, %if.then.102 ], [ 0, %if.then ]

  %conv.104 = or i64 %tmp2, %AtomicLVal.QualType
; CHECK:    = or i64 %tmp2, [[QUAL1]]

  %conv.105 = trunc i64 %conv.104 to i32
  %or.106 = and i32 %conv.105, 7
  %or.107 = or i32 %retval.103, %or.106
  br label %if.end35

if.then.63:                                  ; preds = %if.then11
  %and.60 = and i64 %tmp6, -16
  %tmp8 = inttoptr i64 %and.60 to %ExtQuals*
  %retval.61 = getelementptr inbounds %ExtQuals, %ExtQuals* %tmp8, i64 0, i32 2, i32 0
  %retval.62 = load i32, i32* %retval.61
  br label %MakeVectorElt.exit

; CHECK-LABEL:MakeVectorElt.exit:
MakeVectorElt.exit: ; preds = %if.then.63, %if.then11
  %retval.64 = phi i32 [ %retval.62, %if.then.63 ], [ 0, %if.then11 ]

  %conv.65 = or i64 %tmp6, %AtomicLVal.QualType
; CHECK:   = or i64 %tmp6, [[QUAL1_11]]

  %conv.66 = trunc i64 %conv.65 to i32
  %or.67 = and i32 %conv.66, 7
  %or.68 = or i32 %retval.64, %or.67
  br label %if.end35

; CHECK-LABEL: if.else23:
if.else23:                                        ; preds = %entry
; CHECK: [[QUAL1_23:%.+]] = extractelement <2 x i64> [[VEC_VALUE_QUALTYPE]], i32 1
  %and.0 = and i64 %AtomicLVal.QualType, -16
; CHECK: = and i64 [[QUAL1_23]], -16

  %tmp9 = inttoptr i64 %and.0 to %ExtQualsTypeCommonBase*
  %Value.9 = getelementptr inbounds %ExtQualsTypeCommonBase, %ExtQualsTypeCommonBase* %tmp9, i64 0, i32 1, i32 0, i32 0
  %tmp10 = load i64, i64* %Value.9, align 8
  %tmp11 = and i64 %tmp10, 8
  %tobool.0 = icmp eq i64 %tmp11, 0
  br i1 %tobool.0, label %MakeExtVectorElt.exit, label %MakeExtVectorElt.exit

; CHECK-LABEL:MakeExtVectorElt.exit:
MakeExtVectorElt.exit: ; preds = %MakeExtVectorElt.exit, %if.else23

  %conv.67 = or i64 %tmp10, %AtomicLVal.QualType
; CHECK:   = or i64 %tmp10, [[QUAL1_23]]

  %or.0 = trunc i64 %conv.67 to i32
  br label %if.end35

; CHECK-LABEL: if.end35:
if.end35:                                         ; preds = %MakeExtVectorElt.exit, %MakeVectorElt.exit, %MakeBitfield.exit
  %DesiredLVal = phi i32 [ %or.107, %MakeBitfield.exit ], [ %or.68, %MakeVectorElt.exit ], [ %or.0, %MakeExtVectorElt.exit ]

  %DesiredLValP.2      = getelementptr inbounds %"class.clang::CodeGen::LValue", %"class.clang::CodeGen::LValue"* %agg, i64 0, i32 2, i32 0
; CHECK: [[VALP2:%.+]] = getelementptr inbounds %"class.clang::CodeGen::LValue", %"class.clang::CodeGen::LValue"* %agg, i64 0, i32 2, i32 0
  %DesiredLValP.3      = getelementptr inbounds %"class.clang::CodeGen::LValue", %"class.clang::CodeGen::LValue"* %agg, i64 0, i32 3, i32 0, i32 0

  %tmp14               = bitcast %"class.llvm::Value"** %DesiredLValP.2 to i64*
; CHECK: [[TMP14:%.+]] = bitcast %"class.llvm::Value"** [[VALP2]] to i64*

  store i64 %AtomicLVal.LValue, i64* %tmp14, align 8
  store i64 %AtomicLVal.QualType, i64* %DesiredLValP.3, align 8
; CHECK: [[LVALUE:%.+]] = bitcast i64* [[TMP14]] to <2 x i64>*
; CHECK: store <2 x i64> [[VEC_VALUE_QUALTYPE]], <2 x i64>* [[LVALUE]], align 8

  %DesiredLValP = getelementptr inbounds %"class.clang::CodeGen::LValue", %"class.clang::CodeGen::LValue"* %agg, i64 0, i32 4, i32 0
  store i32 %DesiredLVal, i32* %DesiredLValP, align 8
  ret void
}
