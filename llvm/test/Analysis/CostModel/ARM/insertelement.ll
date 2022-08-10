; RUN: opt -passes="print<cost-model>" 2>&1 -disable-output -mtriple=thumbv7-apple-ios6.0.0 -mcpu=swift < %s | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:32:64-v128:32:128-a0:0:32-n32-S32"
target triple = "thumbv7-apple-ios6.0.0"

; Multiple insert elements from loads into d subregisters are expensive on swift
; due to renaming constraints.
%T_i8v = type <8 x i8>
%T_i8 = type i8
; CHECK: insertelement_i8
define void @insertelement_i8(%T_i8* %saddr,
                           %T_i8v* %vaddr) {
  %v0 = load %T_i8v, %T_i8v* %vaddr
  %v1 = load %T_i8, %T_i8* %saddr
;CHECK: estimated cost of 3 for {{.*}} insertelement <8 x i8>
  %v2 = insertelement %T_i8v %v0, %T_i8 %v1, i32 1
  store %T_i8v %v2, %T_i8v* %vaddr
  ret void
}


%T_i16v = type <4 x i16>
%T_i16 = type i16
; CHECK: insertelement_i16
define void @insertelement_i16(%T_i16* %saddr,
                           %T_i16v* %vaddr) {
  %v0 = load %T_i16v, %T_i16v* %vaddr
  %v1 = load %T_i16, %T_i16* %saddr
;CHECK: estimated cost of 3 for {{.*}} insertelement <4 x i16>
  %v2 = insertelement %T_i16v %v0, %T_i16 %v1, i32 1
  store %T_i16v %v2, %T_i16v* %vaddr
  ret void
}

%T_i32v = type <2 x i32>
%T_i32 = type i32
; CHECK: insertelement_i32
define void @insertelement_i32(%T_i32* %saddr,
                           %T_i32v* %vaddr) {
  %v0 = load %T_i32v, %T_i32v* %vaddr
  %v1 = load %T_i32, %T_i32* %saddr
;CHECK: estimated cost of 3 for {{.*}} insertelement <2 x i32>
  %v2 = insertelement %T_i32v %v0, %T_i32 %v1, i32 1
  store %T_i32v %v2, %T_i32v* %vaddr
  ret void
}
