; RUN: opt < %s -aa-pipeline=basic-aa -passes=aa-eval -print-all-alias-modref-info 2>&1 | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

; BasicAA's guard against use-def cycles shouldn't prevent it from
; analyzing use-def dags.

; CHECK: MustAlias:  i8* %base, i8* %phi
; CHECK: MustAlias: i8* %phi, i8* %wwa
; CHECK: MustAlias: i8* %phi, i8* %wwb
; CHECK: MustAlias: i16* %bigbase, i8* %phi
define i8 @foo(i8* %base, i1 %x, i1 %w) {
entry:
  br i1 %w, label %wa, label %wb
wa:
  %wwa = bitcast i8* %base to i8*
  br label %wc
wb:
  %wwb = bitcast i8* %base to i8*
  br label %wc
wc:
  %first = phi i8* [ %wwa, %wa ], [ %wwb, %wb ]
  %fc = bitcast i8* %first to i8*
  br i1 %x, label %xa, label %xb
xa:
  %xxa = bitcast i8* %fc to i8*
  br label %xc
xb:
  %xxb = bitcast i8* %fc to i8*
  br label %xc
xc:
  %phi = phi i8* [ %xxa, %xa ], [ %xxb, %xb ]

  store i8 0, i8* %phi

  %bigbase = bitcast i8* %base to i16*
  store i16 -1, i16* %bigbase

  %loaded = load i8, i8* %phi
  ret i8 %loaded
}
