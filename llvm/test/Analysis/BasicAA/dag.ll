; RUN: opt < %s -aa-pipeline=basic-aa -passes=aa-eval -print-all-alias-modref-info 2>&1 | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

; BasicAA's guard against use-def cycles shouldn't prevent it from
; analyzing use-def dags.

; CHECK: MustAlias:  i8* %base, i8* %phi
; CHECK: MustAlias: i16* %base, i8* %base
; CHECK: MustAlias: i16* %base, i8* %phi
define i8 @foo(ptr %base, i1 %x, i1 %w) {
entry:
  load i8, ptr %base
  br i1 %w, label %wa, label %wb
wa:
  load i8, ptr %base
  br label %wc
wb:
  load i8, ptr %base
  br label %wc
wc:
  %first = phi ptr [ %base, %wa ], [ %base, %wb ]
  br i1 %x, label %xa, label %xb
xa:
  br label %xc
xb:
  br label %xc
xc:
  %phi = phi ptr [ %first, %xa ], [ %first, %xb ]

  store i8 0, ptr %phi

  store i16 -1, ptr %base

  %loaded = load i8, ptr %phi
  ret i8 %loaded
}
