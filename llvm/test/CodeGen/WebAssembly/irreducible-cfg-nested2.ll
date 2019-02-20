; RUN: llc < %s -asm-verbose=false -verify-machineinstrs -disable-block-placement -wasm-disable-explicit-locals -wasm-keep-registers | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; Test an interesting pattern of nested irreducibility.
; Just check we resolve all the irreducibility here (if not we'd crash).

; CHECK-LABEL: func_2:

; Function Attrs: noinline nounwind optnone
define void @func_2() {
entry:
  br i1 undef, label %lbl_937, label %if.else787

lbl_937:                                          ; preds = %for.body978, %entry
  br label %if.end965

if.else787:                                       ; preds = %entry
  br label %if.end965

if.end965:                                        ; preds = %if.else787, %lbl_937
  br label %for.cond967

for.cond967:                                      ; preds = %for.end1035, %if.end965
  br label %for.cond975

for.cond975:                                      ; preds = %if.end984, %for.cond967
  br i1 undef, label %for.body978, label %for.end1035

for.body978:                                      ; preds = %for.cond975
  br i1 undef, label %lbl_937, label %if.end984

if.end984:                                        ; preds = %for.body978
  br label %for.cond975

for.end1035:                                      ; preds = %for.cond975
  br label %for.cond967
}
