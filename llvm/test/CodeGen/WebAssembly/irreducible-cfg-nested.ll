; RUN: llc < %s -asm-verbose=false -verify-machineinstrs -disable-block-placement -wasm-disable-explicit-locals -wasm-keep-registers | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"


; Test an interesting pattern of nested irreducibility.
; Just check we resolve all the irreducibility here (if not we'd crash).

; CHECK-LABEL: tre_parse:

define void @tre_parse() {
entry:
  br label %for.cond.outer

for.cond.outer:                                   ; preds = %do.body14, %entry
  br label %for.cond

for.cond:                                         ; preds = %for.cond.backedge, %for.cond.outer
  %nbranch.0 = phi i32* [ null, %for.cond.outer ], [ %call188, %for.cond.backedge ]
  switch i8 undef, label %if.else [
    i8 40, label %do.body14
    i8 41, label %if.then63
  ]

do.body14:                                        ; preds = %for.cond
  br label %for.cond.outer

if.then63:                                        ; preds = %for.cond
  unreachable

if.else:                                          ; preds = %for.cond
  switch i8 undef, label %if.then84 [
    i8 92, label %if.end101
    i8 42, label %if.end101
  ]

if.then84:                                        ; preds = %if.else
  switch i8 undef, label %cleanup.thread [
    i8 43, label %if.end101
    i8 63, label %if.end101
    i8 123, label %if.end101
  ]

if.end101:                                        ; preds = %if.then84, %if.then84, %if.then84, %if.else, %if.else
  unreachable

cleanup.thread:                                   ; preds = %if.then84
  %call188 = tail call i32* undef(i32* %nbranch.0)
  switch i8 undef, label %for.cond.backedge [
    i8 92, label %land.lhs.true208
    i8 0, label %if.else252
  ]

land.lhs.true208:                                 ; preds = %cleanup.thread
  unreachable

for.cond.backedge:                                ; preds = %cleanup.thread
  br label %for.cond

if.else252:                                       ; preds = %cleanup.thread
  unreachable
}
