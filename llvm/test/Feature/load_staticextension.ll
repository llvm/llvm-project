; REQUIRES: x86-registered-target
; RUN: opt-printplugin %s -passes="printpass" -disable-output 2>&1 | FileCheck %s

; REQUIRES: plugins

; CHECK: [PrintPass] Found function: somefunk

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"
@junk = global i32 0

define ptr @somefunk() {
  ret ptr @junk
}

