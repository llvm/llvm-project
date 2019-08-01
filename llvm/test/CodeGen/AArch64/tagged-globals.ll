; RUN: llc < %s | FileCheck %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-android"

@global = external hidden global i32
declare void @func()

define i32* @global_addr() #0 {
  ; CHECK: global_addr:
  ; CHECK: adrp x0, :pg_hi21_nc:global
  ; CHECK: movk x0, #:prel_g3:global+4294967296
  ; CHECK: add x0, x0, :lo12:global
  ret i32* @global
}

define i32 @global_load() #0 {
  ; CHECK: global_load:
  ; CHECK: adrp x8, :pg_hi21_nc:global
  ; CHECK: ldr w0, [x8, :lo12:global]
  %load = load i32, i32* @global
  ret i32 %load
}

define void ()* @func_addr() #0 {
  ; CHECK: func_addr:
  ; CHECK: adrp x0, func
  ; CHECK: add x0, x0, :lo12:func
  ret void ()* @func
}

attributes #0 = { "target-features"="+tagged-globals" }
