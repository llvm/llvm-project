; RUN: llc --relocation-model=pic -mattr=+pauth < %s | FileCheck %s

; RUN: llc -global-isel=0 -fast-isel=0 -O0 --relocation-model=pic < %s -mattr=+pauth | FileCheck %s --check-prefixes=CHECK,DAGISEL
; RUN: llc -global-isel=0 -fast-isel=1 -O0 --relocation-model=pic < %s -mattr=+pauth | FileCheck %s --check-prefixes=CHECK,DAGISEL
; RUN: llc -global-isel=1              -O0 --relocation-model=pic < %s -mattr=+pauth | FileCheck %s --check-prefixes=CHECK,GISEL

;; Note: for FastISel, we fall back to SelectionDAG

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-android"

@global = external global i32
declare void @func()

define ptr @global_addr() #0 {
  ; CHECK: global_addr:
  ; CHECK: adrp  [[REG:x[0-9]+]], :got_auth:global
  ; CHECK: add   [[REG]], [[REG]], :got_auth_lo12:global
  ; CHECK: ldr   x0, [[[REG]]]
  ; CHECK: autda x0, [[REG]]
  ; CHECK: ret

  ret ptr @global
}

define i32 @global_load() #0 {
  ; CHECK: global_load:
  ; CHECK: adrp  [[REG0:x[0-9]+]], :got_auth:global
  ; CHECK: add   [[REG0]], [[REG0]], :got_auth_lo12:global
  ; CHECK: ldr   [[REG1:x[0-9]+]], [[[REG0]]]
  ; CHECK: autda [[REG1]], [[REG0]]
  ; CHECK: ldr   w0, [[[REG1]]]
  ; CHECK: ret
  %load = load i32, ptr @global
  ret i32 %load
}

define void @global_store() #0 {
  ; CHECK:   global_store:
  ; CHECK:   adrp  [[REG0:x[0-9]+]], :got_auth:global
  ; CHECK:   add   [[REG0]], [[REG0]], :got_auth_lo12:global
  ; CHECK:   ldr   [[REG1:x[0-9]+]], [[[REG0]]]
  ; CHECK:   autda [[REG1]], [[REG0]]
  ; GISEL:   str   wzr, [[[REG1]]]
  ; DAGISEL: mov   w8, wzr
  ; DAGISEL: str   w8, [[[REG1]]]
  ; CHECK:   ret
  store i32 0, ptr @global
  ret void
}

define ptr @func_addr() #0 {
  ; CHECK: func_addr:
  ; CHECK: adrp  [[REG:x[0-9]+]], :got_auth:func
  ; CHECK: add   [[REG]], [[REG]], :got_auth_lo12:func
  ; CHECK: ldr   x0, [[[REG]]]
  ; CHECK: autia x0, [[REG]]
  ; CHECK: ret
  ret ptr @func
}

attributes #0 = { "target-features"="+tagged-globals" }

!llvm.module.flags = !{!0, !1}
!0 = !{i32 1, !"aarch64-elf-pauthabi-platform", i32 268435458}
!1 = !{i32 1, !"aarch64-elf-pauthabi-version", i32 128}
