; RUN: llc --relocation-model=pic < %s \
; RUN:   | FileCheck %s --check-prefix=CHECK-PIC

; Ensure that GlobalISel lowers correctly. GlobalISel is the default ISel for
; -O0 on aarch64. GlobalISel lowers the instruction sequence in the static
; relocation model different to SelectionDAGISel. GlobalISel does the lowering
; of AddLow *after* legalization, and thus doesn't differentiate between
; address-taken-only vs. address-taken-for-loadstore. Hence, we generate a movk
; instruction for load/store instructions as well with GlobalISel. GlobalISel
; also doesn't have the scaffolding to correctly check the bounds of the global
; offset, and cannot fold the lo12 bits into the load/store. Neither of these
; things are a problem as GlobalISel is only used by default at -O0, so we don't
; mind the code size and performance increase.

; RUN: llc --aarch64-enable-global-isel-at-O=0 -O0 --relocation-model=pic < %s \
; RUN:   | FileCheck %s --check-prefix=CHECK-PIC

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-android"

@global = external global i32
declare void @func()

define ptr @global_addr() #0 {
  ; CHECK-PIC: global_addr:
  ; CHECK-PIC: adrp [[REG:x[0-9]+]], :got:global
  ; CHECK-PIC: ldr x0, [[[REG]], :got_lo12:global]
  ; CHECK-PIC: ret

  ret ptr @global
}

define i32 @global_load() #0 {
  ; CHECK-PIC: global_load:
  ; CHECK-PIC: adrp [[REG:x[0-9]+]], :got:global
  ; CHECK-PIC: ldr  [[REG]], [[[REG]], :got_lo12:global]
  ; CHECK-PIC: ldr w0, [[[REG]]]
  ; CHECK-PIC: ret

  %load = load i32, ptr @global
  ret i32 %load
}

define void @global_store() #0 {
  ; CHECK-PIC: global_store:
  ; CHECK-PIC: adrp [[REG:x[0-9]+]], :got:global
  ; CHECK-PIC: ldr  [[REG]], [[[REG]], :got_lo12:global]
  ; CHECK-PIC: str wzr, [[[REG]]]
  ; CHECK-PIC: ret

  store i32 0, ptr @global
  ret void
}

define ptr @func_addr() #0 {
  ; CHECK-PIC: func_addr:
  ; CHECK-PIC: adrp [[REG:x[0-9]+]], :got:func
  ; CHECK-PIC: ldr  x0, [[[REG]], :got_lo12:func]
  ; CHECK-PIC: ret

  ret ptr @func
}

attributes #0 = { "target-features"="+tagged-globals" }
