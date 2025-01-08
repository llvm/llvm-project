; RUN: llc --relocation-model=pic < %s | FileCheck %s

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
; RUN:   | FileCheck %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-android"

@global = external global i32
declare void @func()

define ptr @global_addr() #0 {
  ; CHECK-LABEL: global_addr:
  ; CHECK:         adrp [[REG:x[0-9]+]], :got:global
  ; CHECK-NEXT:    ldr x0, [[[REG]], :got_lo12:global]
  ; CHECK-NEXT:    ret

  ret ptr @global
}

define i32 @global_load() #0 {
  ; CHECK-LABEL: global_load:
  ; CHECK:         adrp [[REG:x[0-9]+]], :got:global
  ; CHECK-NEXT:    ldr  [[REG]], [[[REG]], :got_lo12:global]
  ; CHECK-NEXT:    ldr w0, [[[REG]]]
  ; CHECK-NEXT:    ret

  %load = load i32, ptr @global
  ret i32 %load
}

define void @global_store() #0 {
  ; CHECK-LABEL: global_store:
  ; CHECK:         adrp [[REG:x[0-9]+]], :got:global
  ; CHECK-NEXT:    ldr  [[REG]], [[[REG]], :got_lo12:global]
  ; CHECK-NEXT:    str wzr, [[[REG]]]
  ; CHECK-NEXT:    ret

  store i32 0, ptr @global
  ret void
}

define ptr @func_addr() #0 {
  ; CHECK-LABEL: func_addr:
  ; CHECK:         adrp [[REG:x[0-9]+]], :got:func
  ; CHECK-NEXT:    ldr  x0, [[[REG]], :got_lo12:func]
  ; CHECK-NEXT:    ret

  ret ptr @func
}

attributes #0 = { "target-features"="+tagged-globals" }
