; RUN: llc --relocation-model=pic -mattr=+pauth < %s | FileCheck %s --check-prefixes=CHECK,GISEL

; RUN: llc -global-isel=0 -fast-isel=0 -O0         --relocation-model=pic < %s -mattr=+pauth | FileCheck %s --check-prefixes=CHECK,DAGISEL
; RUN: llc -global-isel=0 -fast-isel=1 -O0         --relocation-model=pic < %s -mattr=+pauth | FileCheck %s --check-prefixes=CHECK,DAGISEL
; RUN: llc -global-isel=1 -global-isel-abort=1 -O0 --relocation-model=pic < %s -mattr=+pauth | FileCheck %s --check-prefixes=CHECK,GISEL

;; Note: for FastISel, we fall back to SelectionDAG

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-android"

@global = external global i32
declare void @func()

define ptr @global_addr() #0 {
  ; CHECK-LABEL: global_addr:
  ; CHECK:         adrp  [[REG:x[0-9]+]], :got_auth:global
  ; CHECK-NEXT:    add   [[REG]], [[REG]], :got_auth_lo12:global
  ; CHECK-NEXT:    ldr   x0, [[[REG]]]
  ; CHECK-NEXT:    autda x0, [[REG]]
  ; CHECK-NEXT:    ret

  ret ptr @global
}

define i32 @global_load() #0 {
  ; CHECK-LABEL: global_load:
  ; CHECK:         adrp  [[REG0:x[0-9]+]], :got_auth:global
  ; CHECK-NEXT:    add   [[REG0]], [[REG0]], :got_auth_lo12:global
  ; CHECK-NEXT:    ldr   [[REG1:x[0-9]+]], [[[REG0]]]
  ; CHECK-NEXT:    autda [[REG1]], [[REG0]]
  ; CHECK-NEXT:    ldr   w0, [[[REG1]]]
  ; CHECK-NEXT:    ret
  %load = load i32, ptr @global
  ret i32 %load
}

define void @global_store() #0 {
  ; CHECK-LABEL:   global_store:
  ; CHECK:           adrp  [[REG0:x[0-9]+]], :got_auth:global
  ; CHECK-NEXT:      add   [[REG0]], [[REG0]], :got_auth_lo12:global
  ; CHECK-NEXT:      ldr   [[REG1:x[0-9]+]], [[[REG0]]]
  ; CHECK-NEXT:      autda [[REG1]], [[REG0]]
  ; GISEL-NEXT:      str   wzr, [[[REG1]]]
  ; DAGISEL-NEXT:    mov   w8, wzr
  ; DAGISEL-NEXT:    str   w8, [[[REG1]]]
  ; CHECK-NEXT:      ret
  store i32 0, ptr @global
  ret void
}

define ptr @func_addr() #0 {
  ; CHECK-LABEL: func_addr:
  ; CHECK:         adrp  [[REG:x[0-9]+]], :got_auth:func
  ; CHECK-NEXT:    add   [[REG]], [[REG]], :got_auth_lo12:func
  ; CHECK-NEXT:    ldr   x0, [[[REG]]]
  ; CHECK-NEXT:    autia x0, [[REG]]
  ; CHECK-NEXT:    ret
  ret ptr @func
}

attributes #0 = { "target-features"="+tagged-globals" }

!llvm.module.flags = !{!0, !1}
!0 = !{i32 1, !"aarch64-elf-pauthabi-platform", i32 268435458}
!1 = !{i32 1, !"aarch64-elf-pauthabi-version", i32 256}
