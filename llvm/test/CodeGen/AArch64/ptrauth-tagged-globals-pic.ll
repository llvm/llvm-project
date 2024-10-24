; RUN: llc -global-isel=0 -fast-isel=0 -O0         --relocation-model=pic < %s \
; RUN:   -mattr=+pauth -mattr=+fpac | FileCheck %s --check-prefixes=CHECK,DAGISEL
; RUN: llc -global-isel=0 -fast-isel=1 -O0         --relocation-model=pic < %s \
; RUN:   -mattr=+pauth -mattr=+fpac | FileCheck %s --check-prefixes=CHECK,DAGISEL
; RUN: llc -global-isel=1 -global-isel-abort=1 -O0 --relocation-model=pic < %s \
; RUN:   -mattr=+pauth -mattr=+fpac | FileCheck %s --check-prefixes=CHECK,GISEL

;; Note: for FastISel, we fall back to SelectionDAG

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-android"

@global = external global i32
declare void @func()

define ptr @global_addr() #0 {
; CHECK-LABEL: global_addr:
; CHECK:         adrp  x16, :got_auth:global
; CHECK-NEXT:    add   x16, x16, :got_auth_lo12:global
; CHECK-NEXT:    ldr   x0,  [x16]
; CHECK-NEXT:    autda x0,  x16
; CHECK-NEXT:    ret

  ret ptr @global
}

define i32 @global_load() #0 {
; CHECK-LABEL: global_load:
; CHECK:         adrp  x16, :got_auth:global
; CHECK-NEXT:    add   x16, x16, :got_auth_lo12:global
; CHECK-NEXT:    ldr   x8,  [x16]
; CHECK-NEXT:    autda x8,  x16
; CHECK-NEXT:    ldr   w0,  [x8]
; CHECK-NEXT:    ret

  %load = load i32, ptr @global
  ret i32 %load
}

define void @global_store() #0 {
; CHECK-LABEL:   global_store:
; CHECK:           adrp  x16, :got_auth:global
; CHECK-NEXT:      add   x16, x16, :got_auth_lo12:global
; GISEL-NEXT:      ldr   x8,  [x16]
; GISEL-NEXT:      autda x8,  x16
; GISEL-NEXT:      str   wzr, [x8]
; DAGISEL-NEXT:    ldr   x9,  [x16]
; DAGISEL-NEXT:    autda x9,  x16
; DAGISEL-NEXT:    mov   w8,  wzr
; DAGISEL-NEXT:    str   w8,  [x9]
; CHECK-NEXT:      ret
  store i32 0, ptr @global
  ret void
}

define ptr @func_addr() #0 {
; CHECK-LABEL: func_addr:
; CHECK:         adrp  x16, :got_auth:func
; CHECK-NEXT:    add   x16, x16, :got_auth_lo12:func
; CHECK-NEXT:    ldr   x0,  [x16]
; CHECK-NEXT:    autia x0,  x16
; CHECK-NEXT:    ret

  ret ptr @func
}

attributes #0 = { "target-features"="+tagged-globals" }

!llvm.module.flags = !{!0}
!0 = !{i32 8, !"ptrauth-elf-got", i32 1}
