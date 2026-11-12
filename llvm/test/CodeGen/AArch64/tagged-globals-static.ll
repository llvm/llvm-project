; RUN: llc --relocation-model=static < %s \
; RUN:   | FileCheck %s --check-prefixes=CHECK-STATIC,CHECK-SELECTIONDAGISEL

; RUN: llc --aarch64-enable-global-isel-at-O=0 -O0 < %s \
; RUN:   | FileCheck %s --check-prefixes=CHECK-STATIC,CHECK-GLOBALISEL

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-android"

@global = external dso_local global i32
declare dso_local void @func()

define ptr @global_addr() #0 {
  ; Static relocation model has common codegen between SelectionDAGISel and
  ; GlobalISel when the address-taken of a global isn't folded into a load or
  ; store instruction.
  ; CHECK-STATIC: global_addr:
  ; CHECK-STATIC: adrp [[REG:x[0-9]+]], :pg_hi21_nc:global
  ; CHECK-STATIC: movk [[REG]], #:prel_g3:global+4294967296
  ; CHECK-STATIC: add x0, [[REG]], :lo12:global
  ; CHECK-STATIC: ret

  ret ptr @global
}

define i32 @global_load() #0 {
  ; CHECK-SELECTIONDAGISEL: global_load:
  ; CHECK-SELECTIONDAGISEL: adrp [[REG:x[0-9]+]], :pg_hi21_nc:global
  ; CHECK-SELECTIONDAGISEL: ldr w0, [[[REG]], :lo12:global]
  ; CHECK-SELECTIONDAGISEL: ret

  ; CHECK-GLOBALISEL: global_load:
  ; CHECK-GLOBALISEL: adrp [[REG:x[0-9]+]], :pg_hi21_nc:global
  ; CHECK-GLOBALISEL: movk [[REG]], #:prel_g3:global+4294967296
  ; CHECK-GLOBALISEL: add [[REG]], [[REG]], :lo12:global
  ; CHECK-GLOBALISEL: ldr w0, [[[REG]]]
  ; CHECK-GLOBALISEL: ret

  %load = load i32, ptr @global
  ret i32 %load
}

define void @global_store() #0 {
  ; CHECK-SELECTIONDAGISEL: global_store:
  ; CHECK-SELECTIONDAGISEL: adrp [[REG:x[0-9]+]], :pg_hi21_nc:global
  ; CHECK-SELECTIONDAGISEL: str wzr, [[[REG]], :lo12:global]
  ; CHECK-SELECTIONDAGISEL: ret

  ; CHECK-GLOBALISEL: global_store:
  ; CHECK-GLOBALISEL: adrp [[REG:x[0-9]+]], :pg_hi21_nc:global
  ; CHECK-GLOBALISEL: movk [[REG]], #:prel_g3:global+4294967296
  ; CHECK-GLOBALISEL: add [[REG]], [[REG]], :lo12:global
  ; CHECK-GLOBALISEL: str wzr, [[[REG]]]
  ; CHECK-GLOBALISEL: ret

  store i32 0, ptr @global
  ret void
}

define ptr @func_addr() #0 {
  ; CHECK-STATIC: func_addr:
  ; CHECK-STATIC: adrp [[REG:x[0-9]+]], func
  ; CHECK-STATIC: add x0, [[REG]], :lo12:func
  ; CHECK-STATIC: ret

  ret ptr @func
}

attributes #0 = { "target-features"="+tagged-globals" }
