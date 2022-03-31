; RUN: llc -mtriple aarch64-windows %s -o - | FileCheck %s

@tlsVar = thread_local global i32 0
@tlsVar8 = thread_local global i8 0
@tlsVar64 = thread_local global i64 0

define i32 @getVar() {
  %1 = load i32, i32* @tlsVar
  ret i32 %1
}

define i32* @getPtr() {
  ret i32* @tlsVar
}

define void @setVar(i32 %val) {
  store i32 %val, i32* @tlsVar
  ret void
}

define i8 @getVar8() {
  %1 = load i8, i8* @tlsVar8
  ret i8 %1
}

define i64 @getVar64() {
  %1 = load i64, i64* @tlsVar64
  ret i64 %1
}

; CHECK-LABEL: getVar
; CHECK: adrp [[TLS_INDEX_ADDR:x[0-9]+]], _tls_index
; CHECK: ldr [[TLS_POINTER:x[0-9]+]], [x18, #88]
; CHECK: ldr w[[TLS_INDEX:[0-9]+]], [[[TLS_INDEX_ADDR]], :lo12:_tls_index]

; CHECK: ldr [[TLS:x[0-9]+]], [[[TLS_POINTER]], x[[TLS_INDEX]], lsl #3]
; CHECK: add [[TLS]], [[TLS]], :secrel_hi12:tlsVar
; CHECK: ldr w0, [[[TLS]], :secrel_lo12:tlsVar]

; CHECK-LABEL: getPtr
; CHECK: adrp [[TLS_INDEX_ADDR:x[0-9]+]], _tls_index
; CHECK: ldr [[TLS_POINTER:x[0-9]+]], [x18, #88]
; CHECK: ldr w[[TLS_INDEX:[0-9]+]], [[[TLS_INDEX_ADDR]], :lo12:_tls_index]

; CHECK: ldr [[TLS:x[0-9]+]], [[[TLS_POINTER]], x[[TLS_INDEX]], lsl #3]
; CHECK: add [[TLS]], [[TLS]], :secrel_hi12:tlsVar
; CHECK: add x0, [[TLS]], :secrel_lo12:tlsVar

; CHECK-LABEL: setVar
; CHECK: adrp [[TLS_INDEX_ADDR:x[0-9]+]], _tls_index
; CHECK: ldr [[TLS_POINTER:x[0-9]+]], [x18, #88]
; CHECK: ldr w[[TLS_INDEX:[0-9]+]], [[[TLS_INDEX_ADDR]], :lo12:_tls_index]

; CHECK: ldr [[TLS:x[0-9]+]], [[[TLS_POINTER]], x[[TLS_INDEX]], lsl #3]
; CHECK: add [[TLS]], [[TLS]], :secrel_hi12:tlsVar
; CHECK: str w0, [[[TLS]], :secrel_lo12:tlsVar]

; CHECK-LABEL: getVar8
; CHECK: add [[TLS:x[0-9]+]], [[TLS]], :secrel_hi12:tlsVar8
; CHECK: ldrb w0, [[[TLS]], :secrel_lo12:tlsVar8]

; CHECK-LABEL: getVar64
; CHECK: add [[TLS:x[0-9]+]], [[TLS]], :secrel_hi12:tlsVar64
; CHECK: ldr x0, [[[TLS]], :secrel_lo12:tlsVar64]
