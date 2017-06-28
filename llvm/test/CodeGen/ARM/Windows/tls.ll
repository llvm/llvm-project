; RUN: llc -mtriple thumbv7--windows-itanium %s -o - | FileCheck %s

@i = thread_local global i32 0
@j = external thread_local global i32
@k = internal thread_local global i32 0
@l = hidden thread_local global i32 0
@m = external hidden thread_local global i32
@n = thread_local global i16 0
@o = thread_local global i8 0

define i32 @f() {
  %1 = load i32, i32* @i
  ret i32 %1
}

; CHECK:      mrc p15, #0, [[TEB:r[0-9]]], c13, c0, #2

; CHECK:      ldr [[TLS_POINTER:r[0-9]]], {{\[}}[[TEB]], #44]
; CHECK:      movw [[TLS_INDEX:r[0-9]]], :lower16:_tls_index
; CHECK-NEXT: movt [[TLS_INDEX]], :upper16:_tls_index
; CHECK-NEXT: ldr [[INDEX:r[0-9]]], {{\[}}[[TLS_INDEX]]]

; CHECK-NEXT: ldr{{.w}} [[TLS:r[0-9]]], {{\[}}[[TLS_POINTER]], [[INDEX]], lsl #2]

; CHECK-NEXT: ldr [[SLOT:r[0-9]]], [[CPI:\.LCPI[0-9]+_[0-9]+]]

; CHECK-NEXT: ldr r0, {{\[}}[[TLS]], [[SLOT]]]

; CHECK: [[CPI]]:
; CHECK-NEXT: .long i(SECREL32)

define i32 @e() {
  %1 = load i32, i32* @j
  ret i32 %1
}

; CHECK:      mrc p15, #0, [[TEB:r[0-9]]], c13, c0, #2

; CHECK:      ldr [[TLS_POINTER:r[0-9]]], {{\[}}[[TEB]], #44]
; CHECK:      movw [[TLS_INDEX:r[0-9]]], :lower16:_tls_index
; CHECK-NEXT: movt [[TLS_INDEX]], :upper16:_tls_index
; CHECK-NEXT: ldr [[INDEX:r[0-9]]], {{\[}}[[TLS_INDEX]]]

; CHECK-NEXT: ldr{{.w}} [[TLS:r[0-9]]], {{\[}}[[TLS_POINTER]], [[INDEX]], lsl #2]

; CHECK-NEXT: ldr [[SLOT:r[0-9]]], [[CPI:\.LCPI[0-9]+_[0-9]+]]

; CHECK-NEXT: ldr r0, {{\[}}[[TLS]], [[SLOT]]]

; CHECK: [[CPI]]:
; CHECK-NEXT: .long j(SECREL32)

define i32 @d() {
  %1 = load i32, i32* @k
  ret i32 %1
}

; CHECK:      mrc p15, #0, [[TEB:r[0-9]]], c13, c0, #2

; CHECK:      ldr [[TLS_POINTER:r[0-9]]], {{\[}}[[TEB]], #44]
; CHECK:      movw [[TLS_INDEX:r[0-9]]], :lower16:_tls_index
; CHECK-NEXT: movt [[TLS_INDEX]], :upper16:_tls_index
; CHECK-NEXT: ldr [[INDEX:r[0-9]]], {{\[}}[[TLS_INDEX]]]

; CHECK-NEXT: ldr{{.w}} [[TLS:r[0-9]]], {{\[}}[[TLS_POINTER]], [[INDEX]], lsl #2]

; CHECK-NEXT: ldr [[SLOT:r[0-9]]], [[CPI:\.LCPI[0-9]+_[0-9]+]]

; CHECK-NEXT: ldr r0, {{\[}}[[TLS]], [[SLOT]]]

; CHECK: [[CPI]]:
; CHECK-NEXT: .long k(SECREL32)

define i32 @c() {
  %1 = load i32, i32* @l
  ret i32 %1
}

; CHECK:      mrc p15, #0, [[TEB:r[0-9]]], c13, c0, #2

; CHECK:      ldr [[TLS_POINTER:r[0-9]]], {{\[}}[[TEB]], #44]
; CHECK:      movw [[TLS_INDEX:r[0-9]]], :lower16:_tls_index
; CHECK-NEXT: movt [[TLS_INDEX]], :upper16:_tls_index
; CHECK-NEXT: ldr [[INDEX:r[0-9]]], {{\[}}[[TLS_INDEX]]]

; CHECK-NEXT: ldr{{.w}} [[TLS:r[0-9]]], {{\[}}[[TLS_POINTER]], [[INDEX]], lsl #2]

; CHECK-NEXT: ldr [[SLOT:r[0-9]]], [[CPI:\.LCPI[0-9]+_[0-9]+]]

; CHECK-NEXT: ldr r0, {{\[}}[[TLS]], [[SLOT]]]

; CHECK: [[CPI]]:
; CHECK-NEXT: .long l(SECREL32)

define i32 @b() {
  %1 = load i32, i32* @m
  ret i32 %1
}

; CHECK:      mrc p15, #0, [[TEB:r[0-9]]], c13, c0, #2

; CHECK:      ldr [[TLS_POINTER:r[0-9]]], {{\[}}[[TEB]], #44]
; CHECK:      movw [[TLS_INDEX:r[0-9]]], :lower16:_tls_index
; CHECK-NEXT: movt [[TLS_INDEX]], :upper16:_tls_index
; CHECK-NEXT: ldr [[INDEX:r[0-9]]], {{\[}}[[TLS_INDEX]]]

; CHECK-NEXT: ldr{{.w}} [[TLS:r[0-9]]], {{\[}}[[TLS_POINTER]], [[INDEX]], lsl #2]

; CHECK-NEXT: ldr [[SLOT:r[0-9]]], [[CPI:\.LCPI[0-9]+_[0-9]+]]

; CHECK-NEXT: ldr r0, {{\[}}[[TLS]], [[SLOT]]]

; CHECK: [[CPI]]:
; CHECK: .long m(SECREL32)

define i16 @a() {
  %1 = load i16, i16* @n
  ret i16 %1
}

; CHECK:      mrc p15, #0, [[TEB:r[0-9]]], c13, c0, #2

; CHECK:      ldr [[TLS_POINTER:r[0-9]]], {{\[}}[[TEB]], #44]
; CHECK:      movw [[TLS_INDEX:r[0-9]]], :lower16:_tls_index
; CHECK-NEXT: movt [[TLS_INDEX]], :upper16:_tls_index
; CHECK-NEXT: ldr [[INDEX:r[0-9]]], {{\[}}[[TLS_INDEX]]]

; CHECK-NEXT: ldr{{.w}} [[TLS:r[0-9]]], {{\[}}[[TLS_POINTER]], [[INDEX]], lsl #2]

; CHECK-NEXT: ldr [[SLOT:r[0-9]]], [[CPI:\.LCPI[0-9]+_[0-9]+]]

; CHECK-NEXT: ldrh r0, {{\[}}[[TLS]], [[SLOT]]]

; CHECK: [[CPI]]:
; CHECK: .long n(SECREL32)

define i8 @Z() {
  %1 = load i8, i8* @o
  ret i8 %1
}

; CHECK:      mrc p15, #0, [[TEB:r[0-9]]], c13, c0, #2

; CHECK:      ldr [[TLS_POINTER:r[0-9]]], {{\[}}[[TEB]], #44]
; CHECK:      movw [[TLS_INDEX:r[0-9]]], :lower16:_tls_index
; CHECK-NEXT: movt [[TLS_INDEX]], :upper16:_tls_index
; CHECK-NEXT: ldr [[INDEX:r[0-9]]], {{\[}}[[TLS_INDEX]]]

; CHECK-NEXT: ldr{{.w}} [[TLS:r[0-9]]], {{\[}}[[TLS_POINTER]], [[INDEX]], lsl #2]

; CHECK-NEXT: ldr [[SLOT:r[0-9]]], [[CPI:\.LCPI[0-9]+_[0-9]+]]

; CHECK-NEXT: ldrb r0, {{\[}}[[TLS]], [[SLOT]]]

; CHECK: [[CPI]]:
; CHECK-NEXT: .long o(SECREL32)

