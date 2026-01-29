; RUN: llc -verify-machineinstrs -enable-machine-outliner -mtriple aarch64 %s -o - | FileCheck %s --check-prefix V8A
; RUN: llc -verify-machineinstrs -enable-machine-outliner -mtriple aarch64 -mattr=+v8.3a %s -o - | FileCheck %s --check-prefix V83A

define void @a() "sign-return-address"="all" "sign-return-address-key"="a_key" "sign-return-address-harden"="load-return-address" nounwind {
; V8A-LABEL:     a:
; V8A:           hint #25
; V8A-NOT:       bl OUTLINED_FUNCTION
; V8A:           hint #29
; V8A-NEXT:      mov x8, x30
; V8A-NEXT:      hint #7
; V8A-NEXT:      ldr w30, [x30]
; V8A-NEXT:      mov x30, x8
; V8A-NEXT:      ret{{$}}

; V83A-LABEL:    a:
; V83A:          paciasp
; V83A-NOT:      bl OUTLINED_FUNCTION
; V83A:          autiasp
; V83A-NEXT:     mov x8, x30
; V83A-NEXT:     xpaci x8
; V83A-NEXT:     ldr w8, [x8]
; V83A-NEXT:     ret{{$}}
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  store i32 1, ptr %1, align 4
  store i32 2, ptr %2, align 4
  store i32 3, ptr %3, align 4
  store i32 4, ptr %4, align 4
  store i32 5, ptr %5, align 4
  store i32 6, ptr %6, align 4
  ret void
}

define void @b() "sign-return-address"="all" "sign-return-address-key"="a_key" "sign-return-address-harden"="load-return-address" nounwind {
; V8A-LABEL:     b:
; V8A:           hint #25
; V8A-NOT:       bl OUTLINED_FUNCTION
; V8A:           hint #29
; V8A-NEXT:      mov x8, x30
; V8A-NEXT:      hint #7
; V8A-NEXT:      ldr w30, [x30]
; V8A-NEXT:      mov x30, x8
; V8A-NEXT:      ret{{$}}

; V83A-LABEL:    b:
; V83A:          paciasp
; V83A-NOT:      bl OUTLINED_FUNCTION
; V83A:          autiasp
; V83A-NEXT:     mov x8, x30
; V83A-NEXT:     xpaci x8
; V83A-NEXT:     ldr w8, [x8]
; V83A-NEXT:     ret{{$}}
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  store i32 1, ptr %1, align 4
  store i32 2, ptr %2, align 4
  store i32 3, ptr %3, align 4
  store i32 4, ptr %4, align 4
  store i32 5, ptr %5, align 4
  store i32 6, ptr %6, align 4
  ret void
}

define void @c() "sign-return-address"="all" "sign-return-address-key"="a_key" "sign-return-address-harden"="load-return-address" nounwind {
; V8A-LABEL:     c:
; V8A:           hint #25
; V8A-NOT:       bl OUTLINED_FUNCTION
; V8A:           hint #29
; V8A-NEXT:      mov x8, x30
; V8A-NEXT:      hint #7
; V8A-NEXT:      ldr w30, [x30]
; V8A-NEXT:      mov x30, x8
; V8A-NEXT:      ret{{$}}

; V83A-LABEL:    c:
; V83A:          paciasp
; V83A-NOT:      bl OUTLINED_FUNCTION
; V83A:          autiasp
; V83A-NEXT:     mov x8, x30
; V83A-NEXT:     xpaci x8
; V83A-NEXT:     ldr w8, [x8]
; V83A-NEXT:     ret{{$}}
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  store i32 1, ptr %1, align 4
  store i32 2, ptr %2, align 4
  store i32 3, ptr %3, align 4
  store i32 4, ptr %4, align 4
  store i32 5, ptr %5, align 4
  store i32 6, ptr %6, align 4
  ret void
}

; V8A-NOT:       OUTLINED_FUNCTION_0:
; V83A-NOT:      OUTLINED_FUNCTION_0:
