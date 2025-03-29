; RUN: llc -verify-machineinstrs -enable-machine-outliner -mtriple aarch64 %s -o - | \
; RUN:   FileCheck %s --check-prefixes CHECK,V8A
; RUN: llc -verify-machineinstrs -enable-machine-outliner -mtriple aarch64 -mattr=+v8.3a %s -o - | \
; RUN:   FileCheck %s --check-prefixes CHECK,V83A

define void @a() "sign-return-address"="all" "sign-return-address-key"="b_key" nounwind {
; CHECK-LABEL:      a:                                     // @a
; CHECK-NEXT:       // %bb.0:
; V8A:              hint #27
; V83A:             pacibsp
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
; V8A:              hint #31
; V83A:             retab
  ret void
}

define void @b() "sign-return-address"="all" "sign-return-address-key"="b_key" nounwind {
; CHECK-LABEL:      b:                                     // @b
; CHECK-NEXT:       // %bb.0:
; V8A:              hint #27
; V83A:             pacibsp
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
; V8A:              hint #31
; V83A:             retab
  ret void
}

define void @c() "sign-return-address"="all" "sign-return-address-key"="b_key" nounwind {
; CHECK-LABEL:      c:                                     // @c
; CHECK-NEXT:       // %bb.0:
; V8A:              hint #27
; V83A:             pacibsp
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
; V8A:              hint #31
; V83A:             retab
  ret void
}

; CHECK-LABEL:      OUTLINED_FUNCTION_0:
; CHECK:            // %bb.0:
; V8A-NEXT:             hint #27
; V83A-NEXT:            pacibsp
; V8A:                  hint #31
; V8A-NEXT:             ret
; V83A:                 retab
