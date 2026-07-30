; RUN: llc -verify-machineinstrs -enable-machine-outliner -mtriple \
; RUN: aarch64-arm-linux-gnu %s -o - | FileCheck %s

; Check that outlined functions use the dedicated RETAA/RETAB instructions
; to sign their return address if available.

define void @a() #0 {
; CHECK-LABEL:      a:                                     // @a
; CHECK:            // %bb.0:
; CHECK-NEXT:               pacibsp
; CHECK:                    bl [[OUTLINED_FUNC:OUTLINED_FUNCTION_[0-9]+]]
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
; CHECK-NOT:              auti{{[a,b]}}sp
; CHECK:                  retab
  ret void
}

define void @b() #0 {
; CHECK-LABEL:      b:                                     // @b
; CHECK:            // %bb.0:
; CHECK-NEXT:               pacibsp
; CHECK:                    bl OUTLINED_FUNC
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
; CHECK-NOT:              auti{{[a,b]}}sp
; CHECK:                  retab
  ret void
}

define void @c() #0 {
; CHECK-LABEL:      c:                                     // @c
; CHECK:            // %bb.0:
; CHECK-NEXT:               pacibsp
; CHECK:                    bl OUTLINED_FUNC
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
; CHECK-NOT:              auti{{[a,b]}}sp
; CHECK:                  retab
  ret void
}

attributes #0 = { "sign-return-address"="all"
                  "sign-return-address-key"="b_key"
                  "target-features"="+v8.3a"
                  nounwind }

; CHECK:            OUTLINED_FUNC
; CHECK:            // %bb.0:
; CHECK-NEXT:               pacibsp
; CHECK-NOT:                auti{{[a,b]}}sp
; CHECK:                    retab
