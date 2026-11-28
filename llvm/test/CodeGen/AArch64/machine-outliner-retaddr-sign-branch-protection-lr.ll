; RUN: llc -verify-machineinstrs -enable-machine-outliner -mtriple aarch64-arm-linux-gnu %s -o - | FileCheck %s

@v = common dso_local global ptr null, align 8

define void @no_br_protection_lr_a(i32 %x) #0 {
; CHECK-LABEL:      no_br_protection_lr_a:
; CHECK:            // %bb.0:
; CHECK-NEXT:               hint #27
; CHECK-NOT:                OUTLINED_FUNCTION_
; CHECK:                    hint #31
  %ext = zext i32 %x to i64
  %vla = alloca i32, i64 %ext, align 4
  store volatile ptr %vla, ptr @v, align 8
  store volatile ptr %vla, ptr @v, align 8
  store volatile ptr %vla, ptr @v, align 8
  store volatile ptr %vla, ptr @v, align 8
  store volatile ptr %vla, ptr @v, align 8
  store volatile ptr %vla, ptr @v, align 8
  ret void
}

define void @no_br_protection_lr_b(i32 %x) #0 {
; CHECK-LABEL:      no_br_protection_lr_b:
; CHECK:            // %bb.0:
; CHECK-NEXT:               hint #27
; CHECK-NOT:                OUTLINED_FUNCTION_
; CHECK:                    hint #31
  %ext = zext i32 %x to i64
  %vla = alloca i32, i64 %ext, align 4
  store volatile ptr %vla, ptr @v, align 8
  store volatile ptr %vla, ptr @v, align 8
  store volatile ptr %vla, ptr @v, align 8
  store volatile ptr %vla, ptr @v, align 8
  store volatile ptr %vla, ptr @v, align 8
  store volatile ptr %vla, ptr @v, align 8
  ret void
}

define void @with_br_protection_lr_a(i32 %x) #1 {
; CHECK-LABEL:      with_br_protection_lr_a:
; CHECK:            // %bb.0:
; CHECK-NEXT:      .Ltmp0:
; CHECK-NEXT:               pacibsppc
; CHECK-NOT:                OUTLINED_FUNCTION_
; CHECK:                    autibsppc
  %ext = zext i32 %x to i64
  %vla = alloca i32, i64 %ext, align 4
  store volatile ptr %vla, ptr @v, align 8
  store volatile ptr %vla, ptr @v, align 8
  store volatile ptr %vla, ptr @v, align 8
  store volatile ptr %vla, ptr @v, align 8
  store volatile ptr %vla, ptr @v, align 8
  store volatile ptr %vla, ptr @v, align 8
  ret void
}

define void @with_br_protection_lr_b(i32 %x) #1 {
; CHECK-LABEL:      with_br_protection_lr_b:
; CHECK:            // %bb.0:
; CHECK-NEXT:      .Ltmp1:
; CHECK-NEXT:               pacibsppc
; CHECK-NOT:                OUTLINED_FUNCTION_
; CHECK:                    autibsppc
  %ext = zext i32 %x to i64
  %vla = alloca i32, i64 %ext, align 4
  store volatile ptr %vla, ptr @v, align 8
  store volatile ptr %vla, ptr @v, align 8
  store volatile ptr %vla, ptr @v, align 8
  store volatile ptr %vla, ptr @v, align 8
  store volatile ptr %vla, ptr @v, align 8
  store volatile ptr %vla, ptr @v, align 8
  ret void
}

attributes #0 = { "sign-return-address"="all"
                  "sign-return-address-key"="b_key"
                  "target-features"="+pauth-lr"
                  nounwind }

attributes #1 = { "sign-return-address"="all"
                  "sign-return-address-key"="b_key"
                  "target-features"="+pauth-lr"
                  "branch-protection-pauth-lr"
                  nounwind }

; CHECK-NOT:        OUTLINED_FUNCTION_
