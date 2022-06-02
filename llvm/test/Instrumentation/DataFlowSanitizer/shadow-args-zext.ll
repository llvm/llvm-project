; RUN: opt -mtriple=x86_64-unknown-linux-gnu < %s -dfsan -S --dfsan-abilist=%S/Inputs/shadow-args-abilist.txt | FileCheck %s

; REQUIRES: x86-registered-target

; Test that the custom abi marks shadow parameters as zero extended.

; CHECK: @__dfsan_shadow_width_bits = weak_odr constant i32 [[#SBITS:]]
; CHECK: @__dfsan_shadow_width_bytes = weak_odr constant i32 [[#SBYTES:]]

define i32 @m() {
  ; CHECK-LABEL: @m.dfsan
  ; CHECK: %{{.*}} = call zeroext i16 @__dfsw_dfsan_get_label(i64 signext 56, i[[#SBITS]] zeroext 0, ptr %{{.*}})

entry:
  %call = call zeroext i16 @dfsan_get_label(i64 signext 56)
  %conv = zext i16 %call to i32
  ret i32 %conv
}

define i32 @k() {
  ; CHECK-LABEL: @k.dfsan
  ; CHECK: %{{.*}} = call zeroext i16 @__dfsw_k2(i64 signext 56, i64 signext 67, i[[#SBITS]] zeroext {{.*}}, i[[#SBITS]] zeroext {{.*}}, ptr %{{.*}})

entry:
  %call = call zeroext i16 @k2(i64 signext 56, i64 signext 67)
  %conv = zext i16 %call to i32
  ret i32 %conv
}

define i32 @k3() {
  ; CHECK-LABEL: @k3.dfsan
  ; CHECK: %{{.*}} = call zeroext i16 @__dfsw_k4(i64 signext 56, i64 signext 67, i64 signext 78, i64 signext 89, i[[#SBITS]] zeroext {{.*}}, i[[#SBITS]] zeroext {{.*}}, i[[#SBITS]] zeroext {{.*}}, i[[#SBITS]] zeroext {{.*}}, ptr %{{.*}})

entry:
  %call = call zeroext i16 @k4(i64 signext 56, i64 signext 67, i64 signext 78, i64 signext 89)
  %conv = zext i16 %call to i32
  ret i32 %conv
}

declare zeroext i16 @dfsan_get_label(i64 signext)
; CHECK-LABEL: @"dfsw$dfsan_get_label"
; CHECK: %{{.*}} = call i16 @__dfsw_dfsan_get_label(i64 %0, i[[#SBITS]] zeroext %1, ptr %{{.*}})

declare zeroext i16 @k2(i64 signext, i64 signext)
; CHECK-LABEL: @"dfsw$k2"
; CHECK: %{{.*}} = call i16 @__dfsw_k2(i64 %{{.*}}, i64 %{{.*}}, i[[#SBITS]] zeroext %{{.*}}, i[[#SBITS]] zeroext %{{.*}}, ptr %{{.*}})

declare zeroext i16 @k4(i64 signext, i64 signext, i64 signext, i64 signext)
; CHECK-LABEL: @"dfsw$k4"
; CHECK: %{{.*}} = call i16 @__dfsw_k4(i64 %{{.*}}, i64 %{{.*}}, i64  %{{.*}}, i64 %{{.*}}, i[[#SBITS]] zeroext %{{.*}}, i[[#SBITS]] zeroext %{{.*}}, i[[#SBITS]] zeroext %{{.*}}, i[[#SBITS]] zeroext %{{.*}}, ptr %{{.*}})


; CHECK: declare zeroext i16 @__dfsw_dfsan_get_label(i64 signext, i[[#SBITS]], ptr)
; CHECK: declare zeroext i16 @__dfsw_k2(i64 signext, i64 signext, i[[#SBITS]], i[[#SBITS]], ptr)
; CHECK: declare zeroext i16 @__dfsw_k4(i64 signext, i64 signext, i64 signext, i64 signext, i[[#SBITS]], i[[#SBITS]], i[[#SBITS]], i[[#SBITS]], ptr)
