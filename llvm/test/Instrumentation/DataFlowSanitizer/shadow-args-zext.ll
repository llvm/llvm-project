; RUN: opt -mtriple=x86_64-unknown-linux-gnu < %s -passes=dfsan -S --dfsan-abilist=%S/Inputs/shadow-args-abilist.txt | FileCheck %s
; RUN: opt -mtriple=s390x-unknown-linux < %s -passes=dfsan -S --dfsan-abilist=%S/Inputs/shadow-args-abilist.txt \
; RUN:   | FileCheck %s --check-prefix=SYSTEMZ

; REQUIRES: x86-registered-target, systemz-registered-target

; Test that the custom abi marks shadow parameters as zero extended.

define i32 @m() {
  ; CHECK-LABEL: @m.dfsan
  ; CHECK: %{{.*}} = call zeroext i16 @__dfsw_dfsan_get_label(i64 signext 56, i8 0, ptr %{{.*}})
  ; SYSTEMZ-LABEL: @m.dfsan
  ; SYSTEMZ: %{{.*}} = call zeroext i16 @__dfsw_dfsan_get_label(i64 signext 56, i8 zeroext 0, ptr %{{.*}})

entry:
  %call = call zeroext i16 @dfsan_get_label(i64 signext 56)
  %conv = zext i16 %call to i32
  ret i32 %conv
}

define i32 @k() {
  ; CHECK-LABEL: @k.dfsan
  ; CHECK: %{{.*}} = call zeroext i16 @__dfsw_k2(i64 signext 56, i64 signext 67, i8 {{.*}}, i8 {{.*}}, ptr %{{.*}})
  ; SYSTEMZ-LABEL: @k.dfsan
  ; SYSTEMZ: %{{.*}} = call zeroext i16 @__dfsw_k2(i64 signext 56, i64 signext 67, i8 zeroext {{.*}}, i8 zeroext {{.*}}, ptr %{{.*}})

entry:
  %call = call zeroext i16 @k2(i64 signext 56, i64 signext 67)
  %conv = zext i16 %call to i32
  ret i32 %conv
}

define i32 @k3() {
  ; CHECK-LABEL: @k3.dfsan
  ; CHECK: %{{.*}} = call zeroext i16 @__dfsw_k4(i64 signext 56, i64 signext 67, i64 signext 78, i64 signext 89, i8 {{.*}}, i8 {{.*}}, i8 {{.*}}, i8 {{.*}}, ptr %{{.*}})
  ; SYSTEMZ-LABEL: @k3.dfsan
  ; SYSTEMZ: %{{.*}} = call zeroext i16 @__dfsw_k4(i64 signext 56, i64 signext 67, i64 signext 78, i64 signext 89, i8 zeroext {{.*}}, i8 zeroext {{.*}}, i8 zeroext {{.*}}, i8 zeroext {{.*}}, ptr %{{.*}})

entry:
  %call = call zeroext i16 @k4(i64 signext 56, i64 signext 67, i64 signext 78, i64 signext 89)
  %conv = zext i16 %call to i32
  ret i32 %conv
}

declare zeroext i16 @dfsan_get_label(i64 signext)
; CHECK-LABEL: @"dfsw$dfsan_get_label"
; CHECK: %{{.*}} = call i16 @__dfsw_dfsan_get_label(i64 signext %0, i8 %1, ptr %{{.*}})
; SYSTEMZ-LABEL: @"dfsw$dfsan_get_label"
; SYSTEMZ: %{{.*}} = call i16 @__dfsw_dfsan_get_label(i64 signext %0, i8 zeroext %1, ptr %{{.*}})

declare zeroext i16 @k2(i64 signext, i64 signext)
; CHECK-LABEL: @"dfsw$k2"
; CHECK: %{{.*}} = call i16 @__dfsw_k2(i64 signext %{{.*}}, i64 signext %{{.*}}, i8 %{{.*}}, i8 %{{.*}}, ptr %{{.*}})
; SYSTEMZ-LABEL: @"dfsw$k2"
; SYSTEMZ: %{{.*}} = call i16 @__dfsw_k2(i64 signext %{{.*}}, i64 signext %{{.*}}, i8 zeroext %{{.*}}, i8 zeroext %{{.*}}, ptr %{{.*}})

declare zeroext i16 @k4(i64 signext, i64 signext, i64 signext, i64 signext)
; CHECK-LABEL: @"dfsw$k4"
; CHECK: %{{.*}} = call i16 @__dfsw_k4(i64 signext %{{.*}}, i64 signext %{{.*}}, i64 signext %{{.*}}, i64 signext %{{.*}}, i8 %{{.*}}, i8 %{{.*}}, i8 %{{.*}}, i8 %{{.*}}, ptr %{{.*}})
; SYSTEMZ-LABEL: @"dfsw$k4"
; SYSTEMZ: %{{.*}} = call i16 @__dfsw_k4(i64 signext %{{.*}}, i64 signext %{{.*}}, i64 signext %{{.*}}, i64 signext %{{.*}}, i8 zeroext %{{.*}}, i8 zeroext %{{.*}}, i8 zeroext %{{.*}}, i8 zeroext %{{.*}}, ptr %{{.*}})


; CHECK: declare zeroext i16 @__dfsw_dfsan_get_label(i64 signext, i8, ptr)
; CHECK: declare zeroext i16 @__dfsw_k2(i64 signext, i64 signext, i8, i8, ptr)
; CHECK: declare zeroext i16 @__dfsw_k4(i64 signext, i64 signext, i64 signext, i64 signext, i8, i8, i8, i8, ptr)
; SYSTEMZ: declare zeroext i16 @__dfsw_dfsan_get_label(i64 signext, i8 zeroext, ptr)
; SYSTEMZ: declare zeroext i16 @__dfsw_k2(i64 signext, i64 signext, i8 zeroext, i8 zeroext, ptr)
; SYSTEMZ: declare zeroext i16 @__dfsw_k4(i64 signext, i64 signext, i64 signext, i64 signext, i8 zeroext, i8 zeroext, i8 zeroext, i8 zeroext, ptr)
