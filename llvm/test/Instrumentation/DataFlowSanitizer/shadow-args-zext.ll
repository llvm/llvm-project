; RUN: opt -mtriple=x86_64-unknown-linux-gnu < %s -dfsan -S --dfsan-abilist=%S/Inputs/shadow-args-abilist.txt  | FileCheck %s

; REQUIRES: x86-registered-target

; Test that the custom abi marks shadow parameters as zero extended.

define i32 @m() {
entry:
  %call = call zeroext i16 @dfsan_get_label(i64 signext 56)
  %conv = zext i16 %call to i32
  ret i32 %conv
}

; CHECK-LABEL: @"dfs$m"
; CHECK: %{{.*}} = call zeroext i16 @__dfsw_dfsan_get_label(i64 signext 56, i16 zeroext 0, i16* %{{.*}})

define i32 @k() {
entry:
  %call = call zeroext i16 @k2(i64 signext 56, i64 signext 67)
  %conv = zext i16 %call to i32
  ret i32 %conv
}

; CHECK-LABEL: @"dfs$k"
; CHECK: %{{.*}} = call zeroext i16 @__dfsw_k2(i64 signext 56, i64 signext 67, i16 zeroext {{.*}}, i16 zeroext {{.*}}, i16* %{{.*}})

define i32 @k3() {
entry:
  %call = call zeroext i16 @k4(i64 signext 56, i64 signext 67, i64 signext 78, i64 signext 89)
  %conv = zext i16 %call to i32
  ret i32 %conv
}

; CHECK-LABEL: @"dfs$k3"
; CHECK: %{{.*}} = call zeroext i16 @__dfsw_k4(i64 signext 56, i64 signext 67, i64 signext 78, i64 signext 89, i16 zeroext {{.*}}, i16 zeroext {{.*}}, i16 zeroext {{.*}}, i16 zeroext {{.*}}, i16* %{{.*}})

declare zeroext i16 @dfsan_get_label(i64 signext)

; CHECK-LABEL: @"dfsw$dfsan_get_label"
; CHECK: %{{.*}} = call i16 @__dfsw_dfsan_get_label(i64 %0, i16 zeroext %1, i16* %{{.*}})

declare zeroext i16 @k2(i64 signext, i64 signext)
; CHECK-LABEL: @"dfsw$k2"
; CHECK: %{{.*}} = call i16 @__dfsw_k2(i64 %{{.*}}, i64 %{{.*}}, i16 zeroext %{{.*}}, i16 zeroext %{{.*}}, i16* %{{.*}})

declare zeroext i16 @k4(i64 signext, i64 signext, i64 signext, i64 signext)

; CHECK-LABEL: @"dfsw$k4"
; CHECK: %{{.*}} = call i16 @__dfsw_k4(i64 %{{.*}}, i64 %{{.*}}, i64  %{{.*}}, i64 %{{.*}}, i16 zeroext %{{.*}}, i16 zeroext %{{.*}}, i16 zeroext %{{.*}}, i16 zeroext %{{.*}}, i16* %{{.*}})


; CHECK: declare zeroext i16 @__dfsw_dfsan_get_label(i64 signext, i16, i16*)
; CHECK: declare zeroext i16 @__dfsw_k2(i64 signext, i64 signext, i16, i16, i16*)
; CHECK: declare zeroext i16 @__dfsw_k4(i64 signext, i64 signext, i64 signext, i64 signext, i16, i16, i16, i16, i16*)
