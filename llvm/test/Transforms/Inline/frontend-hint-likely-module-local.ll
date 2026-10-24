; RUN: opt -passes=inline -inline-threshold=0 -S < %s | FileCheck %s

; A linkonce_odr function marked "frontend-hint-likely-module-local" gets the
; last-call-to-static bonus for its sole call, like a local function would.

; CHECK-LABEL: define i32 @caller(
; CHECK-NOT: call i32 @hint_once(
; CHECK-COUNT-2: call i32 @hint_twice(
; CHECK: call i32 @nohint_once(
; CHECK: call i32 @hint_external_once(
; CHECK: ret i32
define i32 @caller(i32 %x) {
  %a = call i32 @hint_once(i32 %x)
  %b1 = call i32 @hint_twice(i32 %a)
  %b2 = call i32 @hint_twice(i32 %b1)
  %c = call i32 @nohint_once(i32 %b2)
  %d = call i32 @hint_external_once(i32 %c)
  ret i32 %d
}

define linkonce_odr i32 @hint_once(i32 %x) "frontend-hint-likely-module-local" {
  %v1 = mul i32 %x, 3
  %v2 = mul i32 %v1, 5
  %v3 = mul i32 %v2, 7
  %v4 = mul i32 %v3, 11
  %v5 = mul i32 %v4, 13
  %v6 = mul i32 %v5, 17
  %v7 = mul i32 %v6, 19
  %v8 = mul i32 %v7, 23
  %v9 = mul i32 %v8, 29
  %v10 = mul i32 %v9, 31
  %v11 = mul i32 %v10, 37
  %v12 = mul i32 %v11, 41
  %v13 = mul i32 %v12, 43
  %v14 = mul i32 %v13, 47
  %v15 = mul i32 %v14, 53
  ret i32 %v15
}

define linkonce_odr i32 @hint_twice(i32 %x) "frontend-hint-likely-module-local" {
  %v1 = mul i32 %x, 3
  %v2 = mul i32 %v1, 5
  %v3 = mul i32 %v2, 7
  %v4 = mul i32 %v3, 11
  %v5 = mul i32 %v4, 13
  %v6 = mul i32 %v5, 17
  %v7 = mul i32 %v6, 19
  %v8 = mul i32 %v7, 23
  %v9 = mul i32 %v8, 29
  %v10 = mul i32 %v9, 31
  %v11 = mul i32 %v10, 37
  %v12 = mul i32 %v11, 41
  %v13 = mul i32 %v12, 43
  %v14 = mul i32 %v13, 47
  %v15 = mul i32 %v14, 53
  ret i32 %v15
}

define linkonce_odr i32 @nohint_once(i32 %x) {
  %v1 = mul i32 %x, 3
  %v2 = mul i32 %v1, 5
  %v3 = mul i32 %v2, 7
  %v4 = mul i32 %v3, 11
  %v5 = mul i32 %v4, 13
  %v6 = mul i32 %v5, 17
  %v7 = mul i32 %v6, 19
  %v8 = mul i32 %v7, 23
  %v9 = mul i32 %v8, 29
  %v10 = mul i32 %v9, 31
  %v11 = mul i32 %v10, 37
  %v12 = mul i32 %v11, 41
  %v13 = mul i32 %v12, 43
  %v14 = mul i32 %v13, 47
  %v15 = mul i32 %v14, 53
  ret i32 %v15
}

define i32 @hint_external_once(i32 %x) "frontend-hint-likely-module-local" {
  %v1 = mul i32 %x, 3
  %v2 = mul i32 %v1, 5
  %v3 = mul i32 %v2, 7
  %v4 = mul i32 %v3, 11
  %v5 = mul i32 %v4, 13
  %v6 = mul i32 %v5, 17
  %v7 = mul i32 %v6, 19
  %v8 = mul i32 %v7, 23
  %v9 = mul i32 %v8, 29
  %v10 = mul i32 %v9, 31
  %v11 = mul i32 %v10, 37
  %v12 = mul i32 %v11, 41
  %v13 = mul i32 %v12, 43
  %v14 = mul i32 %v13, 47
  %v15 = mul i32 %v14, 53
  ret i32 %v15
}
