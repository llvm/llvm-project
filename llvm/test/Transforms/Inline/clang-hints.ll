; RUN: opt -passes=inline -inline-threshold=0 -S < %s | FileCheck %s --check-prefix=OFF
; RUN: opt -passes=inline -inline-threshold=0 -inline-use-clang-hints -S < %s | FileCheck %s --check-prefix=HINTS
; RUN: opt -passes=inline -inline-threshold=0 -inline-use-clang-hints -clang-hints-lambda-main-file-bonus=0 -S < %s | FileCheck %s --check-prefix=LOCAL
; RUN: opt -passes=inline -inline-threshold=0 -inline-use-clang-hints -clang-hints-lambda-main-file-bonus=0 -clang-hints-local-lambdas=false -S < %s | FileCheck %s --check-prefix=NOLOCAL

; OFF-LABEL: define i32 @caller(
; OFF-COUNT-2: call i32 @lambda(
; OFF-COUNT-2: call i32 @main_template(
; OFF-COUNT-2: call i32 @main_plain(
; OFF: call i32 @sole_main_inline(

; HINTS-LABEL: define i32 @caller(
; HINTS-NOT: call i32 @lambda(
; HINTS-COUNT-2: call i32 @main_template(
; HINTS-COUNT-2: call i32 @main_plain(
; HINTS-NOT: call i32 @sole_main_inline(
; HINTS: ret i32

; LOCAL-LABEL: define i32 @caller(
; LOCAL-NOT: call i32 @sole_main_inline(
; LOCAL: ret i32

; NOLOCAL-LABEL: define i32 @caller(
; NOLOCAL: call i32 @sole_main_inline(
define i32 @caller(i32 %x) {
  %a1 = call i32 @lambda(i32 %x)
  %a2 = call i32 @lambda(i32 %a1)
  %b1 = call i32 @main_template(i32 %a2)
  %b2 = call i32 @main_template(i32 %b1)
  %c1 = call i32 @main_plain(i32 %b2)
  %c2 = call i32 @main_plain(i32 %c1)
  %d = call i32 @sole_main_inline(i32 %c2)
  ret i32 %d
}

define linkonce_odr i32 @lambda(i32 %x) "clang-lambda" "function-inline-cost"="300" {
  %r = add i32 %x, 1
  ret i32 %r
}

define linkonce_odr i32 @main_template(i32 %x) "clang-main-file"="inline-or-template" "function-inline-cost"="300" {
  %r = add i32 %x, 2
  ret i32 %r
}

define linkonce_odr i32 @main_plain(i32 %x) "clang-main-file" "function-inline-cost"="300" {
  %r = add i32 %x, 3
  ret i32 %r
}

define linkonce_odr i32 @sole_main_inline(i32 %x) "clang-lambda" "clang-main-file" {
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
  ret i32 %v10
}
