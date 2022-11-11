; Test marking string functions as nobuiltin in address sanitizer.
;
; RUN: opt < %s -passes=asan -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

declare i8* @memchr(i8* %a, i32 %b, i64 %c)
declare i32 @memcmp(i8* %a, i8* %b, i64 %c)
declare i32 @strcmp(i8* %a, i8* %b)
declare i8* @strcpy(i8* %a, i8* %b)
declare i8* @stpcpy(i8* %a, i8* %b)
declare i64 @strlen(i8* %a)
declare i64 @strnlen(i8* %a, i64 %b)

; CHECK: call{{.*}}@memchr{{.*}} #[[ATTR:[0-9]+]]
; CHECK: call{{.*}}@memcmp{{.*}} #[[ATTR]]
; CHECK: call{{.*}}@strcmp{{.*}} #[[ATTR]]
; CHECK: call{{.*}}@strcpy{{.*}} #[[ATTR]]
; CHECK: call{{.*}}@stpcpy{{.*}} #[[ATTR]]
; CHECK: call{{.*}}@strlen{{.*}} #[[ATTR]]
; CHECK: call{{.*}}@strnlen{{.*}} #[[ATTR]]
; attributes #[[ATTR]] = { nobuiltin }

define void @f1(i8* %a, i8* %b) nounwind uwtable sanitize_address {
  tail call i8* @memchr(i8* %a, i32 1, i64 12)
  tail call i32 @memcmp(i8* %a, i8* %b, i64 12)
  tail call i32 @strcmp(i8* %a, i8* %b)
  tail call i8* @strcpy(i8* %a, i8* %b)
  tail call i8* @stpcpy(i8* %a, i8* %b)
  tail call i64 @strlen(i8* %a)
  tail call i64 @strnlen(i8* %a, i64 12)
  ret void
}
