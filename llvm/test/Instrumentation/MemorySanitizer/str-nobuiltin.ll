; Test marking string functions as nobuiltin in memory sanitizer.
;
; RUN: opt < %s -S -passes=msan 2>&1 | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

declare ptr @memchr(ptr %a, i32 %b, i64 %c)
declare i32 @memcmp(ptr %a, ptr %b, i64 %c)
declare i32 @strcmp(ptr %a, ptr %b)
declare ptr @strcpy(ptr %a, ptr %b)
declare ptr @stpcpy(ptr %a, ptr %b)
declare i64 @strlen(ptr %a)
declare i64 @strnlen(ptr %a, i64 %b)

; CHECK: call{{.*}}@memchr{{.*}} #[[ATTR:[0-9]+]]
; CHECK: call{{.*}}@memcmp{{.*}} #[[ATTR]]
; CHECK: call{{.*}}@strcmp{{.*}} #[[ATTR]]
; CHECK: call{{.*}}@strcpy{{.*}} #[[ATTR]]
; CHECK: call{{.*}}@stpcpy{{.*}} #[[ATTR]]
; CHECK: call{{.*}}@strlen{{.*}} #[[ATTR]]
; CHECK: call{{.*}}@strnlen{{.*}} #[[ATTR]]
; attributes #[[ATTR]] = { nobuiltin }

define void @f1(ptr %a, ptr %b) nounwind uwtable sanitize_memory {
  tail call ptr @memchr(ptr %a, i32 1, i64 12)
  tail call i32 @memcmp(ptr %a, ptr %b, i64 12)
  tail call i32 @strcmp(ptr %a, ptr %b)
  tail call ptr @strcpy(ptr %a, ptr %b)
  tail call ptr @stpcpy(ptr %a, ptr %b)
  tail call i64 @strlen(ptr %a)
  tail call i64 @strnlen(ptr %a, i64 12)
  ret void
}
