; RUN: opt < %s -argpromotion -S | FileCheck %s
; RUN: opt < %s -passes=argpromotion -S | FileCheck %s

target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

; CHECK: define internal void @add(i32 %[[THIS1:.*]], i32 %[[THIS2:.*]], i32* noalias %[[SR:.*]])
define internal void @add({i32, i32}* %this, i32* sret %r) {
  %ap = getelementptr {i32, i32}, {i32, i32}* %this, i32 0, i32 0
  %bp = getelementptr {i32, i32}, {i32, i32}* %this, i32 0, i32 1
  %a = load i32, i32* %ap
  %b = load i32, i32* %bp
  ; CHECK: %[[AB:.*]] = add i32 %[[THIS1]], %[[THIS2]]
  %ab = add i32 %a, %b
  ; CHECK: store i32 %[[AB]], i32* %[[SR]]
  store i32 %ab, i32* %r
  ret void
}

; CHECK: define void @f()
define void @f() {
  ; CHECK: %[[R:.*]] = alloca i32
  %r = alloca i32
  %pair = alloca {i32, i32}

  ; CHECK: call void @add(i32 %{{.*}}, i32 %{{.*}}, i32* noalias %[[R]])
  call void @add({i32, i32}* %pair, i32* sret %r)
  ret void
}
