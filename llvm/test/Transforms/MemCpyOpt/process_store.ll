; RUN: opt < %s -memcpyopt -disable-output

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@b = common dso_local local_unnamed_addr global i32 0, align 4
@a = common dso_local local_unnamed_addr global i32 0, align 4

declare dso_local i32 @f1()

; Do not crash due to store first in BB.
define dso_local void @f2() {
for.end:
  %0 = load i32, i32* @b, align 4
  ret void

for.body:
  store i32 %1, i32* @a, align 4
  %call = call i32 @f1()
  %cmp = icmp sge i32 %call, 0
  %1 = load i32, i32* @b, align 4
  br label %for.body
}

; Do not crash due to call not before store in BB.
define dso_local void @f3() {
for.end:
  %0 = load i32, i32* @b, align 4
  ret void

for.body:
  %t = add i32 %t2, 1
  store i32 %1, i32* @a, align 4
  %call = call i32 @f1()
  %cmp = icmp sge i32 %call, 0
  %1 = load i32, i32* @b, align 4
  %t2 = xor i32 %t, 5
  br label %for.body
}
