target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i1 @importf1(ptr %p) {
  %x = call i1 @f1(ptr %p)
  ret i1 %x
}

define i1 @importf2(ptr %p) {
  %x = call i1 @f2(ptr %p)
  ret i1 %x
}

declare i1 @f1(ptr %p)
declare i1 @f2(ptr %p)
