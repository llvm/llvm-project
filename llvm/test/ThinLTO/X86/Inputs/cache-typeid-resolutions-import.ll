target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i1 @importf1(i8* %p) {
  %x = call i1 @f1(i8* %p)
  ret i1 %x
}

define i1 @importf2(i8* %p) {
  %x = call i1 @f2(i8* %p)
  ret i1 %x
}

declare i1 @f1(i8* %p)
declare i1 @f2(i8* %p)
