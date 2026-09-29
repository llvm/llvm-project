target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare !type !0 i8 @bar(ptr)
declare i1 @llvm.type.test(ptr %ptr, metadata %type) nounwind readnone

define i8 @baz(ptr %p) !type !0 {
  %x = call i1 @llvm.type.test(ptr %p, metadata !"t1")
  %1 = select i1 %x, i8 0, i8 3
  ret i8 %1
}

define i8 @qux(ptr %p) !type !0 {
  %x = call i1 @llvm.type.test(ptr %p, metadata !"t1")
  ret i8 4
}

define i8 @g(i1 %i, ptr %p) {
  %1 = select i1 %i, ptr @bar, ptr @qux
  %2 = call i8 %1(ptr %p)
  ret i8 %2
}

!0 = !{i64 0, !"t1"}
