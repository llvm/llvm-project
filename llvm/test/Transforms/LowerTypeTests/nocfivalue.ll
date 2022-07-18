; RUN: opt -S -passes=lowertypetests %s | FileCheck %s

target datalayout = "e-p:64:64"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: @a = global [6 x void ()*] [void ()* no_cfi @f1, void ()* @.cfi.jumptable, void ()* bitcast ([8 x i8]* getelementptr inbounds ([3 x [8 x i8]], [3 x [8 x i8]]* bitcast (void ()* @.cfi.jumptable to [3 x [8 x i8]]*), i64 0, i64 1) to void ()*), void ()* no_cfi @f2, void ()* @f3, void ()* no_cfi @f3.cfi]
@a = global [6 x void ()*] [void ()* no_cfi @f1, void ()* @f1, void ()* @f2, void ()* no_cfi @f2, void ()* @f3, void ()* no_cfi @f3]

; CHECK: define void @f1()
define void @f1() !type !0 {
  ret void
}

; CHECK: define internal void @f2()
define internal void @f2() !type !0 {
  ret void
}

; CHECK: define hidden void @f3.cfi()
define void @f3() #0 !type !0 {
  ret void
}

declare i1 @llvm.type.test(i8* %ptr, metadata %bitset) nounwind readnone

define i1 @foo(i8* %p) {
  %x = call i1 @llvm.type.test(i8* %p, metadata !"typeid1")
  ret i1 %x
}

!llvm.module.flags = !{!1}

attributes #0 = { "cfi-canonical-jump-table" }

!0 = !{i32 0, !"typeid1"}
!1 = !{i32 4, !"CFI Canonical Jump Tables", i32 0}
