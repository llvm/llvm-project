; RUN: opt -S -passes=lowertypetests %s | FileCheck %s

target datalayout = "e-p:64:64"
target triple = "x86_64-unknown-linux"

; CHECK: @0 = private constant { i32, [0 x i8], i32 } { i32 1, [0 x i8] zeroinitializer, i32 2 }
; CHECK: @g1 = alias i32, ptr @0
; CHECK: @g2 = alias i32, getelementptr inbounds ({ i32, [0 x i8], i32 }, ptr @0, i32 0, i32 2)
; CHECK: @f1 = alias void (), ptr @.cfi.jumptable
; CHECK: @f2 = alias void (), getelementptr inbounds ([2 x [8 x i8]], ptr @.cfi.jumptable, i64 0, i64 1)

@g1 = constant i32 1
@g2 = constant i32 2

define void @f1() {
  ret void
}

define void @f2() {
  ret void
}

declare void @g1f()
declare void @g2f()

define void @jt2(ptr nest, ...) {
  musttail call void (...) @llvm.icall.branch.funnel(
      ptr %0,
      ptr @g1, ptr @g1f,
      ptr @g2, ptr @g2f,
      ...
  )
  ret void
}

define void @jt3(ptr nest, ...) {
  musttail call void (...) @llvm.icall.branch.funnel(
      ptr %0,
      ptr @f1, ptr @f1,
      ptr @f2, ptr @f2,
      ...
  )
  ret void
}

declare void @llvm.icall.branch.funnel(...)
