; RUN: opt -S -passes=lowertypetests %s | FileCheck %s

target datalayout = "e-p:64:64"
target triple = "x86_64-unknown-linux"

; CHECK: @0 = private constant { i32, [0 x i8], i32 } { i32 1, [0 x i8] zeroinitializer, i32 2 }
; CHECK: @g1 = alias i32, getelementptr inbounds ({ i32, [0 x i8], i32 }, { i32, [0 x i8], i32 }* @0, i32 0, i32 0)
; CHECK: @g2 = alias i32, getelementptr inbounds ({ i32, [0 x i8], i32 }, { i32, [0 x i8], i32 }* @0, i32 0, i32 2)
; CHECK: @f1 = alias void (), void ()* @.cfi.jumptable
; CHECK: @f2 = alias void (), bitcast ([8 x i8]* getelementptr inbounds ([2 x [8 x i8]], [2 x [8 x i8]]* bitcast (void ()* @.cfi.jumptable to [2 x [8 x i8]]*), i64 0, i64 1) to void ()*)

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

define void @jt2(i8* nest, ...) {
  musttail call void (...) @llvm.icall.branch.funnel(
      i8* %0,
      i32* @g1, void ()* @g1f,
      i32* @g2, void ()* @g2f,
      ...
  )
  ret void
}

define void @jt3(i8* nest, ...) {
  musttail call void (...) @llvm.icall.branch.funnel(
      i8* %0,
      void ()* @f1, void ()* @f1,
      void ()* @f2, void ()* @f2,
      ...
  )
  ret void
}

declare void @llvm.icall.branch.funnel(...)
