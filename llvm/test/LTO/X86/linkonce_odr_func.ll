; RUN: llvm-as < %s >%t1
; RUN: llvm-lto -o %t2 -dso-symbol=foo1 -dso-symbol=foo2 -dso-symbol=foo3 \
; RUN:     -dso-symbol=v1 -dso-symbol=v2 -dso-symbol=v3 \
; RUN:     -dso-symbol=v4 -dso-symbol=v5 -dso-symbol=v6 %t1 -O0
; RUN: llvm-nm %t2 | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: W foo1
define linkonce_odr void @foo1() noinline {
  ret void
}

; CHECK: t foo2
define linkonce_odr void @foo2() local_unnamed_addr noinline {
  ret void
}

; CHECK: t foo3
define linkonce_odr void @foo3() unnamed_addr noinline {
  ret void
}

; CHECK: V v1
@v1 = linkonce_odr constant i32 32

; CHECK: r v2
@v2 = linkonce_odr local_unnamed_addr constant i32 32

; CHECK: r v3
@v3 = linkonce_odr unnamed_addr constant i32 32

; CHECK: V v4
@v4 = linkonce_odr global i32 32

; CHECK: V v5
@v5 = linkonce_odr local_unnamed_addr global i32 32

; CHECK: d v6
@v6 = linkonce_odr unnamed_addr global i32 32

define void @use() {
  call void @foo1()
  call void @foo2()
  call void @foo3()
  %x1 = load i32, ptr @v1
  %x2 = load i32, ptr @v2
  %x3 = load i32, ptr @v3
  %x4 = load i32, ptr @v4
  %x5 = load i32, ptr @v5
  %x6 = load i32, ptr @v6
  ret void
}
