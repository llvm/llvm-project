; RUN: opt -globalopt -S < %s | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin10.0.0"

%0 = type { i32, void ()* }
%struct.foo = type { i32* }
%struct.bar = type { i128 }

@G = global i32 0, align 4
@H = global i32 0, align 4
@X = global %struct.foo zeroinitializer, align 8
@X2 = global %struct.bar zeroinitializer, align 8
@llvm.global_ctors = appending global [2 x %0] [%0 { i32 65535, void ()* @init1 }, %0 { i32 65535, void ()* @init2 }]

; PR8710 - GlobalOpt shouldn't change the global's initializer to have this
; arbitrary constant expression, the code generator can't handle it.
define internal void @init1() {
entry:
  %tmp = getelementptr inbounds %struct.foo, %struct.foo* @X, i32 0, i32 0
  store i32* inttoptr (i64 sdiv (i64 ptrtoint (i32* @G to i64), i64 ptrtoint (i32* @H to i64)) to i32*), i32** %tmp, align 8
  ret void
}
; CHECK-LABEL: @init1(
; CHECK: store i32*

; PR11705 - ptrtoint isn't safe in general in global initializers.
define internal void @init2() {
entry:
  %tmp = getelementptr inbounds %struct.bar, %struct.bar* @X2, i32 0, i32 0
  store i128 ptrtoint (i32* @G to i128), i128* %tmp, align 16
  ret void
}
; CHECK-LABEL: @init2(
; CHECK: store i128
