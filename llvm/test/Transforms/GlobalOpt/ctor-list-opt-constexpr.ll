; RUN: opt -passes=globalopt -S < %s | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin10.0.0"

%0 = type { i32, ptr, ptr }
%struct.foo = type { ptr }
%struct.bar = type { i128 }

@G = global i32 0, align 4
@H = global i32 0, align 4
@X = global %struct.foo zeroinitializer, align 8
@X2 = global %struct.bar zeroinitializer, align 8
@llvm.global_ctors = appending global [2 x %0] [%0 { i32 65535, ptr @init1, ptr null }, %0 { i32 65535, ptr @init2, ptr null }]

; PR8710 - GlobalOpt shouldn't change the global's initializer to have this
; arbitrary constant expression, the code generator can't handle it.
define internal void @init1() {
entry:
  store ptr inttoptr (i64 xor (i64 ptrtoint (ptr @G to i64), i64 ptrtoint (ptr @H to i64)) to ptr), ptr @X, align 8
  ret void
}
; CHECK-LABEL: @init1(
; CHECK: store ptr

; PR11705 - ptrtoint isn't safe in general in global initializers.
define internal void @init2() {
entry:
  store i128 ptrtoint (ptr @G to i128), ptr @X2, align 16
  ret void
}
; CHECK-LABEL: @init2(
; CHECK: store i128
