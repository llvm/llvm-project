; RUN: opt < %s -globalopt -S | FileCheck %s

@c = global i8 42

@llvm.used = appending global [3 x i8*] [i8* bitcast (void ()* @fa to i8*), i8* bitcast (void ()* @f to i8*), i8* @ca], section "llvm.metadata"
; CHECK: @llvm.used = appending global [3 x i8*] [i8* bitcast (void ()* @fa to i8*), i8* bitcast (void ()* @f to i8*), i8* @ca], section "llvm.metadata"

@llvm.compiler_used = appending global [2 x i8*] [i8* bitcast (void ()* @fa to i8*), i8* bitcast (void ()* @fa3 to i8*)], section "llvm.metadata"

@sameAsUsed = global [3 x i8*] [i8* bitcast (void ()* @fa to i8*), i8* bitcast (void ()* @f to i8*), i8* @ca]
; CHECK: @sameAsUsed = global [3 x i8*] [i8* bitcast (void ()* @f to i8*), i8* bitcast (void ()* @f to i8*), i8* @c]

@other = global i32* bitcast (void ()* @fa to i32*)
; CHECK: @other = global i32* bitcast (void ()* @f to i32*)

@fa = alias internal void ()* @f
; CHECK: @fa = alias internal void ()* @f

@fa2 = alias internal void ()* @f
; CHECK-NOT: @fa2

@fa3 = alias internal void ()* @f
; CHECK: @fa3

@ca = alias internal i8* @c
; CHECK: @ca = alias internal i8* @c

define void @f() {
  ret void
}

define i8* @g() {
  ret i8* bitcast (void ()* @fa to i8*);
}

define i8* @g2() {
  ret i8* bitcast (void ()* @fa2 to i8*);
}

define i8* @h() {
  ret i8* @ca
}
