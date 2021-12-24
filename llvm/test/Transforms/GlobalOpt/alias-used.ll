; RUN: opt < %s -passes=globalopt -S | FileCheck %s

@c = global i8 42

@i = internal global i8 42
; CHECK: @ia = internal global i8 42
@ia = internal alias i8, i8* @i

@llvm.used = appending global [3 x i8*] [i8* bitcast (void ()* @fa to i8*), i8* bitcast (void ()* @f to i8*), i8* @ca], section "llvm.metadata"
; CHECK-DAG: @llvm.used = appending global [3 x i8*] [i8* @ca, i8* bitcast (void ()* @f to i8*), i8* bitcast (void ()* @fa to i8*)], section "llvm.metadata"

@llvm.compiler.used = appending global [4 x i8*] [i8* bitcast (void ()* @fa3 to i8*), i8* bitcast (void ()* @fa to i8*), i8* @ia, i8* @i], section "llvm.metadata"
; CHECK-DAG: @llvm.compiler.used = appending global [2 x i8*] [i8* bitcast (void ()* @fa3 to i8*), i8* @ia], section "llvm.metadata"

@sameAsUsed = global [3 x i8*] [i8* bitcast (void ()* @fa to i8*), i8* bitcast (void ()* @f to i8*), i8* @ca]
; CHECK-DAG: @sameAsUsed = local_unnamed_addr global [3 x i8*] [i8* bitcast (void ()* @f to i8*), i8* bitcast (void ()* @f to i8*), i8* @c]

@other = global i32* bitcast (void ()* @fa to i32*)
; CHECK-DAG: @other = local_unnamed_addr global i32* bitcast (void ()* @f to i32*)

@fa = internal alias void (), void ()* @f
; CHECK: @fa = internal alias void (), void ()* @f

@fa2 = internal alias void (), void ()* @f
; CHECK-NOT: @fa2

@fa3 = internal alias void (), void ()* @f
; CHECK: @fa3

@ca = internal alias i8, i8* @c
; CHECK: @ca = internal alias i8, i8* @c

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

; Check that GlobalOpt doesn't try to resolve aliases with GEP operands.

%struct.S = type { i32, i32, i32 }
@s = global %struct.S { i32 1, i32 2, i32 3 }, align 4

@alias1 = alias i32, i32* getelementptr inbounds (%struct.S, %struct.S* @s, i64 0, i32 1)
@alias2 = alias i32, i32* getelementptr inbounds (%struct.S, %struct.S* @s, i64 0, i32 2)

; CHECK: load i32, i32* @alias1, align 4
; CHECK: load i32, i32* @alias2, align 4

define i32 @foo1() {
entry:
  %0 = load i32, i32* @alias1, align 4
  %1 = load i32, i32* @alias2, align 4
  %add = add nsw i32 %1, %0
  ret i32 %add
}
