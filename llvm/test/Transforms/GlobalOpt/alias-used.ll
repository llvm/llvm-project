; RUN: opt < %s -passes=globalopt -S | FileCheck %s

@c = dso_local global i8 42

; CHECK: @i = internal global i8 42
@i = internal global i8 42
; CHECK-DAG: @ia = internal alias i8, ptr @i
@ia = internal alias i8, ptr @i

@llvm.used = appending global [3 x ptr] [ptr @fa, ptr @f, ptr @ca], section "llvm.metadata"
; CHECK-DAG: @llvm.used = appending global [3 x ptr] [ptr @ca, ptr @f, ptr @fa], section "llvm.metadata"

@llvm.compiler.used = appending global [4 x ptr] [ptr @fa3, ptr @fa, ptr @ia, ptr @i], section "llvm.metadata"
; CHECK-DAG: @llvm.compiler.used = appending global [3 x ptr] [ptr @fa3, ptr @i, ptr @ia], section "llvm.metadata"

@sameAsUsed = global [3 x ptr] [ptr @fa, ptr @f, ptr @ca]
; CHECK-DAG: @sameAsUsed = local_unnamed_addr global [3 x ptr] [ptr @f, ptr @f, ptr @c]

@other = global ptr @fa
; CHECK-DAG: @other = local_unnamed_addr global ptr @f

@fa = internal alias void (), ptr @f
; CHECK: @fa = internal alias void (), ptr @f

@fa2 = internal alias void (), ptr @f
; CHECK-NOT: @fa2

@fa3 = internal alias void (), ptr @f
; CHECK: @fa3

@ca = internal alias i8, ptr @c
; CHECK: @ca = internal alias i8, ptr @c

define hidden void @f() {
  ret void
}

define ptr @g() {
  ret ptr @fa;
}

define ptr @g2() {
  ret ptr @fa2;
}

define ptr @h() {
  ret ptr @ca
}

; Check that GlobalOpt doesn't try to resolve aliases with GEP operands.

%struct.S = type { i32, i32, i32 }
@s = global %struct.S { i32 1, i32 2, i32 3 }, align 4

@alias1 = alias i32, ptr getelementptr inbounds (%struct.S, ptr @s, i64 0, i32 1)
@alias2 = alias i32, ptr getelementptr inbounds (%struct.S, ptr @s, i64 0, i32 2)

; CHECK: load i32, ptr @alias1, align 4
; CHECK: load i32, ptr @alias2, align 4

define i32 @foo1() {
entry:
  %0 = load i32, ptr @alias1, align 4
  %1 = load i32, ptr @alias2, align 4
  %add = add nsw i32 %1, %0
  ret i32 %add
}
