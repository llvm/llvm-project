; RUN: opt < %s -S -passes=globalopt | FileCheck %s

; This global is externally_initialized, so if we split it into scalars we
; should keep that flag set on all of the new globals. This will prevent the
; store to @a[0] from being constant propagated to the load in @foo, but will not
; prevent @a[1] from being removed since it is dead.
; CHECK: @a.0 = internal unnamed_addr externally_initialized global i32 undef
; CHECK-NOT: @a.1
@a = internal externally_initialized global [2 x i32] undef, align 4
; This is the same, but a struct rather than an array.
; CHECK: @b.0 = internal unnamed_addr externally_initialized global i32 undef
; CHECK-NOT: @b.1
@b = internal externally_initialized global {i32, i32} undef, align 4

define i32 @foo() {
; CHECK-LABEL: define i32 @foo
entry:
; This load uses the split global, but cannot be constant-propagated away.
; CHECK: %0 = load i32, ptr @a.0
  %0 = load i32, ptr @a, align 4
  ret i32 %0
}

define i32 @bar() {
; CHECK-LABEL: define i32 @bar
entry:
; This load uses the split global, but cannot be constant-propagated away.
; CHECK: %0 = load i32, ptr @b.0
  %0 = load i32, ptr getelementptr inbounds ({i32, i32}, ptr @b, i32 0, i32 0), align 4
  ret i32 %0
}

define void @init() {
; CHECK-LABEL: define void @init
entry:
; This store uses the split global, but cannot be constant-propagated away.
; CHECK: store i32 1, ptr @a.0
  store i32 1, ptr @a, align 4
; This store can be removed, because the second element of @a is never read.
; CHECK-NOT: store i32 2, ptr @a.1
  store i32 2, ptr getelementptr inbounds ([2 x i32], ptr @a, i32 0, i32 1), align 4

; This store uses the split global, but cannot be constant-propagated away.
; CHECK: store i32 3, ptr @b.0
  store i32 3, ptr getelementptr inbounds ({i32, i32}, ptr @b, i32 0, i32 0), align 4
; This store can be removed, because the second element of @b is never read.
; CHECK-NOT: store i32 4, ptr @b.1
  store i32 4, ptr getelementptr inbounds ({i32, i32}, ptr @b, i32 0, i32 1), align 4
  ret void
}
