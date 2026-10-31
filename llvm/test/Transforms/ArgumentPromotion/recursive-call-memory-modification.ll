; RUN: opt -passes=argpromotion -S < %s -o %t
; RUN: FileCheck %s < %t
; RUN: lli %t

@G = global i32 1, align 4

; The recursive call must observe the value stored in memory, not a
; stale value captured when the function was first called.

; CHECK-LABEL: define internal i32 @f(ptr %p, i1 %cond)
; CHECK: store i32 0, ptr @G, align 4
; CHECK: %r = call i32 @f(ptr %p, i1 true)

define internal i32 @f(ptr %p, i1 %cond) {
entry:
  br i1 %cond, label %load, label %store

load:
  %v = load i32, ptr %p, align 4
  ret i32 %v

store:
  store i32 0, ptr @G, align 4
  %r = call i32 @f(ptr %p, i1 true)
  ret i32 %r
}

; CHECK-LABEL: define i32 @main()
; CHECK: %r = call i32 @f(ptr @G, i1 false)

define i32 @main() {
  %r = call i32 @f(ptr @G, i1 false)
  ret i32 %r
}
