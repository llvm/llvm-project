; RUN: opt -S -passes=mergefunc < %s | FileCheck %s

; A function's address is observable when it is used as a value, so it must not
; be replaced with the address of a forwarding thunk's target.
; CHECK-LABEL: define void @f(
; CHECK:         icmp eq ptr {{.*}}, @f
; CHECK:         call void @llvm.assume
; CHECK-LABEL: define void @g(
; CHECK:         icmp eq ptr {{.*}}, @g
; CHECK-NOT:     tail call void @f
; CHECK:         call void @llvm.assume

define void @f(ptr %p) {
  %cmp = icmp eq ptr %p, @f
  call void @llvm.assume(i1 %cmp)
  ret void
}

define void @g(ptr %p) {
  %cmp = icmp eq ptr %p, @g
  call void @llvm.assume(i1 %cmp)
  ret void
}

; A self-reference passed as an argument is also observable and must remain a
; normal value comparison rather than a call-target comparison.
; CHECK-LABEL: define void @arg_f(
; CHECK:         call void @consume(ptr @arg_f)
; CHECK-LABEL: define void @arg_g(
; CHECK:         call void @consume(ptr @arg_g)

declare void @consume(ptr)

define void @arg_f() {
  call void @consume(ptr @arg_f)
  ret void
}

define void @arg_g() {
  call void @consume(ptr @arg_g)
  ret void
}

define i32 @main() {
  call void @f(ptr @f)
  call void @g(ptr @g)
  call void @arg_f()
  call void @arg_g()
  ret i32 0
}
