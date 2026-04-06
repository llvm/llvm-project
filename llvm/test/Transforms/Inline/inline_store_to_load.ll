; RUN: opt < %s -passes=inline -inline-threshold=20 -S | FileCheck %s

; Test that the inliner can use store-to-load forwarding to resolve call
; arguments to constants. We use -inline-threshold=20 so that @callee is
; only inlined when the constant argument enables dead-branch elimination.

target datalayout = "p:64:64"

; Two paths: mode==0 is trivial (ret), otherwise too expensive to inline.
define i32 @callee(i32 %mode, ptr %p) {
entry:
  %cmp = icmp eq i32 %mode, 0
  br i1 %cmp, label %fast, label %slow
fast:
  %v = load i32, ptr %p
  ret i32 %v
slow:
  %a1 = load volatile i32, ptr %p
  %a2 = load volatile i32, ptr %p
  %x1 = add i32 %a1, %a2
  %a3 = load volatile i32, ptr %p
  %x2 = add i32 %x1, %a3
  %a4 = load volatile i32, ptr %p
  %x3 = add i32 %x2, %a4
  %a5 = load volatile i32, ptr %p
  %x4 = add i32 %x3, %a5
  ret i32 %x4
}

; Trivial when called with null, otherwise too expensive to inline.
define void @recursive_callee(ptr %x) {
entry:
  %cmp = icmp eq ptr %x, null
  br i1 %cmp, label %done, label %recurse
recurse:
  %next = load ptr, ptr %x
  call void @recursive_callee(ptr %next)
  %v = load volatile i32, ptr %x
  br label %done
done:
  ret void
}

; Store-to-load forwarding resolves %mode to 0, making only the fast path
; reachable and the callee cheap enough to inline.
; CHECK-LABEL: define i32 @caller_store_forward(
; CHECK: %cmp.i = icmp eq i32 %mode, 0
; CHECK: callee.exit:
; CHECK-NEXT: %{{.*}} = phi i32
; CHECK-NEXT: ret i32
define i32 @caller_store_forward() {
entry:
  %p = alloca i32
  store i32 0, ptr %p
  %mode = load i32, ptr %p
  %r = call i32 @callee(i32 %mode, ptr %p)
  ret i32 %r
}

; Memset-to-load forwarding converts the zero-filled integer to a null
; pointer, making the null check take the early exit.
; CHECK-LABEL: define void @caller_memset_ptr(
; CHECK: %cmp.i = icmp eq ptr %x, null
; CHECK: recursive_callee.exit:
; CHECK-NEXT: ret void
define void @caller_memset_ptr() {
entry:
  %p = alloca ptr
  call void @llvm.memset.p0.i64(ptr %p, i8 0, i64 8, i1 false)
  %x = load ptr, ptr %p
  call void @recursive_callee(ptr %x)
  ret void
}
