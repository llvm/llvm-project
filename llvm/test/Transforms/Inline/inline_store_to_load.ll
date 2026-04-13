; RUN: opt -S -passes="cgscc(inline)" < %s | FileCheck %s

; Test that the CGSCC inliner performs store-to-load forwarding for call
; arguments after inlining a function in the same caller. The first call
; (@init or @init_memset) is inlined, producing stores. Forwarding then
; resolves the subsequent load to a constant, enabling inlining of the
; second callee.

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

; Trivially cheap — inlined first, producing a store.
define internal void @init_i32(ptr %p) {
  store i32 0, ptr %p
  ret void
}

; Trivially cheap — inlined first, producing a memset.
define internal void @init_memset(ptr %p) {
  call void @llvm.memset.p0.i64(ptr %p, i8 0, i64 8, i1 false)
  ret void
}

declare void @llvm.memset.p0.i64(ptr, i8, i64, i1)

; After inlining @init_i32: store 0 → %p.
; Forwarding resolves %mode to 0, making only the fast path reachable.
; CHECK-LABEL: @caller_store_forward(
; CHECK-NOT: call i32 @callee
; CHECK: ret i32
define i32 @caller_store_forward() {
entry:
  %p = alloca i32
  call void @init_i32(ptr %p)
  %mode = load i32, ptr %p
  %r = call i32 @callee(i32 %mode, ptr %p)
  ret i32 %r
}

; Memset-to-load forwarding converts the zero-filled integer to a null
; pointer, making the null check take the early exit.
; CHECK-LABEL: @caller_memset_forward(
; CHECK-NOT: call void @recursive_callee
; CHECK: ret void
define void @caller_memset_forward() {
entry:
  %p = alloca ptr
  call void @init_memset(ptr %p)
  %x = load ptr, ptr %p
  call void @recursive_callee(ptr %x)
  ret void
}
