; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu | FileCheck %s --check-prefix=LIN
; RUN: llc < %s -mtriple=x86_64-pc-windows-msvc | FileCheck %s --check-prefix=WIN

; The callee address computation should get folded into the call.
; CHECK-LABEL: f:
; CHECK-NOT: mov
; LIN: jmpq *(%rdi,%rsi,8)
; WIN: rex64 jmpq *(%rcx,%rdx,8)
define void @f(ptr %table, i64 %idx, i64 %aux1, i64 %aux2, i64 %aux3) {
entry:
  %arrayidx = getelementptr inbounds ptr, ptr %table, i64 %idx
  %funcptr = load ptr, ptr %arrayidx, align 8
  tail call void %funcptr(ptr %table, i64 %idx, i64 %aux1, i64 %aux2, i64 %aux3)
  ret void
}

; Check that we don't assert here. On Win64 this has a TokenFactor with
; multiple uses, which we can't currently fold.
define void @thunk(ptr %this, ...) {
entry:
  %vtable = load ptr, ptr %this, align 8
  %vfn = getelementptr inbounds nuw i8, ptr %vtable, i64 8
  %0 = load ptr, ptr %vfn, align 8
  musttail call void (ptr, ...) %0(ptr %this, ...)
  ret void
}
