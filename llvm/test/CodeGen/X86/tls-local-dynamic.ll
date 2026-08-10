; RUN: llc < %s -mtriple=x86_64-linux-gnu -relocation-model=pic | FileCheck  %s

@x = internal thread_local global i32 0, align 4
@y = internal thread_local global i32 0, align 4

; get_x and get_y are here to prevent x and y to be optimized away as 0

define ptr @get_x() {
entry:
  ret ptr @x
; FIXME: This function uses a single thread-local variable,
; so we might want to fall back to general-dynamic here.
; CHECK-LABEL:       get_x:
; CHECK:       leaq x@TLSLD(%rip), %rdi
; CHECK-NEXT:  callq __tls_get_addr@PLT
; CHECK:       x@DTPOFF
}

define ptr @get_y() {
entry:
  ret ptr @y
}

define i32 @f(i32 %i) {
entry:
  %cmp = icmp eq i32 %i, 1
  br i1 %cmp, label %return, label %if.else
; This bb does not access TLS, so should not call __tls_get_addr.
; CHECK-LABEL:       f:
; CHECK-NOT:   __tls_get_addr
; CHECK:       je


if.else:
  %0 = load i32, ptr @x, align 4
  %cmp1 = icmp eq i32 %i, 2
  br i1 %cmp1, label %if.then2, label %return
; Now we call __tls_get_addr.
; CHECK:       # %if.else
; CHECK:       leaq x@TLSLD(%rip), %rdi
; CHECK-NEXT:  callq __tls_get_addr@PLT
; CHECK:       x@DTPOFF


if.then2:
  %1 = load i32, ptr @y, align 4
  %add = add nsw i32 %1, %0
  br label %return
; This accesses TLS, but is dominated by the previous block,
; so should not have to call __tls_get_addr again.
; CHECK:       # %if.then2
; CHECK-NOT:   __tls_get_addr
; CHECK:       y@DTPOFF


return:
  %retval.0 = phi i32 [ %add, %if.then2 ], [ 5, %entry ], [ %0, %if.else ]
  ret i32 %retval.0
}
