; RUN: llc -mtriple=aarch64_lfi -mattr=+pauth -relocation-model=pic -filetype=obj -o - %s | llvm-objdump -d --no-show-raw-insn - | FileCheck %s

declare void @callee()

@var = thread_local global i32 0

define void @ret_lr() {
; CHECK-LABEL: <ret_lr>:
; CHECK:         ldr x30, [sp], #0x10
; CHECK-NEXT:    add x30, x27, w30, uxtw
; CHECK-NEXT:    ret
  call void @callee()
  ret void
}

define void @direct_tail() {
; CHECK-LABEL: <direct_tail>:
; CHECK:         ldr x30, [sp], #0x10
; CHECK-NEXT:    add x30, x27, w30, uxtw
; CHECK-NEXT:    b
  call void @callee()
  tail call void @callee()
  ret void
}

define void @indirect_tail(ptr %f) {
; CHECK-LABEL: <indirect_tail>:
; CHECK:         ldr x30, [sp], #0x10
; CHECK-NEXT:    add x30, x27, w30, uxtw
; CHECK-NEXT:    add x28, x27, w0, uxtw
; CHECK-NEXT:    br x28
  call void @callee()
  tail call void %f()
  ret void
}

define void @auth_indirect_tail(ptr %f) {
; CHECK-LABEL: <auth_indirect_tail>:
; CHECK:         ldr x30, [sp], #0x10
; CHECK-NEXT:    mov x16, #0x2a
; CHECK-NEXT:    add x30, x27, w30, uxtw
; CHECK-NEXT:    autia x0, x16
; CHECK-NEXT:    add x28, x27, w0, uxtw
; CHECK-NEXT:    br x28
  call void @callee()
  tail call void %f() [ "ptrauth"(i32 0, i64 42) ]
  ret void
}

define i32 @tls_pic() {
; CHECK-LABEL: <tls_pic>:
; CHECK:         adrp x0,
; CHECK-NEXT:    add x28, x27, w0, uxtw
; CHECK-NEXT:    ldr x1, [x28]
; CHECK-NEXT:    add x0, x0, #0x0
; CHECK-NEXT:    add x28, x27, w1, uxtw
; CHECK-NEXT:    blr x28
; CHECK:         ldr x30, [sp], #0x10
; CHECK-NEXT:    add x30, x27, w30, uxtw
; CHECK-NEXT:    ret
  %p = call ptr @llvm.threadlocal.address.p0(ptr @var)
  %v = load i32, ptr %p
  ret i32 %v
}
