; RUN: llc -o - %s | FileCheck %s
target triple="arm--"

@glob = external global ptr

declare void @bar(ptr, ptr byval([20000 x i8]))

; CHECK-LABEL: foo:
; We should see the stack getting additional alignment
; CHECK: sub sp, sp, #16
; CHECK: bic sp, sp, #31
; And a base pointer getting used.
; CHECK: mov r6, sp
; Which is passed to the call
; CHECK: mov r0, r6
; CHECK: bl bar
define void @foo(ptr %addr) {
  %tmp = alloca [4 x i32], align 32
  call void @bar(ptr %tmp, ptr byval([20000 x i8]) %addr)
  ret void
}

