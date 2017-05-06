; RUN: llc -o - %s | FileCheck %s
target triple="arm--"

@glob = external global i32*

declare void @bar(i32*, [20000 x i8]* byval)

; CHECK-LABEL: foo:
; We should see the stack getting additional alignment
; CHECK: sub sp, sp, #16
; CHECK: bic sp, sp, #31
; And a base pointer getting used.
; CHECK: mov r6, sp
; Which is passed to the call
; CHECK: add [[REG:r[0-9]+]], r6, #19456
; CHECK: add r0, [[REG]], #536
; CHECK: bl bar
define void @foo([20000 x i8]* %addr) {
  %tmp = alloca [4 x i32], align 32
  %tmp0 = getelementptr [4 x i32], [4 x i32]* %tmp, i32 0, i32 0
  call void @bar(i32* %tmp0, [20000 x i8]* byval %addr)
  ret void
}

