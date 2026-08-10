; RUN: llvm-as < %s | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s
; Basic smoke test for token type.

; CHECK: declare void @llvm.token.foobar(token)
declare void @llvm.token.foobar(token)

define void @f() {
  call void @llvm.token.foobar(token none)
  ret void
}
