; RUN: opt < %s -globalopt -S | FileCheck %s

; Check that fastccc is not set on
; function "bar" because its address is taken
; when no optional parameters are passed to
; hasAddressTaken() as it returns "true"
; If the optional parameter corresponding to
; ignore llvmassumelike is passed to
; hasAddressTaken() then this test would fail.

declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture)

; CHECK-NOT: define internal fastcc void @bar() {
; CHECK: define internal void @bar() {
define internal void @bar() {
entry:
  ret void
}

; CHECK: define void @foo() local_unnamed_addr {
define void @foo() {
entry:
  %c = bitcast void()* @bar to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %c)
  ret void
}

