; RUN: llc -mtriple=arm64-linux-gnu -o - %s | FileCheck %s
; RUN: llc -mtriple=arm64-linux-gnu -filetype=obj -o - %s | llvm-objdump --triple=arm64-linux-gnu - -r | FileCheck %s --check-prefix=CHECK-OBJ

declare void @callee()

define void @caller() {
  call void @callee()
  ret void
; CHECK-LABEL: caller:
; CHECK:     bl callee
; CHECK-OBJ: R_AARCH64_CALL26 callee
}

define void @tail_caller() {
  tail call void @callee()
  ret void
; CHECK-LABEL: tail_caller:
; CHECK:     b callee
; CHECK-OBJ: R_AARCH64_JUMP26 callee
}
