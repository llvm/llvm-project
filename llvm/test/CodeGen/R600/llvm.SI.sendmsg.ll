;RUN: llc < %s -march=r600 -mcpu=verde -verify-machineinstrs | FileCheck %s

; CHECK-LABEL: @main
; CHECK: S_SENDMSG 34
; CHECK: S_SENDMSG 274
; CHECK: S_SENDMSG 562
; CHECK: S_SENDMSG 3

define void @main() {
main_body:
  call void @llvm.SI.sendmsg(i32 34, i32 0);
  call void @llvm.SI.sendmsg(i32 274, i32 0);
  call void @llvm.SI.sendmsg(i32 562, i32 0);
  call void @llvm.SI.sendmsg(i32 3, i32 0);
  ret void
}

; Function Attrs: nounwind
declare void @llvm.SI.sendmsg(i32, i32) #0

attributes #0 = { nounwind }
