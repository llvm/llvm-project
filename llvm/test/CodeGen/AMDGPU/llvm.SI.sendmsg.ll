;RUN: llc < %s -march=amdgcn -mcpu=verde -verify-machineinstrs | FileCheck %s
;RUN: llc < %s -march=amdgcn -mcpu=tonga -verify-machineinstrs | FileCheck %s

; CHECK-LABEL: {{^}}main:
; CHECK: s_mov_b32 m0, 0
; CHECK-NOT: s_mov_b32 m0
; CHECK: s_sendmsg sendmsg(MSG_GS, GS_OP_EMIT, 0)
; CHECK: s_sendmsg sendmsg(MSG_GS, GS_OP_CUT, 1)
; CHECK: s_sendmsg sendmsg(MSG_GS, GS_OP_EMIT_CUT, 2)
; CHECK: s_sendmsg sendmsg(MSG_GS_DONE, GS_OP_NOP)

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
