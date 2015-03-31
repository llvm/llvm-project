;RUN: llc < %s -march=amdgcn -mcpu=verde -verify-machineinstrs | FileCheck %s
;RUN: llc < %s -march=amdgcn -mcpu=tonga -verify-machineinstrs | FileCheck %s

; CHECK-LABEL: {{^}}main:
; CHECK: s_sendmsg Gs(emit stream 0)
; CHECK: s_sendmsg Gs(cut stream 1)
; CHECK: s_sendmsg Gs(emit-cut stream 2)
; CHECK: s_sendmsg Gs_done(nop)

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
