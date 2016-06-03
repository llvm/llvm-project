; RUN: llc -march=amdgcn -mcpu=kaveri  < %s | FileCheck %s

; CHECK-LABEL: {{^}}test:
; CHECK s_and_saveexec_b64
; CHECK s_xor_b64
; CHECK s_or_b64 exec, exec
; CHECK s_andn2_b64 exec, exec
; CHECK s_cbranch_execnz
define spir_kernel void @test(i32 %arg, i32 %arg1, i32 addrspace(1)* nocapture %arg2, i32 %arg3, i32 %arg4, i32 %arg5, i32 %arg6) {
bb:
  %tmp = icmp ne i32 %arg, 0
  %tmp7 = icmp ne i32 %arg1, 0
  %tmp8 = and i1 %tmp, %tmp7
  br i1 %tmp8, label %bb9, label %bb11

bb9:                                              ; preds = %bb
  br label %bb10

bb10:                                             ; preds = %bb10, %bb9
  br label %bb10

bb11:                                             ; preds = %bb
  ret void
}
