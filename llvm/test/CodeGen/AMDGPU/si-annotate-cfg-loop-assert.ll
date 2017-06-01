; RUN: llc -march=amdgcn -mcpu=kaveri -verify-machineinstrs < %s | FileCheck %s

; CHECK-LABEL: {{^}}test:
; CHECK s_and_saveexec_b64
; CHECK s_xor_b64
; CHECK s_or_b64 exec, exec
; CHECK s_andn2_b64 exec, exec
; CHECK s_cbranch_execnz
define void @test(i32 %arg, i32 %arg1) {
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
