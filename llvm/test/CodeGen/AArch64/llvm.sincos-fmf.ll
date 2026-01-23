; RUN: llc -mtriple=aarch64-gnu-linux -debug-only=isel -o /dev/null < %s 2>&1 | FileCheck %s

; REQUIRES: asserts

define { float, float } @test_sincos_f32_afn(float %a) {
; CHECK-LABEL: Initial selection DAG: %bb.0 'test_sincos_f32_afn:'
; CHECK-NEXT:  SelectionDAG has 9 nodes:
; CHECK-NEXT:    t0: ch,glue = EntryToken
; CHECK-NEXT:      t2: f32,ch = CopyFromReg t0, Register:f32 %0
; CHECK-NEXT:    t3: f32,f32 = fsincos afn t2
; CHECK-NEXT:    t5: ch,glue = CopyToReg t0, Register:f32 $s0, t3
; CHECK-NEXT:    t7: ch,glue = CopyToReg t5, Register:f32 $s1, t3:1, t5:1
; CHECK-NEXT:    t8: ch = AArch64ISD::RET_GLUE t7, Register:f32 $s0, Register:f32 $s1, t7:1
  %result = call afn { float, float } @llvm.sincos.f32(float %a)
  ret { float, float } %result
}
