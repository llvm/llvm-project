; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-unknown %s -o - -filetype=obj | spirv-val %}


; CHECK-LABEL: define void @test_dxil_lowering
; CHECK: call void @dx.op.discard(i32 82, i1 %0)
;
define spir_func void @test_dxil_lowering(float noundef %Buf) #0 {
entry:
  %Buf.addr = alloca float, align 4
  store float %Buf, ptr %Buf.addr, align 4
  %1 = load float, ptr %Buf.addr, align 4
  %2 = fcmp olt float %1, 0.000000e+00
  call void @llvm.spv.clip(i1 %2)
  ret void
}

declare void @llvm.spv.clip(i1) #1
