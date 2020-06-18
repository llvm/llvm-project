; RUN: llc -march=amdgcn -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -enable-var-scope %s

; creating v4i16->v4f16 and v4f16->v4i16 bitcasts in the selection DAG is rather
; difficult, so this test has to throw in some llvm.amdgcn.wqm to get them

; CHECK-LABEL: {{^}}test_to_i16:
; CHECK: s_endpgm
define amdgpu_ps void @test_to_i16(<4 x i32> inreg, <4 x half> inreg) #0 {
  %a_tmp = call <4 x half> @llvm.amdgcn.wqm.v4f16(<4 x half> %1)
  %a_i16_tmp = bitcast <4 x half> %a_tmp to <4 x i16>
  %a_i16 = call <4 x i16> @llvm.amdgcn.wqm.v4i16(<4 x i16> %a_i16_tmp)

  %a_i32 = bitcast <4 x i16> %a_i16 to <2 x i32>
  call void @llvm.amdgcn.raw.buffer.store.v2i32(<2 x i32> %a_i32, <4 x i32> %0, i32 0, i32 0, i32 0)
  ret void
}

; CHECK-LABEL: {{^}}test_to_half:
; CHECK: s_endpgm
define amdgpu_ps void @test_to_half(<4 x i32> inreg, <4 x i16> inreg) #0 {
  %a_tmp = call <4 x i16> @llvm.amdgcn.wqm.v4i16(<4 x i16> %1)
  %a_half_tmp = bitcast <4 x i16> %a_tmp to <4 x half>
  %a_half = call <4 x half> @llvm.amdgcn.wqm.v4f16(<4 x half> %a_half_tmp)

  %a_i32 = bitcast <4 x half> %a_half to <2 x i32>
  call void @llvm.amdgcn.raw.buffer.store.v2i32(<2 x i32> %a_i32, <4 x i32> %0, i32 0, i32 0, i32 0)
  ret void
}

declare <4 x half> @llvm.amdgcn.wqm.v4f16(<4 x half>) #1
declare <4 x i16> @llvm.amdgcn.wqm.v4i16(<4 x i16>) #1
declare void @llvm.amdgcn.raw.buffer.store.v2i32(<2 x i32>, <4 x i32>, i32, i32, i32) #0

attributes #0 = { nounwind }
attributes #1 = { nounwind readonly }
