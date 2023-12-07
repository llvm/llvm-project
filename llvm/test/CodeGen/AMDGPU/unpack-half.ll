; RUN: llc -march=amdgcn -mcpu=gfx600 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -march=amdgcn -mcpu=gfx700 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -march=amdgcn -mcpu=gfx1100 -verify-machineinstrs < %s | FileCheck %s

; On gfx6 and gfx7, this test shows a bug in SelectionDAG where scalarizing the
; extension of a vector of f16 generates an illegal node that errors later.

; CHECK-LABEL: {{^}}main:
; CHECK: v_cvt_f32_f16

define amdgpu_gs void @main(i32 inreg %arg) local_unnamed_addr #0 {
.entry:
  %tmp = load volatile float, ptr addrspace(1) undef
  %tmp1 = bitcast float %tmp to i32
  %im0.i = lshr i32 %tmp1, 16
  %tmp2 = insertelement <2 x i32> undef, i32 %im0.i, i32 1
  %tmp3 = trunc <2 x i32> %tmp2 to <2 x i16>
  %tmp4 = bitcast <2 x i16> %tmp3 to <2 x half>
  %tmp5 = fpext <2 x half> %tmp4 to <2 x float>
  %bc = bitcast <2 x float> %tmp5 to <2 x i32>
  %tmp6 = extractelement <2 x i32> %bc, i32 1
  store volatile i32 %tmp6, ptr addrspace(1) undef
  ret void
}

attributes #0 = { nounwind }
