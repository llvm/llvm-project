; RUN: llc -mtriple=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; We are only checking that instruction selection can succeed in this case. This
; cut down test results in no instructions, but that's fine.
;
; See https://llvm.org/PR33743 for details of the bug being addressed
;
; Checking that shufflevector with 3-vec mask is handled in
; combineShuffleToVectorExtend
;
; GCN: s_endpgm

define amdgpu_ps void @main(i32 %in1) local_unnamed_addr {
.entry:
  br i1 undef, label %bb12, label %bb

bb:
  %__llpc_global_proxy_r5.12.vec.insert = insertelement <4 x i32> undef, i32 %in1, i32 3
  %tmp3 = shufflevector <4 x i32> %__llpc_global_proxy_r5.12.vec.insert, <4 x i32> undef, <3 x i32> <i32 undef, i32 undef, i32 1>
  %tmp4 = bitcast <3 x i32> %tmp3 to <3 x float>
  %a2.i123 = extractelement <3 x float> %tmp4, i32 2
  %tmp5 = bitcast float %a2.i123 to i32
  %__llpc_global_proxy_r2.0.vec.insert196 = insertelement <4 x i32> undef, i32 %tmp5, i32 0
  br label %bb12

bb12:
  %__llpc_global_proxy_r2.0 = phi <4 x i32> [ %__llpc_global_proxy_r2.0.vec.insert196, %bb ], [ undef, %.entry ]
  %tmp6 = shufflevector <4 x i32> %__llpc_global_proxy_r2.0, <4 x i32> undef, <3 x i32> <i32 1, i32 2, i32 3>
  %tmp7 = bitcast <3 x i32> %tmp6 to <3 x float>
  %a0.i = extractelement <3 x float> %tmp7, i32 0
  ret void
}
