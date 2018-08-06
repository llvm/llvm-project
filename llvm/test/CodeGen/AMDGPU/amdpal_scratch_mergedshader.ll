; RUN: llc -mtriple=amdgcn--amdpal -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=GFX9 -enable-var-scope %s

; On gfx9 and later, a HS is a merged shader, in which s0-s7 are reserved by the
; hardware, so the PAL puts the GIT (global information table) in s8 rather
; than s0.

; GCN-LABEL: {{^}}_amdgpu_hs_main:
; GCN: s_getpc_b64 s{{\[}}[[GITPTR:[0-9]+]]:
; PREGFX9: s_mov_b32 s[[GITPTR]], s0
; GFX9: s_mov_b32 s[[GITPTR]], s8

define amdgpu_hs void @_amdgpu_hs_main(i32 inreg %arg, i32 inreg %arg1, i32 inreg %arg2, i32 inreg %arg3, i32 inreg %arg4, i32 inreg %arg5, i32 inreg %arg6, i32 inreg %arg7, <6 x i32> inreg %arg8) {
.entry:
  %__llpc_global_proxy_7.i = alloca [3 x <4 x float>], align 16, addrspace(5)
  %tmp = icmp ult i32 undef, undef
  br i1 %tmp, label %.beginls, label %.endls

.beginls:                                         ; preds = %.entry
  %tmp15 = extractelement <6 x i32> %arg8, i32 3
  %.0.vec.insert.i = insertelement <2 x i32> undef, i32 %tmp15, i32 0
  %.4.vec.insert.i = shufflevector <2 x i32> %.0.vec.insert.i, <2 x i32> undef, <2 x i32> <i32 0, i32 3>
  %tmp16 = bitcast <2 x i32> %.4.vec.insert.i to i64
  br label %.endls

.endls:                                           ; preds = %.beginls, %.entry
  %.fca.2.gep120.i = getelementptr inbounds [3 x <4 x float>], [3 x <4 x float>] addrspace(5)* %__llpc_global_proxy_7.i, i64 0, i64 2
  store volatile <4 x float> <float 9.000000e+00, float 1.000000e+01, float 1.100000e+01, float 1.200000e+01>, <4 x float> addrspace(5)* %.fca.2.gep120.i, align 16
  br label %bb

bb:                                               ; preds = %bb, %.endls
  %lsr.iv182 = phi [3 x <4 x float>] addrspace(5)* [ undef, %bb ], [ %__llpc_global_proxy_7.i, %.endls ]
  %scevgep183 = getelementptr [3 x <4 x float>], [3 x <4 x float>] addrspace(5)* %lsr.iv182, i32 0, i32 1
  br label %bb
}
