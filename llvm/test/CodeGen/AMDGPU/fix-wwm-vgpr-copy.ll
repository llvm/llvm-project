; RUN: llc -mtriple=amdgcn--amdpal -march=amdgcn -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; NOTE: llvm.amdgcn.wwm is deprecated, use llvm.amdgcn.strict.wwm instead.

; GCN-LABEL: wwm:
define amdgpu_hs void @wwm(i32 inreg %arg, <4 x i32> inreg %buffer) {
entry:
  br label %work

bb42:
  br label %bb602

bb602:
  %tmp603 = phi i32 [ 0, %bb42 ], [ 1, %work ]
  %tmp607 = icmp eq i32 %tmp603, %tmp1196
  br i1 %tmp607, label %bb49, label %bb54

bb49:
  call void @llvm.amdgcn.raw.tbuffer.store.f32(float 1.0, <4 x i32> %buffer, i32 4, i32 1, i32 116, i32 1)
  ret void

bb54:
  ret void

work:
; GCN: s_not_b64 exec, exec
; GCN: v_mov_b32_e32 v[[tmp1189:[0-9]+]], 1
; GCN: s_not_b64 exec, exec
  %tmp1189 = tail call i32 @llvm.amdgcn.set.inactive.i32(i32 4, i32 1)

; GCN: s_or_saveexec_b64 s{{\[}}[[LO:[0-9]+]]:[[HI:[0-9]+]]{{\]}}, -1
; GCN: v_lshlrev_b32_e32 v[[tmp1191:[0-9]+]], 2, v[[tmp1189]]
  %tmp1191 = mul i32 %tmp1189, 4

; GCN: s_mov_b64 exec, s{{\[}}[[LO]]:[[HI]]{{\]}}
  %tmp1196 = tail call i32 @llvm.amdgcn.wwm.i32(i32 %tmp1191)

  %tmp34 = icmp eq i32 %arg, 0
  br i1 %tmp34, label %bb602, label %bb42
}

; GCN-LABEL: strict_wwm:
define amdgpu_hs void @strict_wwm(i32 inreg %arg, <4 x i32> inreg %buffer) {
entry:
  br label %work

bb42:
  br label %bb602

bb602:
  %tmp603 = phi i32 [ 0, %bb42 ], [ 1, %work ]
  %tmp607 = icmp eq i32 %tmp603, %tmp1196
  br i1 %tmp607, label %bb49, label %bb54

bb49:
  call void @llvm.amdgcn.raw.tbuffer.store.f32(float 1.0, <4 x i32> %buffer, i32 4, i32 1, i32 116, i32 1)
  ret void

bb54:
  ret void

work:
; GCN: s_not_b64 exec, exec
; GCN: v_mov_b32_e32 v[[tmp1189:[0-9]+]], 1
; GCN: s_not_b64 exec, exec
  %tmp1189 = tail call i32 @llvm.amdgcn.set.inactive.i32(i32 4, i32 1)

; GCN: s_or_saveexec_b64 s{{\[}}[[LO:[0-9]+]]:[[HI:[0-9]+]]{{\]}}, -1
; GCN: v_lshlrev_b32_e32 v[[tmp1191:[0-9]+]], 2, v[[tmp1189]]
  %tmp1191 = mul i32 %tmp1189, 4

; GCN: s_mov_b64 exec, s{{\[}}[[LO]]:[[HI]]{{\]}}
  %tmp1196 = tail call i32 @llvm.amdgcn.strict.wwm.i32(i32 %tmp1191)

  %tmp34 = icmp eq i32 %arg, 0
  br i1 %tmp34, label %bb602, label %bb42
}

declare i32 @llvm.amdgcn.set.inactive.i32(i32, i32) #0
declare i32 @llvm.amdgcn.wwm.i32(i32) #1
declare i32 @llvm.amdgcn.strict.wwm.i32(i32) #1
declare void @llvm.amdgcn.raw.tbuffer.store.f32(float, <4 x i32>, i32, i32, i32 immarg, i32 immarg) #2

attributes #0 = { convergent nounwind readnone willreturn }
attributes #1 = { convergent nounwind readnone speculatable willreturn }
attributes #2 = { nounwind willreturn writeonly }
