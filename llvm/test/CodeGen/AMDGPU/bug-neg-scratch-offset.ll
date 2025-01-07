; RUN: llc < %s -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1200 -stop-after=amdgpu-isel -verify-machineinstrs | FileCheck %s

target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

%union.anon.41 = type { [4 x i64] }
%union.anon.2 = type { i8 }

define fastcc void @_ZN10PrimitivesI12rccl_bfloat810FuncMinMaxIS0_E13FanAsymmetricILi1ELi1EELi1E10ProtoLL128Li0EE9localCopyEPS0_S7_i(i32 %0, i64 %idx.ext62.i.i) {
entry:
  %1 = alloca %union.anon.41, i32 0, align 8, addrspace(5)
  %add.ptr63.i.i3 = getelementptr %union.anon.2, ptr null, i64 %idx.ext62.i.i
  br label %for.body69.i.i.epil3

for.body69.i.i.epil3:                             ; preds = %for.body69.i.i.epil3, %entry
  %i.0117.i.i.epil4 = phi i32 [ %inc.i.i.7.epil, %for.body69.i.i.epil3 ], [ %0, %entry ]
  %conv65.i.i.epil5 = zext i32 %i.0117.i.i.epil4 to i64
  %arrayidx73.i.i.epil6 = getelementptr [32 x i8], ptr addrspace(5) %1, i32 0, i32 %i.0117.i.i.epil4
  %add.ptr75.i.i.epil7 = getelementptr i8, ptr %add.ptr63.i.i3, i64 %conv65.i.i.epil5
  %2 = load <4 x i8>, ptr addrspace(5) %arrayidx73.i.i.epil6, align 8
  store <4 x i8> %2, ptr %add.ptr75.i.i.epil7, align 1
  %inc.i.i.3.epil = or disjoint i32 %i.0117.i.i.epil4, 1
  %conv65.i.i.4.epil = zext i32 %inc.i.i.3.epil to i64
  %arrayidx73.i.i.4.epil = getelementptr [32 x i8], ptr addrspace(5) %1, i32 0, i32 %inc.i.i.3.epil
  %add.ptr75.i.i.4.epil = getelementptr i8, ptr %add.ptr63.i.i3, i64 %conv65.i.i.4.epil
  %3 = load <4 x i8>, ptr addrspace(5) %arrayidx73.i.i.4.epil, align 4
  store <4 x i8> %3, ptr %add.ptr75.i.i.4.epil, align 1
  %inc.i.i.7.epil = add nuw i32 %i.0117.i.i.epil4, 1
  br label %for.body69.i.i.epil3

for.body69.i.i.epil3.for.cond.cleanup68.loopexit.i.i.unr-lcssa_crit_edge: ; No predecessors!
  %conv65.i.i.epil = zext i32 %inc.i.i.7.epil to i64
  ret void
}

; CHECK: SCRATCH_LOAD_DWORD_SVS %{{[0-9]+}}, %{{[0-9]+}}, -1
