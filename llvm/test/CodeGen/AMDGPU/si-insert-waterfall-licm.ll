; RUN: opt -mtriple=amdgcn -mcpu=gfx1100 -amdgpu-codegenprepare -S -o - < %s | FileCheck %s

; Check that llvm.licm.disable is removed for loop forming divergent pointer.
; CHECK-NOT: !{!"llvm.licm.disable"}

define amdgpu_ps <4 x float> @needs_licm(i32 inreg %userdata, i32 %index, float %s, <4 x i32> inreg %samp, i32 inreg %a, i32 inreg %b) {
entry:
  %pc.0 = call i64 @llvm.amdgcn.s.getpc()
  %pc.1 = and i64 %pc.0, -4294967296
  %lo = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %hi = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %lo)
  %cc = icmp uge i32 %hi, 16
  br i1 %cc, label %alt.entry, label %preheader

alt.entry:
  br label %preheader

preheader:
  %limit = phi i32 [ %a, %entry ], [ %b, %alt.entry ]
  br label %loop.body

loop.body:
  %i.0 = phi i32 [ 0, %preheader ], [ %inc, %loop.body ]
  %inc = add nsw i32 %i.0, 1
  %cmp = icmp slt i32 %inc, %limit
  %ptr.0 = zext i32 %userdata to i64
  %ptr.1 = or disjoint i64 %pc.1, %ptr.0
  %loop.ptr = inttoptr i64 %ptr.1 to ptr addrspace(4)
  br i1 %cmp, label %loop.body, label %loop.exit, !llvm.loop !1

loop.exit:
  %wf_token = call i32 @llvm.amdgcn.waterfall.begin.i32(i32 0, i32 %index)
  %s_idx = call i32 @llvm.amdgcn.waterfall.readfirstlane.i32.i32(i32 %wf_token, i32 %index)
  %ptr = getelementptr <8 x i32>, <8 x i32> addrspace(4)* %loop.ptr, i32 %s_idx
  %rsrc = load <8 x i32>, <8 x i32> addrspace(4)* %ptr, align 32
  %r = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float %s, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  %r1 = call <4 x float> @llvm.amdgcn.waterfall.end.v4f32(i32 %wf_token, <4 x float> %r)

  ret <4 x float> %r1
}

!1 = !{!1, !2}
!2 = !{!"llvm.licm.disable"}
