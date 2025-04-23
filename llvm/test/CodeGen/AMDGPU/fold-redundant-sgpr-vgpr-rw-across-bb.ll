; NOTE: Do not autogenerate
; RUN: llc -mtriple=amdgcn -mcpu=gfx942 -verify-machineinstrs < %s | FileCheck --check-prefix=GCN %s

; ModuleID = '<stdin>'
source_filename = "add.cpp"
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

$main = comdat any

; Function Attrs: convergent mustprogress nofree norecurse nounwind
define protected amdgpu_kernel void @main(ptr addrspace(1) noundef %args.coerce, ptr addrspace(1) noundef %args.coerce2, ptr addrspace(1) noundef %args.coerce4, i32 noundef %args10, i32 noundef %args12) local_unnamed_addr #0 comdat {
; GCN-LABEL: main:
; check that non-redundant readfirstlanes are not removed
; GCN:      v_readfirstlane_b32
; check that all redundant readfirstlanes are removed
; GCN-NOT:  v_readfirstlane_b32
; GCN:      s_endpgm
entry:
    %0 = tail call noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x()
    %div1 = lshr i32 %0, 6
    %rfl1 = tail call noundef i32 @llvm.amdgcn.readfirstlane.i32(i32 %div1)
    %sub1 = add nsw i32 %args12, 1023
    %div2 = sdiv i32 %sub1, 1024
    %rfl2 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %div2)
    %cmp24.i = icmp sgt i32 %rfl2, 0
    br i1 %cmp24.i, label %for.body.lr.ph.i, label %add.exit

for.body.lr.ph.i:                                 ; preds = %entry
    %pti1 = ptrtoint ptr addrspace(1) %args.coerce4 to i64
    %pti2 = ptrtoint ptr addrspace(1) %args.coerce2 to i64
    %pti3 = ptrtoint ptr addrspace(1) %args.coerce to i64
    %lshr1 = lshr i32 %rfl1, 2
    %wid1 = tail call noundef i32 @llvm.amdgcn.workgroup.id.x()
    %add7 = add i32 %lshr1, %wid1
    %mbl = tail call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
    %mbh = tail call noundef i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %mbl)
    %lshr2 = lshr i32 %mbh, 6
    %add8 = add i32 %add7, %lshr2
    %sub2 = shl i32 %mbh, 2
    %mul1 = and i32 %sub2, 252
    %sub3 = shl i32 %rfl1, 8
    %mul2 = and i32 %sub3, 768
    %add1 = or disjoint i32 %mul1, %mul2
    %add2 = shl i32 %args12, 1
    %mul3 = mul i32 %add2, %add8
    %add3 = add nsw i32 %add1, %mul3
    %zext1 = zext i32 %args12 to i64
    %sub4 = shl nuw i64 %zext1, 32
    %sext1 = add i64 %sub4, 4611686014132420608
    %conv1 = lshr exact i64 %sext1, 32
    %add4 = add nuw nsw i64 %conv1, 1
    %zext2 = zext i32 %args10 to i64
    %tmp.sroa = add nuw nsw i64 %zext2, 4294967295
    %sub5 = add i64 %tmp.sroa, %sub4
    %sext2 = mul i64 %sub5, %sub4
    %conv2 = lshr exact i64 %sext2, 32
    %add5 = add nuw nsw i64 %add4, %conv2
    %conv3 = trunc i64 %add5 to i32
    %mul4 = shl i32 %conv3, 2
    %bc1 = bitcast i64 %pti3 to <2 x i32>
    %ee1 = extractelement <2 x i32> %bc1, i64 0
    %ee2 = extractelement <2 x i32> %bc1, i64 1
    %bc2 = bitcast i64 %pti2 to <2 x i32>
    %ee3 = extractelement <2 x i32> %bc2, i64 0
    %ee4 = extractelement <2 x i32> %bc2, i64 1
    %bc3 = bitcast i64 %pti1 to <2 x i32>
    %ee5 = extractelement <2 x i32> %bc3, i64 0
    %ee6 = extractelement <2 x i32> %bc3, i64 1
    br label %for.body.i

for.body.i:                                       ; preds = %for.body.i, %for.body.lr.ph.i
    %loopi = phi i32 [ 0, %for.body.lr.ph.i ], [ %inc.i, %for.body.i ]
    %tmp1 = phi i32 [ %add3, %for.body.lr.ph.i ], [ %cnt, %for.body.i ]
    %rfl3 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %ee1)
    %rfl4 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %ee2)
    %rfl5 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %mul4)
    %ie1 = insertelement <4 x i32> <i32 poison, i32 poison, i32 poison, i32 131072>, i32 %rfl3, i64 0
    %ie2 = insertelement <4 x i32> %ie1, i32 %rfl4, i64 1
    %ie3 = insertelement <4 x i32> %ie2, i32 %rfl5, i64 2
    %mul5 = shl i32 %tmp1, 2
    %buffload1 = tail call contract noundef <4 x float> @llvm.amdgcn.raw.buffer.load.v4f32(<4 x i32> noundef %ie3, i32 noundef %mul5, i32 noundef 0, i32 noundef 0) #6
    %add6 = add nsw i32 %tmp1, %args12
    %mul6 = shl i32 %add6, 2
    %buffload2 = tail call contract noundef <4 x float> @llvm.amdgcn.raw.buffer.load.v4f32(<4 x i32> noundef %ie3, i32 noundef %mul6, i32 noundef 0, i32 noundef 0) #6
    %rfl6 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %ee3)
    %rfl7 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %ee4)
    %ie4 = insertelement <4 x i32> <i32 poison, i32 poison, i32 poison, i32 131072>, i32 %rfl6, i64 0
    %ie5 = insertelement <4 x i32> %ie4, i32 %rfl7, i64 1
    %ie6 = insertelement <4 x i32> %ie5, i32 %rfl5, i64 2
    %buffload3 = tail call contract noundef <4 x float> @llvm.amdgcn.raw.buffer.load.v4f32(<4 x i32> noundef %ie6, i32 noundef %mul5, i32 noundef 0, i32 noundef 0) #6
    %buffload4 = tail call contract noundef <4 x float> @llvm.amdgcn.raw.buffer.load.v4f32(<4 x i32> noundef %ie6, i32 noundef %mul6, i32 noundef 0, i32 noundef 0) #6
    %rfl8 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %ee5)
    %rfl9 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %ee6)
    %ie7 = insertelement <4 x i32> <i32 poison, i32 poison, i32 poison, i32 131072>, i32 %rfl8, i64 0
    %ie8 = insertelement <4 x i32> %ie7, i32 %rfl9, i64 1
    %ie9 = insertelement <4 x i32> %ie8, i32 %rfl5, i64 2
    %vec_add1 = fadd contract <4 x float> %buffload1, %buffload3
    %vec_add2 = fadd contract <4 x float> %buffload2, %buffload4
    tail call void @llvm.amdgcn.raw.buffer.store.v4f32(<4 x float> noundef %vec_add1, <4 x i32> noundef %ie9, i32 noundef %mul5, i32 noundef 0, i32 noundef 0) #6
    tail call void @llvm.amdgcn.raw.buffer.store.v4f32(<4 x float> noundef %vec_add2, <4 x i32> noundef %ie9, i32 noundef %mul6, i32 noundef 0, i32 noundef 0) #6
    %cnt = add nsw i32 %tmp1, 1024
    %inc.i = add nuw nsw i32 %loopi, 1
    %exitcond.not.i = icmp eq i32 %inc.i, %rfl2
    br i1 %exitcond.not.i, label %add.exit, label %for.body.i, !llvm.loop !6

    add.exit: ; preds = %for.body.i, %entry
    ret void
}

; Function Attrs: convergent mustprogress nocallback nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.readfirstlane.i32(i32) #1
!6 = distinct !{!6, !7}
!7 = !{!"llvm.loop.mustprogress"}
