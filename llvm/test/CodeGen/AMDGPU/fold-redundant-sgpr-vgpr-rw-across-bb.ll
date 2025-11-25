; RUN: llc -mtriple=amdgcn -mcpu=gfx942 < %s | FileCheck --check-prefix=CHECK %s
; XFAIL: *
define protected amdgpu_kernel void @main(ptr addrspace(1) noundef %args.coerce, ptr addrspace(1) noundef %args.coerce2, ptr addrspace(1) noundef %args.coerce4, i32 noundef %args12) {
; CHECK-LABEL: main:
; check that non-redundant readfirstlanes are not removed
; CHECK:      v_readfirstlane_b32
; check that all redundant readfirstlanes are removed
; CHECK-NOT:  v_readfirstlane_b32
; CHECK:      s_endpgm

entry:
    %wid = tail call noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x()
    %div1 = lshr i32 %wid, 6
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
    %mbl = tail call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
    %mbh = tail call noundef i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %mbl)
    %lshr2 = lshr i32 %mbh, 6
    %add8 = add i32 %lshr1, %lshr2
    %sub3 = shl i32 %rfl1, 8
    %mul2 = and i32 %sub3, 768
    %add1 = or disjoint i32 %mbh, %mul2
    %add3 = add nsw i32 %add1, %add8
    %sext1 = add i64 4294967296, 4611686014132420608
    %conv1 = lshr exact i64 64, 32
    %add4 = add nuw nsw i64 %conv1, 1
    %zext2 = zext i32 1 to i64
    %tmp.sroa = add nuw nsw i64 %zext2, 4294967295
    %sub5 = add i64 %tmp.sroa, 4294967296
    %sext2 = mul i64 %sub5, 4294967296
    %conv2 = lshr exact i64 %sext2, 32
    %add5 = add nuw nsw i64 %add4, %conv2
    %conv3 = trunc i64 %add5 to i32
    %mul4 = shl i32 %conv3, 2
    %bc1 = bitcast i64 %pti3 to <2 x i32>
    %ee1 = extractelement <2 x i32> %bc1, i64 0
    %ee2 = extractelement <2 x i32> %bc1, i64 1
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
    %buffload1 = tail call contract noundef <4 x float> @llvm.amdgcn.raw.buffer.load.v4f32(<4 x i32> noundef %ie2, i32 noundef %mul5, i32 noundef 0, i32 noundef 0) #6
    %add6 = add nsw i32 %tmp1, 1
    %buffload3 = tail call contract noundef <4 x float> @llvm.amdgcn.raw.buffer.load.v4f32(<4 x i32> noundef %ie3, i32 noundef %mul5, i32 noundef 0, i32 noundef 0) #6
    %vec_add1 = fadd contract <4 x float> %buffload1, %buffload3
    tail call void @llvm.amdgcn.raw.buffer.store.v4f32(<4 x float> noundef %vec_add1, <4 x i32> noundef %ie3, i32 noundef %mul5, i32 noundef 0, i32 noundef 0) #6
    %cnt = add nsw i32 %tmp1, 1024
    %inc.i = add nuw nsw i32 %loopi, 1
    %exitcond.not.i = icmp eq i32 %inc.i, %rfl2
    br i1 %exitcond.not.i, label %add.exit, label %for.body.i

    add.exit: ; preds = %for.body.i, %entry
    ret void
}
