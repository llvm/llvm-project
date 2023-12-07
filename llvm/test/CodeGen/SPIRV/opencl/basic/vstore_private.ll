; RUN: llc -O0 -opaque-pointers=0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s

; CHECK: %[[#i16_ty:]] = OpTypeInt 16 0
; CHECK: %[[#v4xi16_ty:]] = OpTypeVector %[[#i16_ty]] 4
; CHECK: %[[#pv4xi16_ty:]] = OpTypePointer Function %[[#v4xi16_ty]]
; CHECK: %[[#i16_const0:]] = OpConstant %[[#i16_ty]] 0
; CHECK: %[[#i16_undef:]] = OpUndef %[[#i16_ty]]
; CHECK: %[[#comp_const:]] = OpConstantComposite %[[#v4xi16_ty]] %[[#i16_const0]] %[[#i16_const0]] %[[#i16_const0]] %[[#i16_undef]]

; CHECK: %[[#r:]] = OpInBoundsPtrAccessChain
; CHECK: %[[#r2:]] = OpBitcast %[[#pv4xi16_ty]] %[[#r]]
; CHECK: OpStore %[[#r2]] %[[#comp_const]] Aligned 8

define spir_kernel void @test_fn(i16 addrspace(1)* %srcValues, i32 addrspace(1)* %offsets, <3 x i16> addrspace(1)* %destBuffer, i32 %alignmentOffset) {
entry:
  %sPrivateStorage = alloca [42 x <3 x i16>], align 8
  %0 = bitcast [42 x <3 x i16>]* %sPrivateStorage to i8*
  %1 = bitcast i8* %0 to i8*
  call void @llvm.lifetime.start.p0i8(i64 336, i8* %1)
  %2 = call spir_func <3 x i64> @BuiltInGlobalInvocationId()
  %call = extractelement <3 x i64> %2, i32 0
  %conv = trunc i64 %call to i32
  %idxprom = sext i32 %conv to i64
  %arrayidx = getelementptr inbounds [42 x <3 x i16>], [42 x <3 x i16>]* %sPrivateStorage, i64 0, i64 %idxprom
  %storetmp = bitcast <3 x i16>* %arrayidx to <4 x i16>*
  store <4 x i16> <i16 0, i16 0, i16 0, i16 undef>, <4 x i16>* %storetmp, align 8
  %conv1 = sext i32 %conv to i64
  %call2 = call spir_func <3 x i16> @OpenCL_vload3_i64_p1i16_i32(i64 %conv1, i16 addrspace(1)* %srcValues, i32 3)
  %idxprom3 = sext i32 %conv to i64
  %arrayidx4 = getelementptr inbounds i32, i32 addrspace(1)* %offsets, i64 %idxprom3
  %3 = load i32, i32 addrspace(1)* %arrayidx4, align 4
  %conv5 = zext i32 %3 to i64
  %arraydecay = getelementptr inbounds [42 x <3 x i16>], [42 x <3 x i16>]* %sPrivateStorage, i64 0, i64 0
  %4 = bitcast <3 x i16>* %arraydecay to i16*
  %idx.ext = zext i32 %alignmentOffset to i64
  %add.ptr = getelementptr inbounds i16, i16* %4, i64 %idx.ext
  call spir_func void @OpenCL_vstore3_v3i16_i64_p0i16(<3 x i16> %call2, i64 %conv5, i16* %add.ptr)
  %arraydecay6 = getelementptr inbounds [42 x <3 x i16>], [42 x <3 x i16>]* %sPrivateStorage, i64 0, i64 0
  %5 = bitcast <3 x i16>* %arraydecay6 to i16*
  %idxprom7 = sext i32 %conv to i64
  %arrayidx8 = getelementptr inbounds i32, i32 addrspace(1)* %offsets, i64 %idxprom7
  %6 = load i32, i32 addrspace(1)* %arrayidx8, align 4
  %mul = mul i32 3, %6
  %idx.ext9 = zext i32 %mul to i64
  %add.ptr10 = getelementptr inbounds i16, i16* %5, i64 %idx.ext9
  %idx.ext11 = zext i32 %alignmentOffset to i64
  %add.ptr12 = getelementptr inbounds i16, i16* %add.ptr10, i64 %idx.ext11
  %7 = bitcast <3 x i16> addrspace(1)* %destBuffer to i16 addrspace(1)*
  %idxprom13 = sext i32 %conv to i64
  %arrayidx14 = getelementptr inbounds i32, i32 addrspace(1)* %offsets, i64 %idxprom13
  %8 = load i32, i32 addrspace(1)* %arrayidx14, align 4
  %mul15 = mul i32 3, %8
  %idx.ext16 = zext i32 %mul15 to i64
  %add.ptr17 = getelementptr inbounds i16, i16 addrspace(1)* %7, i64 %idx.ext16
  %idx.ext18 = zext i32 %alignmentOffset to i64
  %add.ptr19 = getelementptr inbounds i16, i16 addrspace(1)* %add.ptr17, i64 %idx.ext18
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %cmp = icmp ult i32 %i.0, 3
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %idxprom21 = zext i32 %i.0 to i64
  %arrayidx22 = getelementptr inbounds i16, i16* %add.ptr12, i64 %idxprom21
  %9 = load i16, i16* %arrayidx22, align 2
  %idxprom23 = zext i32 %i.0 to i64
  %arrayidx24 = getelementptr inbounds i16, i16 addrspace(1)* %add.ptr19, i64 %idxprom23
  store i16 %9, i16 addrspace(1)* %arrayidx24, align 2
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %inc = add i32 %i.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  %10 = bitcast [42 x <3 x i16>]* %sPrivateStorage to i8*
  %11 = bitcast i8* %10 to i8*
  call void @llvm.lifetime.end.p0i8(i64 336, i8* %11)
  ret void
}

declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture)

declare spir_func <3 x i16> @OpenCL_vload3_i64_p1i16_i32(i64, i16 addrspace(1)*, i32)

declare spir_func void @OpenCL_vstore3_v3i16_i64_p0i16(<3 x i16>, i64, i16*)

declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture)

declare spir_func <3 x i64> @BuiltInGlobalInvocationId()
