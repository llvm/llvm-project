; RUN: not llc -O0 -mtriple=spirv64-unknown-unknown %s -o %t.spvt 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR
; RUN: llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_cooperative_matrix %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_cooperative_matrix %s -o - -filetype=obj | spirv-val %}

; CHECK-ERROR: LLVM ERROR: OpTypeCooperativeMatrixKHR type requires the following SPIR-V extension: SPV_KHR_cooperative_matrix

; CHECK: OpCapability CooperativeMatrixKHR
; CHECK: OpExtension "SPV_KHR_cooperative_matrix"

; CHECK-DAG: %[[#Int8Ty:]] = OpTypeInt 8 0
; CHECK-DAG: %[[#Int32Ty:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#Const12:]] = OpConstant %[[#Int32Ty]] 12
; CHECK-DAG: %[[#Const48:]] = OpConstant %[[#Int32Ty]] 48
; CHECK-DAG: %[[#Const0:]] = OpConstant %[[#Int32Ty]] 0
; CHECK-DAG: %[[#Const3:]] = OpConstant %[[#Int32Ty]] 3
; CHECK-DAG: %[[#Const2:]] = OpConstant %[[#Int32Ty]] 2
; CHECK-DAG: %[[#Const1:]] = OpConstant %[[#Int32Ty]] 1
; CHECK-DAG: %[[#MatTy1:]] = OpTypeCooperativeMatrixKHR %[[#Int32Ty]] %[[#Const3]] %[[#Const12]] %[[#Const12]] %[[#Const2]]
; CHECK-DAG: %[[#MatTy2:]] = OpTypeCooperativeMatrixKHR %[[#Int8Ty]] %[[#Const0]] %[[#Const12]] %[[#Const48]] %[[#Const0]]
; CHECK-DAG: %[[#MatTy3:]] = OpTypeCooperativeMatrixKHR %[[#Int8Ty]] %[[#Const2]] %[[#Const48]] %[[#Const12]] %[[#Const1]]
; CHECK: OpCompositeConstruct %[[#MatTy1]]
; CHECK: %[[#Load1:]] = OpCooperativeMatrixLoadKHR %[[#MatTy2]]
; CHECK: OpCooperativeMatrixLengthKHR %[[#Int32Ty]] %[[#Load1]]
; CHECK: OpCooperativeMatrixLoadKHR %[[#MatTy3]]
; CHECK: OpCooperativeMatrixMulAddKHR %[[#MatTy1]]
; CHECK: OpCooperativeMatrixStoreKHR

target triple = "spir64-unknown-unknown"

%"class.sycl::_V1::range" = type { %"class.sycl::_V1::detail::array" }
%"class.sycl::_V1::detail::array" = type { [2 x i64] }
%"class.sycl::_V1::id" = type { %"class.sycl::_V1::detail::array" }

@__spirv_BuiltInGlobalInvocationId = external dso_local local_unnamed_addr addrspace(1) constant <3 x i64>, align 32
@__spirv_BuiltInLocalInvocationId = external dso_local local_unnamed_addr addrspace(1) constant <3 x i64>, align 32

define weak_odr dso_local spir_kernel void @matr_mult(ptr addrspace(1) align 1 %_arg_accA, ptr addrspace(1) align 1 %_arg_accB, ptr byval(%"class.sycl::_V1::range") align 8 %_arg_accB5, ptr byval(%"class.sycl::_V1::id") align 8 %_arg_accB6, ptr addrspace(1) align 4 %_arg_accC, i64 %_arg_N, i64 %_arg_K) {
entry:
  %sub_c.sroa.0.i = alloca target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2), align 8
  %ref.tmp29.sroa.0.i = alloca target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2), align 8
  %agg.tmp15.sroa.0.sroa.2.0..sroa_idx = getelementptr inbounds %"class.sycl::_V1::range", ptr %_arg_accB5, i64 0, i32 0, i32 0, i64 1
  %agg.tmp15.sroa.0.sroa.2.0.copyload = load i64, ptr %agg.tmp15.sroa.0.sroa.2.0..sroa_idx, align 8
  %agg.tmp16.sroa.0.sroa.0.0.copyload = load i64, ptr %_arg_accB6, align 8
  %agg.tmp16.sroa.0.sroa.2.0..sroa_idx = getelementptr inbounds %"class.sycl::_V1::id", ptr %_arg_accB6, i64 0, i32 0, i32 0, i64 1
  %agg.tmp16.sroa.0.sroa.2.0.copyload = load i64, ptr %agg.tmp16.sroa.0.sroa.2.0..sroa_idx, align 8
  %mul.i4.i.i.i.i45 = mul i64 %agg.tmp16.sroa.0.sroa.0.0.copyload, %agg.tmp15.sroa.0.sroa.2.0.copyload
  %add.i6.i.i.i.i46 = add i64 %mul.i4.i.i.i.i45, %agg.tmp16.sroa.0.sroa.2.0.copyload
  %add.ptr.i47 = getelementptr inbounds i8, ptr addrspace(1) %_arg_accB, i64 %add.i6.i.i.i.i46
  %0 = load <3 x i64>, ptr addrspace(1) @__spirv_BuiltInGlobalInvocationId, align 32
  %1 = extractelement <3 x i64> %0, i64 1
  %2 = extractelement <3 x i64> %0, i64 0
  %3 = load <3 x i64>, ptr addrspace(1) @__spirv_BuiltInLocalInvocationId, align 32
  %4 = extractelement <3 x i64> %3, i64 1
  %5 = extractelement <3 x i64> %3, i64 0
  %cmp.i.i = icmp ult i64 %1, 2147483648
  %cmp.i54.i = icmp ult i64 %2, 2147483648
  %cmp.i56.i = icmp ult i64 %4, 2147483648
  %sub.i = sub nsw i64 %1, %4
  %cmp.i58.i = icmp ult i64 %5, 2147483648
  %sub5.i = sub nsw i64 %2, %5
  call void @llvm.lifetime.start.p0(i64 8, ptr nonnull %sub_c.sroa.0.i)
  %call.i.i = tail call spir_func target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) @_Z26__spirv_CompositeConstruct(i32 0)
  store target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) %call.i.i, ptr %sub_c.sroa.0.i, align 8
  %mul.i = mul nsw i64 %sub.i, 12
  %div2452.i = lshr i64 %sub5.i, 4
  %mul26.i = mul i64 %div2452.i, 48
  %div.i = udiv i64 %_arg_K, 48
  %mul11.i = mul i64 %mul.i, %_arg_K
  %add.ptr.i93.i = getelementptr inbounds i8, ptr addrspace(1) %_arg_accA, i64 %mul11.i
  %idx.neg.i.i104.i = sub i64 0, %add.i6.i.i.i.i46
  %add.ptr.i.i105141.i = getelementptr i8, ptr addrspace(1) %add.ptr.i47, i64 %mul26.i
  %mul22.i = shl i64 %_arg_N, 2
  %add.ptr.i108140.i = getelementptr i8, ptr addrspace(1) %add.ptr.i.i105141.i, i64 %idx.neg.i.i104.i
  br label %for.cond.i

for.cond.i:
  %k.0.i = phi i32 [ 0, %entry ], [ %add.i, %for.body.i ]
  %conv.i = zext i32 %k.0.i to i64
  %cmp.i = icmp ugt i64 %div.i, %conv.i
  br i1 %cmp.i, label %for.body.i, label %exit

for.body.i:
  %mul12.i = mul nsw i32 %k.0.i, 48
  %conv13.i = zext i32 %mul12.i to i64
  %add.ptr.i96.i = getelementptr inbounds i8, ptr addrspace(1) %add.ptr.i93.i, i64 %conv13.i
  %call.ascast.i66.i = addrspacecast ptr addrspace(1) %add.ptr.i96.i to ptr addrspace(3)
  %call1.i.i = tail call spir_func target("spirv.CooperativeMatrixKHR", i8, 0, 12, 48, 0) @_Z32__spirv_CooperativeMatrixLoadKHR_1(ptr addrspace(3) %call.ascast.i66.i, i32 0, i64 %_arg_K, i32 1)
  %len = tail call spir_func i32 @_Z34__spirv_CooperativeMatrixLengthKHR(target("spirv.CooperativeMatrixKHR", i8, 0, 12, 48, 0) %call1.i.i)
  %div20.i = mul nsw i32 %k.0.i, 12
  %conv21.i = zext i32 %div20.i to i64
  %mul23.i = mul i64 %mul22.i, %conv21.i
  %add.ptr.i111.i = getelementptr i8, ptr addrspace(1) %add.ptr.i108140.i, i64 %mul23.i
  %call.ascast.i72.i = addrspacecast ptr addrspace(1) %add.ptr.i111.i to ptr addrspace(3)
  %call1.i73.i = tail call spir_func target("spirv.CooperativeMatrixKHR", i8, 2, 48, 12, 1) @_Z32__spirv_CooperativeMatrixLoadKHR_2(ptr addrspace(3) %call.ascast.i72.i, i32 0, i64 %mul22.i)
  call void @llvm.lifetime.start.p0(i64 8, ptr nonnull %ref.tmp29.sroa.0.i)
  %sub_c.sroa.0.i.0.sub_c.sroa.0.i.0.sub_c.sroa.0.0.sub_c.sroa.0.0.sub_c.sroa.0.0.125.i = load target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2), ptr %sub_c.sroa.0.i, align 8
  %call.i77.i = tail call spir_func target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) @_Z34__spirv_CooperativeMatrixMulAddKHR(target("spirv.CooperativeMatrixKHR", i8, 0, 12, 48, 0) %call1.i.i, target("spirv.CooperativeMatrixKHR", i8, 2, 48, 12, 1) %call1.i73.i, target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) %sub_c.sroa.0.i.0.sub_c.sroa.0.i.0.sub_c.sroa.0.0.sub_c.sroa.0.0.sub_c.sroa.0.0.125.i, i32 12)
  store target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) %call.i77.i, ptr %ref.tmp29.sroa.0.i, align 8
  %ref.tmp29.sroa.0.i.0.ref.tmp29.sroa.0.i.0.ref.tmp29.sroa.0.0.ref.tmp29.sroa.0.0.ref.tmp29.sroa.0.0..i = load i64, ptr %ref.tmp29.sroa.0.i, align 8
  store i64 %ref.tmp29.sroa.0.i.0.ref.tmp29.sroa.0.i.0.ref.tmp29.sroa.0.0.ref.tmp29.sroa.0.0.ref.tmp29.sroa.0.0..i, ptr %sub_c.sroa.0.i, align 8
  call void @llvm.lifetime.end.p0(i64 8, ptr nonnull %ref.tmp29.sroa.0.i)
  %add.i = add nuw nsw i32 %k.0.i, 1
  br label %for.cond.i

exit:
  %mul37.i = mul i64 %mul.i, %_arg_N
  %add.ptr.i.i = getelementptr inbounds i32, ptr addrspace(1) %_arg_accC, i64 %mul37.i
  %mul39.i = mul nuw i64 %div2452.i, 12
  %add.ptr.i81.i = getelementptr inbounds i32, ptr addrspace(1) %add.ptr.i.i, i64 %mul39.i
  %call.ascast.i.i = addrspacecast ptr addrspace(1) %add.ptr.i81.i to ptr addrspace(3)
  %sub_c.sroa.0.i.0.sub_c.sroa.0.i.0.sub_c.sroa.0.0.sub_c.sroa.0.0.sub_c.sroa.0.0..i = load target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2), ptr %sub_c.sroa.0.i, align 8
  tail call spir_func void @_Z33__spirv_CooperativeMatrixStoreKHR(ptr addrspace(3) %call.ascast.i.i, target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) %sub_c.sroa.0.i.0.sub_c.sroa.0.i.0.sub_c.sroa.0.0.sub_c.sroa.0.0.sub_c.sroa.0.0..i, i32 0, i64 %_arg_N, i32 1)
  call void @llvm.lifetime.end.p0(i64 8, ptr nonnull %sub_c.sroa.0.i)
  ret void
}

declare dso_local spir_func target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) @_Z26__spirv_CompositeConstruct(i32)
declare dso_local spir_func i32 @_Z34__spirv_CooperativeMatrixLengthKHR(target("spirv.CooperativeMatrixKHR", i8, 0, 12, 48, 0))
declare dso_local spir_func target("spirv.CooperativeMatrixKHR", i8, 0, 12, 48, 0) @_Z32__spirv_CooperativeMatrixLoadKHR_1(ptr addrspace(3), i32, i64, i32)
declare dso_local spir_func target("spirv.CooperativeMatrixKHR", i8, 2, 48, 12, 1) @_Z32__spirv_CooperativeMatrixLoadKHR_2(ptr addrspace(3), i32, i64)
declare dso_local spir_func target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) @_Z34__spirv_CooperativeMatrixMulAddKHR(target("spirv.CooperativeMatrixKHR", i8, 0, 12, 48, 0), target("spirv.CooperativeMatrixKHR", i8, 2, 48, 12, 1), target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2), i32)
declare dso_local spir_func void @_Z33__spirv_CooperativeMatrixStoreKHR(ptr addrspace(3), target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2), i32, i64, i32)

declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture)
