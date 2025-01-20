; This is an excerpt from the SYCL end-to-end test suite, cleaned out from unrelevant details,
; that reproduced multiple cases of the issues when OpPhi's result type mismatches with operand types.
; The only pass criterion is that spirv-val considers output valid.

; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

%struct.PFWGFunctor = type { i64, i64, i32, i32, %"class.sycl::_V1::accessor" }
%"class.sycl::_V1::accessor" = type { %"class.sycl::_V1::detail::AccessorImplDevice", %union.anon }
%"class.sycl::_V1::detail::AccessorImplDevice" = type { %"class.sycl::_V1::range", %"class.sycl::_V1::range", %"class.sycl::_V1::range" }
%"class.sycl::_V1::range" = type { %"class.sycl::_V1::detail::array" }
%"class.sycl::_V1::detail::array" = type { [1 x i64] }
%union.anon = type { ptr addrspace(1) }
%class.anon.2 = type { %"class.sycl::_V1::accessor" }
%"class.sycl::_V1::group" = type { %"class.sycl::_V1::range", %"class.sycl::_V1::range", %"class.sycl::_V1::range", %"class.sycl::_V1::range" }
%"class.sycl::_V1::group.15" = type { %"class.sycl::_V1::range.16", %"class.sycl::_V1::range.16", %"class.sycl::_V1::range.16", %"class.sycl::_V1::range.16" }
%"class.sycl::_V1::range.16" = type { %"class.sycl::_V1::detail::array.17" }
%"class.sycl::_V1::detail::array.17" = type { [2 x i64] }
%"class.sycl::_V1::private_memory" = type { %struct.MyStruct }
%struct.MyStruct = type { i32, i32 }

@GFunctor = internal addrspace(3) global %struct.PFWGFunctor undef, align 8
@WI.0 = internal unnamed_addr addrspace(3) global i64 undef, align 8
@WI.1 = internal unnamed_addr addrspace(3) global i64 undef, align 8
@WI.2 = internal unnamed_addr addrspace(3) global i64 undef, align 8
@WI.3 = internal unnamed_addr addrspace(3) global i64 undef, align 8
@WI.4 = internal unnamed_addr addrspace(3) global i32 undef, align 8
@WI.6 = internal unnamed_addr addrspace(3) global ptr addrspace(4) undef, align 8
@GCnt = internal unnamed_addr addrspace(3) global i32 undef, align 4
@__spirv_BuiltInNumWorkgroups = external dso_local local_unnamed_addr addrspace(1) constant <3 x i64>, align 32
@GKernel1 = internal addrspace(3) global %class.anon.2 undef, align 8
@GCnt2 = internal unnamed_addr addrspace(3) global i32 undef, align 4
@GKernel2 = internal addrspace(3) global %class.anon.2 undef, align 8
@GCnt3 = internal unnamed_addr addrspace(3) global i32 undef, align 4
@GKernel3 = internal addrspace(3) global %class.anon.2 undef, align 8
@GCnt4 = internal unnamed_addr addrspace(3) global i32 undef, align 4
@GKernel4 = internal addrspace(3) global %class.anon.2 undef, align 8
@GCnt5 = internal unnamed_addr addrspace(3) global i32 undef, align 4
@__spirv_BuiltInLocalInvocationIndex = external local_unnamed_addr addrspace(1) constant i64, align 8
@GThis = internal unnamed_addr addrspace(3) global ptr addrspace(4) undef, align 8
@GAsCast = internal unnamed_addr addrspace(3) global ptr addrspace(4) undef, align 8
@GCmp = internal unnamed_addr addrspace(3) global i1 undef, align 1
@WGCopy = internal unnamed_addr addrspace(3) global ptr addrspace(4) undef, align 8
@WGCopy.1.0 = internal unnamed_addr addrspace(3) global i64 undef, align 16
@WGCopy.1.1 = internal unnamed_addr addrspace(3) global i64 undef, align 16
@WGCopy.1.2 = internal unnamed_addr addrspace(3) global i64 undef, align 16
@WGCopy.1.3 = internal unnamed_addr addrspace(3) global i64 undef, align 16
@WGCopy.1.4 = internal unnamed_addr addrspace(3) global i32 undef, align 16
@WGCopy.1.5 = internal unnamed_addr addrspace(3) global i32 undef, align 16
@WGCopy.1.6 = internal unnamed_addr addrspace(3) global ptr addrspace(4) undef, align 16
@ArgShadow = internal unnamed_addr addrspace(3) global %"class.sycl::_V1::group" undef, align 16
@GAsCast2 = internal unnamed_addr addrspace(3) global ptr addrspace(4) undef, align 8
@GCmp2 = internal unnamed_addr addrspace(3) global i1 undef, align 1
@WGCopy.3.0 = internal unnamed_addr addrspace(3) global i64 undef, align 8
@WGCopy.4.0 = internal unnamed_addr addrspace(3) global i64 undef, align 8
@WGCopy.5.0 = internal unnamed_addr addrspace(3) global i64 undef, align 8
@WGCopy.6.0 = internal unnamed_addr addrspace(3) global i64 undef, align 8
@ArgShadow.7 = internal unnamed_addr addrspace(3) global %"class.sycl::_V1::group" undef, align 16
@GAscast3 = internal unnamed_addr addrspace(3) global ptr addrspace(4) undef, align 8
@GCmp3 = internal unnamed_addr addrspace(3) global i1 undef, align 1
@WGCopy.9.0 = internal unnamed_addr addrspace(3) global i64 undef, align 8
@WGCopy.10.0 = internal unnamed_addr addrspace(3) global i64 undef, align 8
@ArgShadow.11 = internal unnamed_addr addrspace(3) global %"class.sycl::_V1::group" undef, align 16
@GAsCast4 = internal unnamed_addr addrspace(3) global ptr addrspace(4) undef, align 8
@GCmp4 = internal unnamed_addr addrspace(3) global i1 undef, align 1
@WGCopy.13.0 = internal unnamed_addr addrspace(3) global i64 undef, align 8
@WGCopy.13.1 = internal unnamed_addr addrspace(3) global i64 undef, align 8
@WGCopy.14.0 = internal unnamed_addr addrspace(3) global ptr addrspace(4) undef, align 8
@WGCopy.14.1 = internal unnamed_addr addrspace(3) global ptr addrspace(4) undef, align 8
@WGCopy.15.0 = internal unnamed_addr addrspace(3) global i64 undef, align 8
@WGCopy.15.1 = internal unnamed_addr addrspace(3) global i64 undef, align 8
@WGCopy.16.0 = internal unnamed_addr addrspace(3) global ptr addrspace(4) undef, align 8
@WGCopy.16.1 = internal unnamed_addr addrspace(3) global ptr addrspace(4) undef, align 8
@ArgShadow.17 = internal unnamed_addr addrspace(3) global %"class.sycl::_V1::group.15" undef, align 16
@GAsCast5 = internal unnamed_addr addrspace(3) global ptr addrspace(4) undef, align 8
@GCmp5 = internal unnamed_addr addrspace(3) global i1 undef, align 1
@WGCopy.19.0 = internal unnamed_addr addrspace(3) global i64 undef, align 8
@WGCopy.20.0 = internal unnamed_addr addrspace(3) global ptr addrspace(4) undef, align 8
@WGCopy.20.1 = internal unnamed_addr addrspace(3) global ptr addrspace(4) undef, align 8
@ArgShadow.21 = internal unnamed_addr addrspace(3) global %"class.sycl::_V1::group" undef, align 16
@__spirv_BuiltInGlobalInvocationId = external dso_local local_unnamed_addr addrspace(1) constant <3 x i64>, align 32
@__spirv_BuiltInGlobalSize = external dso_local local_unnamed_addr addrspace(1) constant <3 x i64>, align 32
@__spirv_BuiltInLocalInvocationId = external dso_local local_unnamed_addr addrspace(1) constant <3 x i64>, align 32
@__spirv_BuiltInWorkgroupId = external dso_local local_unnamed_addr addrspace(1) constant <3 x i64>, align 32
@__spirv_BuiltInWorkgroupSize = external dso_local local_unnamed_addr addrspace(1) constant <3 x i64>, align 32

; Function Attrs: convergent mustprogress norecurse nounwind
define weak_odr dso_local spir_kernel void @_ZTS11PFWGFunctor(i64 noundef %_arg_wg_chunk, i64 noundef %_arg_range_length, i32 noundef %_arg_n_iter, i32 noundef %_arg_addend, ptr addrspace(1) noundef align 4 %_arg_dev_ptr, ptr noundef byval(%"class.sycl::_V1::range") align 8 %_arg_dev_ptr1, ptr noundef byval(%"class.sycl::_V1::range") align 8 %_arg_dev_ptr2, ptr noundef byval(%"class.sycl::_V1::range") align 8 %_arg_dev_ptr3) {
entry:
  %agg.tmp67 = alloca %"class.sycl::_V1::group", align 8
  store i64 %_arg_wg_chunk, ptr addrspace(3) @GFunctor, align 8
  store i64 %_arg_range_length, ptr addrspace(3) undef, align 8
  store i32 %_arg_n_iter, ptr addrspace(3) undef, align 8
  store i32 %_arg_addend, ptr addrspace(3) undef, align 4
  %0 = load i64, ptr %_arg_dev_ptr1, align 8
  %1 = load i64, ptr %_arg_dev_ptr2, align 8
  %2 = load i64, ptr %_arg_dev_ptr3, align 8
  store i64 %2, ptr addrspace(3) undef, align 8
  store i64 %0, ptr addrspace(3) undef, align 8
  store i64 %1, ptr addrspace(3) undef, align 8
  %add.ptr.i = getelementptr inbounds i32, ptr addrspace(1) %_arg_dev_ptr, i64 %2
  store ptr addrspace(1) %add.ptr.i, ptr addrspace(3) undef, align 8
  %3 = load i64, ptr addrspace(1) @__spirv_BuiltInGlobalSize, align 32
  %4 = load i64, ptr addrspace(1) @__spirv_BuiltInWorkgroupSize, align 32
  %5 = load i64, ptr addrspace(1) @__spirv_BuiltInNumWorkgroups, align 32
  %6 = load i64, ptr addrspace(1) @__spirv_BuiltInWorkgroupId, align 32
  call void @llvm.lifetime.start.p0(i64 32, ptr nonnull %agg.tmp67)
  store i64 %3, ptr %agg.tmp67, align 1
  %agg.tmp6.sroa.2.0.agg.tmp67.sroa_idx = getelementptr inbounds i8, ptr %agg.tmp67, i64 8
  store i64 %4, ptr %agg.tmp6.sroa.2.0.agg.tmp67.sroa_idx, align 1
  %agg.tmp6.sroa.3.0.agg.tmp67.sroa_idx = getelementptr inbounds i8, ptr %agg.tmp67, i64 16
  store i64 %5, ptr %agg.tmp6.sroa.3.0.agg.tmp67.sroa_idx, align 1
  %agg.tmp6.sroa.4.0.agg.tmp67.sroa_idx = getelementptr inbounds i8, ptr %agg.tmp67, i64 24
  store i64 %6, ptr %agg.tmp6.sroa.4.0.agg.tmp67.sroa_idx, align 1
  %7 = load i64, ptr addrspace(1) @__spirv_BuiltInLocalInvocationIndex, align 8
  tail call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 2, i32 2, i32 272)
  %cmpz15.i = icmp eq i64 %7, 0
  br i1 %cmpz15.i, label %leader.i, label %merge.i

leader.i:                                         ; preds = %entry
  call void @llvm.memcpy.p3.p0.i64(ptr addrspace(3) noundef align 16 dereferenceable(32) @ArgShadow, ptr noundef nonnull align 8 dereferenceable(32) %agg.tmp67, i64 32, i1 false)
  br label %merge.i

merge.i:                                          ; preds = %leader.i, %entry
  tail call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 2, i32 2, i32 272)
  call void @llvm.memcpy.p0.p3.i64(ptr noundef nonnull align 8 dereferenceable(32) %agg.tmp67, ptr addrspace(3) noundef align 16 dereferenceable(32) @ArgShadow, i64 32, i1 false)
  tail call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 2, i32 2, i32 272)
  br i1 %cmpz15.i, label %wg_leader.i, label %wg_cf.i

wg_leader.i:                                      ; preds = %merge.i
  %g.ascast.i = addrspacecast ptr %agg.tmp67 to ptr addrspace(4)
  store ptr addrspace(4) %g.ascast.i, ptr addrspace(3) @GAsCast, align 8
  store ptr addrspace(4) addrspacecast (ptr addrspace(3) @GFunctor to ptr addrspace(4)), ptr addrspace(3) @GThis, align 8
  %8 = load i32, ptr addrspace(3) undef, align 4
  %9 = load i64, ptr addrspace(3) @GFunctor, align 8
  %index.i = getelementptr inbounds i8, ptr %agg.tmp67, i64 24
  %10 = load i64, ptr %index.i, align 8
  %mul.i = mul i64 %9, %10
  %localRange.i = getelementptr inbounds i8, ptr %agg.tmp67, i64 8
  %11 = load i64, ptr %localRange.i, align 8
  %12 = load i64, ptr addrspace(3) undef, align 8
  store i64 %9, ptr addrspace(3) @WI.0, align 8
  store i64 %11, ptr addrspace(3) @WI.1, align 8
  store i64 %mul.i, ptr addrspace(3) @WI.2, align 8
  store i64 %12, ptr addrspace(3) @WI.3, align 8
  store i32 %8, ptr addrspace(3) @WI.4, align 8
  store ptr addrspace(4) undef, ptr addrspace(3) @WI.6, align 8
  store i32 0, ptr addrspace(3) @GCnt, align 4
  br label %wg_cf.i

wg_cf.i:                                          ; preds = %wg_leader.i, %merge.i
  call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 2, i32 2, i32 272)
  %wg_val_this1.i = load ptr addrspace(4), ptr addrspace(3) @GThis, align 8
  %n_iter.i = getelementptr inbounds i8, ptr addrspace(4) %wg_val_this1.i, i64 16
  %13 = load i64, ptr addrspace(1) @__spirv_BuiltInLocalInvocationId, align 32
  br label %for.cond.i

for.cond.i:                                       ; preds = %wg_cf11.i, %wg_cf.i
  %agg.tmp.i.sroa.0.0 = phi i64 [ undef, %wg_cf.i ], [ %agg.tmp.i.sroa.0.0.copyload13, %wg_cf11.i ]
  %agg.tmp.i.sroa.6.0 = phi i64 [ undef, %wg_cf.i ], [ %agg.tmp.i.sroa.6.0.copyload15, %wg_cf11.i ]
  %agg.tmp.i.sroa.7.0 = phi i64 [ undef, %wg_cf.i ], [ %agg.tmp.i.sroa.7.0.copyload17, %wg_cf11.i ]
  %agg.tmp.i.sroa.8.0 = phi i64 [ undef, %wg_cf.i ], [ %agg.tmp.i.sroa.8.0.copyload19, %wg_cf11.i ]
  %agg.tmp.i.sroa.9.0 = phi i32 [ undef, %wg_cf.i ], [ %agg.tmp.i.sroa.9.0.copyload21, %wg_cf11.i ]
  %agg.tmp.i.sroa.10.0 = phi i32 [ undef, %wg_cf.i ], [ %agg.tmp.i.sroa.10.0.copyload23, %wg_cf11.i ]
  %agg.tmp.i.sroa.11.0 = phi ptr addrspace(4) [ undef, %wg_cf.i ], [ %agg.tmp.i.sroa.11.0.copyload25, %wg_cf11.i ]
  %this.addr.0.i = phi ptr addrspace(4) [ addrspacecast (ptr addrspace(3) @GFunctor to ptr addrspace(4)), %wg_cf.i ], [ %mat_ld13.i, %wg_cf11.i ]
  call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 2, i32 2, i32 272)
  br i1 %cmpz15.i, label %wg_leader4.i, label %wg_cf5.i

wg_leader4.i:                                     ; preds = %for.cond.i
  %14 = load i32, ptr addrspace(3) @GCnt, align 4
  %15 = load i32, ptr addrspace(4) %n_iter.i, align 8
  %cmp.i = icmp slt i32 %14, %15
  store i1 %cmp.i, ptr addrspace(3) @GCmp, align 1
  br label %wg_cf5.i

wg_cf5.i:                                         ; preds = %wg_leader4.i, %for.cond.i
  call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 2, i32 2, i32 272)
  %wg_val_cmp.i = load i1, ptr addrspace(3) @GCmp, align 1
  br i1 %wg_val_cmp.i, label %for.body.i, label %_ZNK11PFWGFunctorclEN4sycl3_V15groupILi1EEE.exit

for.body.i:                                       ; preds = %wg_cf5.i
  call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 2, i32 2, i32 272)
  br i1 %cmpz15.i, label %wg_leader7.i, label %wg_cf8.i

wg_leader7.i:                                     ; preds = %for.body.i
  %agg.tmp.i.sroa.0.0.copyload = load i64, ptr addrspace(3) @WI.0, align 8
  %agg.tmp.i.sroa.6.0.copyload = load i64, ptr addrspace(3) @WI.1, align 8
  %agg.tmp.i.sroa.7.0.copyload = load i64, ptr addrspace(3) @WI.2, align 8
  %agg.tmp.i.sroa.8.0.copyload = load i64, ptr addrspace(3) @WI.3, align 8
  %agg.tmp.i.sroa.9.0.copyload = load i32, ptr addrspace(3) @WI.4, align 8
  %agg.tmp.i.sroa.11.0.copyload = load ptr addrspace(4), ptr addrspace(3) @WI.6, align 8
  br label %wg_cf8.i

wg_cf8.i:                                         ; preds = %wg_leader7.i, %for.body.i
  %agg.tmp.i.sroa.0.1 = phi i64 [ %agg.tmp.i.sroa.0.0.copyload, %wg_leader7.i ], [ %agg.tmp.i.sroa.0.0, %for.body.i ]
  %agg.tmp.i.sroa.6.1 = phi i64 [ %agg.tmp.i.sroa.6.0.copyload, %wg_leader7.i ], [ %agg.tmp.i.sroa.6.0, %for.body.i ]
  %agg.tmp.i.sroa.7.1 = phi i64 [ %agg.tmp.i.sroa.7.0.copyload, %wg_leader7.i ], [ %agg.tmp.i.sroa.7.0, %for.body.i ]
  %agg.tmp.i.sroa.8.1 = phi i64 [ %agg.tmp.i.sroa.8.0.copyload, %wg_leader7.i ], [ %agg.tmp.i.sroa.8.0, %for.body.i ]
  %agg.tmp.i.sroa.9.1 = phi i32 [ %agg.tmp.i.sroa.9.0.copyload, %wg_leader7.i ], [ %agg.tmp.i.sroa.9.0, %for.body.i ]
  %agg.tmp.i.sroa.11.1 = phi ptr addrspace(4) [ %agg.tmp.i.sroa.11.0.copyload, %wg_leader7.i ], [ %agg.tmp.i.sroa.11.0, %for.body.i ]
  call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 2, i32 2, i32 272)
  br i1 %cmpz15.i, label %TestMat.i, label %LeaderMat.i

TestMat.i:                                        ; preds = %wg_cf8.i
  store i64 %agg.tmp.i.sroa.0.1, ptr addrspace(3) @WGCopy.1.0, align 16
  store i64 %agg.tmp.i.sroa.6.1, ptr addrspace(3) @WGCopy.1.1, align 16
  store i64 %agg.tmp.i.sroa.7.1, ptr addrspace(3) @WGCopy.1.2, align 16
  store i64 %agg.tmp.i.sroa.8.1, ptr addrspace(3) @WGCopy.1.3, align 16
  store i32 %agg.tmp.i.sroa.9.1, ptr addrspace(3) @WGCopy.1.4, align 16
  store i32 %agg.tmp.i.sroa.10.0, ptr addrspace(3) @WGCopy.1.5, align 16
  store ptr addrspace(4) %agg.tmp.i.sroa.11.1, ptr addrspace(3) @WGCopy.1.6, align 16
  store ptr addrspace(4) %this.addr.0.i, ptr addrspace(3) @WGCopy, align 8
  br label %LeaderMat.i

LeaderMat.i:                                      ; preds = %TestMat.i, %wg_cf8.i
  call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 2, i32 2, i32 272)
  %mat_ld13.i = load ptr addrspace(4), ptr addrspace(3) @WGCopy, align 8
  %agg.tmp.i.sroa.0.0.copyload13 = load i64, ptr addrspace(3) @WGCopy.1.0, align 16
  %agg.tmp.i.sroa.6.0.copyload15 = load i64, ptr addrspace(3) @WGCopy.1.1, align 16
  %agg.tmp.i.sroa.7.0.copyload17 = load i64, ptr addrspace(3) @WGCopy.1.2, align 16
  %agg.tmp.i.sroa.8.0.copyload19 = load i64, ptr addrspace(3) @WGCopy.1.3, align 16
  %agg.tmp.i.sroa.9.0.copyload21 = load i32, ptr addrspace(3) @WGCopy.1.4, align 16
  %agg.tmp.i.sroa.10.0.copyload23 = load i32, ptr addrspace(3) @WGCopy.1.5, align 16
  %agg.tmp.i.sroa.11.0.copyload25 = load ptr addrspace(4), ptr addrspace(3) @WGCopy.1.6, align 16
  call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 2, i32 2, i32 272)
  call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 noundef 2, i32 noundef 2, i32 noundef 272)
  %cmp.not.i.i = icmp ult i64 %13, %agg.tmp.i.sroa.0.0.copyload13
  br i1 %cmp.not.i.i, label %if.end.i.i, label %lexit1

if.end.i.i:                                       ; preds = %LeaderMat.i
  %add.i.i = add i64 %agg.tmp.i.sroa.0.0.copyload13, %agg.tmp.i.sroa.6.0.copyload15
  %sub.i.i = add i64 %add.i.i, -1
  %div.i.i = udiv i64 %sub.i.i, %agg.tmp.i.sroa.6.0.copyload15
  %mul.i.i = mul i64 %13, %div.i.i
  %add4.i.i = add i64 %agg.tmp.i.sroa.7.0.copyload17, %mul.i.i
  %add6.i.i = add i64 %add4.i.i, %div.i.i
  %.sroa.speculated.i.i = call i64 @llvm.umin.i64(i64 %agg.tmp.i.sroa.8.0.copyload19, i64 %add6.i.i)
  %16 = getelementptr inbounds i8, ptr addrspace(4) %agg.tmp.i.sroa.11.0.copyload25, i64 24
  br label %for.cond.i.i

for.cond.i.i:                                     ; preds = %for.body.i.i, %if.end.i.i
  %ind.0.i.i = phi i64 [ %add4.i.i, %if.end.i.i ], [ %inc.i.i, %for.body.i.i ]
  %cmp8.i.i = icmp ult i64 %ind.0.i.i, %.sroa.speculated.i.i
  br i1 %cmp8.i.i, label %for.body.i.i, label %lexit1

for.body.i.i:                                     ; preds = %for.cond.i.i
  %17 = load ptr addrspace(1), ptr addrspace(4) %16, align 8
  %arrayidx.i.i.i = getelementptr inbounds i32, ptr addrspace(1) %17, i64 %ind.0.i.i
  %18 = load i32, ptr addrspace(1) %arrayidx.i.i.i, align 4
  %add10.i.i = add nsw i32 %18, %agg.tmp.i.sroa.9.0.copyload21
  store i32 %add10.i.i, ptr addrspace(1) %arrayidx.i.i.i, align 4
  %inc.i.i = add nuw i64 %ind.0.i.i, 1
  br label %for.cond.i.i

lexit1: ; preds = %for.cond.i.i, %LeaderMat.i
  call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 noundef 2, i32 noundef 2, i32 noundef 272)
  call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 2, i32 2, i32 272)
  br i1 %cmpz15.i, label %wg_leader10.i, label %wg_cf11.i

wg_leader10.i:                                    ; preds = %lexit1
  %19 = load i32, ptr addrspace(3) @GCnt, align 4
  %inc.i = add nsw i32 %19, 1
  store i32 %inc.i, ptr addrspace(3) @GCnt, align 4
  br label %wg_cf11.i

wg_cf11.i:                                        ; preds = %wg_leader10.i, %lexit1
  call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 2, i32 2, i32 272)
  br label %for.cond.i

_ZNK11PFWGFunctorclEN4sycl3_V15groupILi1EEE.exit: ; preds = %wg_cf5.i
  call void @llvm.lifetime.end.p0(i64 32, ptr nonnull %agg.tmp67)
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture)

; Function Attrs: convergent nounwind
declare dso_local spir_func void @_Z22__spirv_ControlBarrierjjj(i32 noundef, i32 noundef, i32 noundef)

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p3.p0.i64(ptr addrspace(3) noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg)

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p3.i64(ptr noalias nocapture writeonly, ptr addrspace(3) noalias nocapture readonly, i64, i1 immarg)

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.umin.i64(i64, i64)

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture)

; Function Attrs: convergent mustprogress norecurse nounwind
define weak_odr dso_local spir_kernel void @bar(ptr addrspace(1) noundef align 4 %_arg_dev_ptr, ptr noundef byval(%"class.sycl::_V1::range") align 8 %_arg_dev_ptr1, ptr noundef byval(%"class.sycl::_V1::range") align 8 %_arg_dev_ptr2, ptr noundef byval(%"class.sycl::_V1::range") align 8 %_arg_dev_ptr3) {
entry:
  %agg.tmp67 = alloca %"class.sycl::_V1::group", align 8
  %0 = load i64, ptr %_arg_dev_ptr1, align 8
  %1 = load i64, ptr %_arg_dev_ptr2, align 8
  %2 = load i64, ptr %_arg_dev_ptr3, align 8
  store i64 %2, ptr addrspace(3) @GKernel1, align 8
  store i64 %0, ptr addrspace(3) undef, align 8
  store i64 %1, ptr addrspace(3) undef, align 8
  %add.ptr.i = getelementptr inbounds i32, ptr addrspace(1) %_arg_dev_ptr, i64 %2
  store ptr addrspace(1) %add.ptr.i, ptr addrspace(3) undef, align 8
  %3 = load i64, ptr addrspace(1) @__spirv_BuiltInGlobalSize, align 32
  %4 = load i64, ptr addrspace(1) @__spirv_BuiltInWorkgroupSize, align 32
  %5 = load i64, ptr addrspace(1) @__spirv_BuiltInNumWorkgroups, align 32
  %6 = load i64, ptr addrspace(1) @__spirv_BuiltInWorkgroupId, align 32
  call void @llvm.lifetime.start.p0(i64 32, ptr nonnull %agg.tmp67)
  store i64 %3, ptr %agg.tmp67, align 1
  %agg.tmp6.sroa.2.0.agg.tmp67.sroa_idx = getelementptr inbounds i8, ptr %agg.tmp67, i64 8
  store i64 %4, ptr %agg.tmp6.sroa.2.0.agg.tmp67.sroa_idx, align 1
  %agg.tmp6.sroa.3.0.agg.tmp67.sroa_idx = getelementptr inbounds i8, ptr %agg.tmp67, i64 16
  store i64 %5, ptr %agg.tmp6.sroa.3.0.agg.tmp67.sroa_idx, align 1
  %agg.tmp6.sroa.4.0.agg.tmp67.sroa_idx = getelementptr inbounds i8, ptr %agg.tmp67, i64 24
  store i64 %6, ptr %agg.tmp6.sroa.4.0.agg.tmp67.sroa_idx, align 1
  %7 = load i64, ptr addrspace(1) @__spirv_BuiltInLocalInvocationIndex, align 8
  tail call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 2, i32 2, i32 272)
  %cmpz27.i = icmp eq i64 %7, 0
  br i1 %cmpz27.i, label %leader.i, label %merge.i

leader.i:                                         ; preds = %entry
  call void @llvm.memcpy.p3.p0.i64(ptr addrspace(3) noundef align 16 dereferenceable(32) @ArgShadow.7, ptr noundef nonnull align 8 dereferenceable(32) %agg.tmp67, i64 32, i1 false)
  br label %merge.i

merge.i:                                          ; preds = %leader.i, %entry
  tail call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 2, i32 2, i32 272)
  call void @llvm.memcpy.p0.p3.i64(ptr noundef nonnull align 8 dereferenceable(32) %agg.tmp67, ptr addrspace(3) noundef align 16 dereferenceable(32) @ArgShadow.7, i64 32, i1 false)
  tail call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 2, i32 2, i32 272)
  br i1 %cmpz27.i, label %wg_leader.i, label %wg_cf.i

wg_leader.i:                                      ; preds = %merge.i
  %g.ascast.i = addrspacecast ptr %agg.tmp67 to ptr addrspace(4)
  store ptr addrspace(4) %g.ascast.i, ptr addrspace(3) @GAsCast2, align 8
  store i32 0, ptr addrspace(3) @GCnt2, align 4
  br label %wg_cf.i

wg_cf.i:                                          ; preds = %wg_leader.i, %merge.i
  call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 2, i32 2, i32 272)
  %8 = load i64, ptr addrspace(1) @__spirv_BuiltInGlobalInvocationId, align 32
  %9 = load i64, ptr addrspace(1) @__spirv_BuiltInLocalInvocationId, align 32
  %cmp.i.i.i.i.i.i = icmp ult i64 %8, 2147483648
  br label %for.cond.i

for.cond.i:                                       ; preds = %wg_cf18.i, %wg_cf.i
  %agg.tmp5.i.sroa.0.0 = phi i64 [ undef, %wg_cf.i ], [ %18, %wg_cf18.i ]
  %agg.tmp4.i.sroa.0.0 = phi i64 [ undef, %wg_cf.i ], [ %17, %wg_cf18.i ]
  call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 2, i32 2, i32 272)
  br i1 %cmpz27.i, label %wg_leader8.i, label %wg_cf9.i

wg_leader8.i:                                     ; preds = %for.cond.i
  %10 = load i32, ptr addrspace(3) @GCnt2, align 4
  %cmp.i = icmp slt i32 %10, 2
  store i1 %cmp.i, ptr addrspace(3) @GCmp2, align 1
  br label %wg_cf9.i

wg_cf9.i:                                         ; preds = %wg_leader8.i, %for.cond.i
  call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 2, i32 2, i32 272)
  %wg_val_cmp.i = load i1, ptr addrspace(3) @GCmp2, align 1
  br i1 %wg_val_cmp.i, label %for.body.i, label %lexit2

for.body.i:                                       ; preds = %wg_cf9.i
  call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 2, i32 2, i32 272)
  call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 2, i32 2, i32 272)
  br i1 %cmpz27.i, label %TestMat25.i, label %LeaderMat22.i

TestMat25.i:                                      ; preds = %for.body.i
  store i64 %agg.tmp5.i.sroa.0.0, ptr addrspace(3) @WGCopy.6.0, align 8
  store i64 ptrtoint (ptr addrspace(4) addrspacecast (ptr addrspace(3) @GKernel1 to ptr addrspace(4)) to i64), ptr addrspace(3) @WGCopy.4.0, align 8
  store i64 5, ptr addrspace(3) @WGCopy.3.0, align 8
  store i64 %agg.tmp4.i.sroa.0.0, ptr addrspace(3) @WGCopy.5.0, align 8
  br label %LeaderMat22.i

LeaderMat22.i:                                    ; preds = %TestMat25.i, %for.body.i
  call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 2, i32 2, i32 272)
  %11 = load i64, ptr addrspace(3) @WGCopy.3.0, align 8
  %12 = load i64, ptr addrspace(3) @WGCopy.4.0, align 8
  %13 = inttoptr i64 %12 to ptr addrspace(4)
  call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 2, i32 2, i32 272)
  call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 noundef 2, i32 noundef 2, i32 noundef 272)
  %14 = getelementptr inbounds i8, ptr addrspace(4) %13, i64 24
  br label %for.cond.i.i

for.cond.i.i:                                     ; preds = %for.body.i.i, %LeaderMat22.i
  %storemerge.i.i = phi i64 [ %9, %LeaderMat22.i ], [ %add.i.i, %for.body.i.i ]
  %cmp.i.i = icmp ult i64 %storemerge.i.i, %11
  br i1 %cmp.i.i, label %for.body.i.i, label %lexit3

for.body.i.i:                                     ; preds = %for.cond.i.i
  call void @llvm.assume(i1 %cmp.i.i.i.i.i.i)
  %15 = load ptr addrspace(1), ptr addrspace(4) %14, align 8
  %arrayidx.i.i.i.i.i = getelementptr inbounds i32, ptr addrspace(1) %15, i64 %8
  %16 = load i32, ptr addrspace(1) %arrayidx.i.i.i.i.i, align 4
  %inc.i.i.i.i = add nsw i32 %16, 1
  store i32 %inc.i.i.i.i, ptr addrspace(1) %arrayidx.i.i.i.i.i, align 4
  %add.i.i = add i64 %storemerge.i.i, %4
  br label %for.cond.i.i

lexit3: ; preds = %for.cond.i.i
  call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 noundef 2, i32 noundef 2, i32 noundef 272)
  call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 2, i32 2, i32 272)
  call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 2, i32 2, i32 272)
  br i1 %cmpz27.i, label %TestMat.i, label %LeaderMat.i

TestMat.i:                                        ; preds = %lexit3
  store i64 ptrtoint (ptr addrspace(4) addrspacecast (ptr addrspace(3) @GKernel1 to ptr addrspace(4)) to i64), ptr addrspace(3) @WGCopy.6.0, align 8
  store i64 %12, ptr addrspace(3) @WGCopy.4.0, align 8
  store i64 %11, ptr addrspace(3) @WGCopy.3.0, align 8
  store i64 2, ptr addrspace(3) @WGCopy.5.0, align 8
  br label %LeaderMat.i

LeaderMat.i:                                      ; preds = %TestMat.i, %lexit3
  call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 2, i32 2, i32 272)
  %17 = load i64, ptr addrspace(3) @WGCopy.5.0, align 8
  %18 = load i64, ptr addrspace(3) @WGCopy.6.0, align 8
  %19 = inttoptr i64 %18 to ptr addrspace(4)
  call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 2, i32 2, i32 272)
  call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 noundef 2, i32 noundef 2, i32 noundef 272)
  %20 = getelementptr inbounds i8, ptr addrspace(4) %19, i64 24
  br label %for.cond.i.i19

for.cond.i.i19:                                   ; preds = %for.body.i.i22, %LeaderMat.i
  %storemerge.i.i20 = phi i64 [ %9, %LeaderMat.i ], [ %add.i.i26, %for.body.i.i22 ]
  %cmp.i.i21 = icmp ult i64 %storemerge.i.i20, %17
  br i1 %cmp.i.i21, label %for.body.i.i22, label %lexit4

for.body.i.i22:                                   ; preds = %for.cond.i.i19
  call void @llvm.assume(i1 %cmp.i.i.i.i.i.i)
  %21 = load ptr addrspace(1), ptr addrspace(4) %20, align 8
  %arrayidx.i.i.i.i.i23 = getelementptr inbounds i32, ptr addrspace(1) %21, i64 %8
  %22 = load i32, ptr addrspace(1) %arrayidx.i.i.i.i.i23, align 4
  %inc.i.i.i.i25 = add nsw i32 %22, 1
  store i32 %inc.i.i.i.i25, ptr addrspace(1) %arrayidx.i.i.i.i.i23, align 4
  %add.i.i26 = add i64 %storemerge.i.i20, %4
  br label %for.cond.i.i19

lexit4: ; preds = %for.cond.i.i19
  call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 noundef 2, i32 noundef 2, i32 noundef 272)
  call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 2, i32 2, i32 272)
  br i1 %cmpz27.i, label %wg_leader17.i, label %wg_cf18.i

wg_leader17.i:                                    ; preds = %lexit4
  %23 = load i32, ptr addrspace(3) @GCnt2, align 4
  %inc.i = add nsw i32 %23, 1
  store i32 %inc.i, ptr addrspace(3) @GCnt2, align 4
  br label %wg_cf18.i

wg_cf18.i:                                        ; preds = %wg_leader17.i, %lexit4
  call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 2, i32 2, i32 272)
  br label %for.cond.i

lexit2: ; preds = %wg_cf9.i
  call void @llvm.lifetime.end.p0(i64 32, ptr nonnull %agg.tmp67)
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write)
declare void @llvm.assume(i1 noundef)

; Function Attrs: convergent mustprogress norecurse nounwind
define weak_odr dso_local spir_kernel void @test1(ptr addrspace(1) noundef align 4 %_arg_dev_ptr, ptr noundef byval(%"class.sycl::_V1::range") align 8 %_arg_dev_ptr1, ptr noundef byval(%"class.sycl::_V1::range") align 8 %_arg_dev_ptr2, ptr noundef byval(%"class.sycl::_V1::range") align 8 %_arg_dev_ptr3) {
entry:
  %agg.tmp67 = alloca %"class.sycl::_V1::group", align 8
  %0 = load i64, ptr %_arg_dev_ptr1, align 8
  %1 = load i64, ptr %_arg_dev_ptr2, align 8
  %2 = load i64, ptr %_arg_dev_ptr3, align 8
  store i64 %2, ptr addrspace(3) @GKernel2, align 8
  store i64 %0, ptr addrspace(3) undef, align 8
  store i64 %1, ptr addrspace(3) undef, align 8
  %add.ptr.i = getelementptr inbounds i32, ptr addrspace(1) %_arg_dev_ptr, i64 %2
  store ptr addrspace(1) %add.ptr.i, ptr addrspace(3) undef, align 8
  %3 = load i64, ptr addrspace(1) @__spirv_BuiltInGlobalSize, align 32
  %4 = load i64, ptr addrspace(1) @__spirv_BuiltInWorkgroupSize, align 32
  %5 = load i64, ptr addrspace(1) @__spirv_BuiltInNumWorkgroups, align 32
  %6 = load i64, ptr addrspace(1) @__spirv_BuiltInWorkgroupId, align 32
  call void @llvm.lifetime.start.p0(i64 32, ptr nonnull %agg.tmp67)
  store i64 %3, ptr %agg.tmp67, align 1
  %agg.tmp6.sroa.2.0.agg.tmp67.sroa_idx = getelementptr inbounds i8, ptr %agg.tmp67, i64 8
  store i64 %4, ptr %agg.tmp6.sroa.2.0.agg.tmp67.sroa_idx, align 1
  %agg.tmp6.sroa.3.0.agg.tmp67.sroa_idx = getelementptr inbounds i8, ptr %agg.tmp67, i64 16
  store i64 %5, ptr %agg.tmp6.sroa.3.0.agg.tmp67.sroa_idx, align 1
  %agg.tmp6.sroa.4.0.agg.tmp67.sroa_idx = getelementptr inbounds i8, ptr %agg.tmp67, i64 24
  store i64 %6, ptr %agg.tmp6.sroa.4.0.agg.tmp67.sroa_idx, align 1
  %7 = load i64, ptr addrspace(1) @__spirv_BuiltInLocalInvocationIndex, align 8
  tail call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 2, i32 2, i32 272)
  %cmpz15.i = icmp eq i64 %7, 0
  br i1 %cmpz15.i, label %leader.i, label %merge.i

leader.i:                                         ; preds = %entry
  call void @llvm.memcpy.p3.p0.i64(ptr addrspace(3) noundef align 16 dereferenceable(32) @ArgShadow.11, ptr noundef nonnull align 8 dereferenceable(32) %agg.tmp67, i64 32, i1 false)
  br label %merge.i

merge.i:                                          ; preds = %leader.i, %entry
  tail call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 2, i32 2, i32 272)
  call void @llvm.memcpy.p0.p3.i64(ptr noundef nonnull align 8 dereferenceable(32) %agg.tmp67, ptr addrspace(3) noundef align 16 dereferenceable(32) @ArgShadow.11, i64 32, i1 false)
  tail call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 2, i32 2, i32 272)
  br i1 %cmpz15.i, label %wg_leader.i, label %wg_cf.i

wg_leader.i:                                      ; preds = %merge.i
  %g.ascast.i = addrspacecast ptr %agg.tmp67 to ptr addrspace(4)
  store ptr addrspace(4) %g.ascast.i, ptr addrspace(3) @GAscast3, align 8
  store i32 0, ptr addrspace(3) @GCnt3, align 4
  br label %wg_cf.i

wg_cf.i:                                          ; preds = %wg_leader.i, %merge.i
  call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 2, i32 2, i32 272)
  %8 = load i64, ptr addrspace(1) @__spirv_BuiltInGlobalInvocationId, align 32
  %9 = load i64, ptr addrspace(1) @__spirv_BuiltInLocalInvocationId, align 32
  %cmp.i.i.i.i.i.i = icmp ult i64 %8, 2147483648
  br label %for.cond.i

for.cond.i:                                       ; preds = %wg_cf11.i, %wg_cf.i
  call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 2, i32 2, i32 272)
  br i1 %cmpz15.i, label %wg_leader4.i, label %wg_cf5.i

wg_leader4.i:                                     ; preds = %for.cond.i
  %10 = load i32, ptr addrspace(3) @GCnt3, align 4
  %cmp.i = icmp slt i32 %10, 2
  store i1 %cmp.i, ptr addrspace(3) @GCmp3, align 1
  br label %wg_cf5.i

wg_cf5.i:                                         ; preds = %wg_leader4.i, %for.cond.i
  call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 2, i32 2, i32 272)
  %wg_val_cmp.i = load i1, ptr addrspace(3) @GCmp3, align 1
  br i1 %wg_val_cmp.i, label %for.body.i, label %lexit6

for.body.i:                                       ; preds = %wg_cf5.i
  call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 2, i32 2, i32 272)
  call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 2, i32 2, i32 272)
  br i1 %cmpz15.i, label %TestMat.i, label %LeaderMat.i

TestMat.i:                                        ; preds = %for.body.i
  store i64 ptrtoint (ptr addrspace(4) addrspacecast (ptr addrspace(3) @GKernel2 to ptr addrspace(4)) to i64), ptr addrspace(3) @WGCopy.10.0, align 8
  store i64 5, ptr addrspace(3) @WGCopy.9.0, align 8
  br label %LeaderMat.i

LeaderMat.i:                                      ; preds = %TestMat.i, %for.body.i
  call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 2, i32 2, i32 272)
  %11 = load i64, ptr addrspace(3) @WGCopy.9.0, align 8
  %12 = load i64, ptr addrspace(3) @WGCopy.10.0, align 8
  %13 = inttoptr i64 %12 to ptr addrspace(4)
  call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 2, i32 2, i32 272)
  call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 noundef 2, i32 noundef 2, i32 noundef 272)
  %14 = getelementptr inbounds i8, ptr addrspace(4) %13, i64 24
  br label %for.cond.i.i

for.cond.i.i:                                     ; preds = %for.body.i.i, %LeaderMat.i
  %storemerge.i.i = phi i64 [ %9, %LeaderMat.i ], [ %add.i.i, %for.body.i.i ]
  %cmp.i.i = icmp ult i64 %storemerge.i.i, %11
  br i1 %cmp.i.i, label %for.body.i.i, label %lexit7

for.body.i.i:                                     ; preds = %for.cond.i.i
  %cmp5.not.i.i.i.i.i.i = icmp ne i64 %storemerge.i.i, %9
  %cond.i.i.i.i = zext i1 %cmp5.not.i.i.i.i.i.i to i32
  call void @llvm.assume(i1 %cmp.i.i.i.i.i.i)
  %15 = load ptr addrspace(1), ptr addrspace(4) %14, align 8
  %arrayidx.i.i.i.i.i = getelementptr inbounds i32, ptr addrspace(1) %15, i64 %8
  %16 = load i32, ptr addrspace(1) %arrayidx.i.i.i.i.i, align 4
  %add.i.i.i.i = add nsw i32 %16, %cond.i.i.i.i
  store i32 %add.i.i.i.i, ptr addrspace(1) %arrayidx.i.i.i.i.i, align 4
  %add.i.i = add i64 %storemerge.i.i, %4
  br label %for.cond.i.i

lexit7: ; preds = %for.cond.i.i
  call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 noundef 2, i32 noundef 2, i32 noundef 272)
  call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 2, i32 2, i32 272)
  br i1 %cmpz15.i, label %wg_leader10.i, label %wg_cf11.i

wg_leader10.i:                                    ; preds = %lexit7
  %17 = load i32, ptr addrspace(3) @GCnt3, align 4
  %inc.i = add nsw i32 %17, 1
  store i32 %inc.i, ptr addrspace(3) @GCnt3, align 4
  br label %wg_cf11.i

wg_cf11.i:                                        ; preds = %wg_leader10.i, %lexit7
  call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 2, i32 2, i32 272)
  br label %for.cond.i

lexit6: ; preds = %wg_cf5.i
  call void @llvm.lifetime.end.p0(i64 32, ptr nonnull %agg.tmp67)
  ret void
}

; Function Attrs: convergent mustprogress norecurse nounwind
define weak_odr dso_local spir_kernel void @test2(ptr addrspace(1) noundef align 4 %_arg_dev_ptr, ptr noundef byval(%"class.sycl::_V1::range") align 8 %_arg_dev_ptr1, ptr noundef byval(%"class.sycl::_V1::range") align 8 %_arg_dev_ptr2, ptr noundef byval(%"class.sycl::_V1::range") align 8 %_arg_dev_ptr3) {
entry:
  %priv.i = alloca %"class.sycl::_V1::private_memory", align 4
  %agg.tmp67 = alloca %"class.sycl::_V1::group.15", align 8
  %0 = load i64, ptr %_arg_dev_ptr1, align 8
  %1 = load i64, ptr %_arg_dev_ptr2, align 8
  %2 = load i64, ptr %_arg_dev_ptr3, align 8
  store i64 %2, ptr addrspace(3) @GKernel3, align 8
  store i64 %0, ptr addrspace(3) undef, align 8
  store i64 %1, ptr addrspace(3) undef, align 8
  %add.ptr.i = getelementptr inbounds i32, ptr addrspace(1) %_arg_dev_ptr, i64 %2
  store ptr addrspace(1) %add.ptr.i, ptr addrspace(3) undef, align 8
  %3 = load i64, ptr addrspace(1) undef, align 8
  %4 = load i64, ptr addrspace(1) @__spirv_BuiltInGlobalSize, align 32
  %5 = load i64, ptr addrspace(1) undef, align 8
  %6 = load i64, ptr addrspace(1) @__spirv_BuiltInWorkgroupSize, align 32
  %7 = load i64, ptr addrspace(1) undef, align 8
  %8 = load i64, ptr addrspace(1) @__spirv_BuiltInNumWorkgroups, align 32
  %9 = load i64, ptr addrspace(1) undef, align 8
  %10 = load i64, ptr addrspace(1) @__spirv_BuiltInWorkgroupId, align 32
  call void @llvm.lifetime.start.p0(i64 64, ptr nonnull %agg.tmp67)
  store i64 %3, ptr %agg.tmp67, align 1
  %agg.tmp6.sroa.2.0.agg.tmp67.sroa_idx = getelementptr inbounds i8, ptr %agg.tmp67, i64 8
  store i64 %4, ptr %agg.tmp6.sroa.2.0.agg.tmp67.sroa_idx, align 1
  %agg.tmp6.sroa.3.0.agg.tmp67.sroa_idx = getelementptr inbounds i8, ptr %agg.tmp67, i64 16
  store i64 %5, ptr %agg.tmp6.sroa.3.0.agg.tmp67.sroa_idx, align 1
  %agg.tmp6.sroa.4.0.agg.tmp67.sroa_idx = getelementptr inbounds i8, ptr %agg.tmp67, i64 24
  store i64 %6, ptr %agg.tmp6.sroa.4.0.agg.tmp67.sroa_idx, align 1
  %agg.tmp6.sroa.5.0.agg.tmp67.sroa_idx = getelementptr inbounds i8, ptr %agg.tmp67, i64 32
  store i64 %7, ptr %agg.tmp6.sroa.5.0.agg.tmp67.sroa_idx, align 1
  %agg.tmp6.sroa.6.0.agg.tmp67.sroa_idx = getelementptr inbounds i8, ptr %agg.tmp67, i64 40
  store i64 %8, ptr %agg.tmp6.sroa.6.0.agg.tmp67.sroa_idx, align 1
  %agg.tmp6.sroa.7.0.agg.tmp67.sroa_idx = getelementptr inbounds i8, ptr %agg.tmp67, i64 48
  store i64 %9, ptr %agg.tmp6.sroa.7.0.agg.tmp67.sroa_idx, align 1
  %agg.tmp6.sroa.8.0.agg.tmp67.sroa_idx = getelementptr inbounds i8, ptr %agg.tmp67, i64 56
  store i64 %10, ptr %agg.tmp6.sroa.8.0.agg.tmp67.sroa_idx, align 1
  %11 = load i64, ptr addrspace(1) @__spirv_BuiltInLocalInvocationIndex, align 8
  tail call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 2, i32 2, i32 272)
  %cmpz32.i = icmp eq i64 %11, 0
  br i1 %cmpz32.i, label %leader.i, label %merge.i

leader.i:                                         ; preds = %entry
  call void @llvm.memcpy.p3.p0.i64(ptr addrspace(3) noundef align 16 dereferenceable(64) @ArgShadow.17, ptr noundef nonnull align 8 dereferenceable(64) %agg.tmp67, i64 64, i1 false)
  br label %merge.i

merge.i:                                          ; preds = %leader.i, %entry
  tail call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 2, i32 2, i32 272)
  call void @llvm.memcpy.p0.p3.i64(ptr noundef nonnull align 8 dereferenceable(64) %agg.tmp67, ptr addrspace(3) noundef align 16 dereferenceable(64) @ArgShadow.17, i64 64, i1 false)
  %priv.ascast.i = addrspacecast ptr %priv.i to ptr addrspace(4)
  tail call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 2, i32 2, i32 272)
  br i1 %cmpz32.i, label %wg_leader.i, label %wg_cf.i

wg_leader.i:                                      ; preds = %merge.i
  %g.ascast.i = addrspacecast ptr %agg.tmp67 to ptr addrspace(4)
  store ptr addrspace(4) %g.ascast.i, ptr addrspace(3) @GAsCast4, align 8
  call void @llvm.lifetime.start.p0(i64 8, ptr nonnull %priv.i)
  store i32 0, ptr addrspace(3) @GCnt4, align 4
  br label %wg_cf.i

wg_cf.i:                                          ; preds = %wg_leader.i, %merge.i
  call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 2, i32 2, i32 272)
  %12 = load i64, ptr addrspace(1) undef, align 8
  %13 = load i64, ptr addrspace(1) @__spirv_BuiltInGlobalInvocationId, align 32
  %14 = load i64, ptr addrspace(1) undef, align 8
  %15 = load i64, ptr addrspace(1) @__spirv_BuiltInLocalInvocationId, align 32
  %mul.i.i.i.i.i.i = mul i64 %12, %4
  %add.i.i.i.i.i.i = add i64 %mul.i.i.i.i.i.i, %13
  %cmp.i.i.i.i.i.i = icmp ult i64 %add.i.i.i.i.i.i, 2147483648
  %conv.i.i.i.i.i = trunc i64 %add.i.i.i.i.i.i to i32
  %y.i.i.i.i.i = getelementptr inbounds i8, ptr %priv.i, i64 4
  br label %for.cond.i

for.cond.i:                                       ; preds = %wg_cf20.i, %wg_cf.i
  %agg.tmp6.i.sroa.9.0 = phi ptr addrspace(4) [ undef, %wg_cf.i ], [ %agg.tmp6.i.sroa.9.0.copyload40, %wg_cf20.i ]
  %agg.tmp5.i.sroa.0.0 = phi i64 [ undef, %wg_cf.i ], [ %agg.tmp5.i.sroa.0.0.copyload44, %wg_cf20.i ]
  %agg.tmp5.i.sroa.8.0 = phi i64 [ undef, %wg_cf.i ], [ %agg.tmp5.i.sroa.8.0.copyload48, %wg_cf20.i ]
  %agg.tmp2.i.sroa.0.0 = phi ptr addrspace(4) [ undef, %wg_cf.i ], [ %agg.tmp2.i.sroa.0.0.copyload52, %wg_cf20.i ]
  %agg.tmp2.i.sroa.8.0 = phi ptr addrspace(4) [ undef, %wg_cf.i ], [ %agg.tmp2.i.sroa.8.0.copyload56, %wg_cf20.i ]
  %agg.tmp.i.sroa.0.0 = phi i64 [ undef, %wg_cf.i ], [ %agg.tmp.i.sroa.0.0.copyload60, %wg_cf20.i ]
  %agg.tmp.i.sroa.8.0 = phi i64 [ undef, %wg_cf.i ], [ %agg.tmp.i.sroa.8.0.copyload64, %wg_cf20.i ]
  call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 2, i32 2, i32 272)
  br i1 %cmpz32.i, label %wg_leader10.i, label %wg_cf11.i

wg_leader10.i:                                    ; preds = %for.cond.i
  %16 = load i32, ptr addrspace(3) @GCnt4, align 4
  %cmp.i = icmp slt i32 %16, 2
  store i1 %cmp.i, ptr addrspace(3) @GCmp4, align 1
  br label %wg_cf11.i

wg_cf11.i:                                        ; preds = %wg_leader10.i, %for.cond.i
  call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 2, i32 2, i32 272)
  %wg_val_cmp.i = load i1, ptr addrspace(3) @GCmp4, align 1
  br i1 %wg_val_cmp.i, label %for.body.i, label %for.end.i

for.body.i:                                       ; preds = %wg_cf11.i
  call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 2, i32 2, i32 272)
  br i1 %cmpz32.i, label %wg_leader13.i, label %wg_cf14.i

wg_leader13.i:                                    ; preds = %for.body.i
  br label %wg_cf14.i

wg_cf14.i:                                        ; preds = %wg_leader13.i, %for.body.i
  %agg.tmp2.i.sroa.0.1 = phi ptr addrspace(4) [ addrspacecast (ptr addrspace(3) @GKernel3 to ptr addrspace(4)), %wg_leader13.i ], [ %agg.tmp2.i.sroa.0.0, %for.body.i ]
  %agg.tmp2.i.sroa.8.1 = phi ptr addrspace(4) [ %priv.ascast.i, %wg_leader13.i ], [ %agg.tmp2.i.sroa.8.0, %for.body.i ]
  %agg.tmp.i.sroa.0.1 = phi i64 [ 7, %wg_leader13.i ], [ %agg.tmp.i.sroa.0.0, %for.body.i ]
  %agg.tmp.i.sroa.8.1 = phi i64 [ 3, %wg_leader13.i ], [ %agg.tmp.i.sroa.8.0, %for.body.i ]
  call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 2, i32 2, i32 272)
  br i1 %cmpz32.i, label %TestMat30.i, label %LeaderMat27.i

TestMat30.i:                                      ; preds = %wg_cf14.i
  store i64 %agg.tmp.i.sroa.0.1, ptr addrspace(3) @WGCopy.13.0, align 8
  store i64 %agg.tmp.i.sroa.8.1, ptr addrspace(3) @WGCopy.13.1, align 8
  store ptr addrspace(4) %agg.tmp2.i.sroa.0.1, ptr addrspace(3) @WGCopy.14.0, align 8
  store ptr addrspace(4) %agg.tmp2.i.sroa.8.1, ptr addrspace(3) @WGCopy.14.1, align 8
  store i64 %agg.tmp5.i.sroa.0.0, ptr addrspace(3) @WGCopy.15.0, align 8
  store i64 %agg.tmp5.i.sroa.8.0, ptr addrspace(3) @WGCopy.15.1, align 8
  store ptr addrspace(4) %priv.ascast.i, ptr addrspace(3) @WGCopy.16.0, align 8
  store ptr addrspace(4) %agg.tmp6.i.sroa.9.0, ptr addrspace(3) @WGCopy.16.1, align 8
  br label %LeaderMat27.i

LeaderMat27.i:                                    ; preds = %TestMat30.i, %wg_cf14.i
  call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 2, i32 2, i32 272)
  %agg.tmp6.i.sroa.0.0.copyload = load ptr addrspace(4), ptr addrspace(3) @WGCopy.16.0, align 8
  %agg.tmp6.i.sroa.9.0.copyload = load ptr addrspace(4), ptr addrspace(3) @WGCopy.16.1, align 8
  %agg.tmp5.i.sroa.0.0.copyload = load i64, ptr addrspace(3) @WGCopy.15.0, align 8
  %agg.tmp5.i.sroa.8.0.copyload = load i64, ptr addrspace(3) @WGCopy.15.1, align 8
  %agg.tmp2.i.sroa.0.0.copyload = load ptr addrspace(4), ptr addrspace(3) @WGCopy.14.0, align 8
  %agg.tmp.i.sroa.0.0.copyload = load i64, ptr addrspace(3) @WGCopy.13.0, align 8
  %agg.tmp.i.sroa.8.0.copyload = load i64, ptr addrspace(3) @WGCopy.13.1, align 8
  call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 2, i32 2, i32 272)
  call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 noundef 2, i32 noundef 2, i32 noundef 272)
  %17 = getelementptr inbounds i8, ptr addrspace(4) %agg.tmp2.i.sroa.0.0.copyload, i64 24
  br label %for.cond.i.i

for.cond.i.i:                                     ; preds = %lexit10, %LeaderMat27.i
  %storemerge.i.i = phi i64 [ %14, %LeaderMat27.i ], [ %add.i.i, %lexit10 ]
  %cmp.i.i = icmp ult i64 %storemerge.i.i, %agg.tmp.i.sroa.0.0.copyload
  br i1 %cmp.i.i, label %for.cond.i.i.i, label %lexit11

for.cond.i.i.i:                                   ; preds = %for.body.i.i.i, %for.cond.i.i
  %storemerge.i.i.i = phi i64 [ %add.i.i.i, %for.body.i.i.i ], [ %15, %for.cond.i.i ]
  %cmp.i.i.i = icmp ult i64 %storemerge.i.i.i, %agg.tmp.i.sroa.8.0.copyload
  br i1 %cmp.i.i.i, label %for.body.i.i.i, label %lexit10

for.body.i.i.i:                                   ; preds = %for.cond.i.i.i
  call void @llvm.assume(i1 %cmp.i.i.i.i.i.i)
  %18 = load ptr addrspace(1), ptr addrspace(4) %17, align 8
  %arrayidx.i.i.i.i.i.i = getelementptr inbounds i32, ptr addrspace(1) %18, i64 %add.i.i.i.i.i.i
  %19 = load i32, ptr addrspace(1) %arrayidx.i.i.i.i.i.i, align 4
  %inc.i.i.i.i.i = add nsw i32 %19, 1
  store i32 %inc.i.i.i.i.i, ptr addrspace(1) %arrayidx.i.i.i.i.i.i, align 4
  store i32 %conv.i.i.i.i.i, ptr %priv.i, align 4
  store i32 5, ptr %y.i.i.i.i.i, align 4
  %add.i.i.i = add i64 %storemerge.i.i.i, %6
  br label %for.cond.i.i.i

lexit10: ; preds = %for.cond.i.i.i
  %add.i.i = add i64 %storemerge.i.i, %5
  br label %for.cond.i.i

lexit11: ; preds = %for.cond.i.i
  call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 noundef 2, i32 noundef 2, i32 noundef 272)
  call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 2, i32 2, i32 272)
  br i1 %cmpz32.i, label %wg_leader16.i, label %wg_cf17.i

wg_leader16.i:                                    ; preds = %lexit11
  br label %wg_cf17.i

wg_cf17.i:                                        ; preds = %wg_leader16.i, %lexit11
  %agg.tmp6.i.sroa.0.1 = phi ptr addrspace(4) [ %priv.ascast.i, %wg_leader16.i ], [ %agg.tmp6.i.sroa.0.0.copyload, %lexit11 ]
  %agg.tmp6.i.sroa.9.1 = phi ptr addrspace(4) [ addrspacecast (ptr addrspace(3) @GKernel3 to ptr addrspace(4)), %wg_leader16.i ], [ %agg.tmp6.i.sroa.9.0.copyload, %lexit11 ]
  %agg.tmp5.i.sroa.0.1 = phi i64 [ 7, %wg_leader16.i ], [ %agg.tmp5.i.sroa.0.0.copyload, %lexit11 ]
  %agg.tmp5.i.sroa.8.1 = phi i64 [ 3, %wg_leader16.i ], [ %agg.tmp5.i.sroa.8.0.copyload, %lexit11 ]
  call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 2, i32 2, i32 272)
  br i1 %cmpz32.i, label %TestMat.i, label %LeaderMat.i

TestMat.i:                                        ; preds = %wg_cf17.i
  store i64 %agg.tmp.i.sroa.0.0.copyload, ptr addrspace(3) @WGCopy.13.0, align 8
  store i64 %agg.tmp.i.sroa.8.0.copyload, ptr addrspace(3) @WGCopy.13.1, align 8
  store ptr addrspace(4) %agg.tmp2.i.sroa.0.0.copyload, ptr addrspace(3) @WGCopy.14.0, align 8
  store ptr addrspace(4) %priv.ascast.i, ptr addrspace(3) @WGCopy.14.1, align 8
  store i64 %agg.tmp5.i.sroa.0.1, ptr addrspace(3) @WGCopy.15.0, align 8
  store i64 %agg.tmp5.i.sroa.8.1, ptr addrspace(3) @WGCopy.15.1, align 8
  store ptr addrspace(4) %agg.tmp6.i.sroa.0.1, ptr addrspace(3) @WGCopy.16.0, align 8
  store ptr addrspace(4) %agg.tmp6.i.sroa.9.1, ptr addrspace(3) @WGCopy.16.1, align 8
  br label %LeaderMat.i

LeaderMat.i:                                      ; preds = %TestMat.i, %wg_cf17.i
  call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 2, i32 2, i32 272)
  %agg.tmp6.i.sroa.9.0.copyload40 = load ptr addrspace(4), ptr addrspace(3) @WGCopy.16.1, align 8
  %agg.tmp5.i.sroa.0.0.copyload44 = load i64, ptr addrspace(3) @WGCopy.15.0, align 8
  %agg.tmp5.i.sroa.8.0.copyload48 = load i64, ptr addrspace(3) @WGCopy.15.1, align 8
  %agg.tmp2.i.sroa.0.0.copyload52 = load ptr addrspace(4), ptr addrspace(3) @WGCopy.14.0, align 8
  %agg.tmp2.i.sroa.8.0.copyload56 = load ptr addrspace(4), ptr addrspace(3) @WGCopy.14.1, align 8
  %agg.tmp.i.sroa.0.0.copyload60 = load i64, ptr addrspace(3) @WGCopy.13.0, align 8
  %agg.tmp.i.sroa.8.0.copyload64 = load i64, ptr addrspace(3) @WGCopy.13.1, align 8
  call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 2, i32 2, i32 272)
  call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 noundef 2, i32 noundef 2, i32 noundef 272)
  %20 = getelementptr inbounds i8, ptr addrspace(4) %agg.tmp6.i.sroa.9.0.copyload40, i64 24
  br label %for.cond.i.i25

for.cond.i.i25:                                   ; preds = %lexit12, %LeaderMat.i
  %storemerge.i.i26 = phi i64 [ %14, %LeaderMat.i ], [ %add.i.i31, %lexit12 ]
  %cmp.i.i27 = icmp ult i64 %storemerge.i.i26, %agg.tmp5.i.sroa.0.0.copyload44
  br i1 %cmp.i.i27, label %for.cond.i.i.i28, label %lexit13

for.cond.i.i.i28:                                 ; preds = %for.body.i.i.i32, %for.cond.i.i25
  %storemerge.i.i.i29 = phi i64 [ %add.i.i.i35, %for.body.i.i.i32 ], [ %15, %for.cond.i.i25 ]
  %cmp.i.i.i30 = icmp ult i64 %storemerge.i.i.i29, %agg.tmp5.i.sroa.8.0.copyload48
  br i1 %cmp.i.i.i30, label %for.body.i.i.i32, label %lexit12

for.body.i.i.i32:                                 ; preds = %for.cond.i.i.i28
  %21 = load i32, ptr %priv.i, align 4
  %22 = load i32, ptr %y.i.i.i.i.i, align 4
  %add.i.i.i.i.i = add nsw i32 %21, %22
  call void @llvm.assume(i1 %cmp.i.i.i.i.i.i)
  %23 = load ptr addrspace(1), ptr addrspace(4) %20, align 8
  %arrayidx.i.i.i.i.i.i33 = getelementptr inbounds i32, ptr addrspace(1) %23, i64 %add.i.i.i.i.i.i
  %24 = load i32, ptr addrspace(1) %arrayidx.i.i.i.i.i.i33, align 4
  %add4.i.i.i.i.i = add nsw i32 %24, %add.i.i.i.i.i
  store i32 %add4.i.i.i.i.i, ptr addrspace(1) %arrayidx.i.i.i.i.i.i33, align 4
  %add.i.i.i35 = add i64 %storemerge.i.i.i29, %6
  br label %for.cond.i.i.i28

lexit12: ; preds = %for.cond.i.i.i28
  %add.i.i31 = add i64 %storemerge.i.i26, %5
  br label %for.cond.i.i25

lexit13: ; preds = %for.cond.i.i25
  call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 noundef 2, i32 noundef 2, i32 noundef 272)
  call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 2, i32 2, i32 272)
  br i1 %cmpz32.i, label %wg_leader19.i, label %wg_cf20.i

wg_leader19.i:                                    ; preds = %lexit13
  %25 = load i32, ptr addrspace(3) @GCnt4, align 4
  %inc.i = add nsw i32 %25, 1
  store i32 %inc.i, ptr addrspace(3) @GCnt4, align 4
  br label %wg_cf20.i

wg_cf20.i:                                        ; preds = %wg_leader19.i, %lexit13
  call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 2, i32 2, i32 272)
  br label %for.cond.i

for.end.i:                                        ; preds = %wg_cf11.i
  call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 2, i32 2, i32 272)
  br i1 %cmpz32.i, label %wg_leader22.i, label %lexit14

wg_leader22.i:                                    ; preds = %for.end.i
  call void @llvm.lifetime.end.p0(i64 8, ptr nonnull %priv.i)
  br label %lexit14

lexit14: ; preds = %wg_leader22.i, %for.end.i
  call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 2, i32 2, i32 272)
  call void @llvm.lifetime.end.p0(i64 64, ptr nonnull %agg.tmp67)
  ret void
}

; Function Attrs: convergent mustprogress norecurse nounwind
define weak_odr dso_local spir_kernel void @test3(ptr addrspace(1) noundef align 4 %_arg_dev_ptr, ptr noundef byval(%"class.sycl::_V1::range") align 8 %_arg_dev_ptr1, ptr noundef byval(%"class.sycl::_V1::range") align 8 %_arg_dev_ptr2, ptr noundef byval(%"class.sycl::_V1::range") align 8 %_arg_dev_ptr3) {
entry:
  %agg.tmp67 = alloca %"class.sycl::_V1::group", align 8
  %0 = load i64, ptr %_arg_dev_ptr1, align 8
  %1 = load i64, ptr %_arg_dev_ptr2, align 8
  %2 = load i64, ptr %_arg_dev_ptr3, align 8
  store i64 %2, ptr addrspace(3) @GKernel4, align 8
  store i64 %0, ptr addrspace(3) undef, align 8
  store i64 %1, ptr addrspace(3) undef, align 8
  %add.ptr.i = getelementptr inbounds i32, ptr addrspace(1) %_arg_dev_ptr, i64 %2
  store ptr addrspace(1) %add.ptr.i, ptr addrspace(3) undef, align 8
  %3 = load i64, ptr addrspace(1) @__spirv_BuiltInGlobalSize, align 32
  %4 = load i64, ptr addrspace(1) @__spirv_BuiltInWorkgroupSize, align 32
  %5 = load i64, ptr addrspace(1) @__spirv_BuiltInNumWorkgroups, align 32
  %6 = load i64, ptr addrspace(1) @__spirv_BuiltInWorkgroupId, align 32
  call void @llvm.lifetime.start.p0(i64 32, ptr nonnull %agg.tmp67)
  store i64 %3, ptr %agg.tmp67, align 1
  %agg.tmp6.sroa.2.0.agg.tmp67.sroa_idx = getelementptr inbounds i8, ptr %agg.tmp67, i64 8
  store i64 %4, ptr %agg.tmp6.sroa.2.0.agg.tmp67.sroa_idx, align 1
  %agg.tmp6.sroa.3.0.agg.tmp67.sroa_idx = getelementptr inbounds i8, ptr %agg.tmp67, i64 16
  store i64 %5, ptr %agg.tmp6.sroa.3.0.agg.tmp67.sroa_idx, align 1
  %agg.tmp6.sroa.4.0.agg.tmp67.sroa_idx = getelementptr inbounds i8, ptr %agg.tmp67, i64 24
  store i64 %6, ptr %agg.tmp6.sroa.4.0.agg.tmp67.sroa_idx, align 1
  %7 = load i64, ptr addrspace(1) @__spirv_BuiltInLocalInvocationIndex, align 8
  tail call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 2, i32 2, i32 272)
  %cmpz16.i = icmp eq i64 %7, 0
  br i1 %cmpz16.i, label %leader.i, label %merge.i

leader.i:                                         ; preds = %entry
  call void @llvm.memcpy.p3.p0.i64(ptr addrspace(3) noundef align 16 dereferenceable(32) @ArgShadow.21, ptr noundef nonnull align 8 dereferenceable(32) %agg.tmp67, i64 32, i1 false)
  br label %merge.i

merge.i:                                          ; preds = %leader.i, %entry
  tail call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 2, i32 2, i32 272)
  call void @llvm.memcpy.p0.p3.i64(ptr noundef nonnull align 8 dereferenceable(32) %agg.tmp67, ptr addrspace(3) noundef align 16 dereferenceable(32) @ArgShadow.21, i64 32, i1 false)
  tail call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 2, i32 2, i32 272)
  br i1 %cmpz16.i, label %wg_leader.i, label %wg_cf.i

wg_leader.i:                                      ; preds = %merge.i
  %g.ascast.i = addrspacecast ptr %agg.tmp67 to ptr addrspace(4)
  store ptr addrspace(4) %g.ascast.i, ptr addrspace(3) @GAsCast5, align 8
  store i32 0, ptr addrspace(3) @GCnt5, align 4
  br label %wg_cf.i

wg_cf.i:                                          ; preds = %wg_leader.i, %merge.i
  call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 2, i32 2, i32 272)
  %wg_val_g.ascast.i = load ptr addrspace(4), ptr addrspace(3) @GAsCast5, align 8
  %8 = load i64, ptr addrspace(1) @__spirv_BuiltInLocalInvocationId, align 32
  %9 = trunc i64 %4 to i32
  br label %for.cond.i

for.cond.i:                                       ; preds = %wg_cf12.i, %wg_cf.i
  call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 2, i32 2, i32 272)
  br i1 %cmpz16.i, label %wg_leader5.i, label %wg_cf6.i

wg_leader5.i:                                     ; preds = %for.cond.i
  %10 = load i32, ptr addrspace(3) @GCnt5, align 4
  %cmp.i = icmp slt i32 %10, 2
  store i1 %cmp.i, ptr addrspace(3) @GCmp5, align 1
  br label %wg_cf6.i

wg_cf6.i:                                         ; preds = %wg_leader5.i, %for.cond.i
  call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 2, i32 2, i32 272)
  %wg_val_cmp.i = load i1, ptr addrspace(3) @GCmp5, align 1
  br i1 %wg_val_cmp.i, label %for.body.i, label %lexit20

for.body.i:                                       ; preds = %wg_cf6.i
  call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 2, i32 2, i32 272)
  call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 2, i32 2, i32 272)
  br i1 %cmpz16.i, label %TestMat.i, label %LeaderMat.i

TestMat.i:                                        ; preds = %for.body.i
  store ptr addrspace(4) %wg_val_g.ascast.i, ptr addrspace(3) @WGCopy.20.0, align 8
  store ptr addrspace(4) addrspacecast (ptr addrspace(3) @GKernel4 to ptr addrspace(4)), ptr addrspace(3) @WGCopy.20.1, align 8
  store i64 5, ptr addrspace(3) @WGCopy.19.0, align 8
  br label %LeaderMat.i

LeaderMat.i:                                      ; preds = %TestMat.i, %for.body.i
  call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 2, i32 2, i32 272)
  %11 = load i64, ptr addrspace(3) @WGCopy.19.0, align 8
  %agg.tmp2.i.sroa.0.0.copyload = load ptr addrspace(4), ptr addrspace(3) @WGCopy.20.0, align 8
  %agg.tmp2.i.sroa.6.0.copyload = load ptr addrspace(4), ptr addrspace(3) @WGCopy.20.1, align 8
  call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 2, i32 2, i32 272)
  call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 noundef 2, i32 noundef 2, i32 noundef 272)
  %index.i.i.i.i.i = getelementptr inbounds i8, ptr addrspace(4) %agg.tmp2.i.sroa.0.0.copyload, i64 24
  %12 = getelementptr inbounds i8, ptr addrspace(4) %agg.tmp2.i.sroa.6.0.copyload, i64 24
  %13 = trunc i64 %11 to i32
  br label %for.cond.i.i

for.cond.i.i:                                     ; preds = %for.body.i.i, %LeaderMat.i
  %storemerge.i.i = phi i64 [ %8, %LeaderMat.i ], [ %add.i.i, %for.body.i.i ]
  %cmp.i.i = icmp ult i64 %storemerge.i.i, %11
  br i1 %cmp.i.i, label %for.body.i.i, label %lexit21

for.body.i.i:                                     ; preds = %for.cond.i.i
  %14 = load i64, ptr addrspace(4) %index.i.i.i.i.i, align 8
  %mul.i.i.i.i = mul i64 %14, 10
  %mul3.i.i.i.i = shl i64 %storemerge.i.i, 1
  %add.i.i.i.i = add i64 %mul.i.i.i.i, %mul3.i.i.i.i
  %15 = load ptr addrspace(1), ptr addrspace(4) %12, align 8
  %arrayidx.i.i.i.i.i = getelementptr inbounds i32, ptr addrspace(1) %15, i64 %add.i.i.i.i
  %16 = load i32, ptr addrspace(1) %arrayidx.i.i.i.i.i, align 4
  %conv9.i.i.i.i = add i32 %16, %13
  store i32 %conv9.i.i.i.i, ptr addrspace(1) %arrayidx.i.i.i.i.i, align 4
  %add14.i.i.i.i = or disjoint i64 %add.i.i.i.i, 1
  %17 = load ptr addrspace(1), ptr addrspace(4) %12, align 8
  %arrayidx.i25.i.i.i.i = getelementptr inbounds i32, ptr addrspace(1) %17, i64 %add14.i.i.i.i
  %18 = load i32, ptr addrspace(1) %arrayidx.i25.i.i.i.i, align 4
  %conv18.i.i.i.i = add i32 %18, %9
  store i32 %conv18.i.i.i.i, ptr addrspace(1) %arrayidx.i25.i.i.i.i, align 4
  %add.i.i = add i64 %storemerge.i.i, %4
  br label %for.cond.i.i

lexit21: ; preds = %for.cond.i.i
  call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 noundef 2, i32 noundef 2, i32 noundef 272)
  call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 2, i32 2, i32 272)
  br i1 %cmpz16.i, label %wg_leader11.i, label %wg_cf12.i

wg_leader11.i:                                    ; preds = %lexit21
  %19 = load i32, ptr addrspace(3) @GCnt5, align 4
  %inc.i = add nsw i32 %19, 1
  store i32 %inc.i, ptr addrspace(3) @GCnt5, align 4
  br label %wg_cf12.i

wg_cf12.i:                                        ; preds = %wg_leader11.i, %lexit21
  call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 2, i32 2, i32 272)
  br label %for.cond.i

lexit20: ; preds = %wg_cf6.i
  call void @llvm.lifetime.end.p0(i64 32, ptr nonnull %agg.tmp67)
  ret void
}
