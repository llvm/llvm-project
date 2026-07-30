; The goal of the test case is to ensure that correct types are applied to virtual registers which were
; used as return values in call lowering. Pass criterion is that spirv-val considers output valid.

; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

%t_half = type { half }
%t_i17 = type { [17 x i32] }
%t_h17 = type { [17 x %t_half] }

define internal spir_func void @foo(i64 %arrayinit.cur.add_4, half %r1, ptr addrspace(4) noundef align 8 dereferenceable_or_null(72) %this) {
entry:
  %r_3 = alloca %t_h17, align 8
  %p_src = alloca %t_i17, align 4
  %p_src4 = addrspacecast ptr %p_src to ptr addrspace(4)
  %call_2 = call spir_func noundef ptr @_Z42__spirv_GenericCastToPtrExplicit_ToPrivatePvi(ptr addrspace(4) noundef %p_src4, i32 noundef 7)
  br label %l_body

l_body:                                           ; preds = %l_body, %entry
  %l_done = icmp eq i64 %arrayinit.cur.add_4, 34
  br i1 %l_done, label %exit, label %l_body

exit:                                             ; preds = %l_body
  %0 = addrspacecast ptr %call_2 to ptr addrspace(4)
  %call_6 = call spir_func noundef ptr @_Z42__spirv_GenericCastToPtrExplicit_ToPrivatePvi(ptr addrspace(4) noundef %0, i32 noundef 7)
  br label %for.cond_3

for.cond_3:                                       ; preds = %for.body_3, %exit
  %lsr.iv1 = phi ptr [ %scevgep2, %for.body_3 ], [ %call_6, %exit ]
  %lsr.iv = phi ptr [ %scevgep, %for.body_3 ], [ %r_3, %exit ]
  %i.0_3 = phi i64 [ 0, %exit ], [ %inc_3, %for.body_3 ]
  %cmp_3 = icmp ult i64 %i.0_3, 17
  br i1 %cmp_3, label %for.body_3, label %exit2

for.body_3:                                       ; preds = %for.cond_3
  %call2_5 = call spir_func noundef half @_Z17__spirv_ocl_frexpDF16_PU3AS0i(half noundef %r1, ptr noundef %lsr.iv1)
  store half %call2_5, ptr %lsr.iv, align 2
  %inc_3 = add nuw nsw i64 %i.0_3, 1
  %scevgep = getelementptr i8, ptr %lsr.iv, i64 2
  %scevgep2 = getelementptr i8, ptr %lsr.iv1, i64 4
  br label %for.cond_3

exit2:                                            ; preds = %for.cond_3
  ret void
}

declare dso_local spir_func noundef ptr @_Z42__spirv_GenericCastToPtrExplicit_ToPrivatePvi(ptr addrspace(4) noundef, i32 noundef)
declare dso_local spir_func noundef half @_Z17__spirv_ocl_frexpDF16_PU3AS0i(half noundef, ptr noundef)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg)
declare void @llvm.memcpy.p1.p0.i64(ptr addrspace(1) noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg)
