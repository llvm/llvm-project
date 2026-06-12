; The goal of the test case is to ensure that correct types are applied to virtual registers
; which were used as arguments in call lowering and so caused early definition of SPIR-V types.

; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

%t_id = type { %t_arr }
%t_arr = type { [1 x i64] }
%t_bf16 = type { i16 }

define weak_odr dso_local spir_kernel void @foo(ptr addrspace(1) align 4 %_arg_ERR, ptr byval(%t_id) align 8 %_arg_ERR3) {
entry:
  %FloatArray.i = alloca [4 x float], align 4
  %BF16Array.i = alloca [4 x %t_bf16], align 2
  %0 = load i64, ptr %_arg_ERR3, align 8
  %add.ptr.i = getelementptr inbounds i32, ptr addrspace(1) %_arg_ERR, i64 %0
  %FloatArray.ascast.i = addrspacecast ptr %FloatArray.i to ptr addrspace(4)
  %BF16Array.ascast.i = addrspacecast ptr %BF16Array.i to ptr addrspace(4)
  call spir_func void @__devicelib_ConvertFToBF16INTELVec4(ptr addrspace(4) %FloatArray.ascast.i, ptr addrspace(4) %BF16Array.ascast.i)
  br label %for.cond.i

for.cond.i:                                       ; preds = %for.inc.i, %entry
  %lsr.iv1 = phi ptr [ %scevgep2, %for.inc.i ], [ %FloatArray.i, %entry ]
  %lsr.iv = phi ptr addrspace(4) [ %scevgep, %for.inc.i ], [ %BF16Array.ascast.i, %entry ]
  %i.0.i = phi i32 [ 0, %entry ], [ %inc.i, %for.inc.i ]
  %cmp.i = icmp ult i32 %i.0.i, 4
  br i1 %cmp.i, label %for.body.i, label %exit

for.body.i:                                       ; preds = %for.cond.i
  %1 = load float, ptr %lsr.iv1, align 4
  %call.i.i = call spir_func float @__devicelib_ConvertBF16ToFINTEL(ptr addrspace(4) align 2 dereferenceable(2) %lsr.iv)
  %cmp5.i = fcmp une float %1, %call.i.i
  br i1 %cmp5.i, label %if.then.i, label %for.inc.i

if.then.i:                                        ; preds = %for.body.i
  store i32 1, ptr addrspace(1) %add.ptr.i, align 4
  br label %for.inc.i

for.inc.i:                                        ; preds = %if.then.i, %for.body.i
  %inc.i = add nuw nsw i32 %i.0.i, 1
  %scevgep = getelementptr i8, ptr addrspace(4) %lsr.iv, i64 2
  %scevgep2 = getelementptr i8, ptr %lsr.iv1, i64 4
  br label %for.cond.i

exit:                                             ; preds = %for.cond.i
  ret void
}

declare void @llvm.memcpy.p0.p1.i64(ptr noalias nocapture writeonly, ptr addrspace(1) noalias nocapture readonly, i64, i1 immarg)
declare dso_local spir_func void @__devicelib_ConvertFToBF16INTELVec4(ptr addrspace(4), ptr addrspace(4))
declare dso_local spir_func float @__devicelib_ConvertBF16ToFINTEL(ptr addrspace(4) align 2 dereferenceable(2))
