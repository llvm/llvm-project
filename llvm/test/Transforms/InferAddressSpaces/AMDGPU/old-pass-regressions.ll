; RUN: opt -data-layout=A5 -S -mtriple=amdgcn-amd-amdhsa -infer-address-spaces %s | FileCheck %s

; Regression tests from old HSAIL addrspacecast optimization pass

@data = internal addrspace(1) global [100 x double] [double 0.00, double 1.000000e-01, double 2.000000e-01, double 3.000000e-01, double 4.000000e-01, double 5.000000e-01, double 6.000000e-01, double 7.000000e-01, double 8.000000e-01, double 9.000000e-01, double 1.00, double 1.10, double 1.20, double 1.30, double 1.40, double 1.50, double 1.60, double 1.70, double 1.80, double 1.90, double 2.00, double 2.10, double 2.20, double 2.30, double 2.40, double 2.50, double 2.60, double 2.70, double 2.80, double 2.90, double 3.00, double 3.10, double 3.20, double 3.30, double 3.40, double 3.50, double 3.60, double 3.70, double 3.80, double 3.90, double 4.00, double 4.10, double 4.20, double 4.30, double 4.40, double 4.50, double 4.60, double 4.70, double 4.80, double 4.90, double 5.00, double 5.10, double 5.20, double 5.30, double 5.40, double 5.50, double 5.60, double 5.70, double 5.80, double 5.90, double 6.00, double 6.10, double 6.20, double 6.30, double 6.40, double 6.50, double 6.60, double 6.70, double 6.80, double 6.90, double 7.00, double 7.10, double 7.20, double 7.30, double 7.40, double 7.50, double 7.60, double 7.70, double 7.80, double 7.90, double 8.00, double 8.10, double 8.20, double 8.30, double 8.40, double 8.50, double 8.60, double 8.70, double 8.80, double 8.90, double 9.00, double 9.10, double 9.20, double 9.30, double 9.40, double 9.50, double 9.60, double 9.70, double 9.80, double 9.90], align 8


; Should generate flat load

; CHECK-LABEL: @generic_address_bitcast_const(
; CHECK: %vecload1 = load <2 x double>, <2 x double> addrspace(1)* bitcast (double addrspace(1)* getelementptr inbounds ([100 x double], [100 x double] addrspace(1)* @data, i64 0, i64 4) to <2 x double> addrspace(1)*), align 8
define amdgpu_kernel void @generic_address_bitcast_const(i64 %arg0, i32 addrspace(1)* nocapture %results) #0 {
entry:
  %tmp1 = call i32 @llvm.amdgcn.workitem.id.x()
  %tmp2 = zext i32 %tmp1 to i64
  %tmp3 = add i64 %tmp2, %arg0
  %vecload1 = load <2 x double>, <2 x double>* bitcast (double* getelementptr ([100 x double], [100 x double]* addrspacecast ([100 x double] addrspace(1)* @data to [100 x double]*), i64 0, i64 4) to <2 x double>*), align 8
  %cmp = fcmp ord <2 x double> %vecload1, zeroinitializer
  %sext = sext <2 x i1> %cmp to <2 x i64>
  %tmp4 = extractelement <2 x i64> %sext, i64 0
  %tmp5 = extractelement <2 x i64> %sext, i64 1
  %tmp6 = and i64 %tmp4, %tmp5
  %tmp7 = lshr i64 %tmp6, 63
  %tmp8 = trunc i64 %tmp7 to i32
  %idxprom = and i64 %tmp3, 4294967295
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %results, i64 %idxprom
  store i32 %tmp8, i32 addrspace(1)* %arrayidx, align 4
  ret void
}

@generic_address_bug9749.val = internal addrspace(1) global float 0.0, align 4

declare i32 @_Z9get_fencePv(i8*)
%opencl.pipe_t = type opaque

; This is a compile time assert bug, but we still want to check optimization
; is performed to generate ld_global.
; CHECK-LABEL: @generic_address_pipe_bug9673(
; CHECK: %tmp1 = bitcast %opencl.pipe_t addrspace(3)* %in_pipe to i32 addrspace(3)*
; CHECK: %add.ptr = getelementptr inbounds i32, i32 addrspace(3)* %tmp1, i32 2
; CHECK: %tmp2 = load i32, i32 addrspace(3)* %add.ptr, align 4
define amdgpu_kernel void @generic_address_pipe_bug9673(%opencl.pipe_t addrspace(3)* nocapture %in_pipe, i32 addrspace(1)* nocapture %dst) #0 {
entry:
  %tmp = call i32 @llvm.amdgcn.workitem.id.x()
  %tmp1 = bitcast %opencl.pipe_t addrspace(3)* %in_pipe to i32 addrspace(3)*
  %add.ptr = getelementptr inbounds i32, i32 addrspace(3)* %tmp1, i32 2
  %tmp2 = load i32, i32 addrspace(3)* %add.ptr, align 4
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %dst, i32 %tmp
  store i32 %tmp2, i32 addrspace(1)* %arrayidx, align 4
  ret void
}

; Should generate flat load
; CHECK-LABEL: @generic_address_bug9749(
; CHECK: br i1
; CHECK: load float, float*
; CHECK: br label
define amdgpu_kernel void @generic_address_bug9749(i32 addrspace(1)* nocapture %results) #0 {
entry:
  %ptr = alloca float*, align 8, addrspace(5)
  %tmp = call i32 @llvm.amdgcn.workitem.id.x()
  %tmp1 = zext i32 %tmp to i64
  store float 0x3FB99999A0000000, float addrspace(1)* @generic_address_bug9749.val, align 4
  store volatile float* addrspacecast (float addrspace(1)* @generic_address_bug9749.val to float*), float* addrspace(5)* %ptr, align 8
  %tmp2 = load volatile float*, float* addrspace(5)* %ptr, align 8
  %tmp3 = load float, float addrspace(1)* @generic_address_bug9749.val, align 4
  %tmp4 = bitcast float* %tmp2 to i8*
  %call.i = call i32 @_Z9get_fencePv(i8* %tmp4) #1
  %switch.i.i = icmp ult i32 %call.i, 4
  br i1 %switch.i.i, label %if.end.i, label %helperFunction.exit

if.end.i:                                         ; preds = %entry
  %tmp5 = load float, float* %tmp2, align 4
  %not.cmp.i = fcmp oeq float %tmp5, %tmp3
  %phitmp = zext i1 %not.cmp.i to i32
  br label %helperFunction.exit

helperFunction.exit:                              ; preds = %if.end.i, %entry
  %retval.0.i = phi i32 [ 0, %entry ], [ %phitmp, %if.end.i ]
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %results, i64 %tmp1
  store i32 %retval.0.i, i32 addrspace(1)* %arrayidx, align 4
  ret void
}

; CHECK-LABEL: @generic_address_opt_phi_bug9776_simple_phi_kernel(
; CHECK: phi i32 addrspace(3)*
; CHECK: store i32 %i.03, i32 addrspace(3)* %
define amdgpu_kernel void @generic_address_opt_phi_bug9776_simple_phi_kernel(i32 addrspace(3)* nocapture %in, i32 %numElems) #0 {
entry:
  %cmp1 = icmp eq i32 %numElems, 0
  br i1 %cmp1, label %for.end, label %for.body.lr.ph

for.body.lr.ph:                                   ; preds = %entry
  %tmp = addrspacecast i32 addrspace(3)* %in to i32*
  br label %for.body

for.body:                                         ; preds = %for.body, %for.body.lr.ph
  %i.03 = phi i32 [ 0, %for.body.lr.ph ], [ %inc, %for.body ]
  %ptr.02 = phi i32* [ %tmp, %for.body.lr.ph ], [ %add.ptr, %for.body ]
  store i32 %i.03, i32* %ptr.02, align 4
  %add.ptr = getelementptr inbounds i32, i32* %ptr.02, i64 4
  %inc = add nuw i32 %i.03, 1
  %exitcond = icmp eq i32 %inc, %numElems
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

; CHECK-LABEL: @generic_address_bug9899(
; CHECK: %vecload = load <2 x i32>, <2 x i32> addrspace(3)*
; CHECK: store <2 x i32> %tmp16, <2 x i32> addrspace(3)*
define amdgpu_kernel void @generic_address_bug9899(i64 %arg0, i32 addrspace(3)* nocapture %sourceA, i32 addrspace(3)* nocapture %destValues) #0 {
entry:
  %tmp1 = call i32 @llvm.amdgcn.workitem.id.x()
  %tmp2 = zext i32 %tmp1 to i64
  %tmp3 = add i64 %tmp2, %arg0
  %sext = shl i64 %tmp3, 32
  %tmp4 = addrspacecast i32 addrspace(3)* %destValues to i32*
  %tmp5 = addrspacecast i32 addrspace(3)* %sourceA to i32*
  %tmp6 = ashr exact i64 %sext, 31
  %tmp7 = getelementptr inbounds i32, i32* %tmp5, i64 %tmp6
  %arrayidx_v4 = bitcast i32* %tmp7 to <2 x i32>*
  %vecload = load <2 x i32>, <2 x i32>* %arrayidx_v4, align 4
  %tmp8 = extractelement <2 x i32> %vecload, i32 0
  %tmp9 = extractelement <2 x i32> %vecload, i32 1
  %tmp10 = icmp eq i32 %tmp8, 0
  %tmp11 = select i1 %tmp10, i32 32, i32 %tmp8
  %tmp12 = icmp eq i32 %tmp9, 0
  %tmp13 = select i1 %tmp12, i32 32, i32 %tmp9
  %tmp14 = getelementptr inbounds i32, i32* %tmp4, i64 %tmp6
  %tmp15 = insertelement <2 x i32> undef, i32 %tmp11, i32 0
  %tmp16 = insertelement <2 x i32> %tmp15, i32 %tmp13, i32 1
  %arrayidx_v41 = bitcast i32* %tmp14 to <2 x i32>*
  store <2 x i32> %tmp16, <2 x i32>* %arrayidx_v41, align 4
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #2

attributes #0 = { nounwind }
attributes #1 = { nounwind readonly }
attributes #2 = { nounwind readnone }
