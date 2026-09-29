; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: OpName %[[#Foo:]] "foo"
; CHECK-DAG: OpName %[[#CSQRT:]] "csqrt"
; CHECK-DAG: %[[#FloatTy:]] = OpTypeFloat 64
; CHECK-DAG: %[[#StructTy:]] = OpTypeStruct %[[#FloatTy]] %[[#FloatTy]]
; CHECK-DAG: %[[#GPtrStructTy:]] = OpTypePointer Generic %[[#StructTy]]
; CHECK-DAG: %[[#FPtrStructTy:]] = OpTypePointer Function %[[#StructTy]]
; CHECK-DAG: %[[#WrapTy:]] = OpTypeStruct %[[#StructTy]]
; CHECK-DAG: %[[#Int64Ty:]] = OpTypeInt 64 0
; CHECK-DAG: %[[#Int32Ty:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#C1:]] = OpConstant %[[#Int32Ty]] 1
; CHECK-DAG: %[[#ArrayTy:]] = OpTypeArray %[[#Int64Ty]] %[[#C1]]
; CHECK-DAG: %[[#SArrayTy:]] = OpTypeStruct %[[#ArrayTy]]
; CHECK-DAG: %[[#SSArrayTy:]] = OpTypeStruct %[[#SArrayTy]]
; CHECK-DAG: %[[#CWPtrComplexTy:]] = OpTypePointer CrossWorkgroup %[[#WrapTy]]
; CHECK-DAG: %[[#IdTy:]] = OpTypePointer Function %[[#SSArrayTy]]
; CHECK: %[[#Foo]] = OpFunction
; CHECK: OpFunctionParameter %[[#CWPtrComplexTy]]
; CHECK: OpFunctionParameter %[[#IdTy]]
; CHECK: %[[#CSQRT]] = OpFunction
; CHECK: OpFunctionParameter %[[#GPtrStructTy]]
; CHECK: OpFunctionParameter %[[#FPtrStructTy]]

%"class.id" = type { %"class.array" }
%"class.array" = type { [1 x i64] }
%"class.complex" = type { { double, double } }

define weak_odr dso_local spir_kernel void @foo(ptr addrspace(1) align 8 %_arg_buf_out1_access, ptr byval(%"class.id") align 8 %_arg_buf_out1_access3) {
entry:
  %tmp.i.i = alloca { double, double }, align 8
  %byval-temp.i.i = alloca { double, double }, align 8
  %idxvalue = load i64, ptr %_arg_buf_out1_access3, align 8
  %add.ptr.i = getelementptr inbounds %"class.complex", ptr addrspace(1) %_arg_buf_out1_access, i64 %idxvalue
  %tmp.ascast.i.i = addrspacecast ptr %tmp.i.i to ptr addrspace(4)
  %byval-temp.imagp.i.i = getelementptr inbounds i8, ptr %byval-temp.i.i, i64 8
  store double -1.000000e+00, ptr %byval-temp.i.i, align 8
  store double 0.000000e+00, ptr %byval-temp.imagp.i.i, align 8
  call spir_func void @csqrt(ptr addrspace(4) dead_on_unwind writable sret({ double, double }) align 8 %tmp.ascast.i.i, ptr nonnull byval({ double, double }) align 8 %byval-temp.i.i)
  %tmp.ascast.real.i.i = load double, ptr %tmp.i.i, align 8
  %tmp.ascast.imagp.i.i = getelementptr inbounds i8, ptr %tmp.i.i, i64 8
  %tmp.ascast.imag.i.i = load double, ptr %tmp.ascast.imagp.i.i, align 8
  store double %tmp.ascast.real.i.i, ptr addrspace(1) %add.ptr.i, align 8
  %dest = getelementptr inbounds i8, ptr addrspace(1) %add.ptr.i, i64 8
  store double %tmp.ascast.imag.i.i, ptr addrspace(1) %dest, align 8
  ret void
}

define weak dso_local spir_func void @csqrt(ptr addrspace(4) dead_on_unwind noalias writable sret({ double, double }) align 8 %agg.result, ptr byval({ double, double }) align 8 %z) {
entry:
  %tmp = alloca { double, double }, align 8
  %byval-temp = alloca { double, double }, align 8
  %tmp.ascast = addrspacecast ptr %tmp to ptr addrspace(4)
  %z.ascast.real = load double, ptr %z, align 8
  %z.ascast.imagp = getelementptr inbounds i8, ptr %z, i64 8
  %z.ascast.imag = load double, ptr %z.ascast.imagp, align 8
  %byval-temp.imagp = getelementptr inbounds i8, ptr %byval-temp, i64 8
  store double %z.ascast.real, ptr %byval-temp, align 8
  store double %z.ascast.imag, ptr %byval-temp.imagp, align 8
  call spir_func void @__devicelib_csqrt(ptr addrspace(4) dead_on_unwind writable sret({ double, double }) align 8 %tmp.ascast, ptr nonnull byval({ double, double }) align 8 %byval-temp) #7
  %tmp.ascast.real = load double, ptr %tmp, align 8
  %tmp.ascast.imagp = getelementptr inbounds i8, ptr %tmp, i64 8
  %tmp.ascast.imag = load double, ptr %tmp.ascast.imagp, align 8
  %agg.result.imagp = getelementptr inbounds i8, ptr addrspace(4) %agg.result, i64 8
  store double %tmp.ascast.real, ptr addrspace(4) %agg.result, align 8
  store double %tmp.ascast.imag, ptr addrspace(4) %agg.result.imagp, align 8
  ret void
}

declare extern_weak dso_local spir_func void @__devicelib_csqrt(ptr addrspace(4) dead_on_unwind writable sret({ double, double }) align 8, ptr byval({ double, double }) align 8) local_unnamed_addr #3
