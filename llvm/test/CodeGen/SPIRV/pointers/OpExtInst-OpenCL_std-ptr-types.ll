; The goal of the test is to ensure that the output SPIR-V is valid from the perspective of the spirv-val tool.
; RUN: %if spirv-tools %{ llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

%clsid = type { %arr }
%arr = type { [1 x i64] }
%struct_half = type { half }

define weak_odr dso_local spir_kernel void @foo(ptr addrspace(1) noundef readonly align 2 %_acc, ptr noundef byval(%clsid) align 8 %_acc_id, ptr addrspace(3) noundef align 2 %_arg_loc) {
entry:
  %r1 = load i64, ptr %_acc_id, align 8
  %add.ptr.i41 = getelementptr inbounds %struct_half, ptr addrspace(1) %_acc, i64 %r1
  %idx = addrspacecast ptr addrspace(1) %add.ptr.i41 to ptr addrspace(4)
  %call.i.i290 = call spir_func noundef ptr addrspace(1) @_Z41__spirv_GenericCastToPtrExplicit_ToGlobalPKvi(ptr addrspace(4) noundef %idx, i32 noundef 5)
  call spir_func void @_Z20__spirv_ocl_prefetchPU3AS1Kcm(ptr addrspace(1) noundef %call.i.i290, i64 noundef 0)

  %locidx = addrspacecast ptr addrspace(3) %_arg_loc to ptr addrspace(4)
  %ptr1 = tail call spir_func noundef ptr addrspace(3) @_Z40__spirv_GenericCastToPtrExplicit_ToLocalPvi(ptr addrspace(4) noundef %locidx, i32 noundef 4)
  %sincos_r = tail call spir_func noundef half @_Z18__spirv_ocl_sincosDF16_PU3AS3DF16_(half noundef 0xH3145, ptr addrspace(3) noundef %ptr1)

  %p1 = addrspacecast ptr addrspace(1) %_acc to ptr addrspace(4)
  %ptr2 = tail call spir_func noundef ptr addrspace(1) @_Z41__spirv_GenericCastToPtrExplicit_ToGlobalPvi(ptr addrspace(4) noundef %p1, i32 noundef 5)
  %remquo_r = tail call spir_func noundef half @_Z18__spirv_ocl_remquoDF16_DF16_PU3AS1i(half noundef 0xH3A37, half noundef 0xH32F4, ptr addrspace(1) noundef %ptr2)

  ret void
}

declare dso_local spir_func void @_Z20__spirv_ocl_prefetchPU3AS1Kcm(ptr addrspace(1) noundef, i64 noundef)
declare dso_local spir_func noundef ptr addrspace(1) @_Z41__spirv_GenericCastToPtrExplicit_ToGlobalPKvi(ptr addrspace(4) noundef, i32 noundef)

declare dso_local spir_func noundef half @_Z18__spirv_ocl_sincosDF16_PU3AS3DF16_(half noundef, ptr addrspace(3) noundef)
declare dso_local spir_func noundef ptr addrspace(3) @_Z40__spirv_GenericCastToPtrExplicit_ToLocalPvi(ptr addrspace(4) noundef, i32 noundef)

declare dso_local spir_func noundef ptr addrspace(1) @_Z41__spirv_GenericCastToPtrExplicit_ToGlobalPvi(ptr addrspace(4) noundef, i32 noundef)
declare dso_local spir_func noundef half @_Z18__spirv_ocl_remquoDF16_DF16_PU3AS1i(half noundef, half noundef, ptr addrspace(1) noundef)
