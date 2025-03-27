; Modified from: https://github.com/KhronosGroup/SPIRV-LLVM-Translator/test/extensions/INTEL/SPV_INTEL_variable_length_array/vla_spec_const.ll

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_INTEL_variable_length_array %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; TODO: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_INTEL_variable_length_array %s -o - -filetype=obj | spirv-val %}

; CHECK-SPIRV: Capability VariableLengthArrayINTEL
; CHECK-SPIRV: Extension "SPV_INTEL_variable_length_array"
; CHECK-SPIRV: OpDecorate %[[SpecConst:.*]] SpecId 0
; CHECK-SPIRV-DAG: %[[Long:.*]] = OpTypeInt 64 0
; CHECK-SPIRV-DAG: %[[Int:.*]] = OpTypeInt 32 0
; CHECK-SPIRV-DAG: %[[IntPtr:.*]] = OpTypePointer {{[a-zA-Z]+}} %[[Int]]
; CHECK-SPIRV: %[[SpecConst]] = OpSpecConstant %[[Long]]
; CHECK-SPIRV-LABEL: FunctionEnd
; CHECK-SPIRV: %[[SpecConstVal:.*]] = OpFunctionCall %[[Long]]
; CHECK-SPIRV: OpSaveMemoryINTEL
; CHECK-SPIRV: OpVariableLengthArrayINTEL %[[IntPtr]] %[[SpecConstVal]]
; CHECK-SPIRV: OpRestoreMemoryINTEL

; CHECK-SPIRV: OpFunction %[[Long]]
; CHECK-SPIRV: ReturnValue %[[SpecConst]]

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-linux"

%"class._ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEEUlvE_.anon" = type { %"class._ZTSN2cl4sycl12experimental13spec_constantIm13MyUInt64ConstEE.cl::sycl::experimental::spec_constant" }
%"class._ZTSN2cl4sycl12experimental13spec_constantIm13MyUInt64ConstEE.cl::sycl::experimental::spec_constant" = type { i8 }

$_ZTS17SpecializedKernel = comdat any

$_ZNK2cl4sycl12experimental13spec_constantIm13MyUInt64ConstE3getEv = comdat any

; Function Attrs: norecurse
define weak_odr dso_local spir_kernel void @_ZTS17SpecializedKernel() #0 comdat !kernel_arg_addr_space !4 !kernel_arg_access_qual !4 !kernel_arg_type !4 !kernel_arg_base_type !4 !kernel_arg_type_qual !4 {
entry:
  %p = alloca %"class._ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEEUlvE_.anon", align 1
  call void @llvm.lifetime.start.p0(i64 1, ptr %p) #4
  %p4 = addrspacecast ptr %p to ptr addrspace(4)
  call spir_func void @"_ZZZ4mainENK3$_0clERN2cl4sycl7handlerEENKUlvE_clEv"(ptr addrspace(4) %p4)
  call void @llvm.lifetime.end.p0(i64 1, ptr %p) #4
  ret void
}

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture) #1

; Function Attrs: inlinehint norecurse
define internal spir_func void @"_ZZZ4mainENK3$_0clERN2cl4sycl7handlerEENKUlvE_clEv"(ptr addrspace(4) %this) #2 align 2 {
entry:
  %this.addr = alloca ptr addrspace(4), align 8
  %saved_stack = alloca ptr, align 8
  %__vla_expr0 = alloca i64, align 8
  store ptr addrspace(4) %this, ptr %this.addr, align 8, !tbaa !5
  %this1 = load ptr addrspace(4), ptr %this.addr, align 8
  %call = call spir_func i64 @_ZNK2cl4sycl12experimental13spec_constantIm13MyUInt64ConstE3getEv(ptr addrspace(4) %this1)
  %p = call ptr @llvm.stacksave.p0()
  store ptr %p, ptr %saved_stack, align 8
  %vla = alloca i32, i64 %call, align 4
  store i64 %call, ptr %__vla_expr0, align 8
  store i32 42, ptr %vla, align 4, !tbaa !9
  %torestore = load ptr, ptr %saved_stack, align 8
  call void @llvm.stackrestore.p0(ptr %torestore)
  ret void
}

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture) #1

; Function Attrs: norecurse
define linkonce_odr dso_local spir_func i64 @_ZNK2cl4sycl12experimental13spec_constantIm13MyUInt64ConstE3getEv(ptr addrspace(4) %this) #3 comdat align 2 {
entry:
  %this.addr = alloca ptr addrspace(4), align 8
  %TName = alloca ptr addrspace(4), align 8
  store ptr addrspace(4) %this, ptr %this.addr, align 8, !tbaa !5
  call void @llvm.lifetime.start.p0(i64 8, ptr %TName) #4
  %p = call i64 @_Z20__spirv_SpecConstantix(i32 0, i64 0), !SYCL_SPEC_CONST_SYM_ID !11
  call void @llvm.lifetime.end.p0(i64 8, ptr %TName) #4
  ret i64 %p
}

; Function Attrs: nounwind
declare ptr @llvm.stacksave.p0() #4

; Function Attrs: nounwind
declare void @llvm.stackrestore.p0(ptr) #4

declare i64 @_Z20__spirv_SpecConstantix(i32, i64)

attributes #0 = { norecurse "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="/work/intel/vla_spec_const.cpp" "uniform-work-group-size"="true" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind willreturn }
attributes #2 = { inlinehint norecurse "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { norecurse "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind }

!llvm.module.flags = !{!0}
!opencl.spir.version = !{!1}
!spirv.Source = !{!2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 2}
!2 = !{i32 4, i32 100000}
!3 = !{!"clang version 12.0.0"}
!4 = !{}
!5 = !{!6, !6, i64 0}
!6 = !{!"any pointer", !7, i64 0}
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C++ TBAA"}
!9 = !{!10, !10, i64 0}
!10 = !{!"int", !7, i64 0}
!11 = !{!"_ZTS13MyUInt64Const", i32 0}
