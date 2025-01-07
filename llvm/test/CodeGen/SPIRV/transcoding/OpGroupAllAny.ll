; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-SPIRV: OpCapability Groups
; CHECK-SPIRV-DAG: %[[#BoolTypeID:]] = OpTypeBool
; CHECK-SPIRV-DAG: %[[#True:]] = OpConstantTrue %[[#BoolTypeID]]
; CHECK-SPIRV-DAG: %[[#False:]] = OpConstantFalse %[[#BoolTypeID]]

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV: %[[#]] = OpGroupAll %[[#BoolTypeID]] %[[#]] %[[#True]]
; CHECK-SPIRV: %[[#]] = OpGroupAny %[[#BoolTypeID]] %[[#]] %[[#True]]
; CHECK-SPIRV: %[[#]] = OpGroupAll %[[#BoolTypeID]] %[[#]] %[[#True]]
; CHECK-SPIRV: %[[#]] = OpGroupAny %[[#BoolTypeID]] %[[#]] %[[#False]]
define spir_kernel void @test(i32 addrspace(1)* nocapture readnone %i) {
entry:
  %call = tail call spir_func i32 @_Z14work_group_alli(i32 5)
  %call1 = tail call spir_func i32 @_Z14work_group_anyi(i32 5)
  %call3 = tail call spir_func i32 @__spirv_GroupAll(i32 0, i1 1)
  %call4 = tail call spir_func i32 @__spirv_GroupAny(i32 0, i1 0)
  ret void
}

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV: %[[#]] = OpGroupAny %[[#BoolTypeID]] %[[#]] %[[#]]
; CHECK-SPIRV: %[[#]] = OpGroupAll %[[#BoolTypeID]] %[[#]] %[[#]]
define spir_kernel void @test_nonconst_any(ptr addrspace(1) %input, ptr addrspace(1) %output)  #0 !kernel_arg_addr_space !7 !kernel_arg_access_qual !8 !kernel_arg_type !9 !kernel_arg_type_qual !10 !kernel_arg_base_type !9 !spirv.ParameterDecorations !11  {
entry:
  %r0 = call spir_func i64 @_Z13get_global_idj(i32 0)
  %r1 = insertelement <3 x i64> undef, i64 %r0, i32 0
  %r2 = call spir_func i64 @_Z13get_global_idj(i32 1)
  %r3 = insertelement <3 x i64> %r1, i64 %r2, i32 1
  %r4 = call spir_func i64 @_Z13get_global_idj(i32 2)
  %r5 = insertelement <3 x i64> %r3, i64 %r4, i32 2
  %call = extractelement <3 x i64> %r5, i32 0
  %conv = trunc i64 %call to i32
  %idxprom = sext i32 %conv to i64
  %arrayidx = getelementptr inbounds float, ptr addrspace(1) %input, i64 %idxprom
  %r6 = load float, ptr addrspace(1) %arrayidx, align 4
  %add = add nsw i32 %conv, 1
  %idxprom1 = sext i32 %add to i64
  %arrayidx2 = getelementptr inbounds float, ptr addrspace(1) %input, i64 %idxprom1
  %r7 = load float, ptr addrspace(1) %arrayidx2, align 4
  %cmp = fcmp ogt float %r6, %r7
  %conv3 = select i1 %cmp, i32 1, i32 0
  %r8 = icmp ne i32 %conv3, 0
  %r9 = zext i1 %r8 to i32
  %r10 = call spir_func i32 @_Z14work_group_anyi(i32 %r9)
  %call41 = icmp ne i32 %r10, 0
  %call4 = select i1 %call41, i32 1, i32 0
  %idxprom5 = sext i32 %conv to i64
  %arrayidx6 = getelementptr inbounds i32, ptr addrspace(1) %output, i64 %idxprom5
  store i32 %call4, ptr addrspace(1) %arrayidx6, align 4
  %r11 = call spir_func i32 @_Z14work_group_alli(i32 %r9)
  %call42 = icmp ne i32 %r11, 0
  %call5 = select i1 %call42, i32 1, i32 0
  store i32 %call5, ptr addrspace(1) %arrayidx6, align 4
  ret void
}

declare spir_func i64 @_Z13get_global_idj(i32) #1

declare spir_func i32 @_Z14work_group_alli(i32) #2
declare spir_func i32 @_Z14work_group_anyi(i32) #2

declare spir_func i1 @__spirv_GroupAll(i32, i1)
declare spir_func i1 @__spirv_GroupAny(i32, i1)

attributes #0 = { nounwind }
attributes #1 = { nounwind willreturn memory(none) }
attributes #2 = { convergent nounwind }

!spirv.MemoryModel = !{!0}
!opencl.enable.FP_CONTRACT = !{}
!spirv.Source = !{!1}
!opencl.spir.version = !{!2}
!opencl.ocl.version = !{!3}
!opencl.used.extensions = !{!4}
!opencl.used.optional.core.features = !{!5}
!spirv.Generator = !{!6}

!0 = !{i32 2, i32 2}
!1 = !{i32 3, i32 300000}
!2 = !{i32 2, i32 0}
!3 = !{i32 3, i32 0}
!4 = !{!"cl_khr_subgroups"}
!5 = !{}
!6 = !{i16 6, i16 14}
!7 = !{i32 1, i32 1}
!8 = !{!"none", !"none"}
!9 = !{!"float*", !"int*"}
!10 = !{!"", !""}
!11 = !{!5, !5}
