; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val --target-env spv1.4 %}

; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val --target-env spv1.4 %}

; CHECK-SPIRV: OpCapability Groups
; CHECK-SPIRV-DAG: %[[#Int32Ty:]] = OpTypeInt 32 0
; CHECK-SPIRV-DAG: %[[#Int64Ty:]] = OpTypeInt 64 0
; CHECK-SPIRV-DAG: %[[#Float32Ty:]] = OpTypeFloat 32
; CHECK-SPIRV-DAG: %[[#Vec2Int32Ty:]] = OpTypeVector %[[#Int32Ty]] 2
; CHECK-SPIRV-DAG: %[[#Vec3Int32Ty:]] = OpTypeVector %[[#Int32Ty]] 3
; CHECK-SPIRV-DAG: %[[#Vec2Int64Ty:]] = OpTypeVector %[[#Int64Ty]] 2
; CHECK-SPIRV-DAG: %[[#C2:]] = OpConstant %[[#Int32Ty]] 2

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV: %[[#Val:]] = OpFunctionParameter %[[#Int32Ty]]
; CHECK-SPIRV: %[[#X:]] = OpFunctionParameter %[[#Int32Ty]]
; CHECK-SPIRV: %[[#Y:]] = OpFunctionParameter %[[#Int32Ty]]
; CHECK-SPIRV: %[[#Z:]] = OpFunctionParameter %[[#Int32Ty]]
; CHECK-SPIRV: %[[#]] = OpGroupBroadcast %[[#Int32Ty]] %[[#C2]] %[[#Val]] %[[#X]]
; CHECK-SPIRV: %[[#XY:]] = OpCompositeConstruct %[[#Vec2Int32Ty]] %[[#X]] %[[#Y]]
; CHECK-SPIRV: %[[#]] = OpGroupBroadcast %[[#Int32Ty]] %[[#C2]] %[[#Val]] %[[#XY]]
; CHECK-SPIRV: %[[#XYZ:]] = OpCompositeConstruct %[[#Vec3Int32Ty]] %[[#X]] %[[#Y]] %[[#Z]]
; CHECK-SPIRV: %[[#]] = OpGroupBroadcast %[[#Int32Ty]] %[[#C2]] %[[#Val]] %[[#XYZ]]
define spir_kernel void @test_broadcast_xyz(i32 noundef %a, i32 noundef %x, i32 noundef %y, i32 noundef %z) {
entry:
  %call1 = call spir_func i32 @_Z20work_group_broadcastjj(i32 noundef %a, i32 noundef %x)
  %call2 = call spir_func i32 @_Z20work_group_broadcastjj(i32 noundef %a, i32 noundef %x, i32 noundef %y)
  %call3 = call spir_func i32 @_Z20work_group_broadcastjj(i32 noundef %a, i32 noundef %x, i32 noundef %y, i32 noundef %z)
  ret void
}

declare spir_func i32 @_Z20work_group_broadcastjj(i32, i32)
declare spir_func i32 @_Z20work_group_broadcastjjj(i32, i32, i32)
declare spir_func i32 @_Z20work_group_broadcastjjjj(i32, i32, i32, i32)

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV: OpInBoundsPtrAccessChain
; CHECK-SPIRV: %[[#LoadedVal:]] = OpLoad %[[#Float32Ty]] %[[#]]
; CHECK-SPIRV: %[[#IdX:]] = OpCompositeExtract %[[#Int64Ty]] %[[#]] 0
; CHECK-SPIRV: %[[#IdY:]] = OpCompositeExtract %[[#Int64Ty]] %[[#]] 1
; CHECK-SPIRV: %[[#LocIdsVec:]] = OpCompositeConstruct %[[#Vec2Int64Ty]] %[[#IdX]] %[[#IdY]]
; CHECK-SPIRV: %[[#]] = OpGroupBroadcast %[[#Float32Ty]] %[[#C2]] %[[#LoadedVal]] %[[#LocIdsVec]]
define spir_kernel void @test_wg_broadcast_2D(ptr addrspace(1) %input, ptr addrspace(1) %output) #0 !kernel_arg_addr_space !7 !kernel_arg_access_qual !8 !kernel_arg_type !9 !kernel_arg_type_qual !10 !kernel_arg_base_type !9 !spirv.ParameterDecorations !11 {
entry:
  %0 = call spir_func i64 @_Z13get_global_idj(i32 0) #1
  %1 = insertelement <3 x i64> undef, i64 %0, i32 0
  %2 = call spir_func i64 @_Z13get_global_idj(i32 1) #1
  %3 = insertelement <3 x i64> %1, i64 %2, i32 1
  %4 = call spir_func i64 @_Z13get_global_idj(i32 2) #1
  %5 = insertelement <3 x i64> %3, i64 %4, i32 2
  %call = extractelement <3 x i64> %5, i32 0
  %6 = call spir_func i64 @_Z13get_global_idj(i32 0) #1
  %7 = insertelement <3 x i64> undef, i64 %6, i32 0
  %8 = call spir_func i64 @_Z13get_global_idj(i32 1) #1
  %9 = insertelement <3 x i64> %7, i64 %8, i32 1
  %10 = call spir_func i64 @_Z13get_global_idj(i32 2) #1
  %11 = insertelement <3 x i64> %9, i64 %10, i32 2
  %call1 = extractelement <3 x i64> %11, i32 1
  %12 = call spir_func i64 @_Z12get_group_idj(i32 0) #1
  %13 = insertelement <3 x i64> undef, i64 %12, i32 0
  %14 = call spir_func i64 @_Z12get_group_idj(i32 1) #1
  %15 = insertelement <3 x i64> %13, i64 %14, i32 1
  %16 = call spir_func i64 @_Z12get_group_idj(i32 2) #1
  %17 = insertelement <3 x i64> %15, i64 %16, i32 2
  %call2 = extractelement <3 x i64> %17, i32 0
  %18 = call spir_func i64 @_Z14get_local_sizej(i32 0) #1
  %19 = insertelement <3 x i64> undef, i64 %18, i32 0
  %20 = call spir_func i64 @_Z14get_local_sizej(i32 1) #1
  %21 = insertelement <3 x i64> %19, i64 %20, i32 1
  %22 = call spir_func i64 @_Z14get_local_sizej(i32 2) #1
  %23 = insertelement <3 x i64> %21, i64 %22, i32 2
  %call3 = extractelement <3 x i64> %23, i32 0
  %rem = urem i64 %call2, %call3
  %24 = call spir_func i64 @_Z12get_group_idj(i32 0) #1
  %25 = insertelement <3 x i64> undef, i64 %24, i32 0
  %26 = call spir_func i64 @_Z12get_group_idj(i32 1) #1
  %27 = insertelement <3 x i64> %25, i64 %26, i32 1
  %28 = call spir_func i64 @_Z12get_group_idj(i32 2) #1
  %29 = insertelement <3 x i64> %27, i64 %28, i32 2
  %call4 = extractelement <3 x i64> %29, i32 1
  %30 = call spir_func i64 @_Z14get_local_sizej(i32 0) #1
  %31 = insertelement <3 x i64> undef, i64 %30, i32 0
  %32 = call spir_func i64 @_Z14get_local_sizej(i32 1) #1
  %33 = insertelement <3 x i64> %31, i64 %32, i32 1
  %34 = call spir_func i64 @_Z14get_local_sizej(i32 2) #1
  %35 = insertelement <3 x i64> %33, i64 %34, i32 2
  %call5 = extractelement <3 x i64> %35, i32 1
  %rem6 = urem i64 %call4, %call5
  %36 = call spir_func i64 @_Z15get_global_sizej(i32 0) #1
  %37 = insertelement <3 x i64> undef, i64 %36, i32 0
  %38 = call spir_func i64 @_Z15get_global_sizej(i32 1) #1
  %39 = insertelement <3 x i64> %37, i64 %38, i32 1
  %40 = call spir_func i64 @_Z15get_global_sizej(i32 2) #1
  %41 = insertelement <3 x i64> %39, i64 %40, i32 2
  %call7 = extractelement <3 x i64> %41, i32 0
  %mul = mul i64 %call1, %call7
  %add = add i64 %mul, %call
  %arrayidx = getelementptr inbounds float, ptr addrspace(1) %input, i64 %add
  %42 = load float, ptr addrspace(1) %arrayidx, align 4
  %.splatinsert = insertelement <2 x i64> undef, i64 %rem, i32 0
  %.splat = shufflevector <2 x i64> %.splatinsert, <2 x i64> undef, <2 x i32> zeroinitializer
  %43 = insertelement <2 x i64> %.splat, i64 %rem6, i32 1
  %44 = extractelement <2 x i64> %43, i32 0
  %45 = extractelement <2 x i64> %43, i32 1
  %call8 = call spir_func float @_Z20work_group_broadcastfmm(float %42, i64 %44, i64 %45) #2
  %arrayidx9 = getelementptr inbounds float, ptr addrspace(1) %output, i64 %add
  store float %call8, ptr addrspace(1) %arrayidx9, align 4
  ret void
}

; Function Attrs: nounwind willreturn memory(none)
declare spir_func i64 @_Z13get_global_idj(i32) #1

; Function Attrs: nounwind willreturn memory(none)
declare spir_func i64 @_Z12get_group_idj(i32) #1

; Function Attrs: nounwind willreturn memory(none)
declare spir_func i64 @_Z14get_local_sizej(i32) #1

; Function Attrs: nounwind willreturn memory(none)
declare spir_func i64 @_Z15get_global_sizej(i32) #1

; Function Attrs: convergent nounwind
declare spir_func float @_Z20work_group_broadcastfmm(float, i64, i64) #2

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
!9 = !{!"float*", !"float*"}
!10 = !{!"", !""}
!11 = !{!5, !5}
