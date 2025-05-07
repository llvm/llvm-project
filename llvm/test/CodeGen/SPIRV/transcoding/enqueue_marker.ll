; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefixes=CHECK-SPIRV

;CHECK-SPIRV: OpCapability DeviceEnqueue
;CHECK-SPIRV: OpCapability GenericPointer
;CHECK-SPIRV: %[[#queue:]] = OpTypeQueue
;CHECK-SPIRV: %[[#DeviceEvent:]] = OpTypeDeviceEvent
;CHECK-SPIRV: %[[#DefaultQueue:]] = OpGetDefaultQueue %[[#queue]]
;CHECK-SPIRV: OpEnqueueMarker 


target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-G1"
target triple = "spir-unknown-unknown"

; Function Attrs: convergent noinline norecurse nounwind optnone
define dso_local spir_kernel void @test_enqueue_marker(ptr addrspace(1) noundef align 4 %out) #0 !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !5 !kernel_arg_base_type !5 !kernel_arg_type_qual !6 {
entry:
  %out.addr = alloca ptr addrspace(1), align 4
  %queue = alloca target("spirv.Queue"), align 4
  %waitlist = alloca target("spirv.DeviceEvent"), align 4
  %evt = alloca target("spirv.DeviceEvent"), align 4
  store ptr addrspace(1) %out, ptr %out.addr, align 4
  %call = call spir_func target("spirv.Queue") @_Z17get_default_queuev() #2
  store target("spirv.Queue") %call, ptr %queue, align 4
  %0 = load target("spirv.Queue"), ptr %queue, align 4
  %waitlist.ascast = addrspacecast ptr %waitlist to ptr addrspace(4)
  %evt.ascast = addrspacecast ptr %evt to ptr addrspace(4)
  %call1 = call spir_func i32 @_Z14enqueue_marker9ocl_queuejPU3AS4K12ocl_clkeventPU3AS4S0_(target("spirv.Queue") %0, i32 noundef 1, ptr addrspace(4) noundef %waitlist.ascast, ptr addrspace(4) noundef %evt.ascast) #2
  %1 = load ptr addrspace(1), ptr %out.addr, align 4
  store i32 %call1, ptr addrspace(1) %1, align 4
  ret void
}

; Function Attrs: convergent nounwind
declare spir_func target("spirv.Queue") @_Z17get_default_queuev() #1

; Function Attrs: convergent nounwind
declare spir_func i32 @_Z14enqueue_marker9ocl_queuejPU3AS4K12ocl_clkeventPU3AS4S0_(target("spirv.Queue"), i32 noundef, ptr addrspace(4) noundef, ptr addrspace(4) noundef) #1

attributes #0 = { convergent noinline norecurse nounwind optnone "no-trapping-math"="true" "stack-protector-buffer-size"="8" "uniform-work-group-size"="false" }
attributes #1 = { convergent nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { convergent nounwind }

!llvm.module.flags = !{!0}
!opencl.ocl.version = !{!1}
!opencl.spir.version = !{!1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 2, i32 0}
!2 = !{!"clang version 20.0.0git (https://github.com/sumesh-s-2002/llvm-project.git 87b457e23e06c337bf591d007907e4d049af37b4)"}
!3 = !{i32 1}
!4 = !{!"none"}
!5 = !{!"int*"}
!6 = !{!""}
