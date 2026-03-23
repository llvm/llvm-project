; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefixes=CHECK-SPIRV

;CHECK-SPIRV: OpCapability DeviceEnqueue
;CHECK-SPIRV: OpCapability GenericPointer
;CHECK-SPIRV: %[[#queue:]] = OpTypeQueue
;CHECK-SPIRV: %[[#DeviceEvent:]] = OpTypeDeviceEvent
;CHECK-SPIRV: %[[#DefaultQueue:]] = OpGetDefaultQueue %[[#queue]]
;CHECK-SPIRV: OpEnqueueMarker 

; Function Attrs: convergent noinline norecurse nounwind optnone
define dso_local spir_kernel void @test_enqueue_marker(ptr addrspace(1) noundef align 4 %out) #0 !kernel_arg_addr_space !{i32 1} !kernel_arg_access_qual !{!"none"} !kernel_arg_type !{!"int*"} !kernel_arg_base_type !{!"int*"} !kernel_arg_type_qual !{!""} {
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
