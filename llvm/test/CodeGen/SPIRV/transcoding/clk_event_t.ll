; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; CHECK-SPIRV: OpTypeDeviceEvent
; CHECK-SPIRV: OpFunction
; CHECK-SPIRV: OpCreateUserEvent
; CHECK-SPIRV: OpIsValidEvent
; CHECK-SPIRV: OpRetainEvent
; CHECK-SPIRV: OpSetUserEventStatus
; CHECK-SPIRV: OpCaptureEventProfilingInfo
; CHECK-SPIRV: OpReleaseEvent
; CHECK-SPIRV: OpFunctionEnd

;; kernel void clk_event_t_test(global int *res, global void *prof) {
;;   clk_event_t e1 = create_user_event();
;;   *res = is_valid_event(e1);
;;   retain_event(e1);
;;   set_user_event_status(e1, -42);
;;   capture_event_profiling_info(e1, CLK_PROFILING_COMMAND_EXEC_TIME, prof);
;;   release_event(e1);
;; }

%opencl.clk_event_t = type opaque

define dso_local spir_kernel void @clk_event_t_test(i32 addrspace(1)* nocapture noundef writeonly %res, i8 addrspace(1)* noundef %prof) local_unnamed_addr {
entry:
  %call = call spir_func %opencl.clk_event_t* @_Z17create_user_eventv()
  %call1 = call spir_func zeroext i1 @_Z14is_valid_event12ocl_clkevent(%opencl.clk_event_t* %call)
  %conv = zext i1 %call1 to i32
  store i32 %conv, i32 addrspace(1)* %res, align 4
  call spir_func void @_Z12retain_event12ocl_clkevent(%opencl.clk_event_t* %call)
  call spir_func void @_Z21set_user_event_status12ocl_clkeventi(%opencl.clk_event_t* %call, i32 noundef -42)
  call spir_func void @_Z28capture_event_profiling_info12ocl_clkeventiPU3AS1v(%opencl.clk_event_t* %call, i32 noundef 1, i8 addrspace(1)* noundef %prof)
  call spir_func void @_Z13release_event12ocl_clkevent(%opencl.clk_event_t* %call)
  ret void
}

declare spir_func %opencl.clk_event_t* @_Z17create_user_eventv() local_unnamed_addr

declare spir_func zeroext i1 @_Z14is_valid_event12ocl_clkevent(%opencl.clk_event_t*) local_unnamed_addr

declare spir_func void @_Z12retain_event12ocl_clkevent(%opencl.clk_event_t*) local_unnamed_addr

declare spir_func void @_Z21set_user_event_status12ocl_clkeventi(%opencl.clk_event_t*, i32 noundef) local_unnamed_addr

declare spir_func void @_Z28capture_event_profiling_info12ocl_clkeventiPU3AS1v(%opencl.clk_event_t*, i32 noundef, i8 addrspace(1)* noundef) local_unnamed_addr

declare spir_func void @_Z13release_event12ocl_clkevent(%opencl.clk_event_t*) local_unnamed_addr
