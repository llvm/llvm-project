;; This test checks that Invoke parameter of OpEnueueKernel instruction meet the
;; following specification requirements in case of enqueueing empty block:
;; "Invoke must be an OpFunction whose OpTypeFunction operand has:
;; - Result Type must be OpTypeVoid.
;; - The first parameter must have a type of OpTypePointer to an 8-bit OpTypeInt.
;; - An optional list of parameters, each of which must have a type of OpTypePointer to the Workgroup Storage Class.
;; ... "
;; __kernel void test_enqueue_empty() {
;;   enqueue_kernel(get_default_queue(),
;;                  CLK_ENQUEUE_FLAGS_WAIT_KERNEL,
;;                  ndrange_1D(1),
;;                  0, NULL, NULL,
;;                  ^(){});
;; }
; RUN: llvm-as < %s > %t.bc
; RUN: llvm-spirv %t.bc -o - -spirv-text | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

%struct.ndrange_t = type { i32, [3 x i64], [3 x i64], [3 x i64] }
%opencl.queue_t = type opaque
%opencl.clk_event_t = type opaque

@__block_literal_global = internal addrspace(1) constant { i32, i32 } { i32 8, i32 4 }, align 4

; CHECK-SPIRV: Name [[Block:[0-9]+]] "__block_literal_global"
; CHECK-SPIRV: TypeInt [[Int8:[0-9]+]] 8
; CHECK-SPIRV: TypeVoid [[Void:[0-9]+]]
; CHECK-SPIRV: TypePointer [[Int8Ptr:[0-9]+]] 5 [[Int8]]
; CHECK-SPIRV: TypePointer [[Int8PtrGen:[0-9]+]] 8 [[Int8]]
; CHECK-SPIRV: Variable {{[0-9]+}} [[Block:[0-9]+]]

; Function Attrs: convergent nounwind
define spir_kernel void @test_enqueue_empty() #0 !kernel_arg_addr_space !0 !kernel_arg_access_qual !0 !kernel_arg_type !0 !kernel_arg_base_type !0 !kernel_arg_type_qual !0 {
entry:
  %tmp = alloca %struct.ndrange_t, align 8
  %call = call spir_func %opencl.queue_t* @_Z17get_default_queuev() #4
  call spir_func void @_Z10ndrange_1Dm(%struct.ndrange_t* sret %tmp, i64 1) #4
  %0 = call i32 @__enqueue_kernel_basic_events(%opencl.queue_t* %call, i32 1, %struct.ndrange_t* %tmp, i32 0, %opencl.clk_event_t* addrspace(4)* null, %opencl.clk_event_t* addrspace(4)* null, i8 addrspace(4)* addrspacecast (i8* bitcast (void (i8 addrspace(4)*)* @__test_enqueue_empty_block_invoke_kernel to i8*) to i8 addrspace(4)*), i8 addrspace(4)* addrspacecast (i8 addrspace(1)* bitcast ({ i32, i32 } addrspace(1)* @__block_literal_global to i8 addrspace(1)*) to i8 addrspace(4)*))
  ret void
; CHECK-SPIRV: Bitcast [[Int8Ptr]] [[Int8PtrBlock:[0-9]+]] [[Block]]
; CHECK-SPIRV: PtrCastToGeneric [[Int8PtrGen]] [[Int8PtrGenBlock:[0-9]+]] [[Int8PtrBlock]]
; CHECK-SPIRV: EnqueueKernel {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} [[Invoke:[0-9]+]] [[Int8PtrGenBlock]] {{[0-9]+}} {{[0-9]+}}
}

; Function Attrs: convergent
declare spir_func %opencl.queue_t* @_Z17get_default_queuev() #1

; Function Attrs: convergent
declare spir_func void @_Z10ndrange_1Dm(%struct.ndrange_t* sret, i64) #1

; Function Attrs: convergent nounwind
define internal spir_func void @__test_enqueue_empty_block_invoke(i8 addrspace(4)* %.block_descriptor) #2 {
entry:
  %.block_descriptor.addr = alloca i8 addrspace(4)*, align 8
  store i8 addrspace(4)* %.block_descriptor, i8 addrspace(4)** %.block_descriptor.addr, align 8
  %block = bitcast i8 addrspace(4)* %.block_descriptor to <{ i32, i32 }> addrspace(4)*
  ret void
}

; Function Attrs: nounwind
define internal spir_kernel void @__test_enqueue_empty_block_invoke_kernel(i8 addrspace(4)*) #3 {
entry:
  call void @__test_enqueue_empty_block_invoke(i8 addrspace(4)* %0)
  ret void
}

declare i32 @__enqueue_kernel_basic_events(%opencl.queue_t*, i32, %struct.ndrange_t*, i32, %opencl.clk_event_t* addrspace(4)*, %opencl.clk_event_t* addrspace(4)*, i8 addrspace(4)*, i8 addrspace(4)*)

; CHECK-SPIRV: Function [[Void]] [[Invoke]] {{[0-9]+}} {{[0-9]+}}
; CHECK-SPIRV-NEXT: FunctionParameter [[Int8PtrGen]]  {{[0-9]+}}

attributes #0 = { convergent nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "uniform-work-group-size"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { convergent "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { convergent nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind }
attributes #4 = { convergent }

!opencl.enable.FP_CONTRACT = !{}
!opencl.spir.version = !{!1}
!opencl.ocl.version = !{!2}
!opencl.used.extensions = !{!0}
!opencl.used.optional.core.features = !{!0}
!opencl.compiler.options = !{!0}

!0 = !{}
!1 = !{i32 1, i32 2}
!2 = !{i32 2, i32 0}
