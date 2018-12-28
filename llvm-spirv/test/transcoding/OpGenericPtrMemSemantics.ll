; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o %t.txt
; RUN: FileCheck < %t.txt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV: 4 GenericPtrMemSemantics {{[0-9]+}} [[ResID:[0-9]+]] {{[0-9]+}}
; CHECK-SPIRV-NEXT: 5 ShiftRightLogical {{[0-9]+}} {{[0-9]+}} [[ResID]] {{[0-9]+}}

; Note that round-trip conversion replaces 'get_fence (gentype *ptr)' built-in function with 'get_fence (const gentype *ptr)'.
; CHECK-LLVM: call spir_func i32 @_Z9get_fencePU3AS4Kv(i8
; CHECK-LLVM-NEXT: shl
; CHECK-LLVM-NEXT: lshr

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

@gint = addrspace(1) global i32 1, align 4

; Function Attrs: nounwind readnone
define spir_func i32 @isFenceValid(i32 %fence) #0 {
entry:
  %switch = icmp ult i32 %fence, 4
  %. = zext i1 %switch to i32
  ret i32 %.
}

; Function Attrs: nounwind
define spir_func i32 @f4(i32 %val, i32 addrspace(4)* %ptr) #1 {
entry:
  %0 = bitcast i32 addrspace(4)* %ptr to i8 addrspace(4)*
  %call = tail call spir_func i32 @_Z9get_fencePU3AS4v(i8 addrspace(4)* %0) #3
  %switch.i = icmp ult i32 %call, 4
  %1 = load i32, i32 addrspace(4)* %ptr, align 4
  %cmp = icmp eq i32 %1, %val
  %and4 = and i1 %switch.i, %cmp
  %and = zext i1 %and4 to i32
  %2 = xor i32 %and, 1
  ret i32 %2
}

declare spir_func i32 @_Z9get_fencePU3AS4v(i8 addrspace(4)*) #2

; Function Attrs: nounwind
define spir_kernel void @testKernel(i32 addrspace(1)* nocapture %results) #1 !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !3 !kernel_arg_base_type !4 !kernel_arg_type_qual !5 {
entry:
  %call = tail call spir_func i32 @_Z13get_global_idj(i32 0) #3
  %0 = load i32, i32 addrspace(1)* @gint, align 4
  %call.i = tail call spir_func i32 @_Z9get_fencePU3AS4v(i8 addrspace(4)* addrspacecast (i8 addrspace(1)* bitcast (i32 addrspace(1)* @gint to i8 addrspace(1)*) to i8 addrspace(4)*)) #3
  %switch.i.i = icmp ult i32 %call.i, 4
  %1 = load i32, i32 addrspace(4)* addrspacecast (i32 addrspace(1)* @gint to i32 addrspace(4)*), align 4
  %cmp.i = icmp eq i32 %1, %0
  %and4.i = and i1 %switch.i.i, %cmp.i
  %cond = zext i1 %and4.i to i32
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %results, i32 %call
  store i32 %cond, i32 addrspace(1)* %arrayidx, align 4
  ret void
}

declare spir_func i32 @_Z13get_global_idj(i32) #2

attributes #0 = { nounwind readnone "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind }

!opencl.enable.FP_CONTRACT = !{}
!opencl.spir.version = !{!6}
!opencl.ocl.version = !{!7}
!opencl.used.extensions = !{!8}
!opencl.used.optional.core.features = !{!8}
!opencl.compiler.options = !{!8}

!1 = !{i32 1}
!2 = !{!"none"}
!3 = !{!"uint*"}
!4 = !{!"uint*"}
!5 = !{!""}
!6 = !{i32 1, i32 2}
!7 = !{i32 2, i32 0}
!8 = !{}
