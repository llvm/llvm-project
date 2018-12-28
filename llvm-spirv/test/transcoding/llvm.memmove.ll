; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o %t.txt
; RUN: FileCheck < %t.txt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV-NOT: llvm.memmove

; CHECK-SPIRV: Variable {{[0-9]+}} [[mem:[0-9]+]] 7
; CHECK-SPIRV: Bitcast [[i8Ty:[0-9]+]] [[tmp0:[0-9]+]] [[mem]]
; CHECK-SPIRV: LifetimeStart [[tmp0]] [[size:[0-9]+]]
; CHECK-SPIRV: Bitcast [[i8Ty]] [[tmp1:[0-9]+]] [[mem]]
; CHECK-SPIRV: CopyMemorySized [[tmp1]] {{[0-9]+}} {{[0-9]+}}
; CHECK-SPIRV: Bitcast [[i8Ty]] [[tmp2:[0-9]+]] [[mem]]
; CHECK-SPIRV: CopyMemorySized {{[0-9]+}} [[tmp2]] {{[0-9]+}}
; CHECK-SPIRV: Bitcast [[i8Ty]] [[tmp3:[0-9]+]] [[mem]]
; CHECK-SPIRV: LifetimeStop [[tmp3]] [[size]]

; CHECK-LLVM-NOT: llvm.memmove

; CHECK-LLVM: [[local:%[0-9]+]] = alloca %struct.SomeStruct
; CHECK-LLVM: [[tmp1:%[0-9]+]] = bitcast %struct.SomeStruct* [[local]] to [[type:i[0-9]+\*]]
; CHECK-LLVM: call void @llvm.lifetime.start.p0i8({{i[0-9]+}} {{-?[0-9]+}}, [[type]] [[tmp1]])
; CHECK-LLVM: [[tmp2:%[0-9]+]] = bitcast %struct.SomeStruct* [[local]] to [[type]]
; CHECK-LLVM: call void @llvm.memcpy
; CHECK-LLVM:  ([[type]] align 64 [[tmp2]],
; CHECK-LLVM:  {{i[0-9]+}} [[size:[0-9]+]]
; CHECK-LLVM: [[tmp3:%[0-9]+]] = bitcast %struct.SomeStruct* [[local]] to [[type]]
; CHECK-LLVM: call void @llvm.memcpy
; CHECK-LLVM:  , [[type]] align 64 [[tmp3]], {{i[0-9]+}} [[size]]
; CHECK-LLVM: [[tmp4:%[0-9]+]] = bitcast %struct.SomeStruct* [[local]] to [[type]]
; CHECK-LLVM: call void @llvm.lifetime.end.p0i8({{i[0-9]+}} {{-?[0-9]+}}, [[type]] [[tmp4]])

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir-unknown-unknown"

%struct.SomeStruct = type { <16 x float>, i32, [60 x i8] }

; Function Attrs: nounwind
define spir_kernel void @test_struct(%struct.SomeStruct addrspace(1)* nocapture readonly %in, %struct.SomeStruct addrspace(1)* nocapture %out) #0 !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !3 !kernel_arg_base_type !4 !kernel_arg_type_qual !5 {
  %1 = bitcast %struct.SomeStruct addrspace(1)* %in to i8 addrspace(1)*
  %2 = bitcast %struct.SomeStruct addrspace(1)* %out to i8 addrspace(1)*
  call void @llvm.memmove.p1i8.p1i8.i32(i8 addrspace(1)* %2, i8 addrspace(1)* %1, i32 128, i32 64, i1 false)
  ret void
}

; Function Attrs: nounwind
declare void @llvm.memmove.p1i8.p1i8.i32(i8 addrspace(1)* nocapture, i8 addrspace(1)* nocapture readonly, i32, i32, i1) #1

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind }

!opencl.enable.FP_CONTRACT = !{}
!opencl.spir.version = !{!7}
!opencl.ocl.version = !{!7}
!opencl.used.extensions = !{!8}
!opencl.used.optional.core.features = !{!8}
!opencl.compiler.options = !{!8}

!1 = !{i32 1, i32 1}
!2 = !{!"none", !"none"}
!3 = !{!"struct SomeStruct*", !"struct SomeStruct*"}
!4 = !{!"struct SomeStruct*", !"struct SomeStruct*"}
!5 = !{!"const", !""}
!7 = !{i32 1, i32 2}
!8 = !{}
