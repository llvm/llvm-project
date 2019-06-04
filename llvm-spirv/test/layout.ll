; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o %t
; RUN: FileCheck < %t %s
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv

; CHECK: 119734787 {{[0-9]*}} {{[0-9]*}} {{[0-9]*}} 0
; CHECK-NEXT: {{[0-9]*}} Capability
; CHECK: {{[0-9]*}} ExtInstImport
; CHECK-NEXT: {{[0-9]*}} MemoryModel
; CHECK-NEXT: {{[0-9]*}} EntryPoint
; CHECK: {{[0-9]*}} Source

; CHECK-NOT: {{[0-9]*}} Capability
; CHECK-NOT: {{[0-9]*}} ExtInstImport
; CHECK-NOT: {{[0-9]*}} MemoryModel
; CHECK-NOT: {{[0-9]*}} EntryPoint
; CHECK-NOT: {{[0-9]*}} Source
; CHECK-NOT: {{[0-9]*}} Decorate
; CHECK-NOT: {{[0-9]*}} Type
; CHECK-NOT: {{[0-9]*}} Variable
; CHECK-NOT: {{[0-9]*}} Function

; CHECK: {{[0-9]*}} Name

; CHECK-NOT: {{[0-9]*}} Capability
; CHECK-NOT: {{[0-9]*}} ExtInstImport
; CHECK-NOT: {{[0-9]*}} MemoryModel
; CHECK-NOT: {{[0-9]*}} EntryPoint
; CHECK-NOT: {{[0-9]*}} Source
; CHECK-NOT: {{[0-9]*}} Type
; CHECK-NOT: {{[0-9]*}} Variable
; CHECK-NOT: {{[0-9]*}} Function

; CHECK: {{[0-9]*}} Decorate

; CHECK-NOT: {{[0-9]*}} Capability
; CHECK-NOT: {{[0-9]*}} ExtInstImport
; CHECK-NOT: {{[0-9]*}} MemoryModel
; CHECK-NOT: {{[0-9]*}} EntryPoint
; CHECK-NOT: {{[0-9]*}} Source
; CHECK-NOT: {{[0-9]*}} Name
; CHECK-NOT: {{[0-9]*}} Variable
; CHECK-NOT: {{[0-9]*}} Function

; CHECK: {{[0-9]*}} TypeForwardPointer [[AFwdPtr:[0-9]+]]
; CHECK: {{[0-9]*}} TypeInt [[TypeInt:[0-9]+]]
; CHECK: {{[0-9]*}} Constant [[TypeInt]] [[Two:[0-9]+]] 2
; CHECK: {{[0-9]*}} TypePointer [[TPointer:[0-9]+]]
; CHECK: {{[0-9]*}} TypePointer [[SConstOpType:[0-9]+]]
; CHECK: {{[0-9]*}} TypeFloat [[TypeFloat:[0-9]+]]
; CHECK: {{[0-9]*}} TypeArray [[TypeArray:[0-9]+]] [[TypeFloat]] [[Two]]
; CHECK: {{[0-9]*}} TypeVector [[TypeVectorInt3:[0-9]+]] [[TypeInt]] 3
; CHECK: {{[0-9]*}} TypeStruct [[BID:[0-9]+]] {{[0-9]+}} [[AFwdPtr]]
; CHECK: {{[0-9]*}} TypeStruct [[CID:[0-9]+]] {{[0-9]+}} [[BID]]
; CHECK: {{[0-9]*}} TypeStruct [[AID:[0-9]+]] {{[0-9]+}} [[CID]]
; CHECK: {{[0-9]*}} TypePointer [[AFwdPtr]] {{[0-9]*}} [[AID]]
; CHECK: {{[0-9]*}} TypeVoid [[Void:[0-9]+]]
; CHECK: {{[0-9]*}} TypePointer [[Int3Ptr:[0-9]+]] {{[0-9]+}} [[TypeVectorInt3]]
; CHECK: {{[0-9]*}} TypeFunction [[TypeBar1:[0-9]+]] [[Void]] [[Int3Ptr]]
; CHECK: {{[0-9]*}} Variable [[TPointer]] [[Var:[0-9]+]]
; CHECK: {{[0-9]*}} SpecConstantOp [[SConstOpType]] [[SConstOp:[0-9]+]] 70 [[Var]]
; CHECK: {{[0-9]*}} Variable {{[0-9]+}} {{[0-9]+}} 5 [[SConstOp]]

; CHECK-NOT: {{[0-9]*}} Capability
; CHECK-NOT: {{[0-9]*}} ExtInstImport
; CHECK-NOT: {{[0-9]*}} MemoryModel
; CHECK-NOT: {{[0-9]*}} EntryPoint
; CHECK-NOT: {{[0-9]*}} Source
; CHECK-NOT: {{[0-9]*}} Name
; CHECK-NOT: {{[0-9]*}} Decorate

; CHECK: {{[0-9]*}} Function
; CHECK: {{[0-9]*}} FunctionParameter
; CHECK-NOT: {{[0-9]*}} Return
; CHECK: {{[0-9]*}} FunctionEnd

; CHECK-NOT: {{[0-9]*}} Capability
; CHECK-NOT: {{[0-9]*}} ExtInstImport
; CHECK-NOT: {{[0-9]*}} MemoryModel
; CHECK-NOT: {{[0-9]*}} EntryPoint
; CHECK-NOT: {{[0-9]*}} Source
; CHECK-NOT: {{[0-9]*}} Name
; CHECK-NOT: {{[0-9]*}} Type
; CHECK-NOT: {{[0-9]*}} Decorate
; CHECK-NOT: {{[0-9]*}} Variable

; CHECK: {{[0-9]*}} Function
; CHECK: {{[0-9]*}} FunctionParameter
; CHECK: {{[0-9]*}} FunctionParameter
; CHECK-NOT: {{[0-9]*}} Return
; CHECK: {{[0-9]*}} FunctionEnd

; CHECK-NOT: {{[0-9]*}} Capability
; CHECK-NOT: {{[0-9]*}} ExtInstImport
; CHECK-NOT: {{[0-9]*}} MemoryModel
; CHECK-NOT: {{[0-9]*}} EntryPoint
; CHECK-NOT: {{[0-9]*}} Source
; CHECK-NOT: {{[0-9]*}} Name
; CHECK-NOT: {{[0-9]*}} Type
; CHECK-NOT: {{[0-9]*}} Decorate
; CHECK-NOT: {{[0-9]*}} Variable

; CHECK: {{[0-9]*}} Function
; CHECK: {{[0-9]*}} FunctionParameter
; CHECK: {{[0-9]*}} Label
; CHECK: {{[0-9]*}} FunctionCall
; CHECK: {{[0-9]*}} FunctionCall
; CHECK: {{[0-9]*}} Return
; CHECK: {{[0-9]*}} FunctionEnd

; CHECK-NOT: {{[0-9]*}} Capability
; CHECK-NOT: {{[0-9]*}} ExtInstImport
; CHECK-NOT: {{[0-9]*}} MemoryModel
; CHECK-NOT: {{[0-9]*}} EntryPoint
; CHECK-NOT: {{[0-9]*}} Source
; CHECK-NOT: {{[0-9]*}} Name
; CHECK-NOT: {{[0-9]*}} Type
; CHECK-NOT: {{[0-9]*}} Decorate
; CHECK-NOT: {{[0-9]*}} Variable

; ModuleID = 'layout.bc'
target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir"

@v = addrspace(1) global [2 x i32] [i32 1, i32 2], align 4
@s = addrspace(1) global i32 addrspace(1)* getelementptr inbounds ([2 x i32], [2 x i32] addrspace(1)* @v, i32 0, i32 0), align 4

%struct.A = type { i32, %struct.C }
%struct.C = type { i32, %struct.B }
%struct.B = type { i32, %struct.A addrspace(4)* }

@f = addrspace(2) constant [2 x float] zeroinitializer, align 4
@b = external addrspace(2) constant <3 x i32>
@a = common addrspace(1) global %struct.A zeroinitializer, align 4

; Function Attrs: nounwind
define spir_kernel void @foo(<3 x i32> addrspace(1)* %a) #0 !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !3 !kernel_arg_base_type !4 !kernel_arg_type_qual !5 {
entry:
  call spir_func void @bar1(<3 x i32> addrspace(1)* %a)
  %loadVec4 = load <4 x i32> , <4 x i32> addrspace(2)* bitcast (<3 x i32> addrspace(2)* @b to <4 x i32> addrspace(2)*)
  %extractVec = shufflevector <4 x i32> %loadVec4, <4 x i32> undef, <3 x i32> <i32 0, i32 1, i32 2>
  call spir_func void @bar2(<3 x i32> addrspace(1)* %a, <3 x i32> %extractVec)
  ret void
}

declare spir_func void @bar1(<3 x i32> addrspace(1)*) #1

declare spir_func void @bar2(<3 x i32> addrspace(1)*, <3 x i32>) #1

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!opencl.enable.FP_CONTRACT = !{}
!opencl.spir.version = !{!6}
!opencl.ocl.version = !{!7}
!opencl.used.extensions = !{!8}
!opencl.used.optional.core.features = !{!8}
!opencl.compiler.options = !{!8}

!0 = !{void (<3 x i32> addrspace(1)*)* @foo, !1, !2, !3, !4, !5}
!1 = !{i32 1}
!2 = !{!"none"}
!3 = !{!"int3*"}
!4 = !{!"int3*"}
!5 = !{!""}
!6 = !{i32 1, i32 2}
!7 = !{i32 2, i32 0}
!8 = !{}

