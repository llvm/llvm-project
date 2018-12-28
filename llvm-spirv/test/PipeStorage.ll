; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o %t.txt
; RUN: FileCheck < %t.txt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; CHECK-LLVM: %spirv.ConstantPipeStorage = type { i32, i32, i32 }
; CHECK-LLVM: %"[[CL_PIPE_STORAGE_NAME:[^"]+]]" = type { %spirv.PipeStorage addrspace(1)* }
; CHECK-LLVM: %spirv.PipeStorage = type opaque
; CHECK-LLVM: [[CREATOR_NAME:[^ ]+]] = linkonce_odr addrspace(1) global %spirv.ConstantPipeStorage { i32 16, i32 16, i32 1 }, align 4
; CHECK-LLVM: @mygpipe = addrspace(1) global %"[[CL_PIPE_STORAGE_NAME]]" { %spirv.PipeStorage addrspace(1)* bitcast (%spirv.ConstantPipeStorage addrspace(1)* [[CREATOR_NAME]] to %spirv.PipeStorage addrspace(1)*) }, align 4

; check for magic number followed by version 1.1
; CHECK-SPIRV: 119734787 65792

; CHECK-SPIRV: 4 Name [[MYPIPE_ID:[0-9]+]] "mygpipe"

; CHECK-SPIRV: 2 TypePipeStorage [[PIPE_STORAGE_ID:[0-9]+]]
; CHECK-SPIRV-NEXT: 2 TypePipeStorage [[PIPE_STORAGE_ID_2:[0-9]+]]
; CHECK-SPIRV: 3 TypeStruct [[CL_PIPE_STORAGE_ID:[0-9]+]] [[PIPE_STORAGE_ID_2]]
; CHECK-SPIRV: 4 TypePointer [[CL_PIPE_STORAGE_PTR_ID:[0-9]+]] 5 [[CL_PIPE_STORAGE_ID]]

; CHECK-SPIRV: 6 ConstantPipeStorage [[PIPE_STORAGE_ID]] [[CPS_ID:[0-9]+]] 16 16 1
; CHECK-SPIRV: 4 ConstantComposite  [[CL_PIPE_STORAGE_ID]] [[COMPOSITE_ID:[0-9]+]] [[CPS_ID]]
; CHECK-SPIRV: 5 Variable [[CL_PIPE_STORAGE_PTR_ID]] [[MYPIPE_ID]] 5 [[COMPOSITE_ID]]

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

%spirv.ConstantPipeStorage = type { i32, i32, i32 }
%"class.cl::pipe_storage<int __attribute__((ext_vector_type(4))), 1>" = type { %spirv.PipeStorage addrspace(1)* }
%spirv.PipeStorage = type opaque

@_ZN2cl9__details29OpConstantPipeStorage_CreatorILi16ELi16ELi1EE5valueE = linkonce_odr addrspace(1) global %spirv.ConstantPipeStorage { i32 16, i32 16, i32 1 }, align 4
@mygpipe = addrspace(1) global %"class.cl::pipe_storage<int __attribute__((ext_vector_type(4))), 1>" { %spirv.PipeStorage addrspace(1)* bitcast (%spirv.ConstantPipeStorage addrspace(1)* @_ZN2cl9__details29OpConstantPipeStorage_CreatorILi16ELi16ELi1EE5valueE to %spirv.PipeStorage addrspace(1)*) }, align 4

; Function Attrs: nounwind
define spir_kernel void @worker() {
entry:
  ret void
}

!opencl.enable.FP_CONTRACT = !{}
!opencl.spir.version = !{!0}
!opencl.ocl.version = !{!1}
!opencl.used.extensions = !{!2}
!opencl.used.optional.core.features = !{!2}
!opencl.compiler.options = !{!2}
!llvm.ident = !{!3}
!spirv.Source = !{!4}
!spirv.String = !{}

!0 = !{i32 1, i32 2}
!1 = !{i32 2, i32 2}
!2 = !{}
!3 = !{!"clang version 3.6.1 "}
!4 = !{i32 4, i32 202000}
