; RUN: llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_io_pipes %s -o - | FileCheck %s
; XFAIL: *

; CHECK: OpCapability PipeStorage
; CHECK: OpCapability IOPipeINTEL
; CHECK: OpExtension "SPV_INTEL_io_pipes"

; CHECK: OpName %[[#MYPIPE_ID:]] "mygpipe"
; CHECK: OpDecorate %[[#MYPIPE_ID]] IOPipeStorageINTEL 1

; CHECK: %[[#PIPE_STORAGE_ID:]] = OpTypePipeStorage
; TODO: struct should have TypePipeStorage, not TypePointer
; CHECK: %[[#CL_PIPE_STORAGE_ID:]] = OpTypeStruct
; CHECK: %[[#CL_PIPE_STORAGE_PTR_ID:]] = OpTypePointer CrossWorkgroup %[[#CL_PIPE_STORAGE_ID]]

; CHECK: %[[#CPS_ID:]] = OpConstantPipeStorage %[[#PIPE_STORAGE_ID]] 16 16 1
; CHECK: %[[#COMPOSITE_ID:]] = OpConstantComposite  %[[#CL_PIPE_STORAGE_ID]] %[[#CPS_ID]]
; CHECK:  %[[#MYPIPE_ID]] = OpVariable %[[#CL_PIPE_STORAGE_PTR_ID]] CrossWorkgroup %[[#COMPOSITE_ID]]

%spirv.ConstantPipeStorage = type { i32, i32, i32 }
%"class.cl::pipe_storage<int __attribute__((ext_vector_type(4))), 1>" = type { ptr addrspace(1) }
%spirv.PipeStorage = type opaque

@_ZN2cl9__details29OpConstantPipeStorage_CreatorILi16ELi16ELi1EE5valueE = linkonce_odr addrspace(1) global %spirv.ConstantPipeStorage { i32 16, i32 16, i32 1 }, align 4
@mygpipe = addrspace(1) global %"class.cl::pipe_storage<int __attribute__((ext_vector_type(4))), 1>" { ptr addrspace(1) @_ZN2cl9__details29OpConstantPipeStorage_CreatorILi16ELi16ELi1EE5valueE }, align 4, !io_pipe_id !0

define spir_kernel void @worker() {
entry:
  ret void
}

!0 = !{i32 1}
