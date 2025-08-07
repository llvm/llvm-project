; RUN: opt < %s -S -passes=infer-address-spaces | FileCheck %s --check-prefix=INFER
; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_90 -mattr=+ptx80 | FileCheck %s --check-prefix=PTX
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64 -mcpu=sm_90 -mattr=+ptx80 | %ptxas-verify %}

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-unknown-unknown"

@constant_tensormap = addrspace(4) global [64 x i8] zeroinitializer, align 64

; Inference from const address space
define void @test_infer_const_from_cast() {
; INFER-LABEL: @test_infer_const_from_cast
; INFER: call void @llvm.nvvm.prefetch.tensormap.p4(ptr addrspace(4) @constant_tensormap)
; BOTH: call void @llvm.nvvm.prefetch.tensormap.p4(ptr addrspace(4) @constant_tensormap)
; PTX-LABEL: .visible .func test_infer_const_from_cast(
; PTX: mov.b64 %rd{{[0-9]+}}, constant_tensormap;
; PTX: cvta.const.u64 %rd{{[0-9]+}}, %rd{{[0-9]+}};
; PTX: prefetch.tensormap [%rd{{[0-9]+}}];
entry:
  %casted = addrspacecast ptr addrspace(4) @constant_tensormap to ptr
  call void @llvm.nvvm.prefetch.tensormap.p0(ptr %casted)
  ret void
}

; Cast from Const space to Generic
define void @test_const_to_generic_cast(ptr addrspace(4) %const_ptr) {
; INFER-LABEL: @test_const_to_generic_cast
; INFER: call void @llvm.nvvm.prefetch.tensormap.p4(ptr addrspace(4) %const_ptr)
; PTX-LABEL: .visible .func test_const_to_generic_cast(
; PTX: prefetch.const.tensormap [%rd{{[0-9]+}}];
entry:
  %cast = addrspacecast ptr addrspace(4) %const_ptr to ptr
  call void @llvm.nvvm.prefetch.tensormap.p0(ptr %cast)
  ret void
}

; No inference possible 
define void @test_no_inference_possible(ptr %generic_ptr) {
; INFER-LABEL: @test_no_inference_possible
; INFER: call void @llvm.nvvm.prefetch.tensormap.p0(ptr %generic_ptr)
; PTX-LABEL: .visible .func test_no_inference_possible(
; PTX: prefetch.tensormap [%rd{{[0-9]+}}]; 
entry:
  call void @llvm.nvvm.prefetch.tensormap.p0(ptr %generic_ptr)
  ret void
}

; Cast from Parameter space to Generic
define void @test_param_to_generic_cast(ptr addrspace(101) %param_ptr) {
; INFER-LABEL: @test_param_to_generic_cast
; INFER: call void @llvm.nvvm.prefetch.tensormap.p101(ptr addrspace(101) %param_ptr)
; PTX-LABEL: .visible .func test_param_to_generic_cast(
; PTX: prefetch.param.tensormap [%rd{{[0-9]+}}];
entry:
  %cast = addrspacecast ptr addrspace(101) %param_ptr to ptr
  call void @llvm.nvvm.prefetch.tensormap.p0(ptr %cast)
  ret void
}

; Multiple casts in sequence
define void @test_infer_through_multiple_casts() {
; INFER-LABEL: @test_infer_through_multiple_casts
; INFER: call void @llvm.nvvm.prefetch.tensormap.p4(ptr addrspace(4) @constant_tensormap)
; PTX-LABEL: .visible .func test_infer_through_multiple_casts(
; PTX: mov.b64 %rd{{[0-9]+}}, constant_tensormap;
; PTX: cvta.const.u64 %rd{{[0-9]+}}, %rd{{[0-9]+}};
; PTX: prefetch.tensormap [%rd{{[0-9]+}}];
entry:
  %cast1 = addrspacecast ptr addrspace(4) @constant_tensormap to ptr
  %cast2 = addrspacecast ptr %cast1 to ptr addrspace(4)
  %cast3 = addrspacecast ptr addrspace(4) %cast2 to ptr
  call void @llvm.nvvm.prefetch.tensormap(ptr %cast3)
  ret void
}

declare void @llvm.nvvm.prefetch.tensormap.p0(ptr)
declare void @llvm.nvvm.prefetch.tensormap.p4(ptr addrspace(4))
declare void @llvm.nvvm.prefetch.tensormap.p101(ptr addrspace(101))
