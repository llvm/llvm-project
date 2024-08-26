; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefixes=CHECK-SPIRV,CHECK-SPIRV1_4
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; TODO(#60133): Requires updates following opaque pointer migration.
; XFAIL: *

;; There are no blocks in SPIR-V. Therefore they are translated into regular
;; functions. An LLVM module which uses blocks, also contains some auxiliary
;; block-specific instructions, which are redundant in SPIR-V and should be
;; removed

;; kernel void block_kernel(__global int* res) {
;;   typedef int (^block_t)(int);
;;   constant block_t b1 = ^(int i) { return i + 1; };
;;   *res = b1(5);
;; }

; CHECK-SPIRV1_4:   OpEntryPoint Kernel %[[#]] "block_kernel" %[[#InterfaceId:]]
; CHECK-SPIRV1_4:   OpName %[[#InterfaceId]] "__block_literal_global"
; CHECK-SPIRV:      OpName %[[#block_invoke:]] "_block_invoke"
; CHECK-SPIRV:      %[[#int:]] = OpTypeInt 32
; CHECK-SPIRV:      %[[#int8:]] = OpTypeInt 8
; CHECK-SPIRV:      %[[#int8Ptr:]] = OpTypePointer Generic %[[#int8]]
; CHECK-SPIRV:      %[[#block_invoke_type:]] = OpTypeFunction %[[#int]] %[[#int8Ptr]] %[[#int]]
; CHECK-SPIRV:      %[[#five:]] = OpConstant %[[#int]] 5

; CHECK-SPIRV:      %[[#]] = OpFunctionCall %[[#int]] %[[#block_invoke]] %[[#]] %[[#five]]

; CHECK-SPIRV:      %[[#block_invoke]] = OpFunction %[[#int]] DontInline %[[#block_invoke_type]]
; CHECK-SPIRV-NEXT: %[[#]] = OpFunctionParameter %[[#int8Ptr]]
; CHECK-SPIRV-NEXT: %[[#]] = OpFunctionParameter %[[#int]]

%struct.__opencl_block_literal_generic = type { i32, i32, i8 addrspace(4)* }

@block_kernel.b1 = internal addrspace(2) constant %struct.__opencl_block_literal_generic addrspace(4)* addrspacecast (%struct.__opencl_block_literal_generic addrspace(1)* bitcast ({ i32, i32, i8 addrspace(4)* } addrspace(1)* @__block_literal_global to %struct.__opencl_block_literal_generic addrspace(1)*) to %struct.__opencl_block_literal_generic addrspace(4)*), align 4
@__block_literal_global = internal addrspace(1) constant { i32, i32, i8 addrspace(4)* } { i32 12, i32 4, i8 addrspace(4)* addrspacecast (i8* bitcast (i32 (i8 addrspace(4)*, i32)* @_block_invoke to i8*) to i8 addrspace(4)*) }, align 4

define dso_local spir_kernel void @block_kernel(i32 addrspace(1)* noundef %res) {
entry:
  %res.addr = alloca i32 addrspace(1)*, align 4
  store i32 addrspace(1)* %res, i32 addrspace(1)** %res.addr, align 4
  %call = call spir_func i32 @_block_invoke(i8 addrspace(4)* noundef addrspacecast (i8 addrspace(1)* bitcast ({ i32, i32, i8 addrspace(4)* } addrspace(1)* @__block_literal_global to i8 addrspace(1)*) to i8 addrspace(4)*), i32 noundef 5)
  %0 = load i32 addrspace(1)*, i32 addrspace(1)** %res.addr, align 4
  store i32 %call, i32 addrspace(1)* %0, align 4
  ret void
}

define internal spir_func i32 @_block_invoke(i8 addrspace(4)* noundef %.block_descriptor, i32 noundef %i) #0 {
entry:
  %.block_descriptor.addr = alloca i8 addrspace(4)*, align 4
  %i.addr = alloca i32, align 4
  %block.addr = alloca <{ i32, i32, i8 addrspace(4)* }> addrspace(4)*, align 4
  store i8 addrspace(4)* %.block_descriptor, i8 addrspace(4)** %.block_descriptor.addr, align 4
  %block = bitcast i8 addrspace(4)* %.block_descriptor to <{ i32, i32, i8 addrspace(4)* }> addrspace(4)*
  store i32 %i, i32* %i.addr, align 4
  store <{ i32, i32, i8 addrspace(4)* }> addrspace(4)* %block, <{ i32, i32, i8 addrspace(4)* }> addrspace(4)** %block.addr, align 4
  %0 = load i32, i32* %i.addr, align 4
  %add = add nsw i32 %0, 1
  ret i32 %add
}

attributes #0 = { noinline }
