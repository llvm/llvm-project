;; This test checks that the backend doesn't crash if the module has literal
;; structs, i.e. structs whose type has no name. Typicaly clang generate such
;; structs if the kernel contains OpenCL 2.0 blocks. The IR was produced with
;; the following command:
;; clang -cc1 -triple spir -cl-std=cl2.0 -O0 literal-struct.cl -emit-llvm -o test/literal-struct.ll

;; literal-struct.cl:
;; void foo()
;; {
;;   void (^myBlock)(void) = ^{};
;;   myBlock();
;; }

; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s

; CHECK: OpName %[[#StructType0:]] "struct.__opencl_block_literal_generic"
; CHECK: %[[#Int8:]] = OpTypeInt 8 0
; CHECK: %[[#Int8Ptr:]] = OpTypePointer Generic %[[#Int8]]
; CHECK: %[[#Int:]] = OpTypeInt 32 0
; CHECK: %[[#StructType0:]] = OpTypeStruct %[[#Int]] %[[#Int]] %[[#Int8Ptr]]
; CHECK: %[[#StructType:]] = OpTypeStruct %[[#Int]] %[[#Int]] %[[#Int8Ptr]]

%struct.__opencl_block_literal_generic = type { i32, i32, i8 addrspace(4)* }

@__block_literal_global = internal addrspace(1) constant { i32, i32, i8 addrspace(4)* } { i32 12, i32 4, i8 addrspace(4)* addrspacecast (i8* bitcast (void (i8 addrspace(4)*)* @__foo_block_invoke to i8*) to i8 addrspace(4)*) }, align 4
; CHECK: OpConstantComposite %[[#StructType]]

@__block_literal_global.1 = internal addrspace(1) constant { i32, i32, i8 addrspace(4)* } zeroinitializer, align 4
; CHECK: OpConstantNull %[[#StructType]]

define spir_func void @foo() {
entry:
  %myBlock = alloca %struct.__opencl_block_literal_generic addrspace(4)*, align 4
  store %struct.__opencl_block_literal_generic addrspace(4)* addrspacecast (%struct.__opencl_block_literal_generic addrspace(1)* bitcast ({ i32, i32, i8 addrspace(4)* } addrspace(1)* @__block_literal_global to %struct.__opencl_block_literal_generic addrspace(1)*) to %struct.__opencl_block_literal_generic addrspace(4)*), %struct.__opencl_block_literal_generic addrspace(4)** %myBlock, align 4
  call spir_func void @__foo_block_invoke(i8 addrspace(4)* addrspacecast (i8 addrspace(1)* bitcast ({ i32, i32, i8 addrspace(4)* } addrspace(1)* @__block_literal_global to i8 addrspace(1)*) to i8 addrspace(4)*))
  ret void
}

define internal spir_func void @__foo_block_invoke(i8 addrspace(4)* %.block_descriptor) {
entry:
  %.block_descriptor.addr = alloca i8 addrspace(4)*, align 4
  %block.addr = alloca <{ i32, i32, i8 addrspace(4)* }> addrspace(4)*, align 4
  store i8 addrspace(4)* %.block_descriptor, i8 addrspace(4)** %.block_descriptor.addr, align 4
  %block = bitcast i8 addrspace(4)* %.block_descriptor to <{ i32, i32, i8 addrspace(4)* }> addrspace(4)*
  store <{ i32, i32, i8 addrspace(4)* }> addrspace(4)* %block, <{ i32, i32, i8 addrspace(4)* }> addrspace(4)** %block.addr, align 4
  ret void
}
