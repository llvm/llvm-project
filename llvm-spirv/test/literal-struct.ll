; This test checks that the translator doesn't crash if the module has literal
; structs, i.e. structs whose type has no name. Typicaly clang generate such
; structs if the kernel contains OpenCL 2.0 blocks. The IR was produced with
; the following command:
; clang -cc1 -triple spir -cl-std=cl2.0 -O0 literal-struct.cl -emit-llvm -o test/literal-struct.ll

; literal-struct.cl:
; void foo()
; {
;   void (^myBlock)(void) = ^{};
;   myBlock();
; }

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o %t
; RUN: FileCheck < %t %s
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv

; CHECK: TypeInt [[Int:[0-9]+]] 32 0
; CHECK: TypeInt [[Int8:[0-9]+]] 8 0
; CHECK: TypePointer [[Int8Ptr:[0-9]+]] 8 [[Int8]]
; CHECK: TypeStruct [[StructType:[0-9]+]] [[Int]] [[Int]] [[Int8Ptr]]

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir"

%struct.__opencl_block_literal_generic = type { i32, i32, i8 addrspace(4)* }

@__block_literal_global = internal addrspace(1) constant { i32, i32, i8 addrspace(4)* } { i32 12, i32 4, i8 addrspace(4)* addrspacecast (i8* bitcast (void (i8 addrspace(4)*)* @__foo_block_invoke to i8*) to i8 addrspace(4)*) }, align 4
; CHECK: ConstantComposite [[StructType]]

@__block_literal_global.1 = internal addrspace(1) constant { i32, i32, i8 addrspace(4)* } zeroinitializer, align 4
; CHECK: ConstantNull [[StructType]]

; Function Attrs: convergent noinline nounwind optnone
define spir_func void @foo() #0 {
entry:
  %myBlock = alloca %struct.__opencl_block_literal_generic addrspace(4)*, align 4
  store %struct.__opencl_block_literal_generic addrspace(4)* addrspacecast (%struct.__opencl_block_literal_generic addrspace(1)* bitcast ({ i32, i32, i8 addrspace(4)* } addrspace(1)* @__block_literal_global to %struct.__opencl_block_literal_generic addrspace(1)*) to %struct.__opencl_block_literal_generic addrspace(4)*), %struct.__opencl_block_literal_generic addrspace(4)** %myBlock, align 4
  call spir_func void @__foo_block_invoke(i8 addrspace(4)* addrspacecast (i8 addrspace(1)* bitcast ({ i32, i32, i8 addrspace(4)* } addrspace(1)* @__block_literal_global to i8 addrspace(1)*) to i8 addrspace(4)*)) #1
  ret void
}

; Function Attrs: convergent noinline nounwind optnone
define internal spir_func void @__foo_block_invoke(i8 addrspace(4)* %.block_descriptor) #0 {
entry:
  %.block_descriptor.addr = alloca i8 addrspace(4)*, align 4
  %block.addr = alloca <{ i32, i32, i8 addrspace(4)* }> addrspace(4)*, align 4
  store i8 addrspace(4)* %.block_descriptor, i8 addrspace(4)** %.block_descriptor.addr, align 4
  %block = bitcast i8 addrspace(4)* %.block_descriptor to <{ i32, i32, i8 addrspace(4)* }> addrspace(4)*
  store <{ i32, i32, i8 addrspace(4)* }> addrspace(4)* %block, <{ i32, i32, i8 addrspace(4)* }> addrspace(4)** %block.addr, align 4
  ret void
}

attributes #0 = { convergent noinline nounwind optnone "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { convergent }

!llvm.module.flags = !{!0}
!opencl.ocl.version = !{!1}
!opencl.spir.version = !{!1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 2, i32 0}
!2 = !{!"clang version 9.0.0 (https://llvm.org/git/clang 04fb8964a801a5c5d7baa5a22272243a7d183896) (https://llvm.org/git/llvm 384f64397f6ad95a361b72d62c07d7bac9f24163)"}
