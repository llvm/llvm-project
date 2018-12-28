; This test checks that the translator doesn't crash if the module has literal
; structs, i.e. structs whose type has no name. Typicaly clang generate such
; structs if the kernel contains OpenCL 2.0 blocks. The IR was produced with
; the following command:
; clang -cc1 -triple spir -cl-std=cl2.0 -O0 -finclude-default-header literal-struct.cl -emit-llvm -o test/literal-struct.ll

; literal-struct.cl:
; void foo()
; {
;   void (^myBlock)(void) = ^{};
;   myBlock();
; }

; RUN: llvm-as < %s | llvm-spirv -spirv-text -o %t
; RUN: FileCheck < %t %s

; CHECK-DAG: TypeInt [[Int:[0-9]+]] 32 0
; CHECK-DAG: TypeStruct [[StructType:[0-9]+]] [[Int]] [[Int]] {{$}}

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir"

@__block_literal_global = internal addrspace(1) constant { i32, i32 } { i32 8, i32 4 }, align 4
; CHECK: ConstantComposite [[StructType]]

; This is artificial case is added to cover ConstantNull instrucitions with TypeStruct.
@__block_literal_global.1 = internal addrspace(1) constant { i32, i32 } zeroinitializer, align 4
; CHECK: ConstantNull [[StructType]]

; Function Attrs: convergent noinline nounwind optnone
define spir_func void @foo() #0 {
entry:
  %myBlock = alloca void () addrspace(4)*, align 4
  store void () addrspace(4)* addrspacecast (void () addrspace(1)* bitcast ({ i32, i32 } addrspace(1)* @__block_literal_global to void () addrspace(1)*) to void () addrspace(4)*), void () addrspace(4)** %myBlock, align 4
  call spir_func void @__foo_block_invoke(i8 addrspace(4)* addrspacecast (i8 addrspace(1)* bitcast ({ i32, i32 } addrspace(1)* @__block_literal_global to i8 addrspace(1)*) to i8 addrspace(4)*)) #1
  ret void
}

; Function Attrs: convergent noinline nounwind optnone
define internal spir_func void @__foo_block_invoke(i8 addrspace(4)* %.block_descriptor) #0 {
entry:
  %.block_descriptor.addr = alloca i8 addrspace(4)*, align 4
  %block.addr = alloca <{ i32, i32 }> addrspace(4)*, align 4
  store i8 addrspace(4)* %.block_descriptor, i8 addrspace(4)** %.block_descriptor.addr, align 4
  %block = bitcast i8 addrspace(4)* %.block_descriptor to <{ i32, i32 }> addrspace(4)*
  store <{ i32, i32 }> addrspace(4)* %block, <{ i32, i32 }> addrspace(4)** %block.addr, align 4
  ret void
}

attributes #0 = { convergent noinline nounwind optnone "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { convergent }

!llvm.module.flags = !{!0}
!opencl.ocl.version = !{!1}
!opencl.spir.version = !{!1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 2, i32 0}
!2 = !{!"clang version 8.0.0 "}
