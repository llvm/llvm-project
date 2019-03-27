;; There are no blocks in SPIR-V. Therefore they are translated into regular
;; functions. An LLVM module which uses blocks, also contains some auxiliary
;; block-specific instructions, which are redundant in SPIR-V and should be
;; removed

;; file: block.cl
;; kernel void block_kernel(__global int* res) {
;;   typedef int (^block_t)(int);
;;   constant block_t b1 = ^(int i) { return i + 1; };
;;   *res = b1(5);
;; }
;; clang -cc1 -O0 -triple spir-unknown-unknown -cl-std=CL2.0 -x cl block.cl -emit-llvm -o - | opt -mem2reg -S > global_block.ll

; RUN: llvm-as < %s > %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv -o - | llvm-dis | FileCheck %s --check-prefix=CHECK-LLVM

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

; CHECK-SPIRV: Name [[block_invoke:[0-9]+]] "_block_invoke"
; CHECK-SPIRV: TypeInt [[int:[0-9]+]] 32
; CHECK-SPIRV: TypeInt [[int8:[0-9]+]] 8
; CHECK-SPIRV: Constant [[int]] [[five:[0-9]+]] 5
; CHECK-SPIRV: TypePointer [[int8Ptr:[0-9]+]] 8 [[int8]]
; CHECK-SPIRV: TypeFunction [[block_invoke_type:[0-9]+]] [[int]] [[int8Ptr]] [[int]]

%struct.__opencl_block_literal_generic = type { i32, i32, i8 addrspace(4)* }

@block_kernel.b1 = internal addrspace(2) constant %struct.__opencl_block_literal_generic addrspace(4)* addrspacecast (%struct.__opencl_block_literal_generic addrspace(1)* bitcast ({ i32, i32, i8 addrspace(4)* } addrspace(1)* @__block_literal_global to %struct.__opencl_block_literal_generic addrspace(1)*) to %struct.__opencl_block_literal_generic addrspace(4)*), align 4
@__block_literal_global = internal addrspace(1) constant { i32, i32, i8 addrspace(4)* } { i32 12, i32 4, i8 addrspace(4)* addrspacecast (i8* bitcast (i32 (i8 addrspace(4)*, i32)* @_block_invoke to i8*) to i8 addrspace(4)*) }, align 4

; Function Attrs: convergent noinline nounwind optnone
define spir_kernel void @block_kernel(i32 addrspace(1)* %res) #0 !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !5 !kernel_arg_base_type !5 !kernel_arg_type_qual !6 {
entry:
  %res.addr = alloca i32 addrspace(1)*, align 4
  store i32 addrspace(1)* %res, i32 addrspace(1)** %res.addr, align 4
; CHECK-SPIRV: FunctionCall [[int]] {{[0-9]+}} [[block_invoke]] {{[0-9]+}} [[five]]
; CHECK-LLVM: %call = call spir_func i32 @_block_invoke(i8 addrspace(4)* {{.*}}, i32 5)
  %call = call spir_func i32 @_block_invoke(i8 addrspace(4)* addrspacecast (i8 addrspace(1)* bitcast ({ i32, i32, i8 addrspace(4)* } addrspace(1)* @__block_literal_global to i8 addrspace(1)*) to i8 addrspace(4)*), i32 5) #2
  %0 = load i32 addrspace(1)*, i32 addrspace(1)** %res.addr, align 4
  store i32 %call, i32 addrspace(1)* %0, align 4
  ret void
}

; CHECK-SPIRV: 5 Function [[int]] [[block_invoke]] 2 [[block_invoke_type]]
; CHECK-SPIRV-NEXT: 3 FunctionParameter [[int8Ptr]] {{[0-9]+}}
; CHECK-SPIRV-NEXT: 3 FunctionParameter [[int]] {{[0-9]+}}
; CHECK-LLVM: define internal spir_func i32 @_block_invoke(i8 addrspace(4)* {{.*}}, i32 %{{.*}})
; Function Attrs: convergent noinline nounwind optnone
define internal spir_func i32 @_block_invoke(i8 addrspace(4)* %.block_descriptor, i32 %i) #1 {
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

attributes #0 = { convergent noinline nounwind optnone "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "uniform-work-group-size"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { convergent noinline nounwind optnone "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { convergent }

!llvm.module.flags = !{!0}
!opencl.ocl.version = !{!1}
!opencl.spir.version = !{!1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 2, i32 0}
!2 = !{!"clang version 9.0.0 (https://llvm.org/git/clang 04fb8964a801a5c5d7baa5a22272243a7d183896) (https://llvm.org/git/llvm 384f64397f6ad95a361b72d62c07d7bac9f24163)"}
!3 = !{i32 1}
!4 = !{!"none"}
!5 = !{!"int*"}
!6 = !{!""}
