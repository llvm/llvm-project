; Source
; kernel void block_ret_struct(__global int* res)
; {
;   struct A {
;       int a;
;   };
;   struct A (^kernelBlock)(struct A) = ^struct A(struct A a)
;   {
;     a.a = 6;
;     return a;
;   };
;   size_t tid = get_global_id(0);
;   res[tid] = -1;
;   struct A aa;
;   aa.a = 5;
;   res[tid] = kernelBlock(aa).a - 6;
; }

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o %t.spv.txt
; RUN: FileCheck < %t.spv.txt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc
; RUN: FileCheck < %t.rev.ll %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV: Name [[BlockInv:[0-9]+]] "__block_ret_struct_block_invoke"

; CHECK-SPIRV: 4 TypeInt [[IntTy:[0-9]+]] 32
; CHECK-SPIRV: 3 TypeStruct [[StructTy:[0-9]+]] [[IntTy]]
; CHECK-SPIRV: 4 TypePointer [[StructPtrTy:[0-9]+]] 7 [[StructTy]]

; CHECK-SPIRV: 4 Variable [[StructPtrTy]] [[StructArg:[0-9]+]] 7
; CHECK-SPIRV: 4 Variable [[StructPtrTy]] [[StructRet:[0-9]+]] 7
; CHECK-SPIRV: 4 PtrCastToGeneric {{[0-9]+}} [[BlockLit:[0-9]+]] {{[0-9]+}}
; CHECK-SPIRV: 7 FunctionCall {{[0-9]+}} {{[0-9]+}} [[BlockInv]] [[StructRet]] [[BlockLit]] [[StructArg]]

; CHECK-LLVM: %[[StructA:.*]] = type { i32 }
; CHECK-LLVM: call {{.*}} void @__block_ret_struct_block_invoke(%[[StructA]]*

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown"

%struct.A = type { i32 }

@__block_literal_global = internal addrspace(1) constant { i32, i32 } { i32 8, i32 4 }, align 4

; Function Attrs: convergent noinline nounwind optnone
define spir_kernel void @block_ret_struct(i32 addrspace(1)* %res) #0 !kernel_arg_addr_space !4 !kernel_arg_access_qual !5 !kernel_arg_type !6 !kernel_arg_base_type !6 !kernel_arg_type_qual !7 !kernel_arg_host_accessible !8 !kernel_arg_pipe_depth !9 !kernel_arg_pipe_io !7 !kernel_arg_buffer_location !7 {
entry:
  %res.addr = alloca i32 addrspace(1)*, align 8
  %kernelBlock = alloca void (%struct.A*, %struct.A*) addrspace(4)*, align 8
  %tid = alloca i64, align 8
  %aa = alloca %struct.A, align 4
  %tmp = alloca %struct.A, align 4
  store i32 addrspace(1)* %res, i32 addrspace(1)** %res.addr, align 8
  store void (%struct.A*, %struct.A*) addrspace(4)* addrspacecast (void (%struct.A*, %struct.A*) addrspace(1)* bitcast ({ i32, i32 } addrspace(1)* @__block_literal_global to void (%struct.A*, %struct.A*) addrspace(1)*) to void (%struct.A*, %struct.A*) addrspace(4)*), void (%struct.A*, %struct.A*) addrspace(4)** %kernelBlock, align 8
  %call = call spir_func i64 @_Z13get_global_idj(i32 0) #4
  store i64 %call, i64* %tid, align 8
  %0 = load i32 addrspace(1)*, i32 addrspace(1)** %res.addr, align 8
  %1 = load i64, i64* %tid, align 8
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %0, i64 %1
  store i32 -1, i32 addrspace(1)* %arrayidx, align 4
  %a = getelementptr inbounds %struct.A, %struct.A* %aa, i32 0, i32 0
  store i32 5, i32* %a, align 4
  call spir_func void @__block_ret_struct_block_invoke(%struct.A* sret %tmp, i8 addrspace(4)* addrspacecast (i8 addrspace(1)* bitcast ({ i32, i32 } addrspace(1)* @__block_literal_global to i8 addrspace(1)*) to i8 addrspace(4)*), %struct.A* byval align 4 %aa) #5
  %a1 = getelementptr inbounds %struct.A, %struct.A* %tmp, i32 0, i32 0
  %2 = load i32, i32* %a1, align 4
  %sub = sub nsw i32 %2, 6
  %3 = load i32 addrspace(1)*, i32 addrspace(1)** %res.addr, align 8
  %4 = load i64, i64* %tid, align 8
  %arrayidx2 = getelementptr inbounds i32, i32 addrspace(1)* %3, i64 %4
  store i32 %sub, i32 addrspace(1)* %arrayidx2, align 4
  ret void
}

; Function Attrs: convergent noinline nounwind optnone
define internal spir_func void @__block_ret_struct_block_invoke(%struct.A* noalias sret %agg.result, i8 addrspace(4)* %.block_descriptor, %struct.A* byval align 4 %a) #1 {
entry:
  %.block_descriptor.addr = alloca i8 addrspace(4)*, align 8
  %block.addr = alloca <{ i32, i32 }> addrspace(4)*, align 8
  store i8 addrspace(4)* %.block_descriptor, i8 addrspace(4)** %.block_descriptor.addr, align 8
  %block = bitcast i8 addrspace(4)* %.block_descriptor to <{ i32, i32 }> addrspace(4)*
  store <{ i32, i32 }> addrspace(4)* %block, <{ i32, i32 }> addrspace(4)** %block.addr, align 8
  %a1 = getelementptr inbounds %struct.A, %struct.A* %a, i32 0, i32 0
  store i32 6, i32* %a1, align 4
  %0 = bitcast %struct.A* %agg.result to i8*
  %1 = bitcast %struct.A* %a to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %0, i8* align 4 %1, i64 4, i1 false)
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1) #2

; Function Attrs: convergent nounwind readnone
declare spir_func i64 @_Z13get_global_idj(i32) #3

attributes #0 = { convergent noinline nounwind optnone "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "uniform-work-group-size"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { convergent noinline nounwind optnone "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { argmemonly nounwind }
attributes #3 = { convergent nounwind readnone "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { convergent nounwind readnone }
attributes #5 = { convergent }

!llvm.module.flags = !{!0}
!opencl.enable.FP_CONTRACT = !{}
!opencl.ocl.version = !{!1}
!opencl.spir.version = !{!1}
!opencl.used.extensions = !{!2}
!opencl.used.optional.core.features = !{!2}
!opencl.compiler.options = !{!2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 2, i32 0}
!2 = !{}
!3 = !{!"clang version 7.0.0"}
!4 = !{i32 1}
!5 = !{!"none"}
!6 = !{!"int*"}
!7 = !{!""}
!8 = !{i1 false}
!9 = !{i32 0}

