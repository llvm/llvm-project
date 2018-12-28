; ModuleID = ''
target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.bc
; RUN: llvm-dis < %t.bc | FileCheck %s

; Check 'LLVM ==> SPIR-V ==> LLVM' conversion of atomic_compare_exchange_strong and atomic_compare_exchange_weak.

; Function Attrs: nounwind

; CHECK-LABEL:   define spir_func void @test_strong
; CHECK-NEXT:    entry:
; CHECK:         [[PTR_STRONG:%expected[0-9]*]] = alloca i32, align 4
; CHECK:         store i32 {{.*}}, i32* [[PTR_STRONG]]
; CHECK:         call spir_func i1 @_Z39atomic_compare_exchange_strong_explicit{{.*}}(i32 {{.*}}* %object, i32* [[PTR_STRONG]], i32 %desired, i32 5, i32 5, i32 2)
; CHECK:         load i32, i32* [[PTR_STRONG]]

; CHECK-LABEL:   define spir_func void @test_weak
; CHECK-NEXT:    entry:
; CHECK:         [[PTR_WEAK:%expected[0-9]*]] = alloca i32, align 4
; CHECK:         store i32 {{.*}}, i32* [[PTR_WEAK]]
; CHECK:         call spir_func i1 @_Z37atomic_compare_exchange_weak_explicitPU3AS4VU7_AtomiciPii12memory_orderS4_12memory_scope{{.*}}(i32 {{.*}}* %object, i32* [[PTR_WEAK]], i32 %desired, i32 5, i32 5, i32 2)
; CHECK:         load i32, i32* [[PTR_WEAK]]

; Check that alloca for atomic_compare_exchange is being created in the entry block.

; CHECK-LABEL:   @atomic_in_loop
; CHECK-LABEL:   entry:
; CHECK:         %expected{{[0-9]*}} = alloca i32
; CHECK-LABEL:   for.body:
; CHECK-NOT:     %expected{{[0-9]*}} = alloca i32
; CHECK:         call spir_func i1 @_Z39atomic_compare_exchange_strong_explicit{{.*}}(i32 {{.*}}* {{.*}}, i32* {{.*}}, i32 {{.*}}, i32 5, i32 5, i32 2)

; Function Attrs: nounwind
define spir_func void @test_strong(i32 addrspace(4)* %object, i32 addrspace(4)* %expected, i32 %desired) #0 {
entry:
  %call = tail call spir_func zeroext i1 @_Z30atomic_compare_exchange_strongPVU3AS4U7_AtomiciPU3AS4ii(i32 addrspace(4)* %object, i32 addrspace(4)* %expected, i32 %desired) #2
  ret void
}

declare spir_func zeroext i1 @_Z30atomic_compare_exchange_strongPVU3AS4U7_AtomiciPU3AS4ii(i32 addrspace(4)*, i32 addrspace(4)*, i32) #1

; Function Attrs: nounwind
define spir_func void @test_weak(i32 addrspace(4)* %object, i32 addrspace(4)* %expected, i32 %desired) #0 {
entry:
  %call2 = tail call spir_func zeroext i1 @_Z28atomic_compare_exchange_weakPVU3AS4U7_AtomiciPU3AS4ii(i32 addrspace(4)* %object, i32 addrspace(4)* %expected, i32 %desired) #2
  ret void
}

declare spir_func zeroext i1 @_Z28atomic_compare_exchange_weakPVU3AS4U7_AtomiciPU3AS4ii(i32 addrspace(4)*, i32 addrspace(4)*, i32) #1

; Function Attrs: nounwind
define spir_kernel void @atomic_in_loop(i32 addrspace(1)* %destMemory, i32 addrspace(1)* %oldValues) #0 {
entry:
  %destMemory.addr = alloca i32 addrspace(1)*, align 8
  %oldValues.addr = alloca i32 addrspace(1)*, align 8
  %expected = alloca i32, align 4
  %previous = alloca i32, align 4
  %i = alloca i32, align 4
  store i32 addrspace(1)* %destMemory, i32 addrspace(1)** %destMemory.addr, align 8
  store i32 addrspace(1)* %oldValues, i32 addrspace(1)** %oldValues.addr, align 8
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, i32* %i, align 4
  %cmp = icmp slt i32 %0, 100000
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %1 = load i32 addrspace(1)*, i32 addrspace(1)** %destMemory.addr, align 8
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %1, i64 0
  %2 = addrspacecast i32 addrspace(1)* %arrayidx to i32 addrspace(4)*
  %3 = addrspacecast i32* %expected to i32 addrspace(4)*
  %4 = load i32 addrspace(1)*, i32 addrspace(1)** %oldValues.addr, align 8
  %arrayidx1 = getelementptr inbounds i32, i32 addrspace(1)* %4, i64 0
  %5 = load i32, i32 addrspace(1)* %arrayidx1, align 4
  %call = call spir_func zeroext i1 @_Z30atomic_compare_exchange_strongPVU3AS4U7_AtomiciPU3AS4ii(i32 addrspace(4)* %2, i32 addrspace(4)* %3, i32 %5)
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %6 = load i32, i32* %i, align 4
  %inc = add nsw i32 %6, 1
  store i32 %inc, i32* %i, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind }

!opencl.enable.FP_CONTRACT = !{}
!opencl.spir.version = !{!0}
!opencl.ocl.version = !{!1}
!opencl.used.extensions = !{!2}
!opencl.used.optional.core.features = !{!2}
!opencl.compiler.options = !{!2}

!0 = !{i32 1, i32 2}
!1 = !{i32 2, i32 0}
!2 = !{}
