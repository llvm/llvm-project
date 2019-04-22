; RUN: llvm-as < %s | not llvm-spirv -o %t.spv 2>&1 | FileCheck %s

;kernel void test_order(global atomic_intptr_t *a, global ptrdiff_t *b, int x) {
;  memory_order order;
;  if (x)
;    order = memory_order_relaxed;
;  else
;    order = memory_order_seq_cst;
;  atomic_fetch_add_explicit(a, *b, order, memory_scope_device);
;}

; CHECK: error: memory_order argument needs to be constant

; ModuleID = 'mem_order.cl'
target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

; Function Attrs: convergent
declare dso_local spir_func i32 @_Z25atomic_fetch_add_explicitPU3AS4VU7_Atomicii12memory_order12memory_scope(i32 addrspace(4)*, i32, i32, i32) #1

; Function Attrs: convergent noinline nounwind optnone
define dso_local spir_kernel void @test_order(i32 addrspace(1)* %a, i32 addrspace(1)* %b, i32 %x) #0 !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !5 !kernel_arg_base_type !6 !kernel_arg_type_qual !7 {
entry:
  %a.addr = alloca i32 addrspace(1)*, align 4
  %b.addr = alloca i32 addrspace(1)*, align 4
  %x.addr = alloca i32, align 4
  %order = alloca i32, align 4
  store i32 addrspace(1)* %a, i32 addrspace(1)** %a.addr, align 4
  store i32 addrspace(1)* %b, i32 addrspace(1)** %b.addr, align 4
  store i32 %x, i32* %x.addr, align 4
  %0 = load i32, i32* %x.addr, align 4
  %tobool = icmp ne i32 %0, 0
  br i1 %tobool, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  store i32 0, i32* %order, align 4
  br label %if.end

if.else:                                          ; preds = %entry
  store i32 5, i32* %order, align 4
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %1 = load i32 addrspace(1)*, i32 addrspace(1)** %a.addr, align 4
  %2 = addrspacecast i32 addrspace(1)* %1 to i32 addrspace(4)*
  %3 = load i32 addrspace(1)*, i32 addrspace(1)** %b.addr, align 4
  %4 = load i32, i32 addrspace(1)* %3, align 4
  %5 = load i32, i32* %order, align 4
  %call = call spir_func i32 @_Z25atomic_fetch_add_explicitPU3AS4VU7_Atomicii12memory_order12memory_scope(i32 addrspace(4)* %2, i32 %4, i32 %5, i32 2) #2
  ret void
}

attributes #0 = { convergent noinline nounwind optnone "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "uniform-work-group-size"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { convergent "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { convergent }

!llvm.module.flags = !{!0}
!opencl.ocl.version = !{!1}
!opencl.spir.version = !{!1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 2, i32 0}
!2 = !{!"clang version 9.0.0 (https://git.llvm.org/git/clang.git f469a1cb81189ebb874b4a3317a41bd3af743fde) (https://git.llvm.org/git/llvm.git e3928a7598a181ff2a219f95987a72ce8fb16056)"}
!3 = !{i32 1, i32 1, i32 0}
!4 = !{!"none", !"none", !"none"}
!5 = !{!"atomic_intptr_t*", !"ptrdiff_t*", !"int"}
!6 = !{!"_Atomic(int)*", !"int*", !"int"}
!7 = !{!"", !"", !""}
