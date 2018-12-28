; Check that translator doesn't generate atomic instructions for atomic builtins
; which are not defined in the spec.
;
; Source
; #pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
; #pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable
;
; float __attribute__((overloadable)) atomic_add(volatile __global float *p, float val);
; float __attribute__((overloadable)) atomic_sub(volatile __global float *p, float val);
; float __attribute__((overloadable)) atomic_inc(volatile __global float *p, float val);
; float __attribute__((overloadable)) atomic_dec(volatile __global float *p, float val);
; float __attribute__((overloadable)) atomic_cmpxchg(volatile __global float *p, float val);
; double __attribute__((overloadable)) atomic_min(volatile __global double *p, double val);
; double __attribute__((overloadable)) atomic_max(volatile __global double *p, double val);
; double __attribute__((overloadable)) atomic_and(volatile __global double *p, double val);
; double __attribute__((overloadable)) atomic_or(volatile __global double *p, double val);
; double __attribute__((overloadable)) atomic_xor(volatile __global double *p, double val);
;
; float __attribute__((overloadable)) atom_add(volatile __global float *p, float val);
; float __attribute__((overloadable)) atom_sub(volatile __global float *p, float val);
; float __attribute__((overloadable)) atom_inc(volatile __global float *p, float val);
; float __attribute__((overloadable)) atom_dec(volatile __global float *p, float val);
; float __attribute__((overloadable)) atom_cmpxchg(volatile __global float *p, float val);
; double __attribute__((overloadable)) atom_min(volatile __global double *p, double val);
; double __attribute__((overloadable)) atom_max(volatile __global double *p, double val);
; double __attribute__((overloadable)) atom_and(volatile __global double *p, double val);
; double __attribute__((overloadable)) atom_or(volatile __global double *p, double val);
; double __attribute__((overloadable)) atom_xor(volatile __global double *p, double val);
;
; float __attribute__((overloadable)) atomic_fetch_add(volatile generic atomic_float *object, float operand, memory_order order);
; float __attribute__((overloadable)) atomic_fetch_sub(volatile generic atomic_float *object, float operand, memory_order order);
; float __attribute__((overloadable)) atomic_fetch_or(volatile generic atomic_float *object, float operand, memory_order order);
; float __attribute__((overloadable)) atomic_fetch_xor(volatile generic atomic_float *object, float operand, memory_order order);
; double __attribute__((overloadable)) atomic_fetch_and(volatile generic atomic_double *object, double operand, memory_order order);
; double __attribute__((overloadable)) atomic_fetch_max(volatile generic atomic_double *object, double operand, memory_order order);
; double __attribute__((overloadable)) atomic_fetch_min(volatile generic atomic_double *object, double operand, memory_order order);
; float __attribute__((overloadable)) atomic_fetch_add_explicit(volatile generic atomic_float *object, float operand, memory_order order);
; float __attribute__((overloadable)) atomic_fetch_sub_explicit(volatile generic atomic_float *object, float operand, memory_order order);
; float __attribute__((overloadable)) atomic_fetch_or_explicit(volatile generic atomic_float *object, float operand, memory_order order);
; float __attribute__((overloadable)) atomic_fetch_xor_explicit(volatile generic atomic_float *object, float operand, memory_order order);
; double __attribute__((overloadable)) atomic_fetch_and_explicit(volatile generic atomic_double *object, double operand, memory_order order);
; double __attribute__((overloadable)) atomic_fetch_max_explicit(volatile generic atomic_double *object, double operand, memory_order order);
; double __attribute__((overloadable)) atomic_fetch_min_explicit(volatile generic atomic_double *object, double operand, memory_order order);
;
; __kernel void test_atomic_fn(volatile __global float *p,
;                              volatile __global double *pp,
;                              float val,
;                              memory_order order)
; {
;     float f = 0.0f;
;     double d = 0.0;
;
;     f = atomic_add(p, val);
;     f = atomic_sub(p, val);
;     f = atomic_inc(p, val);
;     f = atomic_dec(p, val);
;     f = atomic_cmpxchg(p, val);
;     d = atomic_min(pp, val);
;     d = atomic_max(pp, val);
;     d = atomic_and(pp, val);
;     d = atomic_or(pp, val);
;     d = atomic_xor(pp, val);
;
;     f = atom_add(p, val);
;     f = atom_sub(p, val);
;     f = atom_inc(p, val);
;     f = atom_dec(p, val);
;     f = atom_cmpxchg(p, val);
;     d = atom_min(pp, val);
;     d = atom_max(pp, val);
;     d = atom_and(pp, val);
;     d = atom_or(pp, val);
;     d = atom_xor(pp, val);
;
;     f = atomic_fetch_add(p, val, order);
;     f = atomic_fetch_sub(p, val, order);
;     f = atomic_fetch_or(p, val, order);
;     f = atomic_fetch_xor(p, val, order);
;     d = atomic_fetch_and(pp, val, order);
;     d = atomic_fetch_min(pp, val, order);
;     d = atomic_fetch_max(pp, val, order);
;     f = atomic_fetch_add_explicit(p, val, order);
;     f = atomic_fetch_sub_explicit(p, val, order);
;     f = atomic_fetch_or_explicit(p, val, order);
;     f = atomic_fetch_xor_explicit(p, val, order);
;     d = atomic_fetch_and_explicit(pp, val, order);
;     d = atomic_fetch_min_explicit(pp, val, order);
;     d = atomic_fetch_max_explicit(pp, val, order);
; }
; Comand
; clang -cc1 -triple spir -O1 -cl-std=cl2.0 -finclude-default-header  -x cl /work/tmp/tmp.cl -emit-llvm -o >> AtomicInvalidBulitins.ll

; RUN: llvm-as < %s > %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o - | FileCheck %s

; CHECK-LABEL: Label
; CHECK-NOT: Atomic

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir"

; Function Attrs: convergent nounwind
define spir_kernel void @test_atomic_fn(float addrspace(1)* %p, double addrspace(1)* %pp, float %val, i32 %order) local_unnamed_addr #0 !kernel_arg_addr_space !5 !kernel_arg_access_qual !6 !kernel_arg_type !7 !kernel_arg_base_type !8 !kernel_arg_type_qual !9 !kernel_arg_host_accessible !10 !kernel_arg_pipe_depth !11 !kernel_arg_pipe_io !12 !kernel_arg_buffer_location !12 {
entry:
  %call = tail call spir_func float @_Z10atomic_addPU3AS1Vff(float addrspace(1)* %p, float %val) #2
  %call1 = tail call spir_func float @_Z10atomic_subPU3AS1Vff(float addrspace(1)* %p, float %val) #2
  %call2 = tail call spir_func float @_Z10atomic_incPU3AS1Vff(float addrspace(1)* %p, float %val) #2
  %call3 = tail call spir_func float @_Z10atomic_decPU3AS1Vff(float addrspace(1)* %p, float %val) #2
  %call4 = tail call spir_func float @_Z14atomic_cmpxchgPU3AS1Vff(float addrspace(1)* %p, float %val) #2
  %conv = fpext float %val to double
  %call5 = tail call spir_func double @_Z10atomic_minPU3AS1Vdd(double addrspace(1)* %pp, double %conv) #2
  %call7 = tail call spir_func double @_Z10atomic_maxPU3AS1Vdd(double addrspace(1)* %pp, double %conv) #2
  %call9 = tail call spir_func double @_Z10atomic_andPU3AS1Vdd(double addrspace(1)* %pp, double %conv) #2
  %call11 = tail call spir_func double @_Z9atomic_orPU3AS1Vdd(double addrspace(1)* %pp, double %conv) #2
  %call13 = tail call spir_func double @_Z10atomic_xorPU3AS1Vdd(double addrspace(1)* %pp, double %conv) #2
  %call14 = tail call spir_func float @_Z8atom_addPU3AS1Vff(float addrspace(1)* %p, float %val) #2
  %call15 = tail call spir_func float @_Z8atom_subPU3AS1Vff(float addrspace(1)* %p, float %val) #2
  %call16 = tail call spir_func float @_Z8atom_incPU3AS1Vff(float addrspace(1)* %p, float %val) #2
  %call17 = tail call spir_func float @_Z8atom_decPU3AS1Vff(float addrspace(1)* %p, float %val) #2
  %call18 = tail call spir_func float @_Z12atom_cmpxchgPU3AS1Vff(float addrspace(1)* %p, float %val) #2
  %call20 = tail call spir_func double @_Z8atom_minPU3AS1Vdd(double addrspace(1)* %pp, double %conv) #2
  %call22 = tail call spir_func double @_Z8atom_maxPU3AS1Vdd(double addrspace(1)* %pp, double %conv) #2
  %call24 = tail call spir_func double @_Z8atom_andPU3AS1Vdd(double addrspace(1)* %pp, double %conv) #2
  %call26 = tail call spir_func double @_Z7atom_orPU3AS1Vdd(double addrspace(1)* %pp, double %conv) #2
  %call28 = tail call spir_func double @_Z8atom_xorPU3AS1Vdd(double addrspace(1)* %pp, double %conv) #2
  %0 = addrspacecast float addrspace(1)* %p to float addrspace(4)*
  %call29 = tail call spir_func float @_Z16atomic_fetch_addPU3AS4VU7_Atomicff(float addrspace(4)* %0, float %val) #2
  %call30 = tail call spir_func float @_Z16atomic_fetch_subPU3AS4VU7_Atomicff(float addrspace(4)* %0, float %val) #2
  %call31 = tail call spir_func float @_Z15atomic_fetch_orPU3AS4VU7_Atomicff(float addrspace(4)* %0, float %val) #2
  %call32 = tail call spir_func float @_Z16atomic_fetch_xorPU3AS4VU7_Atomicff(float addrspace(4)* %0, float %val) #2
  %1 = addrspacecast double addrspace(1)* %pp to double addrspace(4)*
  %call34 = tail call spir_func double @_Z16atomic_fetch_andPU3AS4VU7_Atomicdd(double addrspace(4)* %1, double %conv) #2
  %call36 = tail call spir_func double @_Z16atomic_fetch_minPU3AS4VU7_Atomicdd(double addrspace(4)* %1, double %conv) #2
  %call38 = tail call spir_func double @_Z16atomic_fetch_maxPU3AS4VU7_Atomicdd(double addrspace(4)* %1, double %conv) #2
  %call39 = tail call spir_func float @_Z25atomic_fetch_add_explicitPU3AS4VU7_Atomicff12memory_order(float addrspace(4)* %0, float %val, i32 %order) #2
  %call40 = tail call spir_func float @_Z25atomic_fetch_sub_explicitPU3AS4VU7_Atomicff12memory_order(float addrspace(4)* %0, float %val, i32 %order) #2
  %call41 = tail call spir_func float @_Z24atomic_fetch_or_explicitPU3AS4VU7_Atomicff12memory_order(float addrspace(4)* %0, float %val, i32 %order) #2
  %call42 = tail call spir_func float @_Z25atomic_fetch_xor_explicitPU3AS4VU7_Atomicff12memory_order(float addrspace(4)* %0, float %val, i32 %order) #2
  %call44 = tail call spir_func double @_Z25atomic_fetch_and_explicitPU3AS4VU7_Atomicdd12memory_order(double addrspace(4)* %1, double %conv, i32 %order) #2
  %call46 = tail call spir_func double @_Z25atomic_fetch_min_explicitPU3AS4VU7_Atomicdd12memory_order(double addrspace(4)* %1, double %conv, i32 %order) #2
  %call48 = tail call spir_func double @_Z25atomic_fetch_max_explicitPU3AS4VU7_Atomicdd12memory_order(double addrspace(4)* %1, double %conv, i32 %order) #2
  ret void
}

; Function Attrs: convergent
declare spir_func float @_Z10atomic_addPU3AS1Vff(float addrspace(1)*, float) local_unnamed_addr #1

; Function Attrs: convergent
declare spir_func float @_Z10atomic_subPU3AS1Vff(float addrspace(1)*, float) local_unnamed_addr #1

; Function Attrs: convergent
declare spir_func float @_Z10atomic_incPU3AS1Vff(float addrspace(1)*, float) local_unnamed_addr #1

; Function Attrs: convergent
declare spir_func float @_Z10atomic_decPU3AS1Vff(float addrspace(1)*, float) local_unnamed_addr #1

; Function Attrs: convergent
declare spir_func float @_Z14atomic_cmpxchgPU3AS1Vff(float addrspace(1)*, float) local_unnamed_addr #1

; Function Attrs: convergent
declare spir_func double @_Z10atomic_minPU3AS1Vdd(double addrspace(1)*, double) local_unnamed_addr #1

; Function Attrs: convergent
declare spir_func double @_Z10atomic_maxPU3AS1Vdd(double addrspace(1)*, double) local_unnamed_addr #1

; Function Attrs: convergent
declare spir_func double @_Z10atomic_andPU3AS1Vdd(double addrspace(1)*, double) local_unnamed_addr #1

; Function Attrs: convergent
declare spir_func double @_Z9atomic_orPU3AS1Vdd(double addrspace(1)*, double) local_unnamed_addr #1

; Function Attrs: convergent
declare spir_func double @_Z10atomic_xorPU3AS1Vdd(double addrspace(1)*, double) local_unnamed_addr #1

; Function Attrs: convergent
declare spir_func float @_Z8atom_addPU3AS1Vff(float addrspace(1)*, float) local_unnamed_addr #1

; Function Attrs: convergent
declare spir_func float @_Z8atom_subPU3AS1Vff(float addrspace(1)*, float) local_unnamed_addr #1

; Function Attrs: convergent
declare spir_func float @_Z8atom_incPU3AS1Vff(float addrspace(1)*, float) local_unnamed_addr #1

; Function Attrs: convergent
declare spir_func float @_Z8atom_decPU3AS1Vff(float addrspace(1)*, float) local_unnamed_addr #1

; Function Attrs: convergent
declare spir_func float @_Z12atom_cmpxchgPU3AS1Vff(float addrspace(1)*, float) local_unnamed_addr #1

; Function Attrs: convergent
declare spir_func double @_Z8atom_minPU3AS1Vdd(double addrspace(1)*, double) local_unnamed_addr #1

; Function Attrs: convergent
declare spir_func double @_Z8atom_maxPU3AS1Vdd(double addrspace(1)*, double) local_unnamed_addr #1

; Function Attrs: convergent
declare spir_func double @_Z8atom_andPU3AS1Vdd(double addrspace(1)*, double) local_unnamed_addr #1

; Function Attrs: convergent
declare spir_func double @_Z7atom_orPU3AS1Vdd(double addrspace(1)*, double) local_unnamed_addr #1

; Function Attrs: convergent
declare spir_func double @_Z8atom_xorPU3AS1Vdd(double addrspace(1)*, double) local_unnamed_addr #1

; Function Attrs: convergent
declare spir_func float @_Z16atomic_fetch_addPU3AS4VU7_Atomicff(float addrspace(4)*, float) local_unnamed_addr #1

; Function Attrs: convergent
declare spir_func float @_Z16atomic_fetch_subPU3AS4VU7_Atomicff(float addrspace(4)*, float) local_unnamed_addr #1

; Function Attrs: convergent
declare spir_func float @_Z15atomic_fetch_orPU3AS4VU7_Atomicff(float addrspace(4)*, float) local_unnamed_addr #1

; Function Attrs: convergent
declare spir_func float @_Z16atomic_fetch_xorPU3AS4VU7_Atomicff(float addrspace(4)*, float) local_unnamed_addr #1

; Function Attrs: convergent
declare spir_func double @_Z16atomic_fetch_andPU3AS4VU7_Atomicdd(double addrspace(4)*, double) local_unnamed_addr #1

; Function Attrs: convergent
declare spir_func double @_Z16atomic_fetch_minPU3AS4VU7_Atomicdd(double addrspace(4)*, double) local_unnamed_addr #1

; Function Attrs: convergent
declare spir_func double @_Z16atomic_fetch_maxPU3AS4VU7_Atomicdd(double addrspace(4)*, double) local_unnamed_addr #1

; Function Attrs: convergent
declare spir_func float @_Z25atomic_fetch_add_explicitPU3AS4VU7_Atomicff12memory_order(float addrspace(4)*, float, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare spir_func float @_Z25atomic_fetch_sub_explicitPU3AS4VU7_Atomicff12memory_order(float addrspace(4)*, float, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare spir_func float @_Z24atomic_fetch_or_explicitPU3AS4VU7_Atomicff12memory_order(float addrspace(4)*, float, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare spir_func float @_Z25atomic_fetch_xor_explicitPU3AS4VU7_Atomicff12memory_order(float addrspace(4)*, float, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare spir_func double @_Z25atomic_fetch_and_explicitPU3AS4VU7_Atomicdd12memory_order(double addrspace(4)*, double, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare spir_func double @_Z25atomic_fetch_min_explicitPU3AS4VU7_Atomicdd12memory_order(double addrspace(4)*, double, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare spir_func double @_Z25atomic_fetch_max_explicitPU3AS4VU7_Atomicdd12memory_order(double addrspace(4)*, double, i32) local_unnamed_addr #1

attributes #0 = { convergent nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "uniform-work-group-size"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { convergent "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { convergent nounwind }

!llvm.module.flags = !{!0}
!opencl.enable.FP_CONTRACT = !{}
!opencl.ocl.version = !{!1}
!opencl.spir.version = !{!1}
!opencl.used.extensions = !{!2}
!opencl.used.optional.core.features = !{!3}
!opencl.compiler.options = !{!2}
!llvm.ident = !{!4}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 2, i32 0}
!2 = !{}
!3 = !{!"cl_doubles"}
!4 = !{!"clang version 7.0.0 (cfe/trunk)"}
!5 = !{i32 1, i32 1, i32 0, i32 0}
!6 = !{!"none", !"none", !"none", !"none"}
!7 = !{!"float*", !"double*", !"float", !"memory_order"}
!8 = !{!"float*", !"double*", !"float", !"enum memory_order"}
!9 = !{!"volatile", !"volatile", !"", !""}
!10 = !{i1 false, i1 false, i1 false, i1 false}
!11 = !{i32 0, i32 0, i32 0, i32 0}
!12 = !{!"", !"", !"", !""}
