; RUN: llc < %s -march=nvptx -mcpu=sm_20 | FileCheck %s --check-prefix PTX
; RUN: llc < %s -march=nvptx64 -mcpu=sm_20 | FileCheck %s --check-prefix PTX
; RUN: llc < %s  -march=nvptx64 -mcpu=sm_20 -nvptx-use-infer-addrspace | FileCheck %s --check-prefix PTX
; RUN: opt < %s -S -nvptx-favor-non-generic -dce | FileCheck %s --check-prefix IR
; RUN: opt < %s -S -nvptx-infer-addrspace | FileCheck %s --check-prefix IR --check-prefix IR-WITH-LOOP

@array = internal addrspace(3) global [10 x float] zeroinitializer, align 4
@scalar = internal addrspace(3) global float 0.000000e+00, align 4
@generic_scalar = internal global float 0.000000e+00, align 4

define float @ld_from_shared() {
  %1 = addrspacecast float* @generic_scalar to float addrspace(3)*
  %2 = load float, float addrspace(3)* %1
  ret float %2
}

; Verifies nvptx-favor-non-generic correctly optimizes generic address space
; usage to non-generic address space usage for the patterns we claim to handle:
; 1. load cast
; 2. store cast
; 3. load gep cast
; 4. store gep cast
; gep and cast can be an instruction or a constant expression. This function
; tries all possible combinations.
define void @ld_st_shared_f32(i32 %i, float %v) {
; IR-LABEL: @ld_st_shared_f32
; IR-NOT: addrspacecast
; PTX-LABEL: ld_st_shared_f32(
  ; load cast
  %1 = load float, float* addrspacecast (float addrspace(3)* @scalar to float*), align 4
  call void @use(float %1)
; PTX: ld.shared.f32 %f{{[0-9]+}}, [scalar];
  ; store cast
  store float %v, float* addrspacecast (float addrspace(3)* @scalar to float*), align 4
; PTX: st.shared.f32 [scalar], %f{{[0-9]+}};
  ; use syncthreads to disable optimizations across components
  call void @llvm.nvvm.barrier0()
; PTX: bar.sync 0;

  ; cast; load
  %2 = addrspacecast float addrspace(3)* @scalar to float*
  %3 = load float, float* %2, align 4
  call void @use(float %3)
; PTX: ld.shared.f32 %f{{[0-9]+}}, [scalar];
  ; cast; store
  store float %v, float* %2, align 4
; PTX: st.shared.f32 [scalar], %f{{[0-9]+}};
  call void @llvm.nvvm.barrier0()
; PTX: bar.sync 0;

  ; load gep cast
  %4 = load float, float* getelementptr inbounds ([10 x float], [10 x float]* addrspacecast ([10 x float] addrspace(3)* @array to [10 x float]*), i32 0, i32 5), align 4
  call void @use(float %4)
; PTX: ld.shared.f32 %f{{[0-9]+}}, [array+20];
  ; store gep cast
  store float %v, float* getelementptr inbounds ([10 x float], [10 x float]* addrspacecast ([10 x float] addrspace(3)* @array to [10 x float]*), i32 0, i32 5), align 4
; PTX: st.shared.f32 [array+20], %f{{[0-9]+}};
  call void @llvm.nvvm.barrier0()
; PTX: bar.sync 0;

  ; gep cast; load
  %5 = getelementptr inbounds [10 x float], [10 x float]* addrspacecast ([10 x float] addrspace(3)* @array to [10 x float]*), i32 0, i32 5
  %6 = load float, float* %5, align 4
  call void @use(float %6)
; PTX: ld.shared.f32 %f{{[0-9]+}}, [array+20];
  ; gep cast; store
  store float %v, float* %5, align 4
; PTX: st.shared.f32 [array+20], %f{{[0-9]+}};
  call void @llvm.nvvm.barrier0()
; PTX: bar.sync 0;

  ; cast; gep; load
  %7 = addrspacecast [10 x float] addrspace(3)* @array to [10 x float]*
  %8 = getelementptr inbounds [10 x float], [10 x float]* %7, i32 0, i32 %i
  %9 = load float, float* %8, align 4
  call void @use(float %9)
; PTX: ld.shared.f32 %f{{[0-9]+}}, [%{{(r|rl|rd)[0-9]+}}];
  ; cast; gep; store
  store float %v, float* %8, align 4
; PTX: st.shared.f32 [%{{(r|rl|rd)[0-9]+}}], %f{{[0-9]+}};
  call void @llvm.nvvm.barrier0()
; PTX: bar.sync 0;

  ret void
}

; When hoisting an addrspacecast between different pointer types, replace the
; addrspacecast with a bitcast.
define i32 @ld_int_from_float() {
; IR-LABEL: @ld_int_from_float
; IR: load i32, i32 addrspace(3)* bitcast (float addrspace(3)* @scalar to i32 addrspace(3)*)
; PTX-LABEL: ld_int_from_float(
; PTX: ld.shared.u{{(32|64)}}
  %1 = load i32, i32* addrspacecast(float addrspace(3)* @scalar to i32*), align 4
  ret i32 %1
}

define i32 @ld_int_from_global_float(float addrspace(1)* %input, i32 %i, i32 %j) {
; IR-LABEL: @ld_int_from_global_float(
; PTX-LABEL: ld_int_from_global_float(
  %1 = addrspacecast float addrspace(1)* %input to float*
  %2 = getelementptr float, float* %1, i32 %i
; IR-NEXT: getelementptr float, float addrspace(1)* %input, i32 %i
  %3 = getelementptr float, float* %2, i32 %j
; IR-NEXT: getelementptr float, float addrspace(1)* {{%[^,]+}}, i32 %j
  %4 = bitcast float* %3 to i32*
; IR-NEXT: bitcast float addrspace(1)* {{%[^ ]+}} to i32 addrspace(1)*
  %5 = load i32, i32* %4
; IR-NEXT: load i32, i32 addrspace(1)* {{%.+}}
; PTX-LABEL: ld.global
  ret i32 %5
}

define void @nested_const_expr() {
; PTX-LABEL: nested_const_expr(
  ; store 1 to bitcast(gep(addrspacecast(array), 0, 1))
  store i32 1, i32* bitcast (float* getelementptr ([10 x float], [10 x float]* addrspacecast ([10 x float] addrspace(3)* @array to [10 x float]*), i64 0, i64 1) to i32*), align 4
; PTX: mov.u32 %r1, 1;
; PTX-NEXT: st.shared.u32 [array+4], %r1;
  ret void
}

define void @rauw(float addrspace(1)* %input) {
  %generic_input = addrspacecast float addrspace(1)* %input to float*
  %addr = getelementptr float, float* %generic_input, i64 10
  %v = load float, float* %addr
  store float %v, float* %addr
  ret void
; IR-LABEL: @rauw(
; IR-NEXT: %addr = getelementptr float, float addrspace(1)* %input, i64 10
; IR-NEXT: %v = load float, float addrspace(1)* %addr
; IR-NEXT: store float %v, float addrspace(1)* %addr
; IR-NEXT: ret void
}

define void @loop() {
; IR-WITH-LOOP-LABEL: @loop(
entry:
  %p = addrspacecast [10 x float] addrspace(3)* @array to float*
  %end = getelementptr float, float* %p, i64 10
  br label %loop

loop:
  %i = phi float* [ %p, %entry ], [ %i2, %loop ]
; IR-WITH-LOOP: phi float addrspace(3)* [ %p, %entry ], [ %i2, %loop ]
  %v = load float, float* %i
; IR-WITH-LOOP: %v = load float, float addrspace(3)* %i
  call void @use(float %v)
  %i2 = getelementptr float, float* %i, i64 1
; IR-WITH-LOOP: %i2 = getelementptr float, float addrspace(3)* %i, i64 1
  %exit_cond = icmp eq float* %i2, %end
  br i1 %exit_cond, label %exit, label %loop

exit:
  ret void
}

@generic_end = external global float*

define void @loop_with_generic_bound() {
; IR-WITH-LOOP-LABEL: @loop_with_generic_bound(
entry:
  %p = addrspacecast [10 x float] addrspace(3)* @array to float*
  %end = load float*, float** @generic_end
  br label %loop

loop:
  %i = phi float* [ %p, %entry ], [ %i2, %loop ]
; IR-WITH-LOOP: phi float addrspace(3)* [ %p, %entry ], [ %i2, %loop ]
  %v = load float, float* %i
; IR-WITH-LOOP: %v = load float, float addrspace(3)* %i
  call void @use(float %v)
  %i2 = getelementptr float, float* %i, i64 1
; IR-WITH-LOOP: %i2 = getelementptr float, float addrspace(3)* %i, i64 1
  %exit_cond = icmp eq float* %i2, %end
; IR-WITH-LOOP: addrspacecast float addrspace(3)* %i2 to float*
; IR-WITH-LOOP: icmp eq float* %{{[0-9]+}}, %end
  br i1 %exit_cond, label %exit, label %loop

exit:
  ret void
}

declare void @llvm.nvvm.barrier0() #3

declare void @use(float)

attributes #3 = { noduplicate nounwind }
