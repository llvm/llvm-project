; RUN: llc < %s -mtriple=nvptx -mcpu=sm_20 | FileCheck %s --check-prefix PTX
; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_20 | FileCheck %s --check-prefix PTX
; RUN: opt -mtriple=nvptx-- < %s -S -passes=infer-address-spaces | FileCheck %s --check-prefix IR
; RUN: opt -mtriple=nvptx64-- < %s -S -passes=infer-address-spaces | FileCheck %s --check-prefix IR
; RUN: %if ptxas-ptr32 %{ llc < %s -mtriple=nvptx -mcpu=sm_20 | %ptxas-verify %}
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64 -mcpu=sm_20 | %ptxas-verify %}

@array = internal addrspace(3) global [10 x float] zeroinitializer, align 4
@scalar = internal addrspace(3) global float 0.000000e+00, align 4

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
  %1 = load float, ptr addrspacecast (ptr addrspace(3) @scalar to ptr), align 4
  call void @use(float %1)
; PTX: ld.shared.b32 %r{{[0-9]+}}, [scalar];
  ; store cast
  store float %v, ptr addrspacecast (ptr addrspace(3) @scalar to ptr), align 4
; PTX: st.shared.b32 [scalar], %r{{[0-9]+}};
  ; use syncthreads to disable optimizations across components
  call void @llvm.nvvm.barrier0()
; PTX: bar.sync 0;

  ; cast; load
  %2 = addrspacecast ptr addrspace(3) @scalar to ptr
  %3 = load float, ptr %2, align 4
  call void @use(float %3)
; PTX: ld.shared.b32 %r{{[0-9]+}}, [scalar];
  ; cast; store
  store float %v, ptr %2, align 4
; PTX: st.shared.b32 [scalar], %r{{[0-9]+}};
  call void @llvm.nvvm.barrier0()
; PTX: bar.sync 0;

  ; load gep cast
  %4 = load float, ptr getelementptr inbounds ([10 x float], ptr addrspacecast (ptr addrspace(3) @array to ptr), i32 0, i32 5), align 4
  call void @use(float %4)
; PTX: ld.shared.b32 %r{{[0-9]+}}, [array+20];
  ; store gep cast
  store float %v, ptr getelementptr inbounds ([10 x float], ptr addrspacecast (ptr addrspace(3) @array to ptr), i32 0, i32 5), align 4
; PTX: st.shared.b32 [array+20], %r{{[0-9]+}};
  call void @llvm.nvvm.barrier0()
; PTX: bar.sync 0;

  ; gep cast; load
  %5 = getelementptr inbounds [10 x float], ptr addrspacecast (ptr addrspace(3) @array to ptr), i32 0, i32 5
  %6 = load float, ptr %5, align 4
  call void @use(float %6)
; PTX: ld.shared.b32 %r{{[0-9]+}}, [array+20];
  ; gep cast; store
  store float %v, ptr %5, align 4
; PTX: st.shared.b32 [array+20], %r{{[0-9]+}};
  call void @llvm.nvvm.barrier0()
; PTX: bar.sync 0;

  ; cast; gep; load
  %7 = addrspacecast ptr addrspace(3) @array to ptr
  %8 = getelementptr inbounds [10 x float], ptr %7, i32 0, i32 %i
  %9 = load float, ptr %8, align 4
  call void @use(float %9)
; PTX: ld.shared.b32 %r{{[0-9]+}}, [%{{(r|rl|rd)[0-9]+}}];
  ; cast; gep; store
  store float %v, ptr %8, align 4
; PTX: st.shared.b32 [%{{(r|rl|rd)[0-9]+}}], %r{{[0-9]+}};
  call void @llvm.nvvm.barrier0()
; PTX: bar.sync 0;

  ret void
}

; When hoisting an addrspacecast between different pointer types, replace the
; addrspacecast with a bitcast.
define i32 @ld_int_from_float() {
; IR-LABEL: @ld_int_from_float
; IR: load i32, ptr addrspace(3) @scalar
; PTX-LABEL: ld_int_from_float(
; PTX: ld.shared.b{{(32|64)}}
  %1 = load i32, ptr addrspacecast(ptr addrspace(3) @scalar to ptr), align 4
  ret i32 %1
}

define i32 @ld_int_from_global_float(ptr addrspace(1) %input, i32 %i, i32 %j) {
; IR-LABEL: @ld_int_from_global_float(
; PTX-LABEL: ld_int_from_global_float(
  %1 = addrspacecast ptr addrspace(1) %input to ptr
  %2 = getelementptr float, ptr %1, i32 %i
; IR-NEXT: getelementptr float, ptr addrspace(1) %input, i32 %i
  %3 = getelementptr float, ptr %2, i32 %j
; IR-NEXT: getelementptr float, ptr addrspace(1) {{%[^,]+}}, i32 %j
  %4 = load i32, ptr %3
; IR-NEXT: load i32, ptr addrspace(1) {{%.+}}
; PTX-LABEL: ld.global
  ret i32 %4
}

define void @nested_const_expr() {
; PTX-LABEL: nested_const_expr(
  ; store 1 to bitcast(gep(addrspacecast(array), 0, 1))
  store i32 1, ptr getelementptr ([10 x float], ptr addrspacecast (ptr addrspace(3) @array to ptr), i64 0, i64 1), align 4
; PTX: st.shared.b32 [array+4], 1;
  ret void
}

define void @rauw(ptr addrspace(1) %input) {
  %generic_input = addrspacecast ptr addrspace(1) %input to ptr
  %addr = getelementptr float, ptr %generic_input, i64 10
  %v = load float, ptr %addr
  store float %v, ptr %addr
  ret void
; IR-LABEL: @rauw(
; IR-NEXT: %addr = getelementptr float, ptr addrspace(1) %input, i64 10
; IR-NEXT: %v = load float, ptr addrspace(1) %addr
; IR-NEXT: store float %v, ptr addrspace(1) %addr
; IR-NEXT: ret void
}

define void @loop() {
; IR-LABEL: @loop(
entry:
  %p = addrspacecast ptr addrspace(3) @array to ptr
  %end = getelementptr float, ptr %p, i64 10
  br label %loop

loop:
  %i = phi ptr [ %p, %entry ], [ %i2, %loop ]
; IR: phi ptr addrspace(3) [ @array, %entry ], [ %i2, %loop ]
  %v = load float, ptr %i
; IR: %v = load float, ptr addrspace(3) %i
  call void @use(float %v)
  %i2 = getelementptr float, ptr %i, i64 1
; IR: %i2 = getelementptr float, ptr addrspace(3) %i, i64 1
  %exit_cond = icmp eq ptr %i2, %end
  br i1 %exit_cond, label %exit, label %loop

exit:
  ret void
}

@generic_end = external global ptr

define void @loop_with_generic_bound() {
; IR-LABEL: @loop_with_generic_bound(
entry:
  %p = addrspacecast ptr addrspace(3) @array to ptr
  %end = load ptr, ptr @generic_end
  br label %loop

loop:
  %i = phi ptr [ %p, %entry ], [ %i2, %loop ]
; IR: phi ptr addrspace(3) [ @array, %entry ], [ %i2, %loop ]
  %v = load float, ptr %i
; IR: %v = load float, ptr addrspace(3) %i
  call void @use(float %v)
  %i2 = getelementptr float, ptr %i, i64 1
; IR: %i2 = getelementptr float, ptr addrspace(3) %i, i64 1
  %exit_cond = icmp eq ptr %i2, %end
; IR: addrspacecast ptr addrspace(3) %i2 to ptr
; IR: icmp eq ptr %{{[0-9]+}}, %end
  br i1 %exit_cond, label %exit, label %loop

exit:
  ret void
}

declare void @llvm.nvvm.barrier0() #3

declare void @use(float)

attributes #3 = { noduplicate nounwind }
