; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#VOID:]] = OpTypeVoid

; CHECK-DAG: OpEntryPoint Kernel %[[#WORKER:]] "worker"
; CHECK-DAG: OpEntryPoint Kernel %[[#INIT:]] "_SPIRV_GLOBAL__I_45b04794_Test_attr.cl"
; CHECK-DAG: OpEntryPoint Kernel %[[#FIN:]] "_SPIRV_GLOBAL__D_45b04794_Test_attr.cl"

; CHECK-DAG: OpExecutionMode %[[#WORKER]] LocalSize 10 10 10
; CHECK-DAG: OpExecutionMode %[[#WORKER]] LocalSizeHint 12 10 1
; CHECK-DAG: OpExecutionMode %[[#WORKER]] VecTypeHint 262149
; CHECK-DAG: OpExecutionMode %[[#WORKER]] SubgroupsPerWorkgroup 4
; CHECK-DAG: OpExecutionMode %[[#INIT]] LocalSize 1 1 1
; CHECK-DAG: OpExecutionMode %[[#INIT]] Initializer
; CHECK-DAG: OpExecutionMode %[[#FIN]] LocalSize 1 1 1
; CHECK-DAG: OpExecutionMode %[[#FIN]] Finalizer

%struct.global_ctor_dtor = type { i32 }

@g = addrspace(1) global %struct.global_ctor_dtor zeroinitializer, align 4

define internal spir_func void @__cxx_global_var_init() {
entry:
  call spir_func void @_ZNU3AS416global_ctor_dtorC1Ei(ptr addrspace(4) addrspacecast (ptr addrspace(1) @g to ptr addrspace(4)), i32 12)
  ret void
}

define linkonce_odr spir_func void @_ZNU3AS416global_ctor_dtorC1Ei(ptr addrspace(4) %this, i32 %i) unnamed_addr align 2 {
entry:
  %this.addr = alloca ptr addrspace(4), align 4
  %i.addr = alloca i32, align 4
  store ptr addrspace(4) %this, ptr %this.addr, align 4
  store i32 %i, ptr %i.addr, align 4
  %this1 = load ptr addrspace(4), ptr %this.addr
  %0 = load i32, ptr %i.addr, align 4
  call spir_func void @_ZNU3AS416global_ctor_dtorC2Ei(ptr addrspace(4) %this1, i32 %0)
  ret void
}

define linkonce_odr spir_func void @_ZNU3AS416global_ctor_dtorD1Ev(ptr addrspace(4) %this) unnamed_addr align 2 {
entry:
  %this.addr = alloca ptr addrspace(4), align 4
  store ptr addrspace(4) %this, ptr %this.addr, align 4
  %this1 = load ptr addrspace(4), ptr %this.addr
  call spir_func void @_ZNU3AS416global_ctor_dtorD2Ev(ptr addrspace(4) %this1)
  ret void
}

define internal spir_func void @__dtor_g() {
entry:
  call spir_func void @_ZNU3AS416global_ctor_dtorD1Ev(ptr addrspace(4) addrspacecast (ptr addrspace(1) @g to ptr addrspace(4)))
  ret void
}

; CHECK: %[[#WORKER]] = OpFunction %[[#VOID]]

define spir_kernel void @worker() {
entry:
  ret void
}

define linkonce_odr spir_func void @_ZNU3AS416global_ctor_dtorD2Ev(ptr addrspace(4) %this) unnamed_addr align 2 {
entry:
  %this.addr = alloca ptr addrspace(4), align 4
  store ptr addrspace(4) %this, ptr %this.addr, align 4
  %this1 = load ptr addrspace(4), ptr %this.addr
  %a = getelementptr inbounds %struct.global_ctor_dtor, ptr addrspace(4) %this1, i32 0, i32 0
  store i32 0, ptr addrspace(4) %a, align 4
  ret void
}

define linkonce_odr spir_func void @_ZNU3AS416global_ctor_dtorC2Ei(ptr addrspace(4) %this, i32 %i) unnamed_addr align 2 {
entry:
  %this.addr = alloca ptr addrspace(4), align 4
  %i.addr = alloca i32, align 4
  store ptr addrspace(4) %this, ptr %this.addr, align 4
  store i32 %i, ptr %i.addr, align 4
  %this1 = load ptr addrspace(4), ptr %this.addr
  %0 = load i32, ptr %i.addr, align 4
  %a = getelementptr inbounds %struct.global_ctor_dtor, ptr addrspace(4) %this1, i32 0, i32 0
  store i32 %0, ptr addrspace(4) %a, align 4
  ret void
}

define internal spir_func void @_GLOBAL__sub_I_Test_attr.cl() {
entry:
  call spir_func void @__cxx_global_var_init()
  ret void
}

; CHECK: %[[#INIT]] = OpFunction %[[#VOID]]

define spir_kernel void @_SPIRV_GLOBAL__I_45b04794_Test_attr.cl() {
entry:
  call spir_func void @_GLOBAL__sub_I_Test_attr.cl()
  ret void
}

; CHECK: %[[#FIN]] = OpFunction %[[#VOID]]

define spir_kernel void @_SPIRV_GLOBAL__D_45b04794_Test_attr.cl() {
entry:
  call spir_func void @__dtor_g()
  ret void
}

!spirv.ExecutionMode = !{!0, !1, !2, !3, !4, !5, !6, !7}

!0 = !{ptr @worker, i32 30, i32 262149}
!1 = !{ptr @worker, i32 18, i32 12, i32 10, i32 1}
!2 = !{ptr @worker, i32 17, i32 10, i32 10, i32 10}
!3 = !{ptr @worker, i32 36, i32 4}
!4 = !{ptr @_SPIRV_GLOBAL__I_45b04794_Test_attr.cl, i32 33}
!5 = !{ptr @_SPIRV_GLOBAL__I_45b04794_Test_attr.cl, i32 17, i32 1, i32 1, i32 1}
!6 = !{ptr @_SPIRV_GLOBAL__D_45b04794_Test_attr.cl, i32 34}
!7 = !{ptr @_SPIRV_GLOBAL__D_45b04794_Test_attr.cl, i32 17, i32 1, i32 1, i32 1}
