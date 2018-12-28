; RUN: llvm-as < %s | llvm-spirv -spirv-text -o %t
; RUN: FileCheck < %t %s

; check for magic number followed by version 1.1
; CHECK: 119734787 65792

; CHECK-DAG: TypeVoid [[VOID:[0-9]+]]

; CHECK-DAG: EntryPoint 6 [[WORKER:[0-9]+]] "worker"
; CHECK-DAG: EntryPoint 6 [[INIT:[0-9]+]] "_SPIRV_GLOBAL__I_45b04794_Test_attr.cl"
; CHECK-DAG: EntryPoint 6 [[FIN:[0-9]+]] "_SPIRV_GLOBAL__D_45b04794_Test_attr.cl"

; CHECK-DAG: ExecutionMode [[WORKER]] 17 10 10 10
; CHECK-DAG: ExecutionMode [[WORKER]] 18 12 10 1
; CHECK-DAG: ExecutionMode [[WORKER]] 30 262149
; CHECK-DAG: ExecutionMode [[WORKER]] 36 4
; CHECK-DAG: ExecutionMode [[INIT]] 17 1 1 1
; CHECK-DAG: ExecutionMode [[INIT]] 33
; CHECK-DAG: ExecutionMode [[FIN]] 17 1 1 1
; CHECK-DAG: ExecutionMode [[FIN]] 34

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

%struct.global_ctor_dtor = type { i32 }

@g = addrspace(1) global %struct.global_ctor_dtor zeroinitializer, align 4

; Function Attrs: nounwind
define internal spir_func void @__cxx_global_var_init() #0 {
entry:
  call spir_func void @_ZNU3AS416global_ctor_dtorC1Ei(%struct.global_ctor_dtor addrspace(4)* addrspacecast (%struct.global_ctor_dtor addrspace(1)* @g to %struct.global_ctor_dtor addrspace(4)*), i32 12)
  ret void
}

; Function Attrs: nounwind
define linkonce_odr spir_func void @_ZNU3AS416global_ctor_dtorC1Ei(%struct.global_ctor_dtor addrspace(4)* %this, i32 %i) unnamed_addr #1 align 2 {
entry:
  %this.addr = alloca %struct.global_ctor_dtor addrspace(4)*, align 4
  %i.addr = alloca i32, align 4
  store %struct.global_ctor_dtor addrspace(4)* %this, %struct.global_ctor_dtor addrspace(4)** %this.addr, align 4
  store i32 %i, i32* %i.addr, align 4
  %this1 = load %struct.global_ctor_dtor addrspace(4)*, %struct.global_ctor_dtor addrspace(4)** %this.addr
  %0 = load i32, i32* %i.addr, align 4
  call spir_func void @_ZNU3AS416global_ctor_dtorC2Ei(%struct.global_ctor_dtor addrspace(4)* %this1, i32 %0)
  ret void
}

; Function Attrs: nounwind
define linkonce_odr spir_func void @_ZNU3AS416global_ctor_dtorD1Ev(%struct.global_ctor_dtor addrspace(4)* %this) unnamed_addr #1 align 2 {
entry:
  %this.addr = alloca %struct.global_ctor_dtor addrspace(4)*, align 4
  store %struct.global_ctor_dtor addrspace(4)* %this, %struct.global_ctor_dtor addrspace(4)** %this.addr, align 4
  %this1 = load %struct.global_ctor_dtor addrspace(4)*, %struct.global_ctor_dtor addrspace(4)** %this.addr
  call spir_func void @_ZNU3AS416global_ctor_dtorD2Ev(%struct.global_ctor_dtor addrspace(4)* %this1) #0
  ret void
}

; Function Attrs: nounwind
define internal spir_func void @__dtor_g() #0 {
entry:
  call spir_func void @_ZNU3AS416global_ctor_dtorD1Ev(%struct.global_ctor_dtor addrspace(4)* addrspacecast (%struct.global_ctor_dtor addrspace(1)* @g to %struct.global_ctor_dtor addrspace(4)*))
  ret void
}

; CHECK: Function [[VOID]] [[WORKER]]

; Function Attrs: nounwind
define spir_kernel void @worker() #1 {
entry:
  ret void
}

; Function Attrs: nounwind
define linkonce_odr spir_func void @_ZNU3AS416global_ctor_dtorD2Ev(%struct.global_ctor_dtor addrspace(4)* %this) unnamed_addr #1 align 2 {
entry:
  %this.addr = alloca %struct.global_ctor_dtor addrspace(4)*, align 4
  store %struct.global_ctor_dtor addrspace(4)* %this, %struct.global_ctor_dtor addrspace(4)** %this.addr, align 4
  %this1 = load %struct.global_ctor_dtor addrspace(4)*, %struct.global_ctor_dtor addrspace(4)** %this.addr
  %a = getelementptr inbounds %struct.global_ctor_dtor, %struct.global_ctor_dtor addrspace(4)* %this1, i32 0, i32 0
  store i32 0, i32 addrspace(4)* %a, align 4
  ret void
}

; Function Attrs: nounwind
define linkonce_odr spir_func void @_ZNU3AS416global_ctor_dtorC2Ei(%struct.global_ctor_dtor addrspace(4)* %this, i32 %i) unnamed_addr #1 align 2 {
entry:
  %this.addr = alloca %struct.global_ctor_dtor addrspace(4)*, align 4
  %i.addr = alloca i32, align 4
  store %struct.global_ctor_dtor addrspace(4)* %this, %struct.global_ctor_dtor addrspace(4)** %this.addr, align 4
  store i32 %i, i32* %i.addr, align 4
  %this1 = load %struct.global_ctor_dtor addrspace(4)*, %struct.global_ctor_dtor addrspace(4)** %this.addr
  %0 = load i32, i32* %i.addr, align 4
  %a = getelementptr inbounds %struct.global_ctor_dtor, %struct.global_ctor_dtor addrspace(4)* %this1, i32 0, i32 0
  store i32 %0, i32 addrspace(4)* %a, align 4
  ret void
}

; Function Attrs: nounwind
define internal spir_func void @_GLOBAL__sub_I_Test_attr.cl() #0 {
entry:
  call spir_func void @__cxx_global_var_init()
  ret void
}

; CHECK: Function [[VOID]] [[INIT]]

; Function Attrs: noinline nounwind
define spir_kernel void @_SPIRV_GLOBAL__I_45b04794_Test_attr.cl() #2 {
entry:
  call spir_func void @_GLOBAL__sub_I_Test_attr.cl()
  ret void
}

; CHECK: Function [[VOID]] [[FIN]]

; Function Attrs: noinline nounwind
define spir_kernel void @_SPIRV_GLOBAL__D_45b04794_Test_attr.cl() #2 {
entry:
  call spir_func void @__dtor_g()
  ret void
}

attributes #0 = { nounwind }
attributes #1 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { noinline nounwind }

!spirv.ExecutionMode = !{!0, !1, !2, !3, !4, !5, !6, !7}
!opencl.enable.FP_CONTRACT = !{}
!opencl.spir.version = !{!8}
!opencl.ocl.version = !{!9}
!opencl.used.extensions = !{!10}
!opencl.used.optional.core.features = !{!10}
!opencl.compiler.options = !{!10}
!llvm.ident = !{!11}
!spirv.Source = !{!12}

!0 = !{void ()* @worker, i32 30, i32 262149}
!1 = !{void ()* @worker, i32 18, i32 12, i32 10, i32 1}
!2 = !{void ()* @worker, i32 17, i32 10, i32 10, i32 10}
!3 = !{void ()* @worker, i32 36, i32 4}
!4 = !{void ()* @_SPIRV_GLOBAL__I_45b04794_Test_attr.cl, i32 33}
!5 = !{void ()* @_SPIRV_GLOBAL__I_45b04794_Test_attr.cl, i32 17, i32 1, i32 1, i32 1}
!6 = !{void ()* @_SPIRV_GLOBAL__D_45b04794_Test_attr.cl, i32 34}
!7 = !{void ()* @_SPIRV_GLOBAL__D_45b04794_Test_attr.cl, i32 17, i32 1, i32 1, i32 1}
!8 = !{i32 1, i32 2}
!9 = !{i32 2, i32 2}
!10 = !{}
!11 = !{!"clang version 3.6.1 "}
!12 = !{i32 4, i32 202000}
