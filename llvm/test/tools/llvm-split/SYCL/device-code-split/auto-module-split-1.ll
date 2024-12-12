; RUN: llvm-split -sycl-split=source -S < %s -o %t
; RUN: FileCheck %s -input-file=%t_0.ll --check-prefixes CHECK-TU0,CHECK
; RUN: FileCheck %s -input-file=%t_1.ll --check-prefixes CHECK-TU1,CHECK
; RUN: FileCheck %s -input-file=%t_0.sym --check-prefixes CHECK-TU0-TXT
; RUN: FileCheck %s -input-file=%t_1.sym --check-prefixes CHECK-TU1-TXT

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-linux"

$_Z3barIiET_S0_ = comdat any

; CHECK-TU1-NOT: @{{.*}}GV{{.*}}
; CHECK-TU0: @{{.*}}GV{{.*}} = internal addrspace(1) constant [1 x i32] [i32 42], align 4
@_ZL2GV = internal addrspace(1) constant [1 x i32] [i32 42], align 4

; CHECK-TU1: define dso_local spir_kernel void @{{.*}}TU0_kernel0{{.*}}
; CHECK-TU1-TXT: {{.*}}TU0_kernel0{{.*}}
; CHECK-TU0-NOT: define dso_local spir_kernel void @{{.*}}TU0_kernel0{{.*}}
; CHECK-TU0-TXT-NOT: {{.*}}TU0_kernel0{{.*}}

; CHECK-TU1: call spir_func void @{{.*}}foo{{.*}}()

define dso_local spir_kernel void @_ZTSZ4mainE11TU0_kernel0() #0 {
entry:
  call spir_func void @_Z3foov()
  ret void
}

; CHECK-TU1: define {{.*}} spir_func void @{{.*}}foo{{.*}}()
; CHECK-TU0-NOT: define {{.*}} spir_func void @{{.*}}foo{{.*}}()

; CHECK-TU1: call spir_func i32 @{{.*}}bar{{.*}}(i32 1)

define dso_local spir_func void @_Z3foov() {
entry:
  %a = alloca i32, align 4
  %call = call spir_func i32 @_Z3barIiET_S0_(i32 1)
  %add = add nsw i32 2, %call
  store i32 %add, ptr %a, align 4
  ret void
}

; CHECK-TU1: define {{.*}} spir_func i32 @{{.*}}bar{{.*}}(i32 %arg)
; CHECK-TU0-NOT: define {{.*}} spir_func i32 @{{.*}}bar{{.*}}(i32 %arg)

; Function Attrs: nounwind
define linkonce_odr dso_local spir_func i32 @_Z3barIiET_S0_(i32 %arg) comdat {
entry:
  %arg.addr = alloca i32, align 4
  store i32 %arg, ptr %arg.addr, align 4
  %0 = load i32, ptr %arg.addr, align 4
  ret i32 %0
}

; CHECK-TU1: define dso_local spir_kernel void @{{.*}}TU0_kernel1{{.*}}()
; CHECK-TU1-TXT: {{.*}}TU0_kernel1{{.*}}
; CHECK-TU0-NOT: define dso_local spir_kernel void @{{.*}}TU0_kernel1{{.*}}()
; CHECK-TU0-TXT-NOT: {{.*}}TU0_kernel1{{.*}}

; CHECK-TU1: call spir_func void @{{.*}}foo1{{.*}}()

define dso_local spir_kernel void @_ZTSZ4mainE11TU0_kernel1() #0 {
entry:
  call spir_func void @_Z4foo1v()
  ret void
}

; CHECK-TU1: define {{.*}} spir_func void @{{.*}}foo1{{.*}}()
; CHECK-TU0-NOT: define {{.*}} spir_func void @{{.*}}foo1{{.*}}()

; Function Attrs: nounwind
define dso_local spir_func void @_Z4foo1v() {
entry:
  %a = alloca i32, align 4
  store i32 2, ptr %a, align 4
  ret void
}

; CHECK-TU1-NOT: define dso_local spir_kernel void @{{.*}}TU1_kernel{{.*}}()
; CHECK-TU1-TXT-NOT: {{.*}}TU1_kernel{{.*}}
; CHECK-TU0: define dso_local spir_kernel void @{{.*}}TU1_kernel{{.*}}()
; CHECK-TU0-TXT: {{.*}}TU1_kernel{{.*}}

; CHECK-TU0: call spir_func void @{{.*}}foo2{{.*}}()

define dso_local spir_kernel void @_ZTSZ4mainE10TU1_kernel() #1 {
entry:
  call spir_func void @_Z4foo2v()
  ret void
}

; CHECK-TU1-NOT: define {{.*}} spir_func void @{{.*}}foo2{{.*}}()
; CHECK-TU0: define {{.*}} spir_func void @{{.*}}foo2{{.*}}()

; Function Attrs: nounwind
define dso_local spir_func void @_Z4foo2v() {
entry:
  %a = alloca i32, align 4
; CHECK-TU0: %0 = load i32, ptr addrspace(4) addrspacecast (ptr addrspace(1) @{{.*}}GV{{.*}} to ptr addrspace(4)), align 4
  %0 = load i32, ptr addrspace(4) getelementptr inbounds ([1 x i32], ptr addrspace(4) addrspacecast (ptr addrspace(1) @_ZL2GV to ptr addrspace(4)), i64 0, i64 0), align 4
  %add = add nsw i32 4, %0
  store i32 %add, ptr %a, align 4
  ret void
}

attributes #0 = { "sycl-module-id"="TU1.cpp" }
attributes #1 = { "sycl-module-id"="TU2.cpp" }

; Metadata is saved in both modules.
; CHECK: !opencl.spir.version = !{!0, !0}
; CHECK: !spirv.Source = !{!1, !1}

!opencl.spir.version = !{!0, !0}
!spirv.Source = !{!1, !1}

; CHECK: !0 = !{i32 1, i32 2}
; CHECK: !1 = !{i32 4, i32 100000}

!0 = !{i32 1, i32 2}
!1 = !{i32 4, i32 100000}
