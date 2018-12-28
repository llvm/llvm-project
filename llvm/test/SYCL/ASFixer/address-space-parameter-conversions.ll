; RUN: opt -asfix %s -S -o - | FileCheck %s
; ModuleID = 'address-space-parameter-conversions.cpp'
source_filename = "address-space-parameter-conversions.cpp"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-linux-sycldevice"

; Function Attrs: noinline nounwind optnone
define dso_local spir_func void @_Z3fooPi(i32* %Data) #1 {
entry:
  %Data.addr = alloca i32*, align 8
  store i32* %Data, i32** %Data.addr, align 8
  ret void
}

; Function Attrs: noinline nounwind optnone
define dso_local spir_func void @_Z3fooPU3AS3i(i32 addrspace(3)* %Data) #1 {
entry:
  %Data.addr = alloca i32 addrspace(3)*, align 8
  store i32 addrspace(3)* %Data, i32 addrspace(3)** %Data.addr, align 8
  ret void
}

; Function Attrs: noinline nounwind optnone
define dso_local spir_func void @_Z6usagesv() #1 {
entry:
  %GLOB = alloca i32 addrspace(1)*, align 8
  %LOC = alloca i32 addrspace(3)*, align 8
  %NoAS = alloca i32*, align 8
; CHECK: %[[GLOB:.*]] = alloca i32 addrspace(1)*, align 8
; CHECK: %[[LOC:.*]] = alloca i32 addrspace(3)*, align 8
; CHECK: %[[NoAS:.*]] = alloca i32*, align 8
  %0 = load i32 addrspace(1)*, i32 addrspace(1)** %GLOB, align 8
  %1 = addrspacecast i32 addrspace(1)* %0 to i32*
; CHECK: %[[GLOB_LOAD:.*]] = load i32 addrspace(1)*, i32 addrspace(1)** %[[GLOB]], align 8
; CHECK: %[[NEW_CAST:.*]] = addrspacecast i32 addrspace(1)* %[[GLOB_LOAD]] to i32 addrspace(4)*
; CHECK: %[[OLD_CAST:.*]] = addrspacecast i32 addrspace(1)* %[[GLOB_LOAD]] to i32*
; CHECK: call spir_func void @new.[[FOO:.*]](i32 addrspace(4)* %[[NEW_CAST]])
; CHECK-NOT: call spir_func void @[[FOO]](i32* %[[OLD_CAST]])
  call spir_func void @_Z3fooPi(i32* %1)
  %2 = load i32 addrspace(3)*, i32 addrspace(3)** %LOC, align 8
; CHECK: %[[LOC_LOAD:.*]] = load i32 addrspace(3)*, i32 addrspace(3)** %[[LOC]], align 8
; CHECK: call spir_func void @[[BAR:.*]](i32 addrspace(3)* %[[LOC_LOAD]])
  call spir_func void @_Z3fooPU3AS3i(i32 addrspace(3)* %2)
  %3 = load i32*, i32** %NoAS, align 8
; CHECK: %[[NoAS_LOAD:.*]] = load i32*, i32** %[[NoAS]], align 8
; CHECK: call spir_func void @[[FOO]](i32* %[[NoAS_LOAD]])
  call spir_func void @_Z3fooPi(i32* %3)
  ret void
}

; CHECK: define dso_local spir_func void @new.[[FOO]](i32 addrspace(4)*
; CHECK: %[[PAR_ALLOC:.*]] = alloca i32 addrspace(4)*
; CHECK: store i32 addrspace(4)* %{{.*}}, i32 addrspace(4)** %[[PAR_ALLOC]], align 8

attributes #0 = { noinline optnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "uniform-work-group-size"="true" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { noinline nounwind optnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0}
!opencl.spir.version = !{!1}
!spirv.Source = !{!2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 2}
!2 = !{i32 4, i32 100000}
!3 = !{!"clang version 8.0.0"}
!4 = !{}
