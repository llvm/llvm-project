;__kernel void test_switch(__global int* res, uchar val)
;{
;  switch(val)
;  {
;  case 0:
;    *res = 1;
;    break;
;  case 1:
;    *res = 2;
;    break;
;  case 2:
;    *res = 3;
;    break;
;  }
;}

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.bc
; RUN: llvm-dis < %t.bc | FileCheck %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV: 9 Switch {{[0-9]+}} {{[0-9]+}} 0 {{[0-9]+}} 1 {{[0-9]+}} 2 {{[0-9]+}}

; ModuleID = 'switch.cl'
target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

;CHECK-LLVM-LABEL: @test_switch
;CHECK-LLVM: switch i8 %0, label %sw.epilog
;CHECK-LLVM: i8 0, label %sw.bb
;CHECK-LLVM: i8 1, label %sw.bb1
;CHECK-LLVM: i8 2, label %sw.bb2

; Function Attrs: convergent noinline nounwind optnone
define spir_kernel void @test_switch(i32 addrspace(1)* %res, i8 zeroext %val) #0 !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !5 !kernel_arg_base_type !5 !kernel_arg_type_qual !6 {
entry:
  %res.addr = alloca i32 addrspace(1)*, align 4
  %val.addr = alloca i8, align 1
  store i32 addrspace(1)* %res, i32 addrspace(1)** %res.addr, align 4
  store i8 %val, i8* %val.addr, align 1
  %0 = load i8, i8* %val.addr, align 1
  switch i8 %0, label %sw.epilog [
    i8 0, label %sw.bb
    i8 1, label %sw.bb1
    i8 2, label %sw.bb2
  ]

sw.bb:                                            ; preds = %entry
  %1 = load i32 addrspace(1)*, i32 addrspace(1)** %res.addr, align 4
  store i32 1, i32 addrspace(1)* %1, align 4
  br label %sw.epilog

sw.bb1:                                           ; preds = %entry
  %2 = load i32 addrspace(1)*, i32 addrspace(1)** %res.addr, align 4
  store i32 2, i32 addrspace(1)* %2, align 4
  br label %sw.epilog

sw.bb2:                                           ; preds = %entry
  %3 = load i32 addrspace(1)*, i32 addrspace(1)** %res.addr, align 4
  store i32 3, i32 addrspace(1)* %3, align 4
  br label %sw.epilog

sw.epilog:                                        ; preds = %entry, %sw.bb2, %sw.bb1, %sw.bb
  ret void
}

attributes #0 = { convergent noinline nounwind optnone "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "uniform-work-group-size"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0}
!opencl.ocl.version = !{!1}
!opencl.spir.version = !{!1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 2, i32 0}
!2 = !{!"clang version 9.0.0"}
!3 = !{i32 1, i32 0}
!4 = !{!"none", !"none"}
!5 = !{!"int*", !"uchar"}
!6 = !{!"", !""}
