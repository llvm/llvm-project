; kernel
; __attribute__((vec_type_hint(float4)))
; void test_float() {}
;
; kernel
; __attribute__((vec_type_hint(double)))
; void test_double() {}
;
; kernel
; __attribute__((vec_type_hint(uint4)))
; void test_uint() {}
;
; kernel
; __attribute__((vec_type_hint(int8)))
; void test_int() {}
; bash$ clang -cc1 -triple spir64-unknown-unknown -x cl -cl-std=CL2.0 -O0 -include opencl.h -emit-llvm vec_type_hint.cl -o vec_type_hint.ll

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.bc
; RUN: llvm-dis < %t.bc | FileCheck %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV: {{[0-9]+}} EntryPoint {{[0-9]+}} {{[0-9]+}} "test_float"
; CHECK-SPIRV: {{[0-9]+}} EntryPoint {{[0-9]+}} {{[0-9]+}} "test_double"
; CHECK-SPIRV: {{[0-9]+}} EntryPoint {{[0-9]+}} {{[0-9]+}} "test_uint"
; CHECK-SPIRV: {{[0-9]+}} EntryPoint {{[0-9]+}} {{[0-9]+}} "test_int"
; CHECK-SPIRV: {{[0-9]+}} ExecutionMode {{[0-9]+}} 30 {{[0-9]+}}
; CHECK-SPIRV: {{[0-9]+}} ExecutionMode {{[0-9]+}} 30 {{[0-9]+}}
; CHECK-SPIRV: {{[0-9]+}} ExecutionMode {{[0-9]+}} 30 {{[0-9]+}}
; CHECK-SPIRV: {{[0-9]+}} ExecutionMode {{[0-9]+}} 30 {{[0-9]+}}

; ModuleID = 'vec_type_hint.cl'
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown"

; CHECK-LLVM: define spir_kernel void @test_float()
; CHECK-LLVM-SAME: !vec_type_hint [[VFLOAT:![0-9]+]]
; Function Attrs: norecurse nounwind readnone
define spir_kernel void @test_float() local_unnamed_addr #0 !kernel_arg_addr_space !2 !kernel_arg_access_qual !2 !kernel_arg_type !2 !kernel_arg_base_type !2 !kernel_arg_type_qual !2 !vec_type_hint !4 {
entry:
  ret void
}

; CHECK-LLVM: define spir_kernel void @test_double()
; CHECK-LLVM-SAME: !vec_type_hint [[VDOUBLE:![0-9]+]]
; Function Attrs: norecurse nounwind readnone
define spir_kernel void @test_double() local_unnamed_addr #0 !kernel_arg_addr_space !2 !kernel_arg_access_qual !2 !kernel_arg_type !2 !kernel_arg_base_type !2 !kernel_arg_type_qual !2 !vec_type_hint !5 {
entry:
  ret void
}

; CHECK-LLVM: define spir_kernel void @test_uint()
; CHECK-LLVM-SAME: !vec_type_hint [[VUINT:![0-9]+]]
; Function Attrs: norecurse nounwind readnone
define spir_kernel void @test_uint() local_unnamed_addr #0 !kernel_arg_addr_space !2 !kernel_arg_access_qual !2 !kernel_arg_type !2 !kernel_arg_base_type !2 !kernel_arg_type_qual !2 !vec_type_hint !6 {
entry:
  ret void
}

; CHECK-LLVM: define spir_kernel void @test_int()
; CHECK-LLVM-SAME: !vec_type_hint [[VINT:![0-9]+]]
; Function Attrs: norecurse nounwind readnone
define spir_kernel void @test_int() local_unnamed_addr #0 !kernel_arg_addr_space !2 !kernel_arg_access_qual !2 !kernel_arg_type !2 !kernel_arg_base_type !2 !kernel_arg_type_qual !2 !vec_type_hint !7 {
entry:
  ret void
}

attributes #0 = { norecurse nounwind readnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0}
!opencl.enable.FP_CONTRACT = !{}
!opencl.ocl.version = !{!1}
!opencl.spir.version = !{!1}
!opencl.used.extensions = !{!2}
!opencl.used.optional.core.features = !{!2}
!opencl.compiler.options = !{!2}
!llvm.ident = !{!3}

; CHECK-LLVM: [[VFLOAT]] = !{<4 x float> undef, i32 1}
; CHECK-LLVM: [[VDOUBLE]] = !{double undef, i32 1}
; CHECK-LLVM: [[VUINT]] = !{<4 x i32> undef, i32 1}
; CHECK-LLVM: [[VINT]] = !{<8 x i32> undef, i32 1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 2, i32 0}
!2 = !{}
!3 = !{!"clang version 6.0.0 (cfe/trunk)"}
!4 = !{<4 x float> undef, i32 0}
!5 = !{double undef, i32 0}
!6 = !{<4 x i32> undef, i32 0}
!7 = !{<8 x i32> undef, i32 1}
