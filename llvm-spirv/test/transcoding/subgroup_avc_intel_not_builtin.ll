; Source:
; void __attribute__((overloadable)) intel_sub_group_avc_mce_ime_boo();
; void foo() {
;   intel_sub_group_avc_mce_ime_boo();
; }

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o - -spirv-text | FileCheck %s

; Checks that a function with a name started from 'intel_sub_group_avc_' prefix,
; but which is not a part of 'cl_intel_device_side_avc_motion_estimation'
; extension specification, is being translated to a regular FunctionCall.

; CHECK: Name [[Name:[0-9]+]] "_Z31intel_sub_group_avc_mce_ime_boo"
; CHECK: FunctionCall {{[0-9]+}} {{[0-9]+}} [[Name]]

target triple = "spir"

; Function Attrs: noinline nounwind optnone
define spir_func void @foo() #0 {
entry:
  call spir_func void @_Z31intel_sub_group_avc_mce_ime_boo()
  ret void
}

declare spir_func void @_Z31intel_sub_group_avc_mce_ime_boo() #1

attributes #0 = { noinline nounwind optnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0}
!opencl.enable.FP_CONTRACT = !{}
!opencl.ocl.version = !{!1}
!opencl.spir.version = !{!1}
!opencl.used.extensions = !{!2}
!opencl.used.optional.core.features = !{!2}
!opencl.compiler.options = !{!2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 2, i32 0}
!2 = !{}
!3 = !{!"clang version 5.0.1 (cfe/trunk)"}
