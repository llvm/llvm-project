; RUN: llvm-as %s -o - | llvm-spirv -o %t.spv
; RUN: llvm-spirv %t.spv -spirv-gen-kernel-arg-name-md -r -o - | llvm-dis -o - | FileCheck %s

; CHECK: spir_kernel void @named_arg(float %f) {{.*}} !kernel_arg_name ![[MD_named:[0-9]+]]
; CHECK: spir_kernel void @unnamed_arg(float) {{.*}} !kernel_arg_name ![[MD_unnamed:[0-9]+]]
; CHECK: spir_kernel void @one_unnamed_arg(i8 %a, i8 %b, i8) {{.*}} !kernel_arg_name ![[MD_one_unnamed:[0-9]+]]

; CHECK: ![[MD_unnamed]] = !{!""}
; CHECK: ![[MD_named]] = !{!"f"}
; CHECK: ![[MD_one_unnamed]] = !{!"a", !"b", !""}

; ModuleID = 'kernel_arg_name.ll'
source_filename = "kernel_arg_name.ll"
target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir"

; Function Attrs: norecurse nounwind readnone
define spir_kernel void @named_arg(float %f) local_unnamed_addr #0 !kernel_arg_addr_space !0 !kernel_arg_access_qual !0 !kernel_arg_type !0 !kernel_arg_base_type !0 !kernel_arg_type_qual !0 {
entry:
  ret void
}

; Function Attrs: norecurse nounwind readnone
define spir_kernel void @unnamed_arg(float) local_unnamed_addr #0 !kernel_arg_addr_space !0 !kernel_arg_access_qual !0 !kernel_arg_type !0 !kernel_arg_base_type !0 !kernel_arg_type_qual !0 {
entry:
  ret void
}

; Function Attrs: norecurse nounwind readnone
define spir_kernel void @one_unnamed_arg(i8 %a, i8 %b, i8) local_unnamed_addr #0 !kernel_arg_addr_space !0 !kernel_arg_access_qual !0 !kernel_arg_type !0 !kernel_arg_base_type !0 !kernel_arg_type_qual !0 {
entry:
  ret void
}

attributes #0 = { norecurse nounwind readnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!0 = !{}
