; Source:
; void kk(int x){
;   switch(x) {
;     default: return;
;   }
; }

; Command:
; clang -cc1 -triple spir -emit-llvm -o test/SPIRV/OpSwitchEmpty.ll OpSwitchEmpty.cl -disable-llvm-passes

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o - | FileCheck %s  --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv -o - | llvm-dis -o - | FileCheck %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV: FunctionParameter {{[0-9]+}} [[X:[0-9]+]]
; CHECK-SPIRV: Switch [[X]] [[DEFAULT:[0-9]+]] {{$}}
; CHECK-SPIRV: Label [[DEFAULT]]

; ModuleID = 'test/SPIRV/OpSwitchEmpty.ll'
target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir"

; Function Attrs: nounwind
define spir_func void @kk(i32 %x) #0 {
entry:
  switch i32 %x, label %sw.default [
  ]
; CHECK-LLVM: switch i32 %x, label %sw.default [
; CHECK-LLVM-NEXT ]

sw.default:                                       ; preds = %entry
  ret void
}

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0}
!opencl.enable.FP_CONTRACT = !{}
!opencl.ocl.version = !{!1}
!opencl.spir.version = !{!2}
!opencl.used.extensions = !{!3}
!opencl.used.optional.core.features = !{!3}
!opencl.compiler.options = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 0}
!2 = !{i32 1, i32 2}
!3 = !{}
