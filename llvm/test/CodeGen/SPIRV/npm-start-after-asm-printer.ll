; RUN: llc -mtriple=spirv64-unknown-unknown -O0 -enable-new-pm \
; RUN:   -start-after=spirv-prepare-module-analysis -print-pipeline-passes \
; RUN:   %s -o /dev/null | FileCheck %s --check-prefix=PIPELINE
; RUN: llc -mtriple=spirv64-unknown-unknown -O0 -enable-new-pm \
; RUN:   -stop-after=spirv-prepare-module-analysis %s -o - \
; RUN:   | llc -mtriple=spirv64-unknown-unknown -O0 -enable-new-pm -x mir \
; RUN:     -start-after=spirv-prepare-module-analysis -o - | FileCheck %s

; PIPELINE: spirv-asm-printer-begin,{{.*}}spirv-asm-printer{{.*}},spirv-asm-printer-end

; CHECK: OpEntryPoint Kernel %[[#K:]] "k"
; CHECK: %[[#K]] = OpFunction
; CHECK: OpReturn
; CHECK: OpFunctionEnd

define spir_kernel void @k() {
  ret void
}
