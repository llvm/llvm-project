; REQUIRES: spirv-tools
; RUN: llc -O0 -mtriple=spirv-unknown-unknown %s -o - --filetype=obj | spirv-dis | FileCheck %s
; RUN: llc -O0 -mtriple=spirv64-amd-amdhsa %s -o - --filetype=obj | spirv-dis | FileCheck --check-prefix=AMDGCNSPIRV %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK: Generator: {{.*}}{{43|LLVM SPIR-V Backend}}{{.*}}
; AMDGCNSPIRV: Generator: {{.*}}{{65535|LLVM SPIR-V Backend}}{{.*}}
