; REQUIRES: spirv-tools
; RUN: llc -O0 -mtriple=spirv-unknown-unknown %s -o - --filetype=obj | spirv-dis | FileCheck %s

; CHECK: Generator: {{.*}}{{43|LLVM SPIR-V Backend}}{{.*}}
