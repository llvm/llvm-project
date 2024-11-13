; REQUIRES: spirv-tools
; RUN: llc -O0 -mtriple=spirv64v1.0-unknown-unknown %s -o - --filetype=obj | spirv-dis | FileCheck %s --check-prefix=CHECK-SPIRV10
; RUN: llc -O0 -mtriple=spirv64v1.1-unknown-unknown %s -o - --filetype=obj | spirv-dis | FileCheck %s --check-prefix=CHECK-SPIRV11
; RUN: llc -O0 -mtriple=spirv64v1.2-unknown-unknown %s -o - --filetype=obj | spirv-dis | FileCheck %s --check-prefix=CHECK-SPIRV12
; RUN: llc -O0 -mtriple=spirv64v1.3-unknown-unknown %s -o - --filetype=obj | spirv-dis | FileCheck %s --check-prefix=CHECK-SPIRV13
; RUN: llc -O0 -mtriple=spirv64v1.4-unknown-unknown %s -o - --filetype=obj | spirv-dis | FileCheck %s --check-prefix=CHECK-SPIRV14
; RUN: llc -O0 -mtriple=spirv64v1.5-unknown-unknown %s -o - --filetype=obj | spirv-dis | FileCheck %s --check-prefix=CHECK-SPIRV15
; RUN: llc -O0 -mtriple=spirv64v1.6-unknown-unknown %s -o - --filetype=obj | spirv-dis | FileCheck %s --check-prefix=CHECK-SPIRV16

; CHECK-SPIRV10: Version: 1.0
; CHECK-SPIRV11: Version: 1.1
; CHECK-SPIRV12: Version: 1.2
; CHECK-SPIRV13: Version: 1.3
; CHECK-SPIRV14: Version: 1.4
; CHECK-SPIRV15: Version: 1.5
; CHECK-SPIRV16: Version: 1.6
