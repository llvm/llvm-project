; RUN: split-file %s %t
; RUN: llc -O0 %t/metadata-opencl12.ll -o - | FileCheck %t/metadata-opencl12.ll
; RUN: llc -O0 %t/metadata-opencl20.ll -o - | FileCheck %t/metadata-opencl20.ll
; RUN: llc -O0 %t/metadata-opencl22.ll -o - | FileCheck %t/metadata-opencl22.ll

;--- metadata-opencl12.ll
target triple = "spirv32-unknown-unknown"

!opencl.ocl.version = !{!0}
!0 = !{i32 1, i32 2}

; We assume the SPIR-V 2.2 environment spec's version format: 0|Maj|Min|Rev|
; CHECK: OpSource OpenCL_C 66048
;--- metadata-opencl20.ll
target triple = "spirv32-unknown-unknown"

!opencl.ocl.version = !{!0}
!0 = !{i32 2, i32 0}

; We assume the SPIR-V 2.2 environment spec's version format: 0|Maj|Min|Rev|
; CHECK: OpSource OpenCL_C 131072
;--- metadata-opencl22.ll
target triple = "spirv32-unknown-unknown"

!opencl.ocl.version = !{!0}
!0 = !{i32 2, i32 2}

; We assume the SPIR-V 2.2 environment spec's version format: 0|Maj|Min|Rev|
; CHECK: OpSource OpenCL_C 131584
