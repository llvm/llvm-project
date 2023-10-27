; RUN: llc -mtriple=amdgcn-amd-amdhsa -filetype=obj -o - < %s | llvm-readelf --notes - | FileCheck %s

; CHECK: amdhsa.kernels:
; CHECK-NEXT:  - .args:
; CHECK-NEXT:      - .actual_access:  read_only
; CHECK-LABEL:   .name:          test_noalias_ro_arg
define amdgpu_kernel void @test_noalias_ro_arg(ptr noalias readonly %in) {
  ret void
}

; CHECK:       - .args:
; CHECK-NOT:     read_only
; CHECK-LABEL:   .name:          test_only_ro_arg
define amdgpu_kernel void @test_only_ro_arg(ptr readonly %in) {
  ret void
}

; CHECK:       - .args:
; CHECK-NEXT:      - .actual_access:  write_only
; CHECK-LABEL:   .name:          test_noalias_wo_arg
define amdgpu_kernel void @test_noalias_wo_arg(ptr noalias writeonly %out) {
  ret void
}

; CHECK:       - .args:
; CHECK-NOT:     write_only
; CHECK-LABEL:   .name:          test_only_wo_arg
define amdgpu_kernel void @test_only_wo_arg(ptr writeonly %out) {
  ret void
}
