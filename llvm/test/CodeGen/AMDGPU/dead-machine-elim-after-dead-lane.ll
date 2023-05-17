; RUN: llc -march=amdgcn -verify-machineinstrs %s -o - | FileCheck %s

; CHECK-LABEL: foo
; CHECK-NOT: BUFFER_LOAD_DWORDX2_OFFSET
; After dead code elimination, that buffer load should be eliminated finally
; after dead lane detection.
define amdgpu_kernel void @foo() {
entry:
  switch i8 undef, label %foo.exit [
    i8 4, label %sw.bb4
    i8 10, label %sw.bb10
  ]

sw.bb4:
  %x = load i64, ptr addrspace(1) undef, align 8
  %c = sitofp i64 %x to float
  %v = insertelement <2 x float> <float undef, float 0.000000e+00>, float %c, i32 0
  br label %foo.exit

sw.bb10:
  unreachable

foo.exit:
  %agg = phi <2 x float> [ %v, %sw.bb4 ], [ zeroinitializer, %entry ]
  %s = extractelement <2 x float> %agg, i32 1
  store float %s, ptr addrspace(1) undef, align 4
  ret void
}
