; RUN: llc -march=amdgcn -stop-after=amdgpu-isel -verify-machineinstrs -o - %s | FileCheck %s

; CHECK-LABEL: vcopy_i1_undef
; CHECK: [[IMPDEF0:%[0-9]+]]:sreg_64 = IMPLICIT_DEF
; CHECK-NOT: COPY
; CHECK: [[IMPDEF1:%[0-9]+]]:sreg_64 = IMPLICIT_DEF
; CHECK-NOT: COPY [[IMPDEF0]]
; CHECK-NOT: COPY [[IMPDEF1]]
; CHECK: .false:
define <2 x float> @vcopy_i1_undef(<2 x float> addrspace(1)* %p) {
entry:
  br i1 undef, label %exit, label %false

false:
  %x = load <2 x float>, <2 x float> addrspace(1)* %p
  %cmp = fcmp one <2 x float> %x, zeroinitializer
  br label %exit

exit:
  %c = phi <2 x i1> [ undef, %entry ], [ %cmp, %false ]
  %ret = select <2 x i1> %c, <2 x float> <float 2.0, float 2.0>, <2 x float> <float 4.0, float 4.0>
  ret <2 x float> %ret
}
