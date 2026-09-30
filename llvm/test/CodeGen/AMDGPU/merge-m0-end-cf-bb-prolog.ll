; RUN: llc -mtriple=amdgcn -mcpu=tahiti -stop-after=si-fix-sgpr-copies -o - %s | FileCheck %s

; The M0 initialization merged into a join block stays behind the exec restore.

; CHECK-LABEL: name: m0_init_after_end_cf
; CHECK: bb.{{[0-9]+}}.join:
; CHECK: bb-prolog SI_END_CF
; CHECK-NEXT: SI_INIT_M0 -1

@lds = internal addrspace(3) global [4 x i32] poison

define amdgpu_ps float @m0_init_after_end_cf(float %v, i32 inreg %sel, i32 inreg %idx) {
entry:
  %cond = fcmp ogt float %v, 0.0
  br i1 %cond, label %then, label %join

then:
  %add = fadd float %v, 1.0
  br label %join

join:
  %phi = phi float [ %add, %then ], [ %v, %entry ]
  %uniform = icmp eq i32 %sel, 0
  br i1 %uniform, label %left, label %right

left:
  %gep.l = getelementptr [4 x i32], ptr addrspace(3) @lds, i32 0, i32 %idx
  %ld.l = load i32, ptr addrspace(3) %gep.l
  br label %exit

right:
  %idx1 = add i32 %idx, 1
  %gep.r = getelementptr [4 x i32], ptr addrspace(3) @lds, i32 0, i32 %idx1
  %ld.r = load i32, ptr addrspace(3) %gep.r
  br label %exit

exit:
  %ld = phi i32 [ %ld.l, %left ], [ %ld.r, %right ]
  %f = sitofp i32 %ld to float
  %r = fadd float %f, %phi
  ret float %r
}
