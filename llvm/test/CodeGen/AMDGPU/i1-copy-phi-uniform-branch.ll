; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; GCN-LABEL: {{^}}test_dont_clobber_scc:

; GCN: ; %entry
; GCN:      s_cmp_eq_u32    s0, 0
; GCN:      s_cbranch_scc1  [[PREEXIT:BB[0-9_]+]]

; GCN: ; %blocka
; GCN:      s_cmp_eq_u32    s1, 0
; GCN:      s_cbranch_scc1  [[EXIT:BB[0-9_]+]]

; GCN: [[PREEXIT]]:
; GCN: [[EXIT]]:

define amdgpu_vs float @test_dont_clobber_scc(i32 inreg %uni, i32 inreg %uni2) #0 {
entry:
  %cc.uni = icmp eq i32 %uni, 0
  br i1 %cc.uni, label %exit, label %blocka

blocka:
  call void asm sideeffect "; dummy a", ""()
  %cc.uni2 = icmp eq i32 %uni2, 0
  br i1 %cc.uni2, label %exit, label %blockb

blockb:
  call void asm sideeffect "; dummy b", ""()
  br label %exit

exit:
  %cc.phi = phi i1 [ true, %entry ], [ false, %blocka ], [ false, %blockb ]
  call void asm sideeffect "; dummy exit", ""()
  %r = select i1 %cc.phi, float 1.0, float 2.0
  ret float %r
}

attributes #0 = { nounwind }
