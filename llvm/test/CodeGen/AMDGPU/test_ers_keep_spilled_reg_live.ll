; RUN: llc -mtriple=amdgcn -mcpu=gfx1200 -enable-next-use-analysis=true -verify-machineinstrs -dump-next-use-distance < %s 2>&1 | FileCheck %s

;
;       bb.0.entry
;        /      |
;   bb.3.bb2    |
;      /   |    |
; bb.9.bb5 |    |
;      \   |    |
;    bb.1.Flow1 |
;         \     |
;        bb.8.Flow
;         /  |
;  bb.2.bb1  |
;         \  |
;        bb.6.Flow2
;         /  |
;  bb.7.bb4  |
;         \  |
;        bb.4.Flow3
;         /  |
;  bb.5.bb3  |
;         \  |
;        bb.10.exit
;
define amdgpu_ps i64 @test(i1 %cond, ptr addrspace(3) %p, i64 %val) {
; CHECK-LABEL: # Machine code for function test: IsSSA, TracksLiveness
; CHECK-NEXT: Function Live Ins: $vgpr0 in [[Reg1:%[0-9]+]], $vgpr1 in [[Reg2:%[0-9]+]], $vgpr2 in [[Reg3:%[0-9]+]], $vgpr3 in [[Reg4:%[0-9]+]]
; EMPTY:
; CHECK: bb.0.entry:
; CHECK-NEXT:   successors: %bb.3(0x40000000), %bb.8(0x40000000); %bb.3(50.00%), %bb.8(50.00%)
; CHECK-NEXT:   liveins: $vgpr0, $vgpr1, $vgpr2, $vgpr3
; CHECK-NEXT:   [[Reg4]]:vgpr_32 = COPY killed $vgpr3
; CHECK-NEXT:   [[Reg3]]:vgpr_32 = COPY killed $vgpr2
; CHECK-NEXT:   [[Reg2]]:vgpr_32 = COPY killed $vgpr1
; CHECK-NEXT:   [[Reg1]]:vgpr_32 = COPY killed $vgpr0
; CHECK-NEXT:   [[Reg5:%[0-9]+]]:vgpr_32 = V_AND_B32_e64 1, killed [[Reg1]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg6:%[0-9]+]]:sreg_32 = V_CMP_NE_U32_e64 1, killed [[Reg5]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg7:%[0-9]+]]:sreg_32 = S_MOV_B32 0
; CHECK-NEXT:   [[Reg8:%[0-9]+]]:sreg_32 = SI_IF [[Reg6]]:sreg_32, %bb.8, implicit-def dead $exec, implicit-def dead $scc, implicit $exec
; CHECK-NEXT:   S_BRANCH %bb.3
; EMPTY:
; CHECK: bb.1.Flow1:
; CHECK-NEXT: ; predecessors: %bb.3, %bb.9
; CHECK-NEXT:   successors: %bb.8(0x80000000); %bb.8(100.00%)
; EMPTY:
; CHECK:   [[Reg9:%[0-9]+]]:sreg_32 = PHI [[Reg10:%[0-9]+]]:sreg_32, %bb.3, [[Reg11:%[0-9]+]]:sreg_32, %bb.9
; CHECK-NEXT:   [[Reg12:%[0-9]+]]:vreg_64 = PHI undef [[Reg13:%[0-9]+]]:vreg_64, %bb.3, [[Reg14:%[0-9]+]]:vreg_64, %bb.9
; CHECK-NEXT:   SI_END_CF killed [[Reg15:%[0-9]+]]:sreg_32, implicit-def dead $exec, implicit-def dead $scc, implicit $exec
; CHECK-NEXT:   [[Reg16:%[0-9]+]]:sreg_32 = S_AND_B32 killed [[Reg9]]:sreg_32, $exec_lo, implicit-def dead $scc
; CHECK-NEXT:   [[Reg17:%[0-9]+]]:sreg_32 = COPY killed [[Reg16]]:sreg_32
; CHECK-NEXT:   S_BRANCH %bb.8
; EMPTY:
; CHECK: bb.2.bb1:
; CHECK-NEXT: ; predecessors: %bb.8
; CHECK-NEXT:   successors: %bb.6(0x80000000); %bb.6(100.00%)
; EMPTY:
; CHECK:   [[Reg18:%[0-9]+]]:vgpr_32 = DS_READ_U16_gfx9 [[Reg2]]:vgpr_32, 0, 0, implicit $exec :: (load (s16) from %ir.p, addrspace 3)
; CHECK-NEXT:   [[Reg19:%[0-9]+]]:vgpr_32 = DS_READ_U16_gfx9 [[Reg2]]:vgpr_32, 2, 0, implicit $exec :: (load (s16) from %ir.p + 2, addrspace 3)
; CHECK-NEXT:   [[Reg20:%[0-9]+]]:vgpr_32 = DS_READ_U16_gfx9 [[Reg2]]:vgpr_32, 4, 0, implicit $exec :: (load (s16) from %ir.p + 4, addrspace 3)
; CHECK-NEXT:   [[Reg21:%[0-9]+]]:vgpr_32 = DS_READ_U16_gfx9 [[Reg2]]:vgpr_32, 6, 0, implicit $exec :: (load (s16) from %ir.p + 6, addrspace 3)
; CHECK-NEXT:   [[Reg22:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg21]]:vgpr_32, 16, killed [[Reg20]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg23:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg19]]:vgpr_32, 16, killed [[Reg18]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg24:%[0-9]+]]:vreg_64 = REG_SEQUENCE killed [[Reg23]]:vgpr_32, %subreg.sub0, killed [[Reg22]]:vgpr_32, %subreg.sub1
; CHECK-NEXT:   [[Reg25:%[0-9]+]]:vreg_64 = COPY killed [[Reg24]]:vreg_64
; CHECK-NEXT:   [[Reg26:%[0-9]+]]:sreg_32 = COPY $exec_lo
; CHECK-NEXT:   [[Reg27:%[0-9]+]]:sreg_32 = S_ANDN2_B32 killed [[Reg28:%[0-9]+]]:sreg_32, $exec_lo, implicit-def dead $scc
; CHECK-NEXT:   [[Reg29:%[0-9]+]]:sreg_32 = S_AND_B32 killed [[Reg6]]:sreg_32, $exec_lo, implicit-def dead $scc
; CHECK-NEXT:   [[Reg30:%[0-9]+]]:sreg_32 = S_OR_B32 killed [[Reg27]]:sreg_32, killed [[Reg29]]:sreg_32, implicit-def dead $scc
; CHECK-NEXT:   S_BRANCH %bb.6
; EMPTY:
; CHECK: bb.3.bb2:
; CHECK-NEXT: ; predecessors: %bb.0
; CHECK-NEXT:   successors: %bb.9(0x40000000), %bb.1(0x40000000); %bb.9(50.00%), %bb.1(50.00%)
; EMPTY:
; CHECK:   [[Reg31:%[0-9]+]]:vreg_64 = DS_READ_B64_gfx9 [[Reg2]]:vgpr_32, 8, 0, implicit $exec :: (load (s64) from %ir.gep2, addrspace 3)
; CHECK-NEXT:   [[Reg10]]:sreg_32 = S_MOV_B32 -1
; CHECK-NEXT:   [[Reg15]]:sreg_32 = SI_IF [[Reg6]]:sreg_32, %bb.1, implicit-def dead $exec, implicit-def dead $scc, implicit $exec
; CHECK-NEXT:   S_BRANCH %bb.9
; EMPTY:
; CHECK: bb.4.Flow3:
; CHECK-NEXT: ; predecessors: %bb.6, %bb.7
; CHECK-NEXT:   successors: %bb.5(0x40000000), %bb.10(0x40000000); %bb.5(50.00%), %bb.10(50.00%)
; EMPTY:
; CHECK:   [[Reg32:%[0-9]+]]:sreg_32 = PHI [[Reg33:%[0-9]+]]:sreg_32, %bb.6, [[Reg34:%[0-9]+]]:sreg_32, %bb.7
; CHECK-NEXT:   [[Reg35:%[0-9]+]]:vreg_64 = PHI [[Reg36:%[0-9]+]]:vreg_64, %bb.6, [[Reg37:%[0-9]+]]:vreg_64, %bb.7
; CHECK-NEXT:   SI_END_CF killed [[Reg38:%[0-9]+]]:sreg_32, implicit-def dead $exec, implicit-def dead $scc, implicit $exec
; CHECK-NEXT:   [[Reg39:%[0-9]+]]:sreg_32 = SI_IF killed [[Reg32]]:sreg_32, %bb.10, implicit-def dead $exec, implicit-def dead $scc, implicit $exec
; CHECK-NEXT:   S_BRANCH %bb.5
; EMPTY:
; CHECK: bb.5.bb3:
; CHECK-NEXT: ; predecessors: %bb.4
; CHECK-NEXT:   successors: %bb.10(0x80000000); %bb.10(100.00%)
; EMPTY:
; CHECK:   [[Reg40:%[0-9]+]]:vreg_64 = DS_READ2_B32_gfx9 killed [[Reg2]]:vgpr_32, 6, 7, 0, implicit $exec :: (load (s64) from %ir.gep3, align 4, addrspace 3)
; CHECK-NEXT:   [[Reg41:%[0-9]+]]:vgpr_32, [[Reg42:%[0-9]+]]:sreg_32_xm0_xexec = V_ADD_CO_U32_e64 [[Reg40]].sub0:vreg_64, killed [[Reg3]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg43:%[0-9]+]]:vgpr_32, dead $sgpr_null = V_ADDC_U32_e64 killed [[Reg40]].sub1:vreg_64, killed [[Reg4]]:vgpr_32, killed [[Reg42]]:sreg_32_xm0_xexec, 0, implicit $exec
; CHECK-NEXT:   [[Reg44:%[0-9]+]]:vreg_64 = REG_SEQUENCE killed [[Reg41]]:vgpr_32, %subreg.sub0, killed [[Reg43]]:vgpr_32, %subreg.sub1
; CHECK-NEXT:   S_BRANCH %bb.10
; EMPTY:
; CHECK: bb.6.Flow2:
; CHECK-NEXT: ; predecessors: %bb.8, %bb.2
; CHECK-NEXT:   successors: %bb.7(0x40000000), %bb.4(0x40000000); %bb.7(50.00%), %bb.4(50.00%)
; EMPTY:
; CHECK:   [[Reg45:%[0-9]+]]:sreg_32 = PHI [[Reg28]]:sreg_32, %bb.8, [[Reg30]]:sreg_32, %bb.2
; CHECK-NEXT:   [[Reg33]]:sreg_32 = PHI [[Reg7]]:sreg_32, %bb.8, [[Reg26]]:sreg_32, %bb.2
; CHECK-NEXT:   [[Reg46:%[0-9]+]]:vreg_64 = PHI [[Reg47:%[0-9]+]]:vreg_64, %bb.8, [[Reg25]]:vreg_64, %bb.2
; CHECK-NEXT:   SI_END_CF killed [[Reg48:%[0-9]+]]:sreg_32, implicit-def dead $exec, implicit-def dead $scc, implicit $exec
; CHECK-NEXT:   [[Reg38]]:sreg_32 = SI_IF killed [[Reg45]]:sreg_32, %bb.4, implicit-def dead $exec, implicit-def dead $scc, implicit $exec
; CHECK-NEXT:   S_BRANCH %bb.7
; EMPTY:
; CHECK: bb.7.bb4:
; CHECK-NEXT: ; predecessors: %bb.6
; CHECK-NEXT:   successors: %bb.4(0x80000000); %bb.4(100.00%)
; EMPTY:
; CHECK:   [[Reg49:%[0-9]+]]:vgpr_32, [[Reg50:%[0-9]+]]:sreg_32_xm0_xexec = V_ADD_CO_U32_e64 [[Reg46]].sub0:vreg_64, [[Reg3]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg51:%[0-9]+]]:vgpr_32, dead $sgpr_null = V_ADDC_U32_e64 killed [[Reg46]].sub1:vreg_64, [[Reg4]]:vgpr_32, killed [[Reg50]]:sreg_32_xm0_xexec, 0, implicit $exec
; CHECK-NEXT:   [[Reg37]]:vreg_64 = REG_SEQUENCE killed [[Reg49]]:vgpr_32, %subreg.sub0, killed [[Reg51]]:vgpr_32, %subreg.sub1
; CHECK-NEXT:   [[Reg52:%[0-9]+]]:sreg_32 = S_ANDN2_B32 killed [[Reg33]]:sreg_32, $exec_lo, implicit-def dead $scc
; CHECK-NEXT:   [[Reg34]]:sreg_32 = COPY killed [[Reg52]]:sreg_32
; CHECK-NEXT:   S_BRANCH %bb.4
; EMPTY:
; CHECK: bb.8.Flow:
; CHECK-NEXT: ; predecessors: %bb.0, %bb.1
; CHECK-NEXT:   successors: %bb.2(0x40000000), %bb.6(0x40000000); %bb.2(50.00%), %bb.6(50.00%)
; EMPTY:
; CHECK:   [[Reg28]]:sreg_32 = PHI [[Reg7]]:sreg_32, %bb.0, [[Reg17]]:sreg_32, %bb.1
; CHECK-NEXT:   [[Reg36]]:vreg_64 = PHI undef [[Reg53:%[0-9]+]]:vreg_64, %bb.0, [[Reg12]]:vreg_64, %bb.1
; CHECK-NEXT:   [[Reg47]]:vreg_64 = PHI undef [[Reg53]]:vreg_64, %bb.0, [[Reg31]]:vreg_64, %bb.1
; CHECK-NEXT:   [[Reg48]]:sreg_32 = SI_ELSE killed [[Reg8]]:sreg_32, %bb.6, implicit-def dead $exec, implicit-def dead $scc, implicit $exec
; CHECK-NEXT:   S_BRANCH %bb.2
; EMPTY:
; CHECK: bb.9.bb5:
; CHECK-NEXT: ; predecessors: %bb.3
; CHECK-NEXT:   successors: %bb.1(0x80000000); %bb.1(100.00%)
; EMPTY:
; CHECK:   [[Reg54:%[0-9]+]]:vgpr_32 = DS_READ_U8_gfx9 [[Reg2]]:vgpr_32, 32, 0, implicit $exec :: (load (s8) from %ir.gep4, addrspace 3)
; CHECK-NEXT:   [[Reg55:%[0-9]+]]:vgpr_32 = DS_READ_U8_gfx9 [[Reg2]]:vgpr_32, 33, 0, implicit $exec :: (load (s8) from %ir.gep4 + 1, addrspace 3)
; CHECK-NEXT:   [[Reg56:%[0-9]+]]:vgpr_32 = DS_READ_U8_gfx9 [[Reg2]]:vgpr_32, 34, 0, implicit $exec :: (load (s8) from %ir.gep4 + 2, addrspace 3)
; CHECK-NEXT:   [[Reg57:%[0-9]+]]:vgpr_32 = DS_READ_U8_gfx9 [[Reg2]]:vgpr_32, 35, 0, implicit $exec :: (load (s8) from %ir.gep4 + 3, addrspace 3)
; CHECK-NEXT:   [[Reg58:%[0-9]+]]:vgpr_32 = DS_READ_U8_gfx9 [[Reg2]]:vgpr_32, 36, 0, implicit $exec :: (load (s8) from %ir.gep4 + 4, addrspace 3)
; CHECK-NEXT:   [[Reg59:%[0-9]+]]:vgpr_32 = DS_READ_U8_gfx9 [[Reg2]]:vgpr_32, 37, 0, implicit $exec :: (load (s8) from %ir.gep4 + 5, addrspace 3)
; CHECK-NEXT:   [[Reg60:%[0-9]+]]:vgpr_32 = DS_READ_U8_gfx9 [[Reg2]]:vgpr_32, 38, 0, implicit $exec :: (load (s8) from %ir.gep4 + 6, addrspace 3)
; CHECK-NEXT:   [[Reg61:%[0-9]+]]:vgpr_32 = DS_READ_U8_gfx9 [[Reg2]]:vgpr_32, 39, 0, implicit $exec :: (load (s8) from %ir.gep4 + 7, addrspace 3)
; CHECK-NEXT:   [[Reg62:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg59]]:vgpr_32, 8, killed [[Reg58]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg63:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg61]]:vgpr_32, 8, killed [[Reg60]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg64:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg63]]:vgpr_32, 16, killed [[Reg62]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg65:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg55]]:vgpr_32, 8, killed [[Reg54]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg66:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg57]]:vgpr_32, 8, killed [[Reg56]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg67:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg66]]:vgpr_32, 16, killed [[Reg65]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg68:%[0-9]+]]:vgpr_32, [[Reg69:%[0-9]+]]:sreg_32_xm0_xexec = V_ADD_CO_U32_e64 killed [[Reg67]]:vgpr_32, [[Reg3]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg70:%[0-9]+]]:vgpr_32, dead $sgpr_null = V_ADDC_U32_e64 killed [[Reg64]]:vgpr_32, [[Reg4]]:vgpr_32, killed [[Reg69]]:sreg_32_xm0_xexec, 0, implicit $exec
; CHECK-NEXT:   [[Reg14]]:vreg_64 = REG_SEQUENCE killed [[Reg68]]:vgpr_32, %subreg.sub0, killed [[Reg70]]:vgpr_32, %subreg.sub1
; CHECK-NEXT:   [[Reg11]]:sreg_32 = S_XOR_B32 $exec_lo, -1, implicit-def dead $scc
; CHECK-NEXT:   S_BRANCH %bb.1
; EMPTY:
; CHECK: bb.10.exit:
; CHECK-NEXT: ; predecessors: %bb.4, %bb.5
; EMPTY:
; CHECK:   [[Reg71:%[0-9]+]]:vreg_64 = PHI [[Reg35]]:vreg_64, %bb.4, [[Reg44]]:vreg_64, %bb.5
; CHECK-NEXT:   SI_END_CF killed [[Reg39]]:sreg_32, implicit-def dead $exec, implicit-def dead $scc, implicit $exec
; CHECK-NEXT:   [[Reg72:%[0-9]+]]:sreg_32_xm0 = V_READFIRSTLANE_B32 [[Reg71]].sub0:vreg_64, implicit $exec
; CHECK-NEXT:   [[Reg73:%[0-9]+]]:sreg_32_xm0 = V_READFIRSTLANE_B32 killed [[Reg71]].sub1:vreg_64, implicit $exec
; CHECK-NEXT:   $sgpr0 = COPY killed [[Reg72]]:sreg_32_xm0
; CHECK-NEXT:   $sgpr1 = COPY killed [[Reg73]]:sreg_32_xm0
; CHECK-NEXT:   SI_RETURN_TO_EPILOG killed $sgpr0, killed $sgpr1
; EMPTY:
; CHECK: # End machine code for function test.
; EMPTY:
; CHECK: Next-use distance of Register [[Reg4]] = 21.0
; CHECK-NEXT: Next-use distance of Register [[Reg3]] = 19.0
; CHECK-NEXT: Next-use distance of Register [[Reg2]] = 7.0
; CHECK-NEXT: Next-use distance of Register [[Reg1]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg5]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg6]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg7]] = 3.0
; CHECK-NEXT: Next-use distance of Register [[Reg8]] = 5.0
; CHECK-NEXT: Next-use distance of Register [[Reg9]] = 3.0
; CHECK-NEXT: Next-use distance of Register [[Reg12]] = 6.0
; CHECK-NEXT: Next-use distance of Register [[Reg16]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg17]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg18]] = 5.0
; CHECK-NEXT: Next-use distance of Register [[Reg19]] = 4.0
; CHECK-NEXT: Next-use distance of Register [[Reg20]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg21]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg22]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg23]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg24]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg25]] = 8.0
; CHECK-NEXT: Next-use distance of Register [[Reg26]] = 6.0
; CHECK-NEXT: Next-use distance of Register [[Reg27]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg29]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg30]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg31]] = 12.0
; CHECK-NEXT: Next-use distance of Register [[Reg10]] = 3.0
; CHECK-NEXT: Next-use distance of Register [[Reg15]] = 4.0
; CHECK-NEXT: Next-use distance of Register [[Reg32]] = 3.0
; CHECK-NEXT: Next-use distance of Register [[Reg35]] = 4.0
; CHECK-NEXT: Next-use distance of Register [[Reg39]] = 3.0
; CHECK-NEXT: Next-use distance of Register [[Reg40]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg41]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg42]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg43]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg44]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg45]] = 4.0
; CHECK-NEXT: Next-use distance of Register [[Reg33]] = 5.0
; CHECK-NEXT: Next-use distance of Register [[Reg46]] = 4.0
; CHECK-NEXT: Next-use distance of Register [[Reg38]] = 4.0
; CHECK-NEXT: Next-use distance of Register [[Reg49]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg50]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg51]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg37]] = 5.0
; CHECK-NEXT: Next-use distance of Register [[Reg52]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg34]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg28]] = 5.0
; CHECK-NEXT: Next-use distance of Register [[Reg36]] = 11.0
; CHECK-NEXT: Next-use distance of Register [[Reg47]] = 5.0
; CHECK-NEXT: Next-use distance of Register [[Reg48]] = 5.0
; CHECK-NEXT: Next-use distance of Register [[Reg54]] = 11.0
; CHECK-NEXT: Next-use distance of Register [[Reg55]] = 10.0
; CHECK-NEXT: Next-use distance of Register [[Reg56]] = 10.0
; CHECK-NEXT: Next-use distance of Register [[Reg57]] = 9.0
; CHECK-NEXT: Next-use distance of Register [[Reg58]] = 4.0
; CHECK-NEXT: Next-use distance of Register [[Reg59]] = 3.0
; CHECK-NEXT: Next-use distance of Register [[Reg60]] = 3.0
; CHECK-NEXT: Next-use distance of Register [[Reg61]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg62]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg63]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg64]] = 5.0
; CHECK-NEXT: Next-use distance of Register [[Reg65]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg66]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg67]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg68]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg69]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg70]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg14]] = 4.0
; CHECK-NEXT: Next-use distance of Register [[Reg11]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg71]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg72]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg73]] = 2.0
entry:
;      entry
;      /   \
;    bb1   bb2
;    / \   / \
;  bb3  bb4  bb5
;     \  |  /
;      exit
   br i1 %cond, label %bb1, label %bb2

bb1:
   %gep1 = getelementptr inbounds i64, ptr addrspace(3) %p, i64 0
   %ld1 = load i64, ptr addrspace(3) %gep1, align 2
   br i1 %cond, label %bb3, label %bb4

bb2:
   %gep2 = getelementptr inbounds i64, ptr addrspace(3) %p, i64 1
   %ld2 = load i64, ptr addrspace(3) %gep2, align 8
   br i1 %cond, label %bb4, label %bb5

bb3:
   %gep3 = getelementptr inbounds i64, ptr addrspace(3) %p, i64 3
   %ld3 = load i64, ptr addrspace(3) %gep3, align 4
   %add1 = add i64 %ld3, %val
   br label %exit

bb4:
   %phi1 = phi i64 [ %ld1, %bb1 ], [ %ld2, %bb2]
   %add2 = add i64 %phi1, %val
   br label %exit

bb5:
   %gep4 = getelementptr inbounds i64, ptr addrspace(3) %p, i64 4
   %ld4 = load i64, ptr addrspace(3) %gep4, align 1
   %add3 = add i64 %ld4, %val
   br label %exit

exit:
   %phi2 = phi i64 [ %add1, %bb3 ], [ %add2, %bb4 ], [ %add3, %bb5 ]
   ret i64 %phi2
}
