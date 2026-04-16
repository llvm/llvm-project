; RUN: llc -mtriple=amdgcn -mcpu=gfx1200 -enable-next-use-analysis=true -verify-machineinstrs -dump-next-use-distance < %s 2>&1 | FileCheck %s

;
;           bb.0.entry
;               |
;       bb.1.loop1.header<---+
;          /        |        |
;bb.4.loop1.latch2  |        |
;          \        |        |
;           bb.2.Flow        |
;           /       |        |
;bb.3.loop1.latch1  |        |
;           \       |        |
;           bb.5.Flow1-------+
;               |
;            bb.6.bb
;               |
;           bb.7.loop2<------+
;               |            |
;               +------------+
;               |
;           bb.8.exit
;
define amdgpu_ps void @test(ptr addrspace(1) %p1, ptr addrspace(1) %p2, ptr addrspace(1) %p3, ptr addrspace(1) %p4, ptr addrspace(1) %p5, i1 %cond, i32 %TC1, i32 %TC2, i32 %TC3) {
; CHECK-LABEL: # Machine code for function test: IsSSA, TracksLiveness
; CHECK-NEXT: Function Live Ins: $vgpr0 in [[Reg1:%[0-9]+]], $vgpr1 in [[Reg2:%[0-9]+]], $vgpr2 in [[Reg3:%[0-9]+]], $vgpr3 in [[Reg4:%[0-9]+]], $vgpr4 in [[Reg5:%[0-9]+]], $vgpr5 in [[Reg6:%[0-9]+]], $vgpr6 in [[Reg7:%[0-9]+]], $vgpr7 in [[Reg8:%[0-9]+]], $vgpr8 in [[Reg9:%[0-9]+]], $vgpr9 in [[Reg10:%[0-9]+]], $vgpr10 in [[Reg11:%[0-9]+]], $vgpr11 in [[Reg12:%[0-9]+]], $vgpr12 in [[Reg13:%[0-9]+]], $vgpr13 in [[Reg14:%[0-9]+]]
; EMPTY:
; CHECK: bb.0.entry:
; CHECK-NEXT:   successors: %bb.1(0x80000000); %bb.1(100.00%)
; CHECK-NEXT:   liveins: $vgpr0, $vgpr1, $vgpr2, $vgpr3, $vgpr4, $vgpr5, $vgpr6, $vgpr7, $vgpr8, $vgpr9, $vgpr10, $vgpr11, $vgpr12, $vgpr13
; CHECK-NEXT:   [[Reg14]]:vgpr_32 = COPY killed $vgpr13
; CHECK-NEXT:   [[Reg13]]:vgpr_32 = COPY killed $vgpr12
; CHECK-NEXT:   [[Reg12]]:vgpr_32 = COPY killed $vgpr11
; CHECK-NEXT:   [[Reg11]]:vgpr_32 = COPY killed $vgpr10
; CHECK-NEXT:   [[Reg10]]:vgpr_32 = COPY killed $vgpr9
; CHECK-NEXT:   [[Reg9]]:vgpr_32 = COPY killed $vgpr8
; CHECK-NEXT:   [[Reg8]]:vgpr_32 = COPY killed $vgpr7
; CHECK-NEXT:   [[Reg7]]:vgpr_32 = COPY killed $vgpr6
; CHECK-NEXT:   [[Reg6]]:vgpr_32 = COPY killed $vgpr5
; CHECK-NEXT:   [[Reg5]]:vgpr_32 = COPY killed $vgpr4
; CHECK-NEXT:   [[Reg4]]:vgpr_32 = COPY killed $vgpr3
; CHECK-NEXT:   [[Reg3]]:vgpr_32 = COPY killed $vgpr2
; CHECK-NEXT:   [[Reg2]]:vgpr_32 = COPY killed $vgpr1
; CHECK-NEXT:   [[Reg1]]:vgpr_32 = COPY killed $vgpr0
; CHECK-NEXT:   [[Reg15:%[0-9]+]]:vreg_64 = REG_SEQUENCE killed [[Reg9]]:vgpr_32, %subreg.sub0, killed [[Reg10]]:vgpr_32, %subreg.sub1
; CHECK-NEXT:   [[Reg16:%[0-9]+]]:vreg_64 = REG_SEQUENCE killed [[Reg7]]:vgpr_32, %subreg.sub0, killed [[Reg8]]:vgpr_32, %subreg.sub1
; CHECK-NEXT:   [[Reg17:%[0-9]+]]:vreg_64 = REG_SEQUENCE killed [[Reg5]]:vgpr_32, %subreg.sub0, killed [[Reg6]]:vgpr_32, %subreg.sub1
; CHECK-NEXT:   [[Reg18:%[0-9]+]]:vreg_64 = REG_SEQUENCE killed [[Reg3]]:vgpr_32, %subreg.sub0, killed [[Reg4]]:vgpr_32, %subreg.sub1
; CHECK-NEXT:   [[Reg19:%[0-9]+]]:vreg_64 = REG_SEQUENCE killed [[Reg1]]:vgpr_32, %subreg.sub0, killed [[Reg2]]:vgpr_32, %subreg.sub1
; CHECK-NEXT:   [[Reg20:%[0-9]+]]:vgpr_32 = V_AND_B32_e64 1, killed [[Reg11]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg21:%[0-9]+]]:sreg_32 = V_CMP_NE_U32_e64 1, killed [[Reg20]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg22:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg19]]:vreg_64, 0, 0, implicit $exec :: (load (s8) from %ir.p1, addrspace 1)
; CHECK-NEXT:   [[Reg23:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg19]]:vreg_64, 1, 0, implicit $exec :: (load (s8) from %ir.p1 + 1, addrspace 1)
; CHECK-NEXT:   [[Reg24:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg23]]:vgpr_32, 8, killed [[Reg22]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg25:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg19]]:vreg_64, 2, 0, implicit $exec :: (load (s8) from %ir.p1 + 2, addrspace 1)
; CHECK-NEXT:   [[Reg26:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE killed [[Reg19]]:vreg_64, 3, 0, implicit $exec :: (load (s8) from %ir.p1 + 3, addrspace 1)
; CHECK-NEXT:   [[Reg27:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg26]]:vgpr_32, 8, killed [[Reg25]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg28:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg27]]:vgpr_32, 16, killed [[Reg24]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg29:%[0-9]+]]:vgpr_32 = V_MOV_B32_e32 10, implicit $exec
; CHECK-NEXT:   [[Reg30:%[0-9]+]]:sreg_32 = S_MOV_B32 0
; CHECK-NEXT:   [[Reg31:%[0-9]+]]:vgpr_32 = V_MOV_B32_e32 0, implicit $exec
; EMPTY:
; CHECK: bb.1.loop1.header:
; CHECK-NEXT: ; predecessors: %bb.0, %bb.5
; CHECK-NEXT:   successors: %bb.4(0x40000000), %bb.2(0x40000000); %bb.4(50.00%), %bb.2(50.00%)
; EMPTY:
; CHECK:   [[Reg32:%[0-9]+]]:sreg_32 = PHI [[Reg30]]:sreg_32, %bb.0, [[Reg33:%[0-9]+]]:sreg_32, %bb.5
; CHECK-NEXT:   [[Reg34:%[0-9]+]]:vgpr_32 = PHI [[Reg31]]:vgpr_32, %bb.0, [[Reg35:%[0-9]+]]:vgpr_32, %bb.5
; CHECK-NEXT:   [[Reg36:%[0-9]+]]:vgpr_32 = PHI [[Reg29]]:vgpr_32, %bb.0, [[Reg37:%[0-9]+]]:vgpr_32, %bb.5
; CHECK-NEXT:   [[Reg38:%[0-9]+]]:vgpr_32 = PHI [[Reg28]]:vgpr_32, %bb.0, [[Reg39:%[0-9]+]]:vgpr_32, %bb.5
; CHECK-NEXT:   [[Reg40:%[0-9]+]]:sreg_32 = S_MOV_B32 -1
; CHECK-NEXT:   [[Reg41:%[0-9]+]]:sreg_32 = SI_IF [[Reg21]]:sreg_32, %bb.2, implicit-def dead $exec, implicit-def dead $scc, implicit $exec
; CHECK-NEXT:   S_BRANCH %bb.4
; EMPTY:
; CHECK: bb.2.Flow:
; CHECK-NEXT: ; predecessors: %bb.1, %bb.4
; CHECK-NEXT:   successors: %bb.3(0x40000000), %bb.5(0x40000000); %bb.3(50.00%), %bb.5(50.00%)
; EMPTY:
; CHECK:   [[Reg42:%[0-9]+]]:sreg_32 = PHI [[Reg40]]:sreg_32, %bb.1, [[Reg43:%[0-9]+]]:sreg_32, %bb.4
; CHECK-NEXT:   [[Reg44:%[0-9]+]]:vgpr_32 = PHI undef [[Reg45:%[0-9]+]]:vgpr_32, %bb.1, [[Reg46:%[0-9]+]]:vgpr_32, %bb.4
; CHECK-NEXT:   [[Reg47:%[0-9]+]]:vgpr_32 = PHI undef [[Reg45]]:vgpr_32, %bb.1, [[Reg48:%[0-9]+]]:vgpr_32, %bb.4
; CHECK-NEXT:   [[Reg49:%[0-9]+]]:vgpr_32 = PHI undef [[Reg45]]:vgpr_32, %bb.1, [[Reg50:%[0-9]+]]:vgpr_32, %bb.4
; CHECK-NEXT:   [[Reg51:%[0-9]+]]:vgpr_32 = PHI [[Reg36]]:vgpr_32, %bb.1, undef [[Reg52:%[0-9]+]]:vgpr_32, %bb.4
; CHECK-NEXT:   [[Reg53:%[0-9]+]]:vgpr_32 = PHI [[Reg34]]:vgpr_32, %bb.1, undef [[Reg54:%[0-9]+]]:vgpr_32, %bb.4
; CHECK-NEXT:   [[Reg55:%[0-9]+]]:vgpr_32 = V_MOV_B32_e32 0, implicit $exec
; CHECK-NEXT:   [[Reg56:%[0-9]+]]:sreg_32 = SI_ELSE killed [[Reg41]]:sreg_32, %bb.5, implicit-def dead $exec, implicit-def dead $scc, implicit $exec
; CHECK-NEXT:   S_BRANCH %bb.3
; EMPTY:
; CHECK: bb.3.loop1.latch1:
; CHECK-NEXT: ; predecessors: %bb.2
; CHECK-NEXT:   successors: %bb.5(0x80000000); %bb.5(100.00%)
; EMPTY:
; CHECK:   [[Reg57:%[0-9]+]]:vgpr_32 = V_SUB_U32_e64 [[Reg28]]:vgpr_32, killed [[Reg51]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg58:%[0-9]+]]:vgpr_32 = V_ADD_U32_e64 1, killed [[Reg53]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg59:%[0-9]+]]:sreg_32 = V_CMP_GE_U32_e64 [[Reg58]]:vgpr_32, [[Reg12]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg60:%[0-9]+]]:vgpr_32 = V_MOV_B32_e32 0, implicit $exec
; CHECK-NEXT:   [[Reg61:%[0-9]+]]:sreg_32 = S_ANDN2_B32 killed [[Reg42]]:sreg_32, $exec_lo, implicit-def dead $scc
; CHECK-NEXT:   [[Reg62:%[0-9]+]]:sreg_32 = S_AND_B32 killed [[Reg59]]:sreg_32, $exec_lo, implicit-def dead $scc
; CHECK-NEXT:   [[Reg63:%[0-9]+]]:sreg_32 = S_OR_B32 killed [[Reg61]]:sreg_32, killed [[Reg62]]:sreg_32, implicit-def dead $scc
; CHECK-NEXT:   S_BRANCH %bb.5
; EMPTY:
; CHECK: bb.4.loop1.latch2:
; CHECK-NEXT: ; predecessors: %bb.1
; CHECK-NEXT:   successors: %bb.2(0x80000000); %bb.2(100.00%)
; EMPTY:
; CHECK:   [[Reg48]]:vgpr_32 = GLOBAL_LOAD_DWORD [[Reg18]]:vreg_64, 0, 0, implicit $exec :: (load (s32) from %ir.p2, addrspace 1)
; CHECK-NEXT:   [[Reg46]]:vgpr_32 = V_ADD_U32_e64 1, killed [[Reg36]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg50]]:vgpr_32 = V_ADD_U32_e64 [[Reg48]]:vgpr_32, [[Reg46]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg64:%[0-9]+]]:sreg_32 = V_CMP_GE_U32_e64 [[Reg46]]:vgpr_32, [[Reg13]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg43]]:sreg_32 = S_ORN2_B32 killed [[Reg64]]:sreg_32, $exec_lo, implicit-def dead $scc
; CHECK-NEXT:   S_BRANCH %bb.2
; EMPTY:
; CHECK: bb.5.Flow2:
; CHECK-NEXT: ; predecessors: %bb.2, %bb.3
; CHECK-NEXT:   successors: %bb.6(0x04000000), %bb.1(0x7c000000); %bb.6(3.12%), %bb.1(96.88%)
; EMPTY:
; CHECK:   [[Reg65:%[0-9]+]]:sreg_32 = PHI [[Reg42]]:sreg_32, %bb.2, [[Reg63]]:sreg_32, %bb.3
; CHECK-NEXT:   [[Reg37]]:vgpr_32 = PHI [[Reg44]]:vgpr_32, %bb.2, [[Reg60]]:vgpr_32, %bb.3
; CHECK-NEXT:   [[Reg35]]:vgpr_32 = PHI [[Reg55]]:vgpr_32, %bb.2, [[Reg58]]:vgpr_32, %bb.3
; CHECK-NEXT:   [[Reg66:%[0-9]+]]:vgpr_32 = PHI [[Reg47]]:vgpr_32, %bb.2, [[Reg28]]:vgpr_32, %bb.3
; CHECK-NEXT:   [[Reg39]]:vgpr_32 = PHI [[Reg49]]:vgpr_32, %bb.2, [[Reg57]]:vgpr_32, %bb.3
; CHECK-NEXT:   SI_END_CF killed [[Reg56]]:sreg_32, implicit-def dead $exec, implicit-def dead $scc, implicit $exec
; CHECK-NEXT:   [[Reg33]]:sreg_32 = SI_IF_BREAK killed [[Reg65]]:sreg_32, killed [[Reg32]]:sreg_32, implicit-def dead $scc
; CHECK-NEXT:   SI_LOOP [[Reg33]]:sreg_32, %bb.1, implicit-def dead $exec, implicit-def dead $scc, implicit $exec
; CHECK-NEXT:   S_BRANCH %bb.6
; EMPTY:
; CHECK: bb.6.bb1:
; CHECK-NEXT: ; predecessors: %bb.5
; CHECK-NEXT:   successors: %bb.7(0x80000000); %bb.7(100.00%)
; EMPTY:
; CHECK:   SI_END_CF killed [[Reg33]]:sreg_32, implicit-def dead $exec, implicit-def dead $scc, implicit $exec
; CHECK-NEXT:   [[Reg67:%[0-9]+]]:vgpr_32 = V_MUL_LO_U32_e64 [[Reg39]]:vgpr_32, [[Reg28]]:vgpr_32, implicit $exec
; CHECK-NEXT:   GLOBAL_STORE_DWORD killed [[Reg17]]:vreg_64, [[Reg67]]:vgpr_32, 0, 0, implicit $exec :: (store (s32) into %ir.p3, addrspace 1)
; CHECK-NEXT:   [[Reg68:%[0-9]+]]:sreg_32 = S_MOV_B32 0
; EMPTY:
; CHECK: bb.7.loop2:
; CHECK-NEXT: ; predecessors: %bb.6, %bb.7
; CHECK-NEXT:   successors: %bb.8(0x04000000), %bb.7(0x7c000000); %bb.8(3.12%), %bb.7(96.88%)
; EMPTY:
; CHECK:   [[Reg69:%[0-9]+]]:sreg_32 = PHI [[Reg68]]:sreg_32, %bb.6, [[Reg70:%[0-9]+]]:sreg_32, %bb.7
; CHECK-NEXT:   [[Reg71:%[0-9]+]]:sreg_32 = PHI [[Reg68]]:sreg_32, %bb.6, [[Reg72:%[0-9]+]]:sreg_32, %bb.7
; CHECK-NEXT:   [[Reg73:%[0-9]+]]:vgpr_32 = PHI [[Reg39]]:vgpr_32, %bb.6, [[Reg28]]:vgpr_32, %bb.7
; CHECK-NEXT:   [[Reg72]]:sreg_32 = S_ADD_I32 killed [[Reg71]]:sreg_32, 2, implicit-def dead $scc
; CHECK-NEXT:   [[Reg74:%[0-9]+]]:sreg_32 = V_CMP_GE_U32_e64 [[Reg72]]:sreg_32, [[Reg14]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg70]]:sreg_32 = SI_IF_BREAK killed [[Reg74]]:sreg_32, killed [[Reg69]]:sreg_32, implicit-def dead $scc
; CHECK-NEXT:   [[Reg75:%[0-9]+]]:vgpr_32 = COPY [[Reg72]]:sreg_32, implicit $exec
; CHECK-NEXT:   SI_LOOP [[Reg70]]:sreg_32, %bb.7, implicit-def dead $exec, implicit-def dead $scc, implicit $exec
; CHECK-NEXT:   S_BRANCH %bb.8
; EMPTY:
; CHECK: bb.8.exit:
; CHECK-NEXT: ; predecessors: %bb.7
; EMPTY:
; CHECK:   SI_END_CF killed [[Reg70]]:sreg_32, implicit-def dead $exec, implicit-def dead $scc, implicit $exec
; CHECK-NEXT:   [[Reg76:%[0-9]+]]:vgpr_32 = V_ADD_U32_e64 [[Reg73]]:vgpr_32, killed [[Reg75]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg77:%[0-9]+]]:vgpr_32 = V_ADD_U32_e64 [[Reg76]]:vgpr_32, killed [[Reg39]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg78:%[0-9]+]]:vgpr_32 = V_ADD3_U32_e64 [[Reg77]]:vgpr_32, [[Reg67]]:vgpr_32, killed [[Reg76]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg79:%[0-9]+]]:vgpr_32 = V_ADD3_U32_e64 killed [[Reg28]]:vgpr_32, killed [[Reg78]]:vgpr_32, 100, implicit $exec
; CHECK-NEXT:   [[Reg80:%[0-9]+]]:vgpr_32 = V_MUL_LO_U32_e64 killed [[Reg79]]:vgpr_32, killed [[Reg73]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg81:%[0-9]+]]:vgpr_32 = V_SUB_U32_e64 [[Reg77]]:vgpr_32, killed [[Reg66]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg82:%[0-9]+]]:vgpr_32 = V_MUL_LO_U32_e64 killed [[Reg80]]:vgpr_32, killed [[Reg81]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg83:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_DWORD [[Reg16]]:vreg_64, 0, 0, implicit $exec :: (load (s32) from %ir.p4, addrspace 1)
; CHECK-NEXT:   [[Reg84:%[0-9]+]]:vgpr_32 = V_MUL_LO_U32_e64 killed [[Reg82]]:vgpr_32, [[Reg83]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg85:%[0-9]+]]:vgpr_32 = V_SUB_U32_e64 killed [[Reg77]]:vgpr_32, killed [[Reg83]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg86:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_DWORD [[Reg15]]:vreg_64, 0, 0, implicit $exec :: (load (s32) from %ir.p5, addrspace 1)
; CHECK-NEXT:   [[Reg87:%[0-9]+]]:vgpr_32 = V_SUB_U32_e64 killed [[Reg85]]:vgpr_32, killed [[Reg86]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg88:%[0-9]+]]:vgpr_32 = V_ADD3_U32_e64 killed [[Reg84]]:vgpr_32, killed [[Reg87]]:vgpr_32, killed [[Reg38]]:vgpr_32, implicit $exec
; CHECK-NEXT:   GLOBAL_STORE_DWORD killed [[Reg16]]:vreg_64, killed [[Reg88]]:vgpr_32, 0, 0, implicit $exec :: (store (s32) into %ir.p4, addrspace 1)
; CHECK-NEXT:   GLOBAL_STORE_DWORD killed [[Reg15]]:vreg_64, killed [[Reg67]]:vgpr_32, 0, 0, implicit $exec :: (store (s32) into %ir.p5, addrspace 1)
; CHECK-NEXT:   S_ENDPGM 0
; EMPTY:
; CHECK: # End machine code for function test.
; EMPTY:
; CHECK: Next-use distance of Register [[Reg14]] = 25039.0
; CHECK-NEXT: Next-use distance of Register [[Reg13]] = 40.0
; CHECK-NEXT: Next-use distance of Register [[Reg12]] = 47.0
; CHECK-NEXT: Next-use distance of Register [[Reg11]] = 16.0
; CHECK-NEXT: Next-use distance of Register [[Reg10]] = 10.0
; CHECK-NEXT: Next-use distance of Register [[Reg9]] = 9.0
; CHECK-NEXT: Next-use distance of Register [[Reg8]] = 9.0
; CHECK-NEXT: Next-use distance of Register [[Reg7]] = 8.0
; CHECK-NEXT: Next-use distance of Register [[Reg6]] = 8.0
; CHECK-NEXT: Next-use distance of Register [[Reg5]] = 7.0
; CHECK-NEXT: Next-use distance of Register [[Reg4]] = 7.0
; CHECK-NEXT: Next-use distance of Register [[Reg3]] = 6.0
; CHECK-NEXT: Next-use distance of Register [[Reg2]] = 6.0
; CHECK-NEXT: Next-use distance of Register [[Reg1]] = 5.0
; CHECK-NEXT: Next-use distance of Register [[Reg15]] = 34032.0
; CHECK-NEXT: Next-use distance of Register [[Reg16]] = 34028.0
; CHECK-NEXT: Next-use distance of Register [[Reg17]] = 25017.0
; CHECK-NEXT: Next-use distance of Register [[Reg18]] = 21.0
; CHECK-NEXT: Next-use distance of Register [[Reg19]] = 3.0
; CHECK-NEXT: Next-use distance of Register [[Reg20]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg21]] = 16.0
; CHECK-NEXT: Next-use distance of Register [[Reg22]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg23]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg24]] = 4.0
; CHECK-NEXT: Next-use distance of Register [[Reg25]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg26]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg27]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg28]] = 7.0
; CHECK-NEXT: Next-use distance of Register [[Reg29]] = 5.0
; CHECK-NEXT: Next-use distance of Register [[Reg30]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg31]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg32]] = 22.0
; CHECK-NEXT: Next-use distance of Register [[Reg34]] = 11.0
; CHECK-NEXT: Next-use distance of Register [[Reg36]] = 6.0
; CHECK-NEXT: Next-use distance of Register [[Reg38]] = 34018.0
; CHECK-NEXT: Next-use distance of Register [[Reg40]] = 3.0
; CHECK-NEXT: Next-use distance of Register [[Reg41]] = 9.0
; CHECK-NEXT: Next-use distance of Register [[Reg42]] = 9.0
; CHECK-NEXT: Next-use distance of Register [[Reg44]] = 9.0
; CHECK-NEXT: Next-use distance of Register [[Reg47]] = 10.0
; CHECK-NEXT: Next-use distance of Register [[Reg49]] = 10.0
; CHECK-NEXT: Next-use distance of Register [[Reg51]] = 5.0
; CHECK-NEXT: Next-use distance of Register [[Reg53]] = 5.0
; CHECK-NEXT: Next-use distance of Register [[Reg55]] = 5.0
; CHECK-NEXT: Next-use distance of Register [[Reg56]] = 7.0
; CHECK-NEXT: Next-use distance of Register [[Reg57]] = 12.0
; CHECK-NEXT: Next-use distance of Register [[Reg58]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg59]] = 3.0
; CHECK-NEXT: Next-use distance of Register [[Reg60]] = 6.0
; CHECK-NEXT: Next-use distance of Register [[Reg61]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg62]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg63]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg48]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg46]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg50]] = 7.0
; CHECK-NEXT: Next-use distance of Register [[Reg64]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg43]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg65]] = 6.0
; CHECK-NEXT: Next-use distance of Register [[Reg37]] = 10.0
; CHECK-NEXT: Next-use distance of Register [[Reg35]] = 8.0
; CHECK-NEXT: Next-use distance of Register [[Reg66]] = 34011.0
; CHECK-NEXT: Next-use distance of Register [[Reg39]] = 8.0
; CHECK-NEXT: Next-use distance of Register [[Reg33]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg67]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg68]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg69]] = 5.0
; CHECK-NEXT: Next-use distance of Register [[Reg71]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg73]] = 9002.0
; CHECK-NEXT: Next-use distance of Register [[Reg72]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg74]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg70]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg75]] = 9002.0
; CHECK-NEXT: Next-use distance of Register [[Reg76]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg77]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg78]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg79]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg80]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg81]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg82]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg83]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg84]] = 4.0
; CHECK-NEXT: Next-use distance of Register [[Reg85]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg86]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg87]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg88]] = 1.0
entry:
;              entry
;                |
;  +-------->loop1.header<--------+
;  |        /           \         |
;  +--loop1.latch1  loop1.latch2--+
;                \  /
;                 bb1
;                 |
;                 +<-----+
;               loop2    |
;                 +------+
;                 |
;                exit
   %ld1 = load i32, ptr addrspace(1) %p1, align 1
   %add1 = add i32 %ld1, 100
   br label %loop1.header

loop1.header:
   %phi.inc1 = phi i32 [ 0, %entry ], [ %inc1, %loop1.latch1 ], [ 0, %loop1.latch2 ]
   %phi.inc2 = phi i32 [ 10, %entry ], [ 0, %loop1.latch1 ], [ %inc2, %loop1.latch2 ]
   %phi1 = phi i32 [ %ld1, %entry ], [ %sub, %loop1.latch1 ], [ %add2, %loop1.latch2 ]
   br i1 %cond, label %loop1.latch1, label %loop1.latch2

loop1.latch1:
   %sub = sub i32 %ld1, %phi.inc2
   %inc1 = add i32 %phi.inc1, 1
   %cond1 = icmp ult i32 %inc1, %TC1
   br i1 %cond1, label %loop1.header, label %bb1

loop1.latch2:
   %ld2 = load i32, ptr addrspace(1) %p2, align 4
   %inc2 = add i32 %phi.inc2, 1
   %add2 = add i32 %ld2, %inc2
   %cond2 = icmp ult i32 %inc2, %TC2
   br i1 %cond2, label %loop1.header, label %bb1

bb1:
   %phi2 = phi i32 [ %sub, %loop1.latch1 ], [ %add2, %loop1.latch2 ]
   %ld3 = phi i32 [ %ld1, %loop1.latch1 ], [ %ld2, %loop1.latch2 ]
   %mul = mul i32 %phi2, %ld1
   store i32 %mul, ptr addrspace(1) %p3
   br label %loop2

loop2:
   %phi.inc3 = phi i32 [ 0, %bb1 ], [ %inc3, %loop2 ]
   %phi3 = phi i32 [ %phi2, %bb1 ], [ %ld1, %loop2 ]
   %inc3 = add i32 %phi.inc3, 2
   %add3 = add i32 %phi3, %inc3
   %cond3 = icmp ult i32 %inc3, %TC3
   br i1 %cond3, label %loop2, label %exit

exit:
   %add4 = add i32 %add3, %phi2
   %add5 = add i32 %add4, %mul
   %add6 = add i32 %add5, %add3
   %add7 = add i32 %add6, %add1
   %mul2 = mul i32 %add7, %phi3
   %sub1 = sub i32 %add4, %ld3
   %mul3 = mul i32 %mul2, %sub1
   %ld4 = load i32, ptr addrspace(1) %p4, align 4
   %mul4 = mul i32 %mul3, %ld4
   %sub2 = sub i32 %add4, %ld4
   %ld5 = load i32, ptr addrspace(1) %p5, align 4
   %sub3 = sub i32 %sub2, %ld5
   %add8 = add i32 %mul4, %sub3
   %add9 = add i32 %add8, %phi1
   store i32 %add9, ptr addrspace(1) %p4, align 4
   store i32 %mul, ptr addrspace(1) %p5, align 4
   ret void
}
