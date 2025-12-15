; RUN: llc -mtriple=amdgcn -mcpu=gfx1200 -enable-next-use-analysis=true -verify-machineinstrs -dump-next-use-distance < %s 2>&1 | FileCheck %s

;
;       bb.0.entry
;        /    |
;   bb.1.bb1  |
;        \    |
;       bb.2.bb2
;        /    |
;   bb.5.bb4  |
;        \    |
;      bb.3.Flow3
;        /    |
;   bb.4.bb3  |
;        \    |
;      bb.6.bb5
;        /    |
;   bb.12.bb7 |
;        \    |
;      bb.7.Flow2
;        /    |
;   bb.8.bb6  |
;    /     |  |
;bb.11.bb9 |  |
;    \     |  |
;  bb.9.Flow  |
;    /     |  |
;bb.10.bb8 |  |
;    \     |  |
; bb.13.Flow1 |
;       \     |
;      bb.14.exit
;
define amdgpu_ps i32 @test3(ptr addrspace(1) %p1, ptr addrspace(3) %p2, i1 %cond1, i1 %cond2) {
; CHECK-LABEL: # Machine code for function test3: IsSSA, TracksLiveness
; CHECK-NEXT: Function Live Ins: $vgpr0 in [[Reg1:%[0-9]+]], $vgpr1 in [[Reg2:%[0-9]+]], $vgpr2 in [[Reg3:%[0-9]+]], $vgpr3 in [[Reg4:%[0-9]+]], $vgpr4 in [[Reg5:%[0-9]+]]
; EMPTY:
; CHECK: bb.0.entry:
; CHECK-NEXT:   successors: %bb.1(0x40000000), %bb.2(0x40000000); %bb.1(50.00%), %bb.2(50.00%)
; CHECK-NEXT:   liveins: $vgpr0, $vgpr1, $vgpr2, $vgpr3, $vgpr4
; CHECK-NEXT:   [[Reg5]]:vgpr_32 = COPY killed $vgpr4
; CHECK-NEXT:   [[Reg4]]:vgpr_32 = COPY killed $vgpr3
; CHECK-NEXT:   [[Reg3]]:vgpr_32 = COPY killed $vgpr2
; CHECK-NEXT:   [[Reg2]]:vgpr_32 = COPY killed $vgpr1
; CHECK-NEXT:   [[Reg1]]:vgpr_32 = COPY killed $vgpr0
; CHECK-NEXT:   [[Reg6:%[0-9]+]]:vreg_64 = REG_SEQUENCE killed [[Reg1]]:vgpr_32, %subreg.sub0, killed [[Reg2]]:vgpr_32, %subreg.sub1
; CHECK-NEXT:   [[Reg7:%[0-9]+]]:vgpr_32 = V_AND_B32_e64 1, killed [[Reg4]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg8:%[0-9]+]]:sreg_32 = V_CMP_EQ_U32_e64 1, killed [[Reg7]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg9:%[0-9]+]]:vgpr_32 = V_AND_B32_e64 1, killed [[Reg5]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg10:%[0-9]+]]:sreg_32 = V_CMP_EQ_U32_e64 1, killed [[Reg9]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg11:%[0-9]+]]:sreg_32 = S_XOR_B32 [[Reg10]]:sreg_32, -1, implicit-def dead $scc
; CHECK-NEXT:   [[Reg12:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg6]]:vreg_64, 0, 0, implicit $exec :: (load (s8) from %ir.p1, addrspace 1)
; CHECK-NEXT:   [[Reg13:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg6]]:vreg_64, 1, 0, implicit $exec :: (load (s8) from %ir.p1 + 1, addrspace 1)
; CHECK-NEXT:   [[Reg14:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg13]]:vgpr_32, 8, killed [[Reg12]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg15:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg6]]:vreg_64, 2, 0, implicit $exec :: (load (s8) from %ir.p1 + 2, addrspace 1)
; CHECK-NEXT:   [[Reg16:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg6]]:vreg_64, 3, 0, implicit $exec :: (load (s8) from %ir.p1 + 3, addrspace 1)
; CHECK-NEXT:   [[Reg17:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg16]]:vgpr_32, 8, killed [[Reg15]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg18:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg17]]:vgpr_32, 16, killed [[Reg14]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg19:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg6]]:vreg_64, 12, 0, implicit $exec :: (load (s8) from %ir.gep1, addrspace 1)
; CHECK-NEXT:   [[Reg20:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg6]]:vreg_64, 13, 0, implicit $exec :: (load (s8) from %ir.gep1 + 1, addrspace 1)
; CHECK-NEXT:   [[Reg21:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg6]]:vreg_64, 14, 0, implicit $exec :: (load (s8) from %ir.gep1 + 2, addrspace 1)
; CHECK-NEXT:   [[Reg22:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE killed [[Reg6]]:vreg_64, 15, 0, implicit $exec :: (load (s8) from %ir.gep1 + 3, addrspace 1)
; CHECK-NEXT:   [[Reg23:%[0-9]+]]:vgpr_32 = V_ADD_U32_e64 100, [[Reg18]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg24:%[0-9]+]]:sreg_32 = SI_IF [[Reg8]]:sreg_32, %bb.2, implicit-def dead $exec, implicit-def dead $scc, implicit $exec
; CHECK-NEXT:   S_BRANCH %bb.1
; EMPTY:
; CHECK: bb.1.bb1:
; CHECK-NEXT: ; predecessors: %bb.0
; CHECK-NEXT:   successors: %bb.2(0x80000000); %bb.2(100.00%)
; EMPTY:
; CHECK:   [[Reg25:%[0-9]+]]:vgpr_32 = V_MOV_B32_e32 100, implicit $exec
; EMPTY:
; CHECK: bb.2.bb2:
; CHECK-NEXT: ; predecessors: %bb.0, %bb.1
; CHECK-NEXT:   successors: %bb.5(0x40000000), %bb.3(0x40000000); %bb.5(50.00%), %bb.3(50.00%)
; EMPTY:
; CHECK:   [[Reg26:%[0-9]+]]:vgpr_32 = PHI [[Reg23]]:vgpr_32, %bb.0, [[Reg25]]:vgpr_32, %bb.1
; CHECK-NEXT:   SI_END_CF killed [[Reg24]]:sreg_32, implicit-def dead $exec, implicit-def dead $scc, implicit $exec
; CHECK-NEXT:   [[Reg27:%[0-9]+]]:vgpr_32 = DS_READ_U8_gfx9 [[Reg3]]:vgpr_32, 0, 0, implicit $exec :: (load (s8) from %ir.p2, addrspace 3)
; CHECK-NEXT:   [[Reg28:%[0-9]+]]:vgpr_32 = DS_READ_U8_gfx9 [[Reg3]]:vgpr_32, 1, 0, implicit $exec :: (load (s8) from %ir.p2 + 1, addrspace 3)
; CHECK-NEXT:   [[Reg29:%[0-9]+]]:vgpr_32 = DS_READ_U8_gfx9 [[Reg3]]:vgpr_32, 2, 0, implicit $exec :: (load (s8) from %ir.p2 + 2, addrspace 3)
; CHECK-NEXT:   [[Reg30:%[0-9]+]]:vgpr_32 = DS_READ_U8_gfx9 [[Reg3]]:vgpr_32, 3, 0, implicit $exec :: (load (s8) from %ir.p2 + 3, addrspace 3)
; CHECK-NEXT:   [[Reg31:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg28]]:vgpr_32, 8, killed [[Reg27]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg32:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg30]]:vgpr_32, 8, killed [[Reg29]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg33:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg32]]:vgpr_32, 16, killed [[Reg31]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg34:%[0-9]+]]:sreg_32 = SI_IF killed [[Reg11]]:sreg_32, %bb.3, implicit-def dead $exec, implicit-def dead $scc, implicit $exec
; CHECK-NEXT:   S_BRANCH %bb.5
; EMPTY:
; CHECK: bb.3.Flow3:
; CHECK-NEXT: ; predecessors: %bb.2, %bb.5
; CHECK-NEXT:   successors: %bb.4(0x40000000), %bb.6(0x40000000); %bb.4(50.00%), %bb.6(50.00%)
; EMPTY:
; CHECK:   [[Reg35:%[0-9]+]]:vgpr_32 = PHI undef [[Reg36:%[0-9]+]]:vgpr_32, %bb.2, [[Reg37:%[0-9]+]]:vgpr_32, %bb.5
; CHECK-NEXT:   [[Reg38:%[0-9]+]]:vgpr_32 = PHI [[Reg26]]:vgpr_32, %bb.2, undef [[Reg39:%[0-9]+]]:vgpr_32, %bb.5
; CHECK-NEXT:   [[Reg40:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg20]]:vgpr_32, 8, killed [[Reg19]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg41:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg22]]:vgpr_32, 8, killed [[Reg21]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg42:%[0-9]+]]:sreg_32 = SI_ELSE killed [[Reg34]]:sreg_32, %bb.6, implicit-def dead $exec, implicit-def dead $scc, implicit $exec
; CHECK-NEXT:   S_BRANCH %bb.4
; EMPTY:
; CHECK: bb.4.bb3:
; CHECK-NEXT: ; predecessors: %bb.3
; CHECK-NEXT:   successors: %bb.6(0x80000000); %bb.6(100.00%)
; EMPTY:
; CHECK:   [[Reg43:%[0-9]+]]:vreg_64, $sgpr_null = V_MAD_U64_U32_e64 [[Reg18]]:vgpr_32, killed [[Reg38]]:vgpr_32, 1000, 0, implicit $exec
; CHECK-NEXT:   [[Reg44:%[0-9]+]]:vgpr_32 = COPY killed [[Reg43]].sub0:vreg_64
; CHECK-NEXT:   S_BRANCH %bb.6
; EMPTY:
; CHECK: bb.5.bb4:
; CHECK-NEXT: ; predecessors: %bb.2
; CHECK-NEXT:   successors: %bb.3(0x80000000); %bb.3(100.00%)
; EMPTY:
; CHECK:   [[Reg37]]:vgpr_32 = V_ADD_U32_e64 killed [[Reg23]]:vgpr_32, [[Reg33]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   S_BRANCH %bb.3
; EMPTY:
; CHECK: bb.6.bb5:
; CHECK-NEXT: ; predecessors: %bb.3, %bb.4
; CHECK-NEXT:   successors: %bb.12(0x40000000), %bb.7(0x40000000); %bb.12(50.00%), %bb.7(50.00%)
; EMPTY:
; CHECK:   [[Reg45:%[0-9]+]]:vgpr_32 = PHI [[Reg35]]:vgpr_32, %bb.3, [[Reg44]]:vgpr_32, %bb.4
; CHECK-NEXT:   SI_END_CF killed [[Reg42]]:sreg_32, implicit-def dead $exec, implicit-def dead $scc, implicit $exec
; CHECK-NEXT:   [[Reg46:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg41]]:vgpr_32, 16, killed [[Reg40]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg47:%[0-9]+]]:vgpr_32 = DS_READ_B32_gfx9 killed [[Reg3]]:vgpr_32, 12, 0, implicit $exec :: (load (s32) from %ir.gep2, align 8, addrspace 3)
; CHECK-NEXT:   [[Reg48:%[0-9]+]]:vgpr_32 = V_ADD_U32_e64 [[Reg47]]:vgpr_32, killed [[Reg45]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg49:%[0-9]+]]:sreg_32 = S_XOR_B32 [[Reg8]]:sreg_32, [[Reg10]]:sreg_32, implicit-def dead $scc
; CHECK-NEXT:   [[Reg50:%[0-9]+]]:sreg_32 = S_XOR_B32 killed [[Reg49]]:sreg_32, -1, implicit-def dead $scc
; CHECK-NEXT:   [[Reg51:%[0-9]+]]:sreg_32 = SI_IF killed [[Reg50]]:sreg_32, %bb.7, implicit-def dead $exec, implicit-def dead $scc, implicit $exec
; CHECK-NEXT:   S_BRANCH %bb.12
; EMPTY:
; CHECK: bb.7.Flow2:
; CHECK-NEXT: ; predecessors: %bb.6, %bb.12
; CHECK-NEXT:   successors: %bb.8(0x40000000), %bb.14(0x40000000); %bb.8(50.00%), %bb.14(50.00%)
; EMPTY:
; CHECK:   [[Reg52:%[0-9]+]]:vgpr_32 = PHI undef [[Reg53:%[0-9]+]]:vgpr_32, %bb.6, [[Reg54:%[0-9]+]]:vgpr_32, %bb.12
; CHECK-NEXT:   [[Reg55:%[0-9]+]]:vgpr_32 = PHI [[Reg46]]:vgpr_32, %bb.6, undef [[Reg56:%[0-9]+]]:vgpr_32, %bb.12
; CHECK-NEXT:   [[Reg57:%[0-9]+]]:vgpr_32 = PHI [[Reg48]]:vgpr_32, %bb.6, undef [[Reg58:%[0-9]+]]:vgpr_32, %bb.12
; CHECK-NEXT:   [[Reg59:%[0-9]+]]:sreg_32 = SI_ELSE killed [[Reg51]]:sreg_32, %bb.14, implicit-def dead $exec, implicit-def dead $scc, implicit $exec
; CHECK-NEXT:   S_BRANCH %bb.8
; EMPTY:
; CHECK: bb.8.bb6:
; CHECK-NEXT: ; predecessors: %bb.7
; CHECK-NEXT:   successors: %bb.11(0x40000000), %bb.9(0x40000000); %bb.11(50.00%), %bb.9(50.00%)
; EMPTY:
; CHECK:   [[Reg60:%[0-9]+]]:sreg_32 = S_AND_B32 killed [[Reg8]]:sreg_32, killed [[Reg10]]:sreg_32, implicit-def dead $scc
; CHECK-NEXT:   [[Reg61:%[0-9]+]]:sreg_32 = S_XOR_B32 killed [[Reg60]]:sreg_32, -1, implicit-def dead $scc
; CHECK-NEXT:   [[Reg62:%[0-9]+]]:sreg_32 = SI_IF killed [[Reg61]]:sreg_32, %bb.9, implicit-def dead $exec, implicit-def dead $scc, implicit $exec
; CHECK-NEXT:   S_BRANCH %bb.11
; EMPTY:
; CHECK: bb.9.Flow:
; CHECK-NEXT: ; predecessors: %bb.8, %bb.11
; CHECK-NEXT:   successors: %bb.10(0x40000000), %bb.13(0x40000000); %bb.10(50.00%), %bb.13(50.00%)
; EMPTY:
; CHECK:   [[Reg63:%[0-9]+]]:vgpr_32 = PHI undef [[Reg64:%[0-9]+]]:vgpr_32, %bb.8, [[Reg65:%[0-9]+]]:vgpr_32, %bb.11
; CHECK-NEXT:   [[Reg66:%[0-9]+]]:vgpr_32 = PHI [[Reg55]]:vgpr_32, %bb.8, undef [[Reg67:%[0-9]+]]:vgpr_32, %bb.11
; CHECK-NEXT:   [[Reg68:%[0-9]+]]:vgpr_32 = PHI [[Reg57]]:vgpr_32, %bb.8, undef [[Reg69:%[0-9]+]]:vgpr_32, %bb.11
; CHECK-NEXT:   [[Reg70:%[0-9]+]]:sreg_32 = SI_ELSE killed [[Reg62]]:sreg_32, %bb.13, implicit-def dead $exec, implicit-def dead $scc, implicit $exec
; CHECK-NEXT:   S_BRANCH %bb.10
; EMPTY:
; CHECK: bb.10.bb8:
; CHECK-NEXT: ; predecessors: %bb.9
; CHECK-NEXT:   successors: %bb.13(0x80000000); %bb.13(100.00%)
; EMPTY:
; CHECK:   [[Reg71:%[0-9]+]]:vgpr_32 = V_ADD_U32_e64 killed [[Reg66]]:vgpr_32, killed [[Reg68]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   S_BRANCH %bb.13
; EMPTY:
; CHECK: bb.11.bb9:
; CHECK-NEXT: ; predecessors: %bb.8
; CHECK-NEXT:   successors: %bb.9(0x80000000); %bb.9(100.00%)
; EMPTY:
; CHECK:   [[Reg65]]:vgpr_32 = V_MUL_LO_U32_e64 killed [[Reg55]]:vgpr_32, killed [[Reg57]]:vgpr_32, implicit $exec
; CHECK-NEXT:   S_BRANCH %bb.9
; EMPTY:
; CHECK: bb.12.bb7:
; CHECK-NEXT: ; predecessors: %bb.6
; CHECK-NEXT:   successors: %bb.7(0x80000000); %bb.7(100.00%)
; EMPTY:
; CHECK:   [[Reg72:%[0-9]+]]:vgpr_32 = V_SUB_U32_e64 killed [[Reg47]]:vgpr_32, killed [[Reg48]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg73:%[0-9]+]]:vgpr_32 = V_MUL_LO_U32_e64 killed [[Reg72]]:vgpr_32, [[Reg33]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg74:%[0-9]+]]:vgpr_32 = V_CVT_F32_U32_e64 [[Reg18]]:vgpr_32, 0, 0, implicit $mode, implicit $exec
; CHECK-NEXT:   [[Reg75:%[0-9]+]]:vgpr_32 = nofpexcept V_RCP_IFLAG_F32_e64 0, killed [[Reg74]]:vgpr_32, 0, 0, implicit $mode, implicit $exec
; CHECK-NEXT:   [[Reg76:%[0-9]+]]:vgpr_32 = nnan ninf nsz arcp contract afn reassoc nofpexcept V_MUL_F32_e64 0, 1333788670, 0, killed [[Reg75]]:vgpr_32, 0, 0, implicit $mode, implicit $exec
; CHECK-NEXT:   [[Reg77:%[0-9]+]]:vgpr_32 = nofpexcept V_CVT_U32_F32_e64 0, killed [[Reg76]]:vgpr_32, 0, 0, implicit $mode, implicit $exec
; CHECK-NEXT:   [[Reg78:%[0-9]+]]:vgpr_32 = V_SUB_U32_e64 0, [[Reg18]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg79:%[0-9]+]]:vgpr_32 = V_MUL_LO_U32_e64 killed [[Reg78]]:vgpr_32, [[Reg77]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg80:%[0-9]+]]:vgpr_32 = V_MUL_HI_U32_e64 [[Reg77]]:vgpr_32, killed [[Reg79]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg81:%[0-9]+]]:vgpr_32 = V_ADD_U32_e64 killed [[Reg77]]:vgpr_32, killed [[Reg80]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg82:%[0-9]+]]:vgpr_32 = V_MUL_HI_U32_e64 [[Reg73]]:vgpr_32, killed [[Reg81]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg83:%[0-9]+]]:vgpr_32 = V_MUL_LO_U32_e64 [[Reg82]]:vgpr_32, [[Reg18]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg84:%[0-9]+]]:vgpr_32 = V_SUB_U32_e64 killed [[Reg73]]:vgpr_32, killed [[Reg83]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg85:%[0-9]+]]:sreg_32_xm0_xexec = V_CMP_GE_U32_e64 [[Reg84]]:vgpr_32, [[Reg18]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg86:%[0-9]+]]:vgpr_32 = V_ADD_U32_e64 1, [[Reg82]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg87:%[0-9]+]]:vgpr_32 = V_CNDMASK_B32_e64 0, killed [[Reg82]]:vgpr_32, 0, killed [[Reg86]]:vgpr_32, [[Reg85]]:sreg_32_xm0_xexec, implicit $exec
; CHECK-NEXT:   [[Reg88:%[0-9]+]]:vgpr_32 = V_SUB_U32_e64 [[Reg84]]:vgpr_32, [[Reg18]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg89:%[0-9]+]]:vgpr_32 = V_CNDMASK_B32_e64 0, killed [[Reg84]]:vgpr_32, 0, killed [[Reg88]]:vgpr_32, killed [[Reg85]]:sreg_32_xm0_xexec, implicit $exec
; CHECK-NEXT:   [[Reg90:%[0-9]+]]:sreg_32_xm0_xexec = V_CMP_GE_U32_e64 killed [[Reg89]]:vgpr_32, killed [[Reg18]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg91:%[0-9]+]]:vgpr_32 = V_ADD_U32_e64 1, [[Reg87]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg92:%[0-9]+]]:vgpr_32 = V_CNDMASK_B32_e64 0, killed [[Reg87]]:vgpr_32, 0, killed [[Reg91]]:vgpr_32, killed [[Reg90]]:sreg_32_xm0_xexec, implicit $exec
; CHECK-NEXT:   [[Reg54]]:vgpr_32 = V_ADD_U32_e64 killed [[Reg92]]:vgpr_32, killed [[Reg46]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   S_BRANCH %bb.7
; EMPTY:
; CHECK: bb.13.Flow1:
; CHECK-NEXT: ; predecessors: %bb.9, %bb.10
; CHECK-NEXT:   successors: %bb.14(0x80000000); %bb.14(100.00%)
; EMPTY:
; CHECK:   [[Reg93:%[0-9]+]]:vgpr_32 = PHI [[Reg63]]:vgpr_32, %bb.9, [[Reg71]]:vgpr_32, %bb.10
; CHECK-NEXT:   SI_END_CF killed [[Reg70]]:sreg_32, implicit-def dead $exec, implicit-def dead $scc, implicit $exec
; EMPTY:
; CHECK: bb.14.exit:
; CHECK-NEXT: ; predecessors: %bb.7, %bb.13
; EMPTY:
; CHECK:   [[Reg94:%[0-9]+]]:vgpr_32 = PHI [[Reg52]]:vgpr_32, %bb.7, [[Reg93]]:vgpr_32, %bb.13
; CHECK-NEXT:   SI_END_CF killed [[Reg59]]:sreg_32, implicit-def dead $exec, implicit-def dead $scc, implicit $exec
; CHECK-NEXT:   [[Reg95:%[0-9]+]]:vgpr_32 = V_ADD3_U32_e64 killed [[Reg33]]:vgpr_32, killed [[Reg94]]:vgpr_32, 100, implicit $exec
; CHECK-NEXT:   [[Reg96:%[0-9]+]]:sreg_32_xm0 = V_READFIRSTLANE_B32 killed [[Reg95]]:vgpr_32, implicit $exec
; CHECK-NEXT:   $sgpr0 = COPY killed [[Reg96]]:sreg_32_xm0
; CHECK-NEXT:   SI_RETURN_TO_EPILOG killed $sgpr0
; EMPTY:
; CHECK: # End machine code for function test3.
; EMPTY:
; CHECK: Next-use distance of Register [[Reg5]] = 8.0
; CHECK-NEXT: Next-use distance of Register [[Reg4]] = 5.0
; CHECK-NEXT: Next-use distance of Register [[Reg3]] = 25.0
; CHECK-NEXT: Next-use distance of Register [[Reg2]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg1]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg6]] = 6.0
; CHECK-NEXT: Next-use distance of Register [[Reg7]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg8]] = 16.0
; CHECK-NEXT: Next-use distance of Register [[Reg9]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg10]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg11]] = 24.0
; CHECK-NEXT: Next-use distance of Register [[Reg12]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg13]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg14]] = 4.0
; CHECK-NEXT: Next-use distance of Register [[Reg15]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg16]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg17]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg18]] = 5.0
; CHECK-NEXT: Next-use distance of Register [[Reg19]] = 20.0
; CHECK-NEXT: Next-use distance of Register [[Reg20]] = 19.0
; CHECK-NEXT: Next-use distance of Register [[Reg21]] = 19.0
; CHECK-NEXT: Next-use distance of Register [[Reg22]] = 18.0
; CHECK-NEXT: Next-use distance of Register [[Reg23]] = 3.0
; CHECK-NEXT: Next-use distance of Register [[Reg24]] = 3.0
; CHECK-NEXT: Next-use distance of Register [[Reg25]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg26]] = 12.0
; CHECK-NEXT: Next-use distance of Register [[Reg27]] = 4.0
; CHECK-NEXT: Next-use distance of Register [[Reg28]] = 3.0
; CHECK-NEXT: Next-use distance of Register [[Reg29]] = 3.0
; CHECK-NEXT: Next-use distance of Register [[Reg30]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg31]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg32]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg33]] = 3.0
; CHECK-NEXT: Next-use distance of Register [[Reg34]] = 6.0
; CHECK-NEXT: Next-use distance of Register [[Reg35]] = 6.0
; CHECK-NEXT: Next-use distance of Register [[Reg38]] = 5.0
; CHECK-NEXT: Next-use distance of Register [[Reg40]] = 6.0
; CHECK-NEXT: Next-use distance of Register [[Reg41]] = 5.0
; CHECK-NEXT: Next-use distance of Register [[Reg42]] = 3.0
; CHECK-NEXT: Next-use distance of Register [[Reg43]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg44]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg37]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg45]] = 4.0
; CHECK-NEXT: Next-use distance of Register [[Reg46]] = 8.0
; CHECK-NEXT: Next-use distance of Register [[Reg47]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg48]] = 5.0
; CHECK-NEXT: Next-use distance of Register [[Reg49]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg50]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg51]] = 5.0
; CHECK-NEXT: Next-use distance of Register [[Reg52]] = 5.0
; CHECK-NEXT: Next-use distance of Register [[Reg55]] = 8.0
; CHECK-NEXT: Next-use distance of Register [[Reg57]] = 7.0
; CHECK-NEXT: Next-use distance of Register [[Reg59]] = 3.0
; CHECK-NEXT: Next-use distance of Register [[Reg60]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg61]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg62]] = 5.0
; CHECK-NEXT: Next-use distance of Register [[Reg63]] = 5.0
; CHECK-NEXT: Next-use distance of Register [[Reg66]] = 4.0
; CHECK-NEXT: Next-use distance of Register [[Reg68]] = 3.0
; CHECK-NEXT: Next-use distance of Register [[Reg70]] = 3.0
; CHECK-NEXT: Next-use distance of Register [[Reg71]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg65]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg72]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg73]] = 9.0
; CHECK-NEXT: Next-use distance of Register [[Reg74]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg75]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg76]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg77]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg78]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg79]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg80]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg81]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg82]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg83]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg84]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg85]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg86]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg87]] = 4.0
; CHECK-NEXT: Next-use distance of Register [[Reg88]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg89]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg90]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg91]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg92]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg54]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg93]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg94]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg95]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg96]] = 1.0
entry:
;       entry
;        / \
;      bb1  |
;        \  |
;         BB2
;        /   \
;      BB3   BB4
;         \ /
;         BB5
;        /   \
;      BB6   BB7
;      / \    |
;    BB8 BB9  |
;       \  |  |
;        \ | /
;         exit
   %ld1 = load i32, ptr addrspace(1) %p1, align 1
   %gep1 = getelementptr inbounds i32, ptr addrspace(1) %p1, i64 3
   %ld2 = load i32, ptr addrspace(1) %gep1, align 1
   %add1 = add i32 %ld1, 100
   br i1 %cond1, label %bb1, label %bb2

bb1:
  br label %bb2

bb2:
  %phi0 = phi i32 [ 100, %bb1 ], [ %add1, %entry ]
  %ld3 = load i32, ptr addrspace(3) %p2, align 1
  %add2 = add i32 %ld3, 100
  br i1 %cond2, label %bb3, label %bb4

bb3:
  %mul1 = mul i32 %ld1, %phi0
  %add3 = add i32 %mul1, 1000
  br label %bb5

bb4:
  %add4 = add i32 %add2, %ld1
  br label %bb5

bb5:
  %phi1 = phi i32 [ %add3, %bb3 ], [ %add4, %bb4]
  %gep2 = getelementptr inbounds i32, ptr addrspace(3) %p2, i64 3
  %ld4 = load i32, ptr addrspace(3) %gep2, align 8
  %add5 = add i32 %ld4, %phi1
  %xor = xor i1 %cond1, %cond2
  br i1 %xor, label %bb6, label %bb7

bb6:
  %and = and i1 %cond1, %cond2
  br i1 %and, label %bb8, label %bb9

bb8:
  %add6 = add i32 %ld2, %add5
  br label %exit

bb9:
  %mul2 = mul i32 %ld2, %add5
  br label %exit

bb7:
  %sub1 = sub i32 %ld4, %add5
  %mul3 = mul i32 %sub1, %ld3
  %div = udiv i32 %mul3, %ld1
  %add7 = add i32 %div, %ld2
  br label %exit

exit:
  %phi2 = phi i32 [ %add6, %bb8 ], [ %mul2, %bb9], [ %add7, %bb7 ]
  %add8 = add i32 %add2, %phi2
  ret i32 %add8
}
