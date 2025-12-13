; RUN: llc -mtriple=amdgcn -mcpu=gfx1200 -enable-next-use-analysis=true -verify-machineinstrs -dump-next-use-distance < %s 2>&1 | FileCheck %s

;           bb.0.entry
;            /    |
;       bb.3.bb3  |
;            \    |
;           bb.1.Flow12
;            /    |
;       bb.2.bb2  |
;            \    |
;           bb.4.bb4
;              |
;      bb.5.loop1.header<-------+
;              |                |
;      bb.6.loop2.header<-----+ |
;              |              | |
;      bb.7.loop3.header<---+ | |
;            /   |          | | |
;      bb.8.bb5  |          | | |
;            \   |          | | |
;      bb.9.loop3.latch-----+ | |
;              |              | |
;      bb.10.loop2.latch------+ |
;              |                |
;    bb.11.loop4.preheader      |
;              |                |
;         bb.12.loop4<----+     |
;              +----------+     |
;              |                |
;              |                |
;      bb.13.loop1.latch--------+
;              |
;          bb.14.bb6
;           /      |
;    bb.15.bb7     |
;           \      |
;    bb.16.loop5.preheader
;              |
;     +-->bb.17.loop5
;     +--------+
;              |
;          bb.18.exit
define amdgpu_ps i32 @test12 (ptr addrspace(1) %p1, ptr addrspace(1) %p2, ptr addrspace(1) %p3, ptr addrspace(1) %p4, ptr addrspace(1) %p5, ptr addrspace(1) %p6, ptr addrspace(1) %p7, ptr addrspace(1) %p8, ptr addrspace(1) %p9, ptr addrspace(1) %p10, ptr addrspace(1) %p11, i32 %TC1, i32 %TC2, i32 %TC3, i32 %TC4, i32 %TC5, i32 %Val1, i32 %Val2, i1 %cond1) {
; CHECK-LABEL: # Machine code for function test12: IsSSA, TracksLiveness
; CHECK-NEXT: Function Live Ins: $vgpr0 in [[Reg1:%[0-9]+]], $vgpr1 in [[Reg2:%[0-9]+]], $vgpr2 in [[Reg3:%[0-9]+]], $vgpr3 in [[Reg4:%[0-9]+]], $vgpr4 in [[Reg5:%[0-9]+]], $vgpr5 in [[Reg6:%[0-9]+]], $vgpr6 in [[Reg7:%[0-9]+]], $vgpr7 in [[Reg8:%[0-9]+]], $vgpr8 in [[Reg9:%[0-9]+]], $vgpr9 in [[Reg10:%[0-9]+]], $vgpr10 in [[Reg11:%[0-9]+]], $vgpr11 in [[Reg12:%[0-9]+]], $vgpr12 in [[Reg13:%[0-9]+]], $vgpr13 in [[Reg14:%[0-9]+]], $vgpr14 in [[Reg15:%[0-9]+]], $vgpr15 in [[Reg16:%[0-9]+]], $vgpr16 in [[Reg17:%[0-9]+]], $vgpr17 in [[Reg18:%[0-9]+]], $vgpr18 in [[Reg19:%[0-9]+]], $vgpr19 in [[Reg20:%[0-9]+]], $vgpr20 in [[Reg21:%[0-9]+]], $vgpr21 in [[Reg22:%[0-9]+]], $vgpr22 in [[Reg23:%[0-9]+]], $vgpr23 in [[Reg24:%[0-9]+]], $vgpr24 in [[Reg25:%[0-9]+]], $vgpr25 in [[Reg26:%[0-9]+]], $vgpr26 in [[Reg27:%[0-9]+]], $vgpr27 in [[Reg28:%[0-9]+]]
; EMPTY:
; CHECK: bb.0.entry:
; CHECK-NEXT:   successors: %bb.3(0x40000000), %bb.1(0x40000000); %bb.3(50.00%), %bb.1(50.00%)
; CHECK-NEXT:   liveins: $vgpr0, $vgpr1, $vgpr2, $vgpr3, $vgpr4, $vgpr5, $vgpr6, $vgpr7, $vgpr8, $vgpr9, $vgpr10, $vgpr11, $vgpr12, $vgpr13, $vgpr14, $vgpr15, $vgpr16, $vgpr17, $vgpr18, $vgpr19, $vgpr20, $vgpr21, $vgpr22, $vgpr23, $vgpr24, $vgpr25, $vgpr26, $vgpr27
; CHECK-NEXT:   [[Reg28]]:vgpr_32 = COPY killed $vgpr27
; CHECK-NEXT:   [[Reg27]]:vgpr_32 = COPY killed $vgpr26
; CHECK-NEXT:   [[Reg26]]:vgpr_32 = COPY killed $vgpr25
; CHECK-NEXT:   [[Reg25]]:vgpr_32 = COPY killed $vgpr24
; CHECK-NEXT:   [[Reg24]]:vgpr_32 = COPY killed $vgpr23
; CHECK-NEXT:   [[Reg23]]:vgpr_32 = COPY killed $vgpr22
; CHECK-NEXT:   [[Reg22]]:vgpr_32 = COPY killed $vgpr21
; CHECK-NEXT:   [[Reg21]]:vgpr_32 = COPY killed $vgpr20
; CHECK-NEXT:   [[Reg20]]:vgpr_32 = COPY killed $vgpr19
; CHECK-NEXT:   [[Reg19]]:vgpr_32 = COPY killed $vgpr18
; CHECK-NEXT:   [[Reg18]]:vgpr_32 = COPY killed $vgpr17
; CHECK-NEXT:   [[Reg17]]:vgpr_32 = COPY killed $vgpr16
; CHECK-NEXT:   [[Reg16]]:vgpr_32 = COPY killed $vgpr15
; CHECK-NEXT:   [[Reg15]]:vgpr_32 = COPY killed $vgpr14
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
; CHECK-NEXT:   [[Reg29:%[0-9]+]]:vreg_64 = REG_SEQUENCE killed [[Reg1]]:vgpr_32, %subreg.sub0, killed [[Reg2]]:vgpr_32, %subreg.sub1
; CHECK-NEXT:   [[Reg30:%[0-9]+]]:vgpr_32 = V_AND_B32_e64 1, killed [[Reg28]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg31:%[0-9]+]]:sreg_32 = V_CMP_NE_U32_e64 1, killed [[Reg30]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg32:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg29]]:vreg_64, 0, 0, implicit $exec :: (load (s8) from %ir.p1, addrspace 1)
; CHECK-NEXT:   [[Reg33:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg29]]:vreg_64, 1, 0, implicit $exec :: (load (s8) from %ir.p1 + 1, addrspace 1)
; CHECK-NEXT:   [[Reg34:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg33]]:vgpr_32, 8, killed [[Reg32]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg35:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg29]]:vreg_64, 2, 0, implicit $exec :: (load (s8) from %ir.p1 + 2, addrspace 1)
; CHECK-NEXT:   [[Reg36:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg29]]:vreg_64, 3, 0, implicit $exec :: (load (s8) from %ir.p1 + 3, addrspace 1)
; CHECK-NEXT:   [[Reg37:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg36]]:vgpr_32, 8, killed [[Reg35]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg38:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg37]]:vgpr_32, 16, killed [[Reg34]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg39:%[0-9]+]]:sreg_32 = SI_IF killed [[Reg31]]:sreg_32, %bb.1, implicit-def dead $exec, implicit-def dead $scc, implicit $exec
; CHECK-NEXT:   S_BRANCH %bb.3
; EMPTY:
; CHECK: bb.1.Flow:
; CHECK-NEXT: ; predecessors: %bb.0, %bb.3
; CHECK-NEXT:   successors: %bb.2(0x40000000), %bb.4(0x40000000); %bb.2(50.00%), %bb.4(50.00%)
; EMPTY:
; CHECK:   [[Reg40:%[0-9]+]]:sreg_32 = SI_ELSE killed [[Reg39]]:sreg_32, %bb.4, implicit-def dead $exec, implicit-def dead $scc, implicit $exec
; CHECK-NEXT:   S_BRANCH %bb.2
; EMPTY:
; CHECK: bb.2.bb2:
; CHECK-NEXT: ; predecessors: %bb.1
; CHECK-NEXT:   successors: %bb.4(0x80000000); %bb.4(100.00%)
; EMPTY:
; CHECK:   [[Reg41:%[0-9]+]]:vgpr_32 = V_MUL_LO_U32_e64 [[Reg26]]:vgpr_32, [[Reg38]]:vgpr_32, implicit $exec
; CHECK-NEXT:   GLOBAL_STORE_DWORD [[Reg29]]:vreg_64, killed [[Reg41]]:vgpr_32, 0, 0, implicit $exec :: (store (s32) into %ir.p1, addrspace 1)
; CHECK-NEXT:   S_BRANCH %bb.4
; EMPTY:
; CHECK: bb.3.bb3:
; CHECK-NEXT: ; predecessors: %bb.0
; CHECK-NEXT:   successors: %bb.1(0x80000000); %bb.1(100.00%)
; EMPTY:
; CHECK:   [[Reg42:%[0-9]+]]:vgpr_32 = V_ADD_U32_e64 [[Reg26]]:vgpr_32, [[Reg38]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   GLOBAL_STORE_SHORT_D16_HI [[Reg29]]:vreg_64, [[Reg42]]:vgpr_32, 2, 0, implicit $exec :: (store (s16) into %ir.p1 + 2, addrspace 1)
; CHECK-NEXT:   GLOBAL_STORE_SHORT [[Reg29]]:vreg_64, killed [[Reg42]]:vgpr_32, 0, 0, implicit $exec :: (store (s16) into %ir.p1, addrspace 1)
; CHECK-NEXT:   S_BRANCH %bb.1
; EMPTY:
; CHECK: bb.4.bb4:
; CHECK-NEXT: ; predecessors: %bb.1, %bb.2
; CHECK-NEXT:   successors: %bb.5(0x80000000); %bb.5(100.00%)
; EMPTY:
; CHECK:   SI_END_CF killed [[Reg40]]:sreg_32, implicit-def dead $exec, implicit-def dead $scc, implicit $exec
; CHECK-NEXT:   [[Reg43:%[0-9]+]]:vreg_64 = REG_SEQUENCE killed [[Reg19]]:vgpr_32, %subreg.sub0, killed [[Reg20]]:vgpr_32, %subreg.sub1
; CHECK-NEXT:   [[Reg44:%[0-9]+]]:vreg_64 = REG_SEQUENCE killed [[Reg17]]:vgpr_32, %subreg.sub0, killed [[Reg18]]:vgpr_32, %subreg.sub1
; CHECK-NEXT:   [[Reg45:%[0-9]+]]:vreg_64 = REG_SEQUENCE killed [[Reg15]]:vgpr_32, %subreg.sub0, killed [[Reg16]]:vgpr_32, %subreg.sub1
; CHECK-NEXT:   [[Reg46:%[0-9]+]]:vreg_64 = REG_SEQUENCE killed [[Reg13]]:vgpr_32, %subreg.sub0, killed [[Reg14]]:vgpr_32, %subreg.sub1
; CHECK-NEXT:   [[Reg47:%[0-9]+]]:vreg_64 = REG_SEQUENCE killed [[Reg11]]:vgpr_32, %subreg.sub0, killed [[Reg12]]:vgpr_32, %subreg.sub1
; CHECK-NEXT:   [[Reg48:%[0-9]+]]:vreg_64 = REG_SEQUENCE killed [[Reg9]]:vgpr_32, %subreg.sub0, killed [[Reg10]]:vgpr_32, %subreg.sub1
; CHECK-NEXT:   [[Reg49:%[0-9]+]]:vreg_64 = REG_SEQUENCE killed [[Reg7]]:vgpr_32, %subreg.sub0, killed [[Reg8]]:vgpr_32, %subreg.sub1
; CHECK-NEXT:   [[Reg50:%[0-9]+]]:vreg_64 = REG_SEQUENCE killed [[Reg5]]:vgpr_32, %subreg.sub0, killed [[Reg6]]:vgpr_32, %subreg.sub1
; CHECK-NEXT:   [[Reg51:%[0-9]+]]:vreg_64 = REG_SEQUENCE killed [[Reg3]]:vgpr_32, %subreg.sub0, killed [[Reg4]]:vgpr_32, %subreg.sub1
; CHECK-NEXT:   [[Reg52:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg29]]:vreg_64, 0, 0, implicit $exec :: (load (s8) from %ir.p1, addrspace 1)
; CHECK-NEXT:   [[Reg53:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg29]]:vreg_64, 1, 0, implicit $exec :: (load (s8) from %ir.p1 + 1, addrspace 1)
; CHECK-NEXT:   [[Reg54:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg53]]:vgpr_32, 8, killed [[Reg52]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg55:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg29]]:vreg_64, 2, 0, implicit $exec :: (load (s8) from %ir.p1 + 2, addrspace 1)
; CHECK-NEXT:   [[Reg56:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg29]]:vreg_64, 3, 0, implicit $exec :: (load (s8) from %ir.p1 + 3, addrspace 1)
; CHECK-NEXT:   [[Reg57:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg56]]:vgpr_32, 8, killed [[Reg55]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg58:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg57]]:vgpr_32, 16, killed [[Reg54]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg59:%[0-9]+]]:vgpr_32 = V_ADD_U32_e64 1, [[Reg38]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg60:%[0-9]+]]:sreg_32 = S_MOV_B32 0
; CHECK-NEXT:   [[Reg61:%[0-9]+]]:vgpr_32 = V_MOV_B32_e32 0, implicit $exec
; EMPTY:
; CHECK: bb.5.loop1.header:
; CHECK-NEXT: ; predecessors: %bb.4, %bb.13
; CHECK-NEXT:   successors: %bb.6(0x80000000); %bb.6(100.00%)
; EMPTY:
; CHECK:   [[Reg62:%[0-9]+]]:sreg_32 = PHI [[Reg60]]:sreg_32, %bb.4, [[Reg63:%[0-9]+]]:sreg_32, %bb.13
; CHECK-NEXT:   [[Reg64:%[0-9]+]]:vgpr_32 = PHI [[Reg59]]:vgpr_32, %bb.4, [[Reg65:%[0-9]+]]:vgpr_32, %bb.13
; CHECK-NEXT:   [[Reg66:%[0-9]+]]:vgpr_32 = PHI [[Reg38]]:vgpr_32, %bb.4, [[Reg67:%[0-9]+]]:vgpr_32, %bb.13
; CHECK-NEXT:   [[Reg68:%[0-9]+]]:vgpr_32 = PHI [[Reg61]]:vgpr_32, %bb.4, [[Reg69:%[0-9]+]]:vgpr_32, %bb.13
; CHECK-NEXT:   [[Reg70:%[0-9]+]]:vgpr_32 = V_ASHRREV_I32_e64 31, [[Reg66]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg71:%[0-9]+]]:vreg_64 = REG_SEQUENCE [[Reg66]]:vgpr_32, %subreg.sub0, killed [[Reg70]]:vgpr_32, %subreg.sub1
; CHECK-NEXT:   [[Reg72:%[0-9]+]]:vreg_64 = nsw V_LSHLREV_B64_pseudo_e64 3, killed [[Reg71]]:vreg_64, implicit $exec
; CHECK-NEXT:   [[Reg73:%[0-9]+]]:vgpr_32, [[Reg74:%[0-9]+]]:sreg_32_xm0_xexec = V_ADD_CO_U32_e64 [[Reg29]].sub0:vreg_64, [[Reg72]].sub0:vreg_64, 0, implicit $exec
; CHECK-NEXT:   [[Reg75:%[0-9]+]]:vgpr_32, dead $sgpr_null = V_ADDC_U32_e64 [[Reg29]].sub1:vreg_64, killed [[Reg72]].sub1:vreg_64, killed [[Reg74]]:sreg_32_xm0_xexec, 0, implicit $exec
; CHECK-NEXT:   [[Reg76:%[0-9]+]]:vreg_64 = REG_SEQUENCE killed [[Reg73]]:vgpr_32, %subreg.sub0, killed [[Reg75]]:vgpr_32, %subreg.sub1
; CHECK-NEXT:   [[Reg77:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg76]]:vreg_64, 0, 0, implicit $exec :: (load (s8) from %ir.gep1, addrspace 1)
; CHECK-NEXT:   [[Reg78:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg76]]:vreg_64, 1, 0, implicit $exec :: (load (s8) from %ir.gep1 + 1, addrspace 1)
; CHECK-NEXT:   [[Reg79:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg78]]:vgpr_32, 8, killed [[Reg77]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg80:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg76]]:vreg_64, 2, 0, implicit $exec :: (load (s8) from %ir.gep1 + 2, addrspace 1)
; CHECK-NEXT:   [[Reg81:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE killed [[Reg76]]:vreg_64, 3, 0, implicit $exec :: (load (s8) from %ir.gep1 + 3, addrspace 1)
; CHECK-NEXT:   [[Reg82:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg81]]:vgpr_32, 8, killed [[Reg80]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg83:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg82]]:vgpr_32, 16, killed [[Reg79]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg84:%[0-9]+]]:vgpr_32 = V_MUL_LO_U32_e64 [[Reg26]]:vgpr_32, [[Reg66]]:vgpr_32, implicit $exec
; CHECK-NEXT:   GLOBAL_STORE_SHORT [[Reg29]]:vreg_64, [[Reg84]]:vgpr_32, 0, 0, implicit $exec :: (store (s16) into %ir.p1, addrspace 1)
; CHECK-NEXT:   GLOBAL_STORE_SHORT_D16_HI [[Reg29]]:vreg_64, [[Reg84]]:vgpr_32, 2, 0, implicit $exec :: (store (s16) into %ir.p1 + 2, addrspace 1)
; CHECK-NEXT:   [[Reg85:%[0-9]+]]:vgpr_32 = V_MUL_LO_U32_e64 [[Reg83]]:vgpr_32, [[Reg64]]:vgpr_32, implicit $exec
; EMPTY:
; CHECK: bb.6.loop2.header:
; CHECK-NEXT: ; predecessors: %bb.5, %bb.10
; CHECK-NEXT:   successors: %bb.7(0x80000000); %bb.7(100.00%)
; EMPTY:
; CHECK:   [[Reg86:%[0-9]+]]:sreg_32 = PHI [[Reg60]]:sreg_32, %bb.5, [[Reg87:%[0-9]+]]:sreg_32, %bb.10
; CHECK-NEXT:   [[Reg88:%[0-9]+]]:vgpr_32 = PHI [[Reg85]]:vgpr_32, %bb.5, [[Reg89:%[0-9]+]]:vgpr_32, %bb.10
; CHECK-NEXT:   [[Reg90:%[0-9]+]]:vgpr_32 = PHI [[Reg83]]:vgpr_32, %bb.5, [[Reg91:%[0-9]+]]:vgpr_32, %bb.10
; CHECK-NEXT:   [[Reg92:%[0-9]+]]:vgpr_32 = PHI [[Reg66]]:vgpr_32, %bb.5, [[Reg93:%[0-9]+]]:vgpr_32, %bb.10
; CHECK-NEXT:   [[Reg94:%[0-9]+]]:sreg_32 = S_MOV_B32 0
; EMPTY:
; CHECK: bb.7.loop3.header:
; CHECK-NEXT: ; predecessors: %bb.6, %bb.9
; CHECK-NEXT:   successors: %bb.8(0x40000000), %bb.9(0x40000000); %bb.8(50.00%), %bb.9(50.00%)
; EMPTY:
; CHECK:   [[Reg95:%[0-9]+]]:sreg_32 = PHI [[Reg94]]:sreg_32, %bb.6, [[Reg96:%[0-9]+]]:sreg_32, %bb.9
; CHECK-NEXT:   [[Reg97:%[0-9]+]]:sreg_32 = PHI [[Reg94]]:sreg_32, %bb.6, [[Reg98:%[0-9]+]]:sreg_32, %bb.9
; CHECK-NEXT:   [[Reg99:%[0-9]+]]:vgpr_32 = V_ADD_U32_e64 [[Reg97]]:sreg_32, [[Reg90]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg100:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg51]]:vreg_64, 0, 0, implicit $exec :: (load (s8) from %ir.p2, addrspace 1)
; CHECK-NEXT:   [[Reg101:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg51]]:vreg_64, 1, 0, implicit $exec :: (load (s8) from %ir.p2 + 1, addrspace 1)
; CHECK-NEXT:   [[Reg102:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg101]]:vgpr_32, 8, killed [[Reg100]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg103:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg51]]:vreg_64, 2, 0, implicit $exec :: (load (s8) from %ir.p2 + 2, addrspace 1)
; CHECK-NEXT:   [[Reg104:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg51]]:vreg_64, 3, 0, implicit $exec :: (load (s8) from %ir.p2 + 3, addrspace 1)
; CHECK-NEXT:   [[Reg105:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg104]]:vgpr_32, 8, killed [[Reg103]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg106:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg105]]:vgpr_32, 16, killed [[Reg102]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg107:%[0-9]+]]:sreg_32 = V_CMP_GE_U32_e64 [[Reg99]]:vgpr_32, [[Reg106]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg108:%[0-9]+]]:sreg_32 = SI_IF killed [[Reg107]]:sreg_32, %bb.9, implicit-def dead $exec, implicit-def dead $scc, implicit $exec
; CHECK-NEXT:   S_BRANCH %bb.8
; EMPTY:
; CHECK: bb.8.bb5:
; CHECK-NEXT: ; predecessors: %bb.7
; CHECK-NEXT:   successors: %bb.9(0x80000000); %bb.9(100.00%)
; EMPTY:
; CHECK:   [[Reg109:%[0-9]+]]:vgpr_32 = V_ADD_U32_e64 [[Reg97]]:sreg_32, [[Reg88]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg110:%[0-9]+]]:vgpr_32 = V_MUL_LO_U32_e64 killed [[Reg109]]:vgpr_32, killed [[Reg106]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg111:%[0-9]+]]:vgpr_32 = V_ADD_U32_e64 [[Reg110]]:vgpr_32, [[Reg84]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   GLOBAL_STORE_DWORD [[Reg46]]:vreg_64, [[Reg111]]:vgpr_32, 0, 0, implicit $exec :: (store (s32) into %ir.p7, addrspace 1)
; EMPTY:
; CHECK: bb.9.loop3.latch:
; CHECK-NEXT: ; predecessors: %bb.7, %bb.8
; CHECK-NEXT:   successors: %bb.10(0x04000000), %bb.7(0x7c000000); %bb.10(3.12%), %bb.7(96.88%)
; EMPTY:
; CHECK:   [[Reg69]]:vgpr_32 = PHI [[Reg83]]:vgpr_32, %bb.7, [[Reg111]]:vgpr_32, %bb.8
; CHECK-NEXT:   [[Reg112:%[0-9]+]]:vgpr_32 = PHI [[Reg99]]:vgpr_32, %bb.7, [[Reg110]]:vgpr_32, %bb.8
; CHECK-NEXT:   [[Reg93]]:vgpr_32 = PHI [[Reg99]]:vgpr_32, %bb.7, [[Reg90]]:vgpr_32, %bb.8
; CHECK-NEXT:   SI_END_CF killed [[Reg108]]:sreg_32, implicit-def dead $exec, implicit-def dead $scc, implicit $exec
; CHECK-NEXT:   GLOBAL_STORE_DWORD [[Reg45]]:vreg_64, [[Reg93]]:vgpr_32, 0, 0, implicit $exec :: (store (s32) into %ir.p9, addrspace 1)
; CHECK-NEXT:   [[Reg113:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_DWORD [[Reg44]]:vreg_64, 0, 0, implicit $exec :: (load (s32) from %ir.p10, addrspace 1)
; CHECK-NEXT:   [[Reg114:%[0-9]+]]:vgpr_32 = V_MUL_LO_U32_e64 [[Reg112]]:vgpr_32, [[Reg69]]:vgpr_32, implicit $exec
; CHECK-NEXT:   GLOBAL_STORE_DWORD [[Reg44]]:vreg_64, [[Reg114]]:vgpr_32, 0, 0, implicit $exec :: (store (s32) into %ir.p10, addrspace 1)
; CHECK-NEXT:   [[Reg98]]:sreg_32 = S_ADD_I32 killed [[Reg97]]:sreg_32, 1, implicit-def dead $scc
; CHECK-NEXT:   [[Reg115:%[0-9]+]]:vgpr_32 = V_ADD_U32_e64 [[Reg98]]:sreg_32, [[Reg90]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg116:%[0-9]+]]:sreg_32 = V_CMP_GE_U32_e64 killed [[Reg115]]:vgpr_32, [[Reg23]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg96]]:sreg_32 = SI_IF_BREAK killed [[Reg116]]:sreg_32, killed [[Reg95]]:sreg_32, implicit-def dead $scc
; CHECK-NEXT:   SI_LOOP [[Reg96]]:sreg_32, %bb.7, implicit-def dead $exec, implicit-def dead $scc, implicit $exec
; CHECK-NEXT:   S_BRANCH %bb.10
; EMPTY:
; CHECK: bb.10.loop2.latch:
; CHECK-NEXT: ; predecessors: %bb.9
; CHECK-NEXT:   successors: %bb.11(0x04000000), %bb.6(0x7c000000); %bb.11(3.12%), %bb.6(96.88%)
; EMPTY:
; CHECK:   SI_END_CF killed [[Reg96]]:sreg_32, implicit-def dead $exec, implicit-def dead $scc, implicit $exec
; CHECK-NEXT:   [[Reg91]]:vgpr_32 = V_ADD_U32_e64 1, killed [[Reg90]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg117:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_DWORD [[Reg43]]:vreg_64, 0, 0, implicit $exec :: (load (s32) from %ir.p11, addrspace 1)
; CHECK-NEXT:   [[Reg118:%[0-9]+]]:vgpr_32 = V_ADD_U32_e64 [[Reg91]]:vgpr_32, [[Reg112]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   GLOBAL_STORE_DWORD [[Reg43]]:vreg_64, [[Reg118]]:vgpr_32, 0, 0, implicit $exec :: (store (s32) into %ir.p11, addrspace 1)
; CHECK-NEXT:   [[Reg89]]:vgpr_32 = V_ADD_U32_e64 killed [[Reg88]]:vgpr_32, [[Reg64]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg119:%[0-9]+]]:sreg_32 = V_CMP_GE_U32_e64 [[Reg91]]:vgpr_32, [[Reg22]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg87]]:sreg_32 = SI_IF_BREAK killed [[Reg119]]:sreg_32, killed [[Reg86]]:sreg_32, implicit-def dead $scc
; CHECK-NEXT:   SI_LOOP [[Reg87]]:sreg_32, %bb.6, implicit-def dead $exec, implicit-def dead $scc, implicit $exec
; CHECK-NEXT:   S_BRANCH %bb.11
; EMPTY:
; CHECK: bb.11.loop4.preheader:
; CHECK-NEXT: ; predecessors: %bb.10
; CHECK-NEXT:   successors: %bb.12(0x80000000); %bb.12(100.00%)
; EMPTY:
; CHECK:   SI_END_CF killed [[Reg87]]:sreg_32, implicit-def dead $exec, implicit-def dead $scc, implicit $exec
; CHECK-NEXT:   [[Reg120:%[0-9]+]]:vreg_64 = REG_SEQUENCE [[Reg69]]:vgpr_32, %subreg.sub0, undef [[Reg121:%[0-9]+]]:vgpr_32, %subreg.sub1
; CHECK-NEXT:   [[Reg122:%[0-9]+]]:vreg_64, $sgpr_null = V_MAD_U64_U32_e64 [[Reg69]]:vgpr_32, killed [[Reg112]]:vgpr_32, killed [[Reg120]]:vreg_64, 0, implicit $exec
; CHECK-NEXT:   [[Reg123:%[0-9]+]]:vgpr_32 = COPY killed [[Reg122]].sub0:vreg_64
; CHECK-NEXT:   [[Reg124:%[0-9]+]]:sreg_32 = S_MOV_B32 0
; EMPTY:
; CHECK: bb.12.loop4:
; CHECK-NEXT: ; predecessors: %bb.11, %bb.12
; CHECK-NEXT:   successors: %bb.13(0x04000000), %bb.12(0x7c000000); %bb.13(3.12%), %bb.12(96.88%)
; EMPTY:
; CHECK:   [[Reg125:%[0-9]+]]:sreg_32 = PHI [[Reg124]]:sreg_32, %bb.11, [[Reg126:%[0-9]+]]:sreg_32, %bb.12
; CHECK-NEXT:   [[Reg127:%[0-9]+]]:vgpr_32 = PHI [[Reg123]]:vgpr_32, %bb.11, [[Reg128:%[0-9]+]]:vgpr_32, %bb.12
; CHECK-NEXT:   [[Reg129:%[0-9]+]]:vgpr_32 = PHI [[Reg114]]:vgpr_32, %bb.11, [[Reg130:%[0-9]+]]:vgpr_32, %bb.12
; CHECK-NEXT:   [[Reg131:%[0-9]+]]:vgpr_32 = PHI [[Reg118]]:vgpr_32, %bb.11, [[Reg68]]:vgpr_32, %bb.12
; CHECK-NEXT:   [[Reg132:%[0-9]+]]:vgpr_32 = PHI [[Reg113]]:vgpr_32, %bb.11, [[Reg133:%[0-9]+]]:vgpr_32, %bb.12
; CHECK-NEXT:   [[Reg134:%[0-9]+]]:vgpr_32 = PHI [[Reg117]]:vgpr_32, %bb.11, [[Reg135:%[0-9]+]]:vgpr_32, %bb.12
; CHECK-NEXT:   [[Reg136:%[0-9]+]]:vgpr_32 = V_ADD_U32_e64 [[Reg131]]:vgpr_32, [[Reg69]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg137:%[0-9]+]]:vgpr_32 = V_MUL_LO_U32_e64 killed [[Reg134]]:vgpr_32, [[Reg113]]:vgpr_32, implicit $exec
; CHECK-NEXT:   GLOBAL_STORE_DWORD [[Reg50]]:vreg_64, killed [[Reg137]]:vgpr_32, 0, 0, implicit $exec :: (store (s32) into %ir.p3, addrspace 1)
; CHECK-NEXT:   [[Reg138:%[0-9]+]]:vgpr_32 = V_ADD_U32_e64 killed [[Reg131]]:vgpr_32, [[Reg127]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg139:%[0-9]+]]:vgpr_32 = V_MUL_LO_U32_e64 [[Reg92]]:vgpr_32, killed [[Reg138]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg140:%[0-9]+]]:vgpr_32 = V_MUL_LO_U32_e64 killed [[Reg132]]:vgpr_32, [[Reg117]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg141:%[0-9]+]]:vgpr_32 = V_ADD_U32_e64 [[Reg140]]:vgpr_32, [[Reg139]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   GLOBAL_STORE_DWORD [[Reg49]]:vreg_64, killed [[Reg141]]:vgpr_32, 0, 0, implicit $exec :: (store (s32) into %ir.p4, addrspace 1)
; CHECK-NEXT:   [[Reg130]]:vgpr_32 = V_ADD_U32_e64 4, killed [[Reg129]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg142:%[0-9]+]]:vgpr_32 = V_MUL_HI_U32_e64 -1431655765, killed [[Reg139]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg133]]:vgpr_32 = V_LSHRREV_B32_e64 1, killed [[Reg142]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg143:%[0-9]+]]:vgpr_32 = V_LSHRREV_B32_e64 31, [[Reg136]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg144:%[0-9]+]]:vgpr_32 = V_ADD_U32_e64 killed [[Reg136]]:vgpr_32, killed [[Reg143]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg135]]:vgpr_32 = V_ASHRREV_I32_e64 1, killed [[Reg144]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg128]]:vgpr_32 = V_ADD_U32_e64 4, killed [[Reg127]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg145:%[0-9]+]]:sreg_32 = V_CMP_GE_U32_e64 [[Reg130]]:vgpr_32, [[Reg24]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg126]]:sreg_32 = SI_IF_BREAK killed [[Reg145]]:sreg_32, killed [[Reg125]]:sreg_32, implicit-def dead $scc
; CHECK-NEXT:   SI_LOOP [[Reg126]]:sreg_32, %bb.12, implicit-def dead $exec, implicit-def dead $scc, implicit $exec
; CHECK-NEXT:   S_BRANCH %bb.13
; EMPTY:
; CHECK: bb.13.loop1.latch:
; CHECK-NEXT: ; predecessors: %bb.12
; CHECK-NEXT:   successors: %bb.14(0x04000000), %bb.5(0x7c000000); %bb.14(3.12%), %bb.5(96.88%)
; EMPTY:
; CHECK:   SI_END_CF killed [[Reg126]]:sreg_32, implicit-def dead $exec, implicit-def dead $scc, implicit $exec
; CHECK-NEXT:   [[Reg146:%[0-9]+]]:vgpr_32 = V_ADD_U32_e64 killed [[Reg140]]:vgpr_32, [[Reg27]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   GLOBAL_STORE_DWORD [[Reg48]]:vreg_64, [[Reg146]]:vgpr_32, 0, 0, implicit $exec :: (store (s32) into %ir.p5, addrspace 1)
; CHECK-NEXT:   [[Reg67]]:vgpr_32 = V_ADD_U32_e64 1, killed [[Reg66]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg65]]:vgpr_32 = V_ADD_U32_e64 1, killed [[Reg64]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg147:%[0-9]+]]:sreg_32 = V_CMP_GE_U32_e64 [[Reg67]]:vgpr_32, [[Reg21]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg63]]:sreg_32 = SI_IF_BREAK killed [[Reg147]]:sreg_32, killed [[Reg62]]:sreg_32, implicit-def dead $scc
; CHECK-NEXT:   SI_LOOP [[Reg63]]:sreg_32, %bb.5, implicit-def dead $exec, implicit-def dead $scc, implicit $exec
; CHECK-NEXT:   S_BRANCH %bb.14
; EMPTY:
; CHECK: bb.14.bb6:
; CHECK-NEXT: ; predecessors: %bb.13
; CHECK-NEXT:   successors: %bb.15(0x40000000), %bb.16(0x40000000); %bb.15(50.00%), %bb.16(50.00%)
; EMPTY:
; CHECK:   SI_END_CF killed [[Reg63]]:sreg_32, implicit-def dead $exec, implicit-def dead $scc, implicit $exec
; CHECK-NEXT:   [[Reg148:%[0-9]+]]:vgpr_32 = V_MUL_LO_U32_e64 100, killed [[Reg58]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg149:%[0-9]+]]:sreg_32 = V_CMP_GT_U32_e64 [[Reg148]]:vgpr_32, [[Reg146]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg150:%[0-9]+]]:sreg_32 = SI_IF killed [[Reg149]]:sreg_32, %bb.16, implicit-def dead $exec, implicit-def dead $scc, implicit $exec
; CHECK-NEXT:   S_BRANCH %bb.15
; EMPTY:
; CHECK: bb.15.bb7:
; CHECK-NEXT: ; predecessors: %bb.14
; CHECK-NEXT:   successors: %bb.16(0x80000000); %bb.16(100.00%)
; EMPTY:
; CHECK:   GLOBAL_STORE_DWORD killed [[Reg47]]:vreg_64, [[Reg148]]:vgpr_32, 0, 0, implicit $exec :: (store (s32) into %ir.p6, addrspace 1)
; EMPTY:
; CHECK: bb.16.loop5.preheader:
; CHECK-NEXT: ; predecessors: %bb.14, %bb.15
; CHECK-NEXT:   successors: %bb.17(0x80000000); %bb.17(100.00%)
; EMPTY:
; CHECK:   [[Reg151:%[0-9]+]]:vgpr_32 = PHI [[Reg148]]:vgpr_32, %bb.14, [[Reg146]]:vgpr_32, %bb.15
; CHECK-NEXT:   SI_END_CF killed [[Reg150]]:sreg_32, implicit-def dead $exec, implicit-def dead $scc, implicit $exec
; CHECK-NEXT:   [[Reg152:%[0-9]+]]:sreg_32 = S_MOV_B32 0
; EMPTY:
; CHECK: bb.17.loop5:
; CHECK-NEXT: ; predecessors: %bb.16, %bb.17
; CHECK-NEXT:   successors: %bb.18(0x04000000), %bb.17(0x7c000000); %bb.18(3.12%), %bb.17(96.88%)
; EMPTY:
; CHECK:   [[Reg153:%[0-9]+]]:sreg_32 = PHI [[Reg152]]:sreg_32, %bb.16, [[Reg154:%[0-9]+]]:sreg_32, %bb.17
; CHECK-NEXT:   [[Reg155:%[0-9]+]]:vgpr_32 = PHI [[Reg151]]:vgpr_32, %bb.16, [[Reg156:%[0-9]+]]:vgpr_32, %bb.17
; CHECK-NEXT:   [[Reg156]]:vgpr_32 = V_ADD_U32_e64 2, [[Reg155]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg157:%[0-9]+]]:sreg_32 = V_CMP_GE_U32_e64 [[Reg156]]:vgpr_32, [[Reg25]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg154]]:sreg_32 = SI_IF_BREAK killed [[Reg157]]:sreg_32, killed [[Reg153]]:sreg_32, implicit-def dead $scc
; CHECK-NEXT:   SI_LOOP [[Reg154]]:sreg_32, %bb.17, implicit-def dead $exec, implicit-def dead $scc, implicit $exec
; CHECK-NEXT:   S_BRANCH %bb.18
; EMPTY:
; CHECK: bb.18.exit:
; CHECK-NEXT: ; predecessors: %bb.17
; EMPTY:
; CHECK:   SI_END_CF killed [[Reg154]]:sreg_32, implicit-def dead $exec, implicit-def dead $scc, implicit $exec
; CHECK-NEXT:   [[Reg158:%[0-9]+]]:vgpr_32 = V_ADD_U32_e64 100, killed [[Reg38]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg159:%[0-9]+]]:vgpr_32 = V_MUL_LO_U32_e64 killed [[Reg148]]:vgpr_32, killed [[Reg155]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg160:%[0-9]+]]:vgpr_32 = V_MUL_LO_U32_e64 killed [[Reg158]]:vgpr_32, killed [[Reg159]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg161:%[0-9]+]]:sreg_32_xm0 = V_READFIRSTLANE_B32 killed [[Reg160]]:vgpr_32, implicit $exec
; CHECK-NEXT:   $sgpr0 = COPY killed [[Reg161]]:sreg_32_xm0
; CHECK-NEXT:   SI_RETURN_TO_EPILOG killed $sgpr0
; EMPTY:
; CHECK: # End machine code for function test12.
; EMPTY:
; CHECK: Next-use distance of Register [[Reg28]] = 29
; CHECK-NEXT: Next-use distance of Register [[Reg27]] = 94088
; CHECK-NEXT: Next-use distance of Register [[Reg26]] = 38
; CHECK-NEXT: Next-use distance of Register [[Reg25]] = 196070
; CHECK-NEXT: Next-use distance of Register [[Reg24]] = 69105
; CHECK-NEXT: Next-use distance of Register [[Reg23]] = 106
; CHECK-NEXT: Next-use distance of Register [[Reg22]] = 27088
; CHECK-NEXT: Next-use distance of Register [[Reg21]] = 94086
; CHECK-NEXT: Next-use distance of Register [[Reg20]] = 35
; CHECK-NEXT: Next-use distance of Register [[Reg19]] = 34
; CHECK-NEXT: Next-use distance of Register [[Reg18]] = 34
; CHECK-NEXT: Next-use distance of Register [[Reg17]] = 33
; CHECK-NEXT: Next-use distance of Register [[Reg16]] = 33
; CHECK-NEXT: Next-use distance of Register [[Reg15]] = 32
; CHECK-NEXT: Next-use distance of Register [[Reg14]] = 32
; CHECK-NEXT: Next-use distance of Register [[Reg13]] = 31
; CHECK-NEXT: Next-use distance of Register [[Reg12]] = 31
; CHECK-NEXT: Next-use distance of Register [[Reg11]] = 30
; CHECK-NEXT: Next-use distance of Register [[Reg10]] = 30
; CHECK-NEXT: Next-use distance of Register [[Reg9]] = 29
; CHECK-NEXT: Next-use distance of Register [[Reg8]] = 29
; CHECK-NEXT: Next-use distance of Register [[Reg7]] = 28
; CHECK-NEXT: Next-use distance of Register [[Reg6]] = 28
; CHECK-NEXT: Next-use distance of Register [[Reg5]] = 27
; CHECK-NEXT: Next-use distance of Register [[Reg4]] = 27
; CHECK-NEXT: Next-use distance of Register [[Reg3]] = 26
; CHECK-NEXT: Next-use distance of Register [[Reg2]] = 2
; CHECK-NEXT: Next-use distance of Register [[Reg1]] = 1
; CHECK-NEXT: Next-use distance of Register [[Reg29]] = 3
; CHECK-NEXT: Next-use distance of Register [[Reg30]] = 1
; CHECK-NEXT: Next-use distance of Register [[Reg31]] = 8
; CHECK-NEXT: Next-use distance of Register [[Reg32]] = 2
; CHECK-NEXT: Next-use distance of Register [[Reg33]] = 1
; CHECK-NEXT: Next-use distance of Register [[Reg34]] = 4
; CHECK-NEXT: Next-use distance of Register [[Reg35]] = 2
; CHECK-NEXT: Next-use distance of Register [[Reg36]] = 1
; CHECK-NEXT: Next-use distance of Register [[Reg37]] = 1
; CHECK-NEXT: Next-use distance of Register [[Reg38]] = 3
; CHECK-NEXT: Next-use distance of Register [[Reg39]] = 2
; CHECK-NEXT: Next-use distance of Register [[Reg40]] = 2
; CHECK-NEXT: Next-use distance of Register [[Reg41]] = 1
; CHECK-NEXT: Next-use distance of Register [[Reg42]] = 1
; CHECK-NEXT: Next-use distance of Register [[Reg43]] = 27047
; CHECK-NEXT: Next-use distance of Register [[Reg44]] = 62
; CHECK-NEXT: Next-use distance of Register [[Reg45]] = 60
; CHECK-NEXT: Next-use distance of Register [[Reg46]] = 58
; CHECK-NEXT: Next-use distance of Register [[Reg47]] = 196020
; CHECK-NEXT: Next-use distance of Register [[Reg48]] = 94042
; CHECK-NEXT: Next-use distance of Register [[Reg49]] = 69052
; CHECK-NEXT: Next-use distance of Register [[Reg50]] = 69046
; CHECK-NEXT: Next-use distance of Register [[Reg51]] = 40
; CHECK-NEXT: Next-use distance of Register [[Reg52]] = 2
; CHECK-NEXT: Next-use distance of Register [[Reg53]] = 1
; CHECK-NEXT: Next-use distance of Register [[Reg54]] = 4
; CHECK-NEXT: Next-use distance of Register [[Reg55]] = 2
; CHECK-NEXT: Next-use distance of Register [[Reg56]] = 1
; CHECK-NEXT: Next-use distance of Register [[Reg57]] = 1
; CHECK-NEXT: Next-use distance of Register [[Reg58]] = 196005
; CHECK-NEXT: Next-use distance of Register [[Reg59]] = 4
; CHECK-NEXT: Next-use distance of Register [[Reg60]] = 2
; CHECK-NEXT: Next-use distance of Register [[Reg61]] = 4
; CHECK-NEXT: Next-use distance of Register [[Reg62]] = 94032
; CHECK-NEXT: Next-use distance of Register [[Reg64]] = 19
; CHECK-NEXT: Next-use distance of Register [[Reg66]] = 2
; CHECK-NEXT: Next-use distance of Register [[Reg68]] = 69026
; CHECK-NEXT: Next-use distance of Register [[Reg70]] = 1
; CHECK-NEXT: Next-use distance of Register [[Reg71]] = 1
; CHECK-NEXT: Next-use distance of Register [[Reg72]] = 1
; CHECK-NEXT: Next-use distance of Register [[Reg73]] = 2
; CHECK-NEXT: Next-use distance of Register [[Reg74]] = 1
; CHECK-NEXT: Next-use distance of Register [[Reg75]] = 1
; CHECK-NEXT: Next-use distance of Register [[Reg76]] = 1
; CHECK-NEXT: Next-use distance of Register [[Reg77]] = 2
; CHECK-NEXT: Next-use distance of Register [[Reg78]] = 1
; CHECK-NEXT: Next-use distance of Register [[Reg79]] = 4
; CHECK-NEXT: Next-use distance of Register [[Reg80]] = 2
; CHECK-NEXT: Next-use distance of Register [[Reg81]] = 1
; CHECK-NEXT: Next-use distance of Register [[Reg82]] = 1
; CHECK-NEXT: Next-use distance of Register [[Reg83]] = 4
; CHECK-NEXT: Next-use distance of Register [[Reg84]] = 1
; CHECK-NEXT: Next-use distance of Register [[Reg85]] = 2
; CHECK-NEXT: Next-use distance of Register [[Reg86]] = 27012
; CHECK-NEXT: Next-use distance of Register [[Reg88]] = 17
; CHECK-NEXT: Next-use distance of Register [[Reg90]] = 5
; CHECK-NEXT: Next-use distance of Register [[Reg92]] = 69016
; CHECK-NEXT: Next-use distance of Register [[Reg94]] = 1
; CHECK-NEXT: Next-use distance of Register [[Reg95]] = 24
; CHECK-NEXT: Next-use distance of Register [[Reg97]] = 1
; CHECK-NEXT: Next-use distance of Register [[Reg99]] = 8
; CHECK-NEXT: Next-use distance of Register [[Reg100]] = 2
; CHECK-NEXT: Next-use distance of Register [[Reg101]] = 1
; CHECK-NEXT: Next-use distance of Register [[Reg102]] = 4
; CHECK-NEXT: Next-use distance of Register [[Reg103]] = 2
; CHECK-NEXT: Next-use distance of Register [[Reg104]] = 1
; CHECK-NEXT: Next-use distance of Register [[Reg105]] = 1
; CHECK-NEXT: Next-use distance of Register [[Reg106]] = 1
; CHECK-NEXT: Next-use distance of Register [[Reg107]] = 1
; CHECK-NEXT: Next-use distance of Register [[Reg108]] = 5
; CHECK-NEXT: Next-use distance of Register [[Reg109]] = 1
; CHECK-NEXT: Next-use distance of Register [[Reg110]] = 1
; CHECK-NEXT: Next-use distance of Register [[Reg111]] = 1
; CHECK-NEXT: Next-use distance of Register [[Reg69]] = 6
; CHECK-NEXT: Next-use distance of Register [[Reg112]] = 5
; CHECK-NEXT: Next-use distance of Register [[Reg93]] = 2
; CHECK-NEXT: Next-use distance of Register [[Reg113]] = 69010
; CHECK-NEXT: Next-use distance of Register [[Reg114]] = 1
; CHECK-NEXT: Next-use distance of Register [[Reg98]] = 1
; CHECK-NEXT: Next-use distance of Register [[Reg115]] = 1
; CHECK-NEXT: Next-use distance of Register [[Reg116]] = 1
; CHECK-NEXT: Next-use distance of Register [[Reg96]] = 1
; CHECK-NEXT: Next-use distance of Register [[Reg91]] = 2
; CHECK-NEXT: Next-use distance of Register [[Reg117]] = 69011
; CHECK-NEXT: Next-use distance of Register [[Reg118]] = 1
; CHECK-NEXT: Next-use distance of Register [[Reg89]] = 6
; CHECK-NEXT: Next-use distance of Register [[Reg119]] = 1
; CHECK-NEXT: Next-use distance of Register [[Reg87]] = 1
; CHECK-NEXT: Next-use distance of Register [[Reg120]] = 1
; CHECK-NEXT: Next-use distance of Register [[Reg122]] = 1
; CHECK-NEXT: Next-use distance of Register [[Reg123]] = 3
; CHECK-NEXT: Next-use distance of Register [[Reg124]] = 1
; CHECK-NEXT: Next-use distance of Register [[Reg125]] = 22
; CHECK-NEXT: Next-use distance of Register [[Reg127]] = 8
; CHECK-NEXT: Next-use distance of Register [[Reg129]] = 12
; CHECK-NEXT: Next-use distance of Register [[Reg131]] = 3
; CHECK-NEXT: Next-use distance of Register [[Reg132]] = 7
; CHECK-NEXT: Next-use distance of Register [[Reg134]] = 2
; CHECK-NEXT: Next-use distance of Register [[Reg136]] = 11
; CHECK-NEXT: Next-use distance of Register [[Reg137]] = 1
; CHECK-NEXT: Next-use distance of Register [[Reg138]] = 1
; CHECK-NEXT: Next-use distance of Register [[Reg139]] = 2
; CHECK-NEXT: Next-use distance of Register [[Reg140]] = 1
; CHECK-NEXT: Next-use distance of Register [[Reg141]] = 1
; CHECK-NEXT: Next-use distance of Register [[Reg130]] = 7
; CHECK-NEXT: Next-use distance of Register [[Reg142]] = 1
; CHECK-NEXT: Next-use distance of Register [[Reg133]] = 13
; CHECK-NEXT: Next-use distance of Register [[Reg143]] = 1
; CHECK-NEXT: Next-use distance of Register [[Reg144]] = 1
; CHECK-NEXT: Next-use distance of Register [[Reg135]] = 11
; CHECK-NEXT: Next-use distance of Register [[Reg128]] = 6
; CHECK-NEXT: Next-use distance of Register [[Reg145]] = 1
; CHECK-NEXT: Next-use distance of Register [[Reg126]] = 1
; CHECK-NEXT: Next-use distance of Register [[Reg146]] = 1
; CHECK-NEXT: Next-use distance of Register [[Reg67]] = 2
; CHECK-NEXT: Next-use distance of Register [[Reg65]] = 6
; CHECK-NEXT: Next-use distance of Register [[Reg147]] = 1
; CHECK-NEXT: Next-use distance of Register [[Reg63]] = 1
; CHECK-NEXT: Next-use distance of Register [[Reg148]] = 1
; CHECK-NEXT: Next-use distance of Register [[Reg149]] = 1
; CHECK-NEXT: Next-use distance of Register [[Reg150]] = 3
; CHECK-NEXT: Next-use distance of Register [[Reg151]] = 4
; CHECK-NEXT: Next-use distance of Register [[Reg152]] = 1
; CHECK-NEXT: Next-use distance of Register [[Reg153]] = 4
; CHECK-NEXT: Next-use distance of Register [[Reg155]] = 1
; CHECK-NEXT: Next-use distance of Register [[Reg156]] = 1
; CHECK-NEXT: Next-use distance of Register [[Reg157]] = 1
; CHECK-NEXT: Next-use distance of Register [[Reg154]] = 1
; CHECK-NEXT: Next-use distance of Register [[Reg158]] = 2
; CHECK-NEXT: Next-use distance of Register [[Reg159]] = 1
; CHECK-NEXT: Next-use distance of Register [[Reg160]] = 1
; CHECK-NEXT: Next-use distance of Register [[Reg161]] = 1
entry:
;        entry
;          |
;         bb1
;        /   \
;      bb2   bb3
;        \   /
;         bb4
;          |
;     loop1.header<-------+
;          |              |
;     loop2.header<-----+ |
;          |            | |
;     loop3.header<---+ | |
;         /   |       | | |
;       bb5   |       | | |
;         \   |       | | |
;     loop3.latch-----+ | |
;          |            | |
;     loop2.latch-------+ |
;          |              |
; +-->loop4|              |
; +--------+              |
;          |              |
;     loop1.latch---------+
;          |
;         bb6
;        /  |
;      bb17 |
;        |  |
;   +-->loop5
;   +-----+
;         |
;        exit

   %ld1 = load i32, ptr addrspace(1) %p1, align 1
   %add1 = add i32 %ld1, 100
   br label %bb1

bb1:
   br i1 %cond1, label %bb2, label %bb3

bb2:
   %mul1 = mul i32 %Val1, %ld1
   store i32 %mul1, ptr addrspace(1) %p1, align 4
   br label %bb4

bb3:
   %add2 = add i32 %Val1, %ld1
   store i32 %add2, ptr addrspace(1) %p1, align 2
   br label %bb4

bb4:
   %phi1 = phi i32 [ %mul1, %bb2 ], [ %add2, %bb3 ]
   %ld2 = load i32, ptr addrspace(1) %p1, align 1
   br label %loop1.header

loop1.header:
   %phi.inc1 = phi i32 [ %ld1, %bb4 ], [  %inc1, %loop1.latch ]
   %phi.phi = phi i32 [ 0, %bb4 ], [ %phi2, %loop1.latch ]
   %sext1 = sext i32 %phi.inc1 to i64
   %gep1 = getelementptr inbounds i64, ptr addrspace(1) %p1, i64 %sext1
   %ld3 = load i32, ptr addrspace(1) %gep1, align 1
   %mul2 = mul i32 %Val1, %phi.inc1
   store i32 %mul2, ptr addrspace(1) %p1, align 2
   br label %loop2.header

loop2.header:
   %phi.inc2 = phi i32 [ %ld3, %loop1.header ], [  %inc2, %loop2.latch ]
   %phi6 = phi i32 [ %phi.inc1, %loop1.header ], [ %phi5, %loop2.latch ]
   br label %loop3.header

loop3.header:
   %phi.inc3 = phi i32 [ %phi.inc2, %loop2.header ], [  %inc3, %loop3.latch ]
   %ld4 = load i32, ptr addrspace(1) %p2, align 1
   %cond2 = icmp uge i32 %phi.inc3, %ld4
   br i1 %cond2, label %bb5, label %loop3.latch

bb5:
   %mul3 = mul i32 %phi.inc1, %phi.inc2
   %add3 = add i32 %mul3, %phi.inc3
   %mul4 = mul i32 %add3, %ld4
   %add4 = add i32 %mul4, %mul2
   store i32 %add4, ptr addrspace(1) %p7
   br label %loop3.latch

loop3.latch:
   %phi2 = phi i32 [ %add4, %bb5 ], [ %ld3, %loop3.header ]
   %phi4 = phi i32 [ %mul4, %bb5 ], [ %phi.inc3, %loop3.header ]
   %phi5 = phi i32 [ %phi.inc2, %bb5 ], [ %phi.inc3, %loop3.header ]
   store i32 %phi5, ptr addrspace(1) %p9
   %inc3 = add i32 %phi.inc3, 1
   %ld10 = load i32, ptr addrspace(1) %p10
   %mul11 = mul i32 %phi4, %phi2
   store i32 %mul11, ptr addrspace(1) %p10
   %cond3 = icmp ult i32 %inc3, %TC3
   br i1 %cond3, label %loop3.header, label %loop2.latch

loop2.latch:
   %inc2 = add i32 %phi.inc2, 1
   %ld11 = load i32, ptr addrspace(1) %p11
   %add9 = add i32 %inc2, %phi4
   store i32 %add9, ptr addrspace(1) %p11
   %cond4 = icmp ult i32 %inc2, %TC2
   br i1 %cond4, label %loop2.header, label %loop4

loop4:
   %phi.inc4 = phi i32 [ %mul11, %loop2.latch ], [  %inc4, %loop4 ]
   %phi7 = phi i32 [ %add9, %loop2.latch ], [ %phi.phi, %loop4 ]
   %phi.div1 = phi i32 [ %ld10, %loop2.latch ], [ %div1, %loop4 ]
   %phi.div2 = phi i32 [ %ld11, %loop2.latch ], [ %div2, %loop4 ]
   %add5 = add i32 %phi7, %phi2
   %mul5 = mul i32 %phi.div2, %ld10
   store i32 %mul5, ptr addrspace(1) %p3
   %add6 = add i32 %add5, %phi.inc4
   %mul8 = mul i32 %phi6, %add6
   %mul9 = mul i32 %phi.div1, %ld11
   %add10 = add i32 %mul9, %mul8
   store i32 %add10, ptr addrspace(1) %p4
   %inc4 = add i32 %phi.inc4, 4
   %div1 = udiv i32 %mul8, 3
   %div2 = sdiv i32 %add5, 2
   %cond7 = icmp ult i32 %inc4, %TC4
   br i1 %cond7, label %loop4, label %loop1.latch

loop1.latch:
   %add7 = add i32 %mul9, %Val2
   store i32 %add7, ptr addrspace(1) %p5
   %inc1 = add i32 %phi.inc1, 1
   %cond5 = icmp ult i32 %inc1, %TC1
   br i1 %cond5, label %loop1.header, label %bb6

bb6:
   %mul6 = mul i32 %ld2, 100
   %cond8 = icmp ugt i32 %mul6, %add7
   br i1 %cond8, label %bb7, label %loop5

bb7:
   store i32 %mul6, ptr addrspace(1) %p6
   br label %loop5

loop5:
   %phi.inc5 = phi i32 [ %add7, %bb7 ], [ %mul6, %bb6 ], [  %inc5, %loop5 ]
   %add8 = mul i32 %mul6, %phi.inc5
   %inc5 = add i32 %phi.inc5, 2
   %cond9 = icmp ult i32 %inc5, %TC5
   br i1 %cond9, label %loop5, label %exit

exit:
   %mul7 = mul i32 %add1, %add8
   ret i32 %mul7
}
