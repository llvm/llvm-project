; RUN: llc -mtriple=amdgcn -mcpu=gfx1200 -enable-next-use-analysis=true -verify-machineinstrs -dump-next-use-distance < %s 2>&1 | FileCheck %s

;
;                     bb.0.entry
;                         |
;                 bb.1.loop1.header<-------+
;                         |                |
;                 bb.2.loop2.header<---+   |
;                         |            |   |
;                     bb.3.loop3<--+   |   |
;                         |        |   |   |
;                         +--------+   |   |
;                         |            |   |
;                 bb.4.loop2.latch-----+   |
;                         |                |
;                 bb.5.loop1.latch---------+
;                         |
;                     bb.6.exit
;
define amdgpu_ps i32 @test(ptr addrspace(1) %p1, ptr addrspace(1) %p2, ptr addrspace(1) %p3, ptr addrspace(1) %p4, ptr addrspace(1) %p5, i32 %TC1, i32 %TC2, i32 %TC3) {
; CHECK-LABEL: # Machine code for function test: IsSSA, TracksLiveness
; CHECK-NEXT: Function Live Ins: $vgpr0 in [[Reg1:%[0-9]+]], $vgpr1 in [[Reg2:%[0-9]+]], $vgpr2 in [[Reg3:%[0-9]+]], $vgpr3 in [[Reg4:%[0-9]+]], $vgpr4 in [[Reg5:%[0-9]+]], $vgpr5 in [[Reg6:%[0-9]+]], $vgpr6 in [[Reg7:%[0-9]+]], $vgpr7 in [[Reg8:%[0-9]+]], $vgpr8 in [[Reg9:%[0-9]+]], $vgpr9 in [[Reg10:%[0-9]+]], $vgpr10 in [[Reg11:%[0-9]+]], $vgpr11 in [[Reg12:%[0-9]+]]
; EMPTY:
; CHECK: bb.0.entry:
; CHECK-NEXT:   successors: %bb.1(0x80000000); %bb.1(100.00%)
; CHECK-NEXT:   liveins: $vgpr0, $vgpr1, $vgpr2, $vgpr3, $vgpr4, $vgpr5, $vgpr6, $vgpr7, $vgpr8, $vgpr9, $vgpr10, $vgpr11
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
; CHECK-NEXT:   [[Reg13:%[0-9]+]]:vreg_64 = REG_SEQUENCE killed [[Reg9]]:vgpr_32, %subreg.sub0, killed [[Reg10]]:vgpr_32, %subreg.sub1
; CHECK-NEXT:   [[Reg14:%[0-9]+]]:vreg_64 = REG_SEQUENCE killed [[Reg7]]:vgpr_32, %subreg.sub0, killed [[Reg8]]:vgpr_32, %subreg.sub1
; CHECK-NEXT:   [[Reg15:%[0-9]+]]:vreg_64 = REG_SEQUENCE killed [[Reg5]]:vgpr_32, %subreg.sub0, killed [[Reg6]]:vgpr_32, %subreg.sub1
; CHECK-NEXT:   [[Reg16:%[0-9]+]]:vreg_64 = REG_SEQUENCE killed [[Reg3]]:vgpr_32, %subreg.sub0, killed [[Reg4]]:vgpr_32, %subreg.sub1
; CHECK-NEXT:   [[Reg17:%[0-9]+]]:vreg_64 = REG_SEQUENCE killed [[Reg1]]:vgpr_32, %subreg.sub0, killed [[Reg2]]:vgpr_32, %subreg.sub1
; CHECK-NEXT:   [[Reg18:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg17]]:vreg_64, 0, 0, implicit $exec :: (load (s8) from %ir.p1, addrspace 1)
; CHECK-NEXT:   [[Reg19:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg17]]:vreg_64, 1, 0, implicit $exec :: (load (s8) from %ir.p1 + 1, addrspace 1)
; CHECK-NEXT:   [[Reg20:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg19]]:vgpr_32, 8, killed [[Reg18]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg21:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg17]]:vreg_64, 2, 0, implicit $exec :: (load (s8) from %ir.p1 + 2, addrspace 1)
; CHECK-NEXT:   [[Reg22:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE killed [[Reg17]]:vreg_64, 3, 0, implicit $exec :: (load (s8) from %ir.p1 + 3, addrspace 1)
; CHECK-NEXT:   [[Reg23:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg22]]:vgpr_32, 8, killed [[Reg21]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg24:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg23]]:vgpr_32, 16, killed [[Reg20]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg25:%[0-9]+]]:sreg_32 = S_MOV_B32 0
; EMPTY:
; CHECK: bb.1.loop1.header:
; CHECK-NEXT: ; predecessors: %bb.0, %bb.5
; CHECK-NEXT:   successors: %bb.2(0x80000000); %bb.2(100.00%)
; EMPTY:
; CHECK:   [[Reg26:%[0-9]+]]:sreg_32 = PHI [[Reg25]]:sreg_32, %bb.0, [[Reg27:%[0-9]+]]:sreg_32, %bb.5
; CHECK-NEXT:   [[Reg28:%[0-9]+]]:sreg_32 = PHI [[Reg25]]:sreg_32, %bb.0, [[Reg29:%[0-9]+]]:sreg_32, %bb.5
; CHECK-NEXT:   [[Reg30:%[0-9]+]]:vgpr_32 = PHI [[Reg24]]:vgpr_32, %bb.0, [[Reg31:%[0-9]+]]:vgpr_32, %bb.5
; CHECK-NEXT:   [[Reg32:%[0-9]+]]:vgpr_32 = V_MOV_B32_e32 13, implicit $exec
; EMPTY:
; CHECK: bb.2.loop2.header:
; CHECK-NEXT: ; predecessors: %bb.1, %bb.4
; CHECK-NEXT:   successors: %bb.3(0x80000000); %bb.3(100.00%)
; EMPTY:
; CHECK:   [[Reg33:%[0-9]+]]:sreg_32 = PHI [[Reg25]]:sreg_32, %bb.1, [[Reg34:%[0-9]+]]:sreg_32, %bb.4
; CHECK-NEXT:   [[Reg35:%[0-9]+]]:sreg_32 = PHI [[Reg25]]:sreg_32, %bb.1, [[Reg36:%[0-9]+]]:sreg_32, %bb.4
; CHECK-NEXT:   [[Reg37:%[0-9]+]]:vgpr_32 = PHI [[Reg32]]:vgpr_32, %bb.1, [[Reg38:%[0-9]+]]:vgpr_32, %bb.4
; CHECK-NEXT:   [[Reg39:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg16]]:vreg_64, 0, 0, implicit $exec :: (load (s8) from %ir.p2, addrspace 1)
; CHECK-NEXT:   [[Reg40:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg16]]:vreg_64, 1, 0, implicit $exec :: (load (s8) from %ir.p2 + 1, addrspace 1)
; CHECK-NEXT:   [[Reg41:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg40]]:vgpr_32, 8, killed [[Reg39]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg42:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg16]]:vreg_64, 2, 0, implicit $exec :: (load (s8) from %ir.p2 + 2, addrspace 1)
; CHECK-NEXT:   [[Reg43:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg16]]:vreg_64, 3, 0, implicit $exec :: (load (s8) from %ir.p2 + 3, addrspace 1)
; CHECK-NEXT:   [[Reg44:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg43]]:vgpr_32, 8, killed [[Reg42]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg45:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg44]]:vgpr_32, 16, killed [[Reg41]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg46:%[0-9]+]]:vgpr_32 = V_ADD_U32_e64 [[Reg35]]:sreg_32, [[Reg45]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   GLOBAL_STORE_BYTE_D16_HI [[Reg14]]:vreg_64, [[Reg46]]:vgpr_32, 2, 0, implicit $exec :: (store (s8) into %ir.p4 + 2, addrspace 1)
; CHECK-NEXT:   GLOBAL_STORE_BYTE [[Reg14]]:vreg_64, [[Reg46]]:vgpr_32, 0, 0, implicit $exec :: (store (s8) into %ir.p4, addrspace 1)
; CHECK-NEXT:   [[Reg47:%[0-9]+]]:vgpr_32 = V_LSHRREV_B32_e64 24, [[Reg46]]:vgpr_32, implicit $exec
; CHECK-NEXT:   GLOBAL_STORE_BYTE [[Reg14]]:vreg_64, killed [[Reg47]]:vgpr_32, 3, 0, implicit $exec :: (store (s8) into %ir.p4 + 3, addrspace 1)
; CHECK-NEXT:   [[Reg48:%[0-9]+]]:vgpr_32 = V_LSHRREV_B32_e64 8, killed [[Reg46]]:vgpr_32, implicit $exec
; CHECK-NEXT:   GLOBAL_STORE_BYTE [[Reg14]]:vreg_64, killed [[Reg48]]:vgpr_32, 1, 0, implicit $exec :: (store (s8) into %ir.p4 + 1, addrspace 1)
; CHECK-NEXT:   [[Reg49:%[0-9]+]]:sreg_32 = S_MOV_B32 0
; EMPTY:
; CHECK: bb.3.loop3:
; CHECK-NEXT: ; predecessors: %bb.2, %bb.3
; CHECK-NEXT:   successors: %bb.4(0x04000000), %bb.3(0x7c000000); %bb.4(3.12%), %bb.3(96.88%)
; EMPTY:
; CHECK:   [[Reg50:%[0-9]+]]:sreg_32 = PHI [[Reg49]]:sreg_32, %bb.2, [[Reg51:%[0-9]+]]:sreg_32, %bb.3
; CHECK-NEXT:   [[Reg52:%[0-9]+]]:sreg_32 = PHI [[Reg49]]:sreg_32, %bb.2, [[Reg53:%[0-9]+]]:sreg_32, %bb.3
; CHECK-NEXT:   [[Reg53]]:sreg_32 = S_ADD_I32 killed [[Reg52]]:sreg_32, 3, implicit-def dead $scc
; CHECK-NEXT:   [[Reg54:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg15]]:vreg_64, 0, 0, implicit $exec :: (load (s8) from %ir.p3, addrspace 1)
; CHECK-NEXT:   [[Reg55:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg15]]:vreg_64, 1, 0, implicit $exec :: (load (s8) from %ir.p3 + 1, addrspace 1)
; CHECK-NEXT:   [[Reg56:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg55]]:vgpr_32, 8, killed [[Reg54]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg57:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg15]]:vreg_64, 2, 0, implicit $exec :: (load (s8) from %ir.p3 + 2, addrspace 1)
; CHECK-NEXT:   [[Reg58:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg15]]:vreg_64, 3, 0, implicit $exec :: (load (s8) from %ir.p3 + 3, addrspace 1)
; CHECK-NEXT:   [[Reg59:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg58]]:vgpr_32, 8, killed [[Reg57]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg60:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg59]]:vgpr_32, 16, killed [[Reg56]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg61:%[0-9]+]]:vgpr_32 = V_ADD_U32_e64 [[Reg53]]:sreg_32, [[Reg60]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   GLOBAL_STORE_BYTE_D16_HI [[Reg13]]:vreg_64, [[Reg61]]:vgpr_32, 2, 0, implicit $exec :: (store (s8) into %ir.p5 + 2, addrspace 1)
; CHECK-NEXT:   GLOBAL_STORE_BYTE [[Reg13]]:vreg_64, [[Reg61]]:vgpr_32, 0, 0, implicit $exec :: (store (s8) into %ir.p5, addrspace 1)
; CHECK-NEXT:   [[Reg62:%[0-9]+]]:vgpr_32 = V_LSHRREV_B32_e64 24, [[Reg61]]:vgpr_32, implicit $exec
; CHECK-NEXT:   GLOBAL_STORE_BYTE [[Reg13]]:vreg_64, killed [[Reg62]]:vgpr_32, 3, 0, implicit $exec :: (store (s8) into %ir.p5 + 3, addrspace 1)
; CHECK-NEXT:   [[Reg63:%[0-9]+]]:vgpr_32 = V_LSHRREV_B32_e64 8, killed [[Reg61]]:vgpr_32, implicit $exec
; CHECK-NEXT:   GLOBAL_STORE_BYTE [[Reg13]]:vreg_64, killed [[Reg63]]:vgpr_32, 1, 0, implicit $exec :: (store (s8) into %ir.p5 + 1, addrspace 1)
; CHECK-NEXT:   [[Reg64:%[0-9]+]]:sreg_32 = V_CMP_GE_U32_e64 [[Reg53]]:sreg_32, [[Reg11]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg51]]:sreg_32 = SI_IF_BREAK killed [[Reg64]]:sreg_32, killed [[Reg50]]:sreg_32, implicit-def dead $scc
; CHECK-NEXT:   [[Reg65:%[0-9]+]]:vgpr_32 = COPY [[Reg53]]:sreg_32, implicit $exec
; CHECK-NEXT:   SI_LOOP [[Reg51]]:sreg_32, %bb.3, implicit-def dead $exec, implicit-def dead $scc, implicit $exec
; CHECK-NEXT:   S_BRANCH %bb.4
; EMPTY:
; CHECK: bb.4.loop2.latch:
; CHECK-NEXT: ; predecessors: %bb.3
; CHECK-NEXT:   successors: %bb.5(0x04000000), %bb.2(0x7c000000); %bb.5(3.12%), %bb.2(96.88%)
; EMPTY:
; CHECK:   SI_END_CF killed [[Reg51]]:sreg_32, implicit-def dead $exec, implicit-def dead $scc, implicit $exec
; CHECK-NEXT:   [[Reg66:%[0-9]+]]:vgpr_32 = V_ADD_U32_e64 killed [[Reg60]]:vgpr_32, [[Reg65]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg38]]:vgpr_32 = V_MUL_LO_U32_e64 [[Reg30]]:vgpr_32, [[Reg66]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg36]]:sreg_32 = S_ADD_I32 killed [[Reg35]]:sreg_32, 2, implicit-def dead $scc
; CHECK-NEXT:   [[Reg67:%[0-9]+]]:sreg_32 = V_CMP_GE_U32_e64 [[Reg36]]:sreg_32, [[Reg12]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg34]]:sreg_32 = SI_IF_BREAK killed [[Reg67]]:sreg_32, killed [[Reg33]]:sreg_32, implicit-def dead $scc
; CHECK-NEXT:   SI_LOOP [[Reg34]]:sreg_32, %bb.2, implicit-def dead $exec, implicit-def dead $scc, implicit $exec
; CHECK-NEXT:   S_BRANCH %bb.5
; EMPTY:
; CHECK: bb.5.loop1.latch:
; CHECK-NEXT: ; predecessors: %bb.4
; CHECK-NEXT:   successors: %bb.6(0x04000000), %bb.1(0x7c000000); %bb.6(3.12%), %bb.1(96.88%)
; EMPTY:
; CHECK:   SI_END_CF killed [[Reg34]]:sreg_32, implicit-def dead $exec, implicit-def dead $scc, implicit $exec
; CHECK-NEXT:   [[Reg31]]:vgpr_32 = V_SUB_U32_e64 killed [[Reg45]]:vgpr_32, killed [[Reg65]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg29]]:sreg_32 = S_ADD_I32 killed [[Reg28]]:sreg_32, 1, implicit-def dead $scc
; CHECK-NEXT:   [[Reg68:%[0-9]+]]:sreg_32 = V_CMP_GE_U32_e64 [[Reg29]]:sreg_32, [[Reg11]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg27]]:sreg_32 = SI_IF_BREAK killed [[Reg68]]:sreg_32, killed [[Reg26]]:sreg_32, implicit-def dead $scc
; CHECK-NEXT:   SI_LOOP [[Reg27]]:sreg_32, %bb.1, implicit-def dead $exec, implicit-def dead $scc, implicit $exec
; CHECK-NEXT:   S_BRANCH %bb.6
; EMPTY:
; CHECK: bb.6.exit:
; CHECK-NEXT: ; predecessors: %bb.5
; EMPTY:
; CHECK:   SI_END_CF killed [[Reg27]]:sreg_32, implicit-def dead $exec, implicit-def dead $scc, implicit $exec
; CHECK-NEXT:   [[Reg69:%[0-9]+]]:vgpr_32 = V_MAX_U32_e64 1, killed [[Reg11]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg70:%[0-9]+]]:vgpr_32 = V_ADD_U32_e64 killed [[Reg69]]:vgpr_32, killed [[Reg24]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg71:%[0-9]+]]:vgpr_32 = V_ADD3_U32_e64 [[Reg37]]:vgpr_32, killed [[Reg30]]:vgpr_32, [[Reg66]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg72:%[0-9]+]]:vgpr_32 = V_ADD_U32_e64 killed [[Reg66]]:vgpr_32, killed [[Reg38]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg73:%[0-9]+]]:vgpr_32 = V_ADD3_U32_e64 killed [[Reg72]]:vgpr_32, killed [[Reg71]]:vgpr_32, killed [[Reg37]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg74:%[0-9]+]]:vgpr_32 = V_ADD3_U32_e64 killed [[Reg70]]:vgpr_32, killed [[Reg73]]:vgpr_32, -1, implicit $exec
; CHECK-NEXT:   [[Reg75:%[0-9]+]]:sreg_32_xm0 = V_READFIRSTLANE_B32 killed [[Reg74]]:vgpr_32, implicit $exec
; CHECK-NEXT:   $sgpr0 = COPY killed [[Reg75]]:sreg_32_xm0
; CHECK-NEXT:   SI_RETURN_TO_EPILOG killed $sgpr0
; EMPTY:
; CHECK: # End machine code for function test.
; EMPTY:
; CHECK: Next-use distance of Register [[Reg12]] = 22051.0
; CHECK-NEXT: Next-use distance of Register [[Reg11]] = 63.0
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
; CHECK-NEXT: Next-use distance of Register [[Reg13]] = 46.0
; CHECK-NEXT: Next-use distance of Register [[Reg14]] = 27.0
; CHECK-NEXT: Next-use distance of Register [[Reg15]] = 36.0
; CHECK-NEXT: Next-use distance of Register [[Reg16]] = 17.0
; CHECK-NEXT: Next-use distance of Register [[Reg17]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg18]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg19]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg20]] = 4.0
; CHECK-NEXT: Next-use distance of Register [[Reg21]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg22]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg23]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg24]] = 4.0
; CHECK-NEXT: Next-use distance of Register [[Reg25]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg26]] = 22026008.0
; CHECK-NEXT: Next-use distance of Register [[Reg28]] = 22026005.0
; CHECK-NEXT: Next-use distance of Register [[Reg30]] = 22022.0
; CHECK-NEXT: Next-use distance of Register [[Reg32]] = 3.0
; CHECK-NEXT: Next-use distance of Register [[Reg33]] = 22023.0
; CHECK-NEXT: Next-use distance of Register [[Reg35]] = 9.0
; CHECK-NEXT: Next-use distance of Register [[Reg37]] = 22026011004.0
; CHECK-NEXT: Next-use distance of Register [[Reg39]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg40]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg41]] = 4.0
; CHECK-NEXT: Next-use distance of Register [[Reg42]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg43]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg44]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg45]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg46]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg47]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg48]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg49]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg50]] = 18.0
; CHECK-NEXT: Next-use distance of Register [[Reg52]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg53]] = 8.0
; CHECK-NEXT: Next-use distance of Register [[Reg54]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg55]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg56]] = 4.0
; CHECK-NEXT: Next-use distance of Register [[Reg57]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg58]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg59]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg60]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg61]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg62]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg63]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg64]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg51]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg65]] = 22002.0
; CHECK-NEXT: Next-use distance of Register [[Reg66]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg38]] = 8.0
; CHECK-NEXT: Next-use distance of Register [[Reg36]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg67]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg34]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg31]] = 8.0
; CHECK-NEXT: Next-use distance of Register [[Reg29]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg68]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg27]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg69]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg70]] = 4.0
; CHECK-NEXT: Next-use distance of Register [[Reg71]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg72]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg73]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg74]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg75]] = 1.0
entry:
;              entry
;                |
;           loop1.header<-------+
;                |              |
;           loop2.header<-----+ |
;                |            | |
;              loop3<-------+ | |
;                +----------+ | |
;                |            | |
;           loop2.latch-------+ |
;                |              |
;           loop1.latch---------+
;                |
;               exit
   %ld1 = load i32, ptr addrspace(1) %p1, align 1
   br label %loop1.header

loop1.header:
   %phi.inc1 = phi i32 [ 0, %entry ], [ %inc1, %loop1.latch ]
   %phi1 = phi i32 [ %ld1, %entry ], [ %sub, %loop1.latch ]
   %add1 = add i32 %ld1, %phi.inc1
   br label %loop2.header

loop2.header:
   %phi.inc2 = phi i32 [ 0, %loop1.header ], [ %inc2, %loop2.latch ]
   %phi2 = phi i32 [ 13, %loop1.header ], [ %mul, %loop2.latch ]
   %ld2 = load i32, ptr addrspace(1) %p2, align 1
   %add2 = add i32 %ld2, %phi.inc2
   store i32 %add2, ptr addrspace(1) %p4, align 1
   br label %loop3

loop3:
   %phi.inc3 = phi i32 [ 0, %loop2.header ], [ %inc3, %loop3 ]
   %inc3 = add i32 %phi.inc3, 3
   %sub = sub i32 %ld2, %inc3
   %ld3 = load i32, ptr addrspace(1) %p3, align 1
   %add3 = add i32 %ld3, %inc3
   store i32 %add3, ptr addrspace(1) %p5, align 1
   %cond3 = icmp ult i32 %inc3, %TC1
   br i1 %cond3, label %loop3, label %loop2.latch

loop2.latch:
   %mul = mul i32 %phi1, %add3
   %inc2 = add i32 %phi.inc2, 2
   %cond2 = icmp ult i32 %inc2, %TC2
   br i1 %cond2, label %loop2.header, label %loop1.latch

loop1.latch:
   %add4 = add i32 %phi2, %phi1
   %add5 = add i32 %add3, %add4
   %inc1 = add i32 %phi.inc1, 1
   %cond1 = icmp ult i32 %inc1, %TC1
   br i1 %cond1, label %loop1.header, label %exit

exit:
   %add6 = add i32 %add3, %mul
   %add7 = add i32 %add6, %add5
   %add8 = add i32 %add7, %phi2
   %add9 = add i32 %add8, %add1
   ret i32 %add9
}
