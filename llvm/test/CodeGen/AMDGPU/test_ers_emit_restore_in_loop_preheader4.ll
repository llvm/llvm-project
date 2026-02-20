; RUN: llc -mtriple=amdgcn -mcpu=gfx1200 -enable-next-use-analysis=true -verify-machineinstrs -dump-next-use-distance < %s 2>&1 | FileCheck %s

;
;       bb.0.entry
;           |
;           +<--------+
;       bb.1.loop1    |
;           +---------+
;           |
;        bb.2.bb
;           |
;           +<--------+
;       bb.3.loop2    |
;           +---------+
;           |
;       bb.4.exit
;
define amdgpu_ps i32 @test(ptr addrspace(1) %p1, ptr addrspace(1) %p2, ptr addrspace(1) %p3, ptr addrspace(1) %p4, i32 %TC1, i32 %TC2) {
; CHECK-LABEL: # Machine code for function test: IsSSA, TracksLiveness
; CHECK-NEXT: Function Live Ins: $vgpr0 in [[Reg1:%[0-9]+]], $vgpr1 in [[Reg2:%[0-9]+]], $vgpr2 in [[Reg3:%[0-9]+]], $vgpr3 in [[Reg4:%[0-9]+]], $vgpr4 in [[Reg5:%[0-9]+]], $vgpr5 in [[Reg6:%[0-9]+]], $vgpr6 in [[Reg7:%[0-9]+]], $vgpr7 in [[Reg8:%[0-9]+]], $vgpr8 in [[Reg9:%[0-9]+]], $vgpr9 in [[Reg10:%[0-9]+]]
; EMPTY:
; CHECK: bb.0.entry:
; CHECK-NEXT:   successors: %bb.1(0x80000000); %bb.1(100.00%)
; CHECK-NEXT:   liveins: $vgpr0, $vgpr1, $vgpr2, $vgpr3, $vgpr4, $vgpr5, $vgpr6, $vgpr7, $vgpr8, $vgpr9
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
; CHECK-NEXT:   [[Reg11:%[0-9]+]]:vreg_64 = REG_SEQUENCE killed [[Reg5]]:vgpr_32, %subreg.sub0, killed [[Reg6]]:vgpr_32, %subreg.sub1
; CHECK-NEXT:   [[Reg12:%[0-9]+]]:vreg_64 = REG_SEQUENCE killed [[Reg3]]:vgpr_32, %subreg.sub0, killed [[Reg4]]:vgpr_32, %subreg.sub1
; CHECK-NEXT:   [[Reg13:%[0-9]+]]:vreg_64 = REG_SEQUENCE killed [[Reg1]]:vgpr_32, %subreg.sub0, killed [[Reg2]]:vgpr_32, %subreg.sub1
; CHECK-NEXT:   [[Reg14:%[0-9]+]]:vreg_64 = REG_SEQUENCE killed [[Reg7]]:vgpr_32, %subreg.sub0, killed [[Reg8]]:vgpr_32, %subreg.sub1
; CHECK-NEXT:   [[Reg15:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_USHORT [[Reg14]]:vreg_64, 0, 0, implicit $exec :: (load (s16) from %ir.p4, addrspace 1)
; CHECK-NEXT:   [[Reg16:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_USHORT [[Reg14]]:vreg_64, 2, 0, implicit $exec :: (load (s16) from %ir.p4 + 2, addrspace 1)
; CHECK-NEXT:   [[Reg17:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg16]]:vgpr_32, 16, killed [[Reg15]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg18:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg13]]:vreg_64, 0, 0, implicit $exec :: (load (s8) from %ir.p1, addrspace 1)
; CHECK-NEXT:   [[Reg19:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg13]]:vreg_64, 1, 0, implicit $exec :: (load (s8) from %ir.p1 + 1, addrspace 1)
; CHECK-NEXT:   [[Reg20:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg19]]:vgpr_32, 8, killed [[Reg18]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg21:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg13]]:vreg_64, 2, 0, implicit $exec :: (load (s8) from %ir.p1 + 2, addrspace 1)
; CHECK-NEXT:   [[Reg22:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg13]]:vreg_64, 3, 0, implicit $exec :: (load (s8) from %ir.p1 + 3, addrspace 1)
; CHECK-NEXT:   [[Reg23:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg22]]:vgpr_32, 8, killed [[Reg21]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg24:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg23]]:vgpr_32, 16, killed [[Reg20]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg25:%[0-9]+]]:vgpr_32 = V_MOV_B32_e32 100, implicit $exec
; CHECK-NEXT:   [[Reg26:%[0-9]+]]:vgpr_32 = V_ADD_U32_e64 100, [[Reg24]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   GLOBAL_STORE_DWORD killed [[Reg14]]:vreg_64, [[Reg26]]:vgpr_32, 0, 0, implicit $exec :: (store (s32) into %ir.p4, addrspace 1)
; CHECK-NEXT:   [[Reg27:%[0-9]+]]:sreg_32 = S_MOV_B32 1
; CHECK-NEXT:   [[Reg28:%[0-9]+]]:sreg_32 = S_MOV_B32 0
; EMPTY:
; CHECK: bb.1.loop1:
; CHECK-NEXT: ; predecessors: %bb.0, %bb.1
; CHECK-NEXT:   successors: %bb.2(0x04000000), %bb.1(0x7c000000); %bb.2(3.12%), %bb.1(96.88%)
; EMPTY:
; CHECK:   [[Reg29:%[0-9]+]]:sreg_32 = PHI [[Reg28]]:sreg_32, %bb.0, [[Reg30:%[0-9]+]]:sreg_32, %bb.1
; CHECK-NEXT:   [[Reg31:%[0-9]+]]:sreg_32 = PHI [[Reg28]]:sreg_32, %bb.0, [[Reg32:%[0-9]+]]:sreg_32, %bb.1
; CHECK-NEXT:   [[Reg33:%[0-9]+]]:sreg_32 = PHI [[Reg27]]:sreg_32, %bb.0, [[Reg34:%[0-9]+]]:sreg_32, %bb.1
; CHECK-NEXT:   [[Reg35:%[0-9]+]]:vgpr_32 = PHI [[Reg24]]:vgpr_32, %bb.0, [[Reg36:%[0-9]+]]:vgpr_32, %bb.1
; CHECK-NEXT:   [[Reg37:%[0-9]+]]:vgpr_32 = PHI [[Reg25]]:vgpr_32, %bb.0, [[Reg38:%[0-9]+]]:vgpr_32, %bb.1
; CHECK-NEXT:   [[Reg39:%[0-9]+]]:vgpr_32 = PHI [[Reg24]]:vgpr_32, %bb.0, [[Reg40:%[0-9]+]]:vgpr_32, %bb.1
; CHECK-NEXT:   [[Reg41:%[0-9]+]]:sreg_32 = S_ADD_I32 [[Reg33]]:sreg_32, -1, implicit-def dead $scc
; CHECK-NEXT:   [[Reg42:%[0-9]+]]:sreg_32_xm0 = S_ASHR_I32 [[Reg41]]:sreg_32, 31, implicit-def dead $scc
; CHECK-NEXT:   [[Reg43:%[0-9]+]]:sreg_64 = REG_SEQUENCE killed [[Reg41]]:sreg_32, %subreg.sub0, killed [[Reg42]]:sreg_32_xm0, %subreg.sub1
; CHECK-NEXT:   [[Reg44:%[0-9]+]]:sreg_64 = nsw S_LSHL_B64 killed [[Reg43]]:sreg_64, 2, implicit-def dead $scc
; CHECK-NEXT:   [[Reg45:%[0-9]+]]:vgpr_32, [[Reg46:%[0-9]+]]:sreg_32_xm0_xexec = V_ADD_CO_U32_e64 [[Reg12]].sub0:vreg_64, [[Reg44]].sub0:sreg_64, 0, implicit $exec
; CHECK-NEXT:   [[Reg47:%[0-9]+]]:vgpr_32, dead $sgpr_null = V_ADDC_U32_e64 [[Reg44]].sub1:sreg_64, [[Reg12]].sub1:vreg_64, killed [[Reg46]]:sreg_32_xm0_xexec, 0, implicit $exec
; CHECK-NEXT:   [[Reg48:%[0-9]+]]:vreg_64 = REG_SEQUENCE killed [[Reg45]]:vgpr_32, %subreg.sub0, killed [[Reg47]]:vgpr_32, %subreg.sub1
; CHECK-NEXT:   [[Reg49:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_DWORD killed [[Reg48]]:vreg_64, 0, 0, implicit $exec :: (load (s32) from %ir.gep1, addrspace 1)
; CHECK-NEXT:   [[Reg36]]:vgpr_32 = V_ADD_U32_e64 [[Reg33]]:sreg_32, [[Reg49]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg38]]:vgpr_32 = V_MUL_LO_U32_e64 [[Reg49]]:vgpr_32, [[Reg33]]:sreg_32, implicit $exec
; CHECK-NEXT:   GLOBAL_STORE_SHORT_D16_HI [[Reg13]]:vreg_64, [[Reg38]]:vgpr_32, 2, 0, implicit $exec :: (store (s16) into %ir.p1 + 2, addrspace 1)
; CHECK-NEXT:   GLOBAL_STORE_SHORT [[Reg13]]:vreg_64, [[Reg38]]:vgpr_32, 0, 0, implicit $exec :: (store (s16) into %ir.p1, addrspace 1)
; CHECK-NEXT:   [[Reg50:%[0-9]+]]:vgpr_32, [[Reg51:%[0-9]+]]:sreg_32_xm0_xexec = V_ADD_CO_U32_e64 [[Reg11]].sub0:vreg_64, [[Reg44]].sub0:sreg_64, 0, implicit $exec
; CHECK-NEXT:   [[Reg52:%[0-9]+]]:vgpr_32, dead $sgpr_null = V_ADDC_U32_e64 killed [[Reg44]].sub1:sreg_64, [[Reg11]].sub1:vreg_64, killed [[Reg51]]:sreg_32_xm0_xexec, 0, implicit $exec
; CHECK-NEXT:   [[Reg53:%[0-9]+]]:vreg_64 = REG_SEQUENCE killed [[Reg50]]:vgpr_32, %subreg.sub0, killed [[Reg52]]:vgpr_32, %subreg.sub1
; CHECK-NEXT:   [[Reg54:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_USHORT [[Reg53]]:vreg_64, 0, 0, implicit $exec :: (load (s16) from %ir.gep2, addrspace 1)
; CHECK-NEXT:   [[Reg55:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_USHORT killed [[Reg53]]:vreg_64, 2, 0, implicit $exec :: (load (s16) from %ir.gep2 + 2, addrspace 1)
; CHECK-NEXT:   [[Reg56:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg55]]:vgpr_32, 16, killed [[Reg54]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg34]]:sreg_32 = S_ADD_I32 [[Reg33]]:sreg_32, 1, implicit-def dead $scc
; CHECK-NEXT:   [[Reg32]]:sreg_32 = S_ADD_I32 [[Reg31]]:sreg_32, -1, implicit-def dead $scc
; CHECK-NEXT:   [[Reg40]]:vgpr_32 = V_ADD_U32_e64 killed [[Reg31]]:sreg_32, killed [[Reg56]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg57:%[0-9]+]]:sreg_32 = V_CMP_GE_U32_e64 killed [[Reg33]]:sreg_32, [[Reg9]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg30]]:sreg_32 = SI_IF_BREAK killed [[Reg57]]:sreg_32, killed [[Reg29]]:sreg_32, implicit-def dead $scc
; CHECK-NEXT:   [[Reg58:%[0-9]+]]:vgpr_32 = COPY [[Reg34]]:sreg_32, implicit $exec
; CHECK-NEXT:   SI_LOOP [[Reg30]]:sreg_32, %bb.1, implicit-def dead $exec, implicit-def dead $scc, implicit $exec
; CHECK-NEXT:   S_BRANCH %bb.2
; EMPTY:
; CHECK: bb.2.bb:
; CHECK-NEXT: ; predecessors: %bb.1
; CHECK-NEXT:   successors: %bb.3(0x80000000); %bb.3(100.00%)
; EMPTY:
; CHECK:   SI_END_CF killed [[Reg30]]:sreg_32, implicit-def dead $exec, implicit-def dead $scc, implicit $exec
; CHECK-NEXT:   [[Reg59:%[0-9]+]]:vgpr_32 = V_ADD_U32_e64 -2, killed [[Reg58]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg60:%[0-9]+]]:vgpr_32 = V_MUL_LO_U32_e64 killed [[Reg38]]:vgpr_32, killed [[Reg59]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg61:%[0-9]+]]:vgpr_32 = V_MUL_LO_U32_e64 100, killed [[Reg35]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg62:%[0-9]+]]:vgpr_32 = V_MUL_LO_U32_e64 [[Reg61]]:vgpr_32, [[Reg17]]:vgpr_32, implicit $exec
; CHECK-NEXT:   GLOBAL_STORE_DWORD [[Reg11]]:vreg_64, killed [[Reg62]]:vgpr_32, 0, 0, implicit $exec :: (store (s32) into %ir.p3, addrspace 1)
; CHECK-NEXT:   [[Reg63:%[0-9]+]]:sreg_32 = S_MOV_B32 0
; EMPTY:
; CHECK: bb.3.loop2:
; CHECK-NEXT: ; predecessors: %bb.2, %bb.3
; CHECK-NEXT:   successors: %bb.4(0x04000000), %bb.3(0x7c000000); %bb.4(3.12%), %bb.3(96.88%)
; EMPTY:
; CHECK:   [[Reg64:%[0-9]+]]:sreg_32 = PHI [[Reg63]]:sreg_32, %bb.2, [[Reg65:%[0-9]+]]:sreg_32, %bb.3
; CHECK-NEXT:   [[Reg66:%[0-9]+]]:vgpr_32 = PHI [[Reg17]]:vgpr_32, %bb.2, [[Reg67:%[0-9]+]]:vgpr_32, %bb.3
; CHECK-NEXT:   [[Reg68:%[0-9]+]]:vgpr_32 = PHI [[Reg39]]:vgpr_32, %bb.2, [[Reg60]]:vgpr_32, %bb.3
; CHECK-NEXT:   [[Reg67]]:vgpr_32 = V_ADD_U32_e64 2, killed [[Reg66]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg69:%[0-9]+]]:sreg_32 = V_CMP_GE_U32_e64 [[Reg67]]:vgpr_32, [[Reg10]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg65]]:sreg_32 = SI_IF_BREAK killed [[Reg69]]:sreg_32, killed [[Reg64]]:sreg_32, implicit-def dead $scc
; CHECK-NEXT:   SI_LOOP [[Reg65]]:sreg_32, %bb.3, implicit-def dead $exec, implicit-def dead $scc, implicit $exec
; CHECK-NEXT:   S_BRANCH %bb.4
; EMPTY:
; CHECK: bb.4.exit:
; CHECK-NEXT: ; predecessors: %bb.3
; EMPTY:
; CHECK:   SI_END_CF killed [[Reg65]]:sreg_32, implicit-def dead $exec, implicit-def dead $scc, implicit $exec
; CHECK-NEXT:   GLOBAL_STORE_DWORD killed [[Reg11]]:vreg_64, [[Reg68]]:vgpr_32, 0, 0, implicit $exec :: (store (s32) into %ir.p3, addrspace 1)
; CHECK-NEXT:   [[Reg70:%[0-9]+]]:vgpr_32 = V_ADD_U32_e64 killed [[Reg68]]:vgpr_32, killed [[Reg67]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg71:%[0-9]+]]:vgpr_32 = V_ADD3_U32_e64 killed [[Reg70]]:vgpr_32, killed [[Reg37]]:vgpr_32, killed [[Reg61]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg72:%[0-9]+]]:vgpr_32 = V_ADD3_U32_e64 killed [[Reg71]]:vgpr_32, killed [[Reg49]]:vgpr_32, killed [[Reg26]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg73:%[0-9]+]]:sreg_32_xm0 = V_READFIRSTLANE_B32 killed [[Reg72]]:vgpr_32, implicit $exec
; CHECK-NEXT:   $sgpr0 = COPY killed [[Reg73]]:sreg_32_xm0
; CHECK-NEXT:   SI_RETURN_TO_EPILOG killed $sgpr0
; EMPTY:
; CHECK: # End machine code for function test.
; EMPTY:
; CHECK: Next-use distance of Register [[Reg10]] = 32040.0
; CHECK-NEXT: Next-use distance of Register [[Reg9]] = 55.0
; CHECK-NEXT: Next-use distance of Register [[Reg8]] = 11.0
; CHECK-NEXT: Next-use distance of Register [[Reg7]] = 10.0
; CHECK-NEXT: Next-use distance of Register [[Reg6]] = 6.0
; CHECK-NEXT: Next-use distance of Register [[Reg5]] = 5.0
; CHECK-NEXT: Next-use distance of Register [[Reg4]] = 5.0
; CHECK-NEXT: Next-use distance of Register [[Reg3]] = 4.0
; CHECK-NEXT: Next-use distance of Register [[Reg2]] = 4.0
; CHECK-NEXT: Next-use distance of Register [[Reg1]] = 3.0
; CHECK-NEXT: Next-use distance of Register [[Reg11]] = 37.0
; CHECK-NEXT: Next-use distance of Register [[Reg12]] = 28.0
; CHECK-NEXT: Next-use distance of Register [[Reg13]] = 5.0
; CHECK-NEXT: Next-use distance of Register [[Reg14]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg15]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg16]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg17]] = 32017.0
; CHECK-NEXT: Next-use distance of Register [[Reg18]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg19]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg20]] = 4.0
; CHECK-NEXT: Next-use distance of Register [[Reg21]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg22]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg23]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg24]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg25]] = 9.0
; CHECK-NEXT: Next-use distance of Register [[Reg26]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg27]] = 4.0
; CHECK-NEXT: Next-use distance of Register [[Reg28]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg29]] = 28.0
; CHECK-NEXT: Next-use distance of Register [[Reg31]] = 24.0
; CHECK-NEXT: Next-use distance of Register [[Reg33]] = 4.0
; CHECK-NEXT: Next-use distance of Register [[Reg35]] = 32004.0
; CHECK-NEXT: Next-use distance of Register [[Reg37]] = 40011.0
; CHECK-NEXT: Next-use distance of Register [[Reg39]] = 32010.0
; CHECK-NEXT: Next-use distance of Register [[Reg41]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg42]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg43]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg44]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg45]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg46]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg47]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg48]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg49]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg36]] = 21.0
; CHECK-NEXT: Next-use distance of Register [[Reg38]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg50]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg51]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg52]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg53]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg54]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg55]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg56]] = 3.0
; CHECK-NEXT: Next-use distance of Register [[Reg34]] = 5.0
; CHECK-NEXT: Next-use distance of Register [[Reg32]] = 8.0
; CHECK-NEXT: Next-use distance of Register [[Reg40]] = 11.0
; CHECK-NEXT: Next-use distance of Register [[Reg57]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg30]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg58]] = 32002.0
; CHECK-NEXT: Next-use distance of Register [[Reg59]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg60]] = 7.0
; CHECK-NEXT: Next-use distance of Register [[Reg61]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg62]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg63]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg64]] = 5.0
; CHECK-NEXT: Next-use distance of Register [[Reg66]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg68]] = 8002.0
; CHECK-NEXT: Next-use distance of Register [[Reg67]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg69]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg65]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg70]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg71]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg72]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg73]] = 1.0
entry:
;     entry
;       |
;       +<-----+
;     loop1    |
;       +------+
;       |
;       bb
;       |
;       +<-----+
;     loop2    |
;       +------+
;       |
;      exit
   %ld0 = load i32, ptr addrspace(1) %p4, align 2
   %ld1 = load i32, ptr addrspace(1) %p1, align 1
   %add1 = add i32 %ld1, 100
   store i32 %add1, ptr addrspace(1) %p4, align 4
   br label %loop1

loop1:
   %phi.inc1 = phi i32 [ 0, %entry ], [ %inc1, %loop1 ]
   %phi1 = phi i32 [ %ld1, %entry ], [ %add2, %loop1 ]
   %phi2 = phi i32 [ 100, %entry ], [ %mul1, %loop1 ]
   %phi3 = phi i32 [ %ld1, %entry ], [ %sub, %loop1 ]
   %sext1 = sext i32 %phi.inc1 to i64
   %gep1 = getelementptr inbounds i32, ptr addrspace(1) %p2, i64 %sext1
   %ld2 = load i32, ptr addrspace(1) %gep1, align 4
   %inc1 = add i32 %phi.inc1, 1
   %add2 = add i32 %ld2, %inc1
   %mul1 = mul i32 %ld2, %inc1
   store i32 %mul1, ptr addrspace(1) %p1, align 2
   %mul2 = mul i32 %mul1, %phi.inc1
   %sext2 = sext i32 %inc1 to i64
   %gep2 = getelementptr inbounds i32, ptr addrspace(1) %p3, i64 %sext1
   %ld3 = load i32, ptr addrspace(1) %gep2, align 2
   %sub =  sub i32 %ld3, %phi.inc1
   %cond1 = icmp ult i32 %inc1, %TC1
   br i1 %cond1, label %loop1, label %bb

bb:
   %mul3 = mul i32 %phi1, 100
   %mul4 = mul i32 %mul3, %ld0
   store i32 %mul4, ptr addrspace(1) %p3
   br label %loop2

loop2:
   %phi.inc2 = phi i32 [ %ld0, %bb ], [ %inc2, %loop2 ]
   %phi4 = phi i32 [ %phi3, %bb ], [ %mul2, %loop2 ]
   %inc2 = add i32 %phi.inc2, 2
   store i32 %phi4, ptr addrspace(1) %p3
   %add3 = add i32 %phi4, %inc2
   %cond2 = icmp ult i32 %inc2, %TC2
   br i1 %cond2, label %loop2, label %exit

exit:
   %add4 = add i32 %add3, %phi2
   %add5 = add i32 %add4, %mul3
   %add6 = add i32 %add5, %ld2
   %add7 = add i32 %add6, %add1
   ret i32 %add7
}

