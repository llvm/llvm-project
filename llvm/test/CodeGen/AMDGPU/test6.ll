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
define amdgpu_ps i32 @test6(ptr addrspace(1) %p1, ptr addrspace(1) %p2, ptr addrspace(1) %p3, i32 %TC1, i32 %TC2) {
; CHECK-LABEL: # Machine code for function test6: IsSSA, TracksLiveness
; CHECK-NEXT: Function Live Ins: $vgpr0 in [[Reg1:%[0-9]+]], $vgpr1 in [[Reg2:%[0-9]+]], $vgpr2 in [[Reg3:%[0-9]+]], $vgpr3 in [[Reg4:%[0-9]+]], $vgpr4 in [[Reg5:%[0-9]+]], $vgpr5 in [[Reg6:%[0-9]+]], $vgpr6 in [[Reg7:%[0-9]+]], $vgpr7 in [[Reg8:%[0-9]+]]
; EMPTY:
; CHECK: bb.0.entry:
; CHECK-NEXT:   successors: %bb.1(0x80000000); %bb.1(100.00%)
; CHECK-NEXT:   liveins: $vgpr0, $vgpr1, $vgpr2, $vgpr3, $vgpr4, $vgpr5, $vgpr6, $vgpr7
; CHECK-NEXT:   [[Reg8]]:vgpr_32 = COPY killed $vgpr7
; CHECK-NEXT:   [[Reg7]]:vgpr_32 = COPY killed $vgpr6
; CHECK-NEXT:   [[Reg6]]:vgpr_32 = COPY killed $vgpr5
; CHECK-NEXT:   [[Reg5]]:vgpr_32 = COPY killed $vgpr4
; CHECK-NEXT:   [[Reg4]]:vgpr_32 = COPY killed $vgpr3
; CHECK-NEXT:   [[Reg3]]:vgpr_32 = COPY killed $vgpr2
; CHECK-NEXT:   [[Reg2]]:vgpr_32 = COPY killed $vgpr1
; CHECK-NEXT:   [[Reg1]]:vgpr_32 = COPY killed $vgpr0
; CHECK-NEXT:   [[Reg9:%[0-9]+]]:vreg_64 = REG_SEQUENCE killed [[Reg5]]:vgpr_32, %subreg.sub0, killed [[Reg6]]:vgpr_32, %subreg.sub1
; CHECK-NEXT:   [[Reg10:%[0-9]+]]:vreg_64 = REG_SEQUENCE killed [[Reg3]]:vgpr_32, %subreg.sub0, killed [[Reg4]]:vgpr_32, %subreg.sub1
; CHECK-NEXT:   [[Reg11:%[0-9]+]]:vreg_64 = REG_SEQUENCE killed [[Reg1]]:vgpr_32, %subreg.sub0, killed [[Reg2]]:vgpr_32, %subreg.sub1
; CHECK-NEXT:   [[Reg12:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg11]]:vreg_64, 0, 0, implicit $exec :: (load (s8) from %ir.p1, addrspace 1)
; CHECK-NEXT:   [[Reg13:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg11]]:vreg_64, 1, 0, implicit $exec :: (load (s8) from %ir.p1 + 1, addrspace 1)
; CHECK-NEXT:   [[Reg14:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg13]]:vgpr_32, 8, killed [[Reg12]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg15:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg11]]:vreg_64, 2, 0, implicit $exec :: (load (s8) from %ir.p1 + 2, addrspace 1)
; CHECK-NEXT:   [[Reg16:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg11]]:vreg_64, 3, 0, implicit $exec :: (load (s8) from %ir.p1 + 3, addrspace 1)
; CHECK-NEXT:   [[Reg17:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg16]]:vgpr_32, 8, killed [[Reg15]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg18:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg17]]:vgpr_32, 16, killed [[Reg14]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg19:%[0-9]+]]:vgpr_32 = V_MOV_B32_e32 100, implicit $exec
; CHECK-NEXT:   [[Reg20:%[0-9]+]]:sreg_32 = S_MOV_B32 1
; CHECK-NEXT:   [[Reg21:%[0-9]+]]:sreg_32 = S_MOV_B32 0
; EMPTY:
; CHECK: bb.1.loop1:
; CHECK-NEXT: ; predecessors: %bb.0, %bb.1
; CHECK-NEXT:   successors: %bb.2(0x04000000), %bb.1(0x7c000000); %bb.2(3.12%), %bb.1(96.88%)
; EMPTY:
; CHECK:   [[Reg22:%[0-9]+]]:sreg_32 = PHI [[Reg21]]:sreg_32, %bb.0, [[Reg23:%[0-9]+]]:sreg_32, %bb.1
; CHECK-NEXT:   [[Reg24:%[0-9]+]]:sreg_32 = PHI [[Reg21]]:sreg_32, %bb.0, [[Reg25:%[0-9]+]]:sreg_32, %bb.1
; CHECK-NEXT:   [[Reg26:%[0-9]+]]:sreg_32 = PHI [[Reg20]]:sreg_32, %bb.0, [[Reg27:%[0-9]+]]:sreg_32, %bb.1
; CHECK-NEXT:   [[Reg28:%[0-9]+]]:vgpr_32 = PHI [[Reg18]]:vgpr_32, %bb.0, [[Reg29:%[0-9]+]]:vgpr_32, %bb.1
; CHECK-NEXT:   [[Reg30:%[0-9]+]]:vgpr_32 = PHI [[Reg19]]:vgpr_32, %bb.0, [[Reg31:%[0-9]+]]:vgpr_32, %bb.1
; CHECK-NEXT:   [[Reg32:%[0-9]+]]:vgpr_32 = PHI [[Reg18]]:vgpr_32, %bb.0, [[Reg33:%[0-9]+]]:vgpr_32, %bb.1
; CHECK-NEXT:   [[Reg34:%[0-9]+]]:sreg_32 = S_ADD_I32 [[Reg26]]:sreg_32, -1, implicit-def dead $scc
; CHECK-NEXT:   [[Reg35:%[0-9]+]]:sreg_32_xm0 = S_ASHR_I32 [[Reg34]]:sreg_32, 31, implicit-def dead $scc
; CHECK-NEXT:   [[Reg36:%[0-9]+]]:sreg_64 = REG_SEQUENCE killed [[Reg34]]:sreg_32, %subreg.sub0, killed [[Reg35]]:sreg_32_xm0, %subreg.sub1
; CHECK-NEXT:   [[Reg37:%[0-9]+]]:sreg_64 = nsw S_LSHL_B64 killed [[Reg36]]:sreg_64, 2, implicit-def dead $scc
; CHECK-NEXT:   [[Reg38:%[0-9]+]]:vgpr_32, [[Reg39:%[0-9]+]]:sreg_32_xm0_xexec = V_ADD_CO_U32_e64 [[Reg10]].sub0:vreg_64, [[Reg37]].sub0:sreg_64, 0, implicit $exec
; CHECK-NEXT:   [[Reg40:%[0-9]+]]:vgpr_32, dead $sgpr_null = V_ADDC_U32_e64 [[Reg37]].sub1:sreg_64, [[Reg10]].sub1:vreg_64, killed [[Reg39]]:sreg_32_xm0_xexec, 0, implicit $exec
; CHECK-NEXT:   [[Reg41:%[0-9]+]]:vreg_64 = REG_SEQUENCE killed [[Reg38]]:vgpr_32, %subreg.sub0, killed [[Reg40]]:vgpr_32, %subreg.sub1
; CHECK-NEXT:   [[Reg42:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_DWORD killed [[Reg41]]:vreg_64, 0, 0, implicit $exec :: (load (s32) from %ir.gep1, addrspace 1)
; CHECK-NEXT:   [[Reg29]]:vgpr_32 = V_ADD_U32_e64 [[Reg26]]:sreg_32, [[Reg42]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg31]]:vgpr_32 = V_MUL_LO_U32_e64 [[Reg42]]:vgpr_32, [[Reg26]]:sreg_32, implicit $exec
; CHECK-NEXT:   GLOBAL_STORE_SHORT_D16_HI [[Reg11]]:vreg_64, [[Reg31]]:vgpr_32, 2, 0, implicit $exec :: (store (s16) into %ir.p1 + 2, addrspace 1)
; CHECK-NEXT:   GLOBAL_STORE_SHORT [[Reg11]]:vreg_64, [[Reg31]]:vgpr_32, 0, 0, implicit $exec :: (store (s16) into %ir.p1, addrspace 1)
; CHECK-NEXT:   [[Reg43:%[0-9]+]]:vgpr_32, [[Reg44:%[0-9]+]]:sreg_32_xm0_xexec = V_ADD_CO_U32_e64 [[Reg9]].sub0:vreg_64, [[Reg37]].sub0:sreg_64, 0, implicit $exec
; CHECK-NEXT:   [[Reg45:%[0-9]+]]:vgpr_32, dead $sgpr_null = V_ADDC_U32_e64 killed [[Reg37]].sub1:sreg_64, [[Reg9]].sub1:vreg_64, killed [[Reg44]]:sreg_32_xm0_xexec, 0, implicit $exec
; CHECK-NEXT:   [[Reg46:%[0-9]+]]:vreg_64 = REG_SEQUENCE killed [[Reg43]]:vgpr_32, %subreg.sub0, killed [[Reg45]]:vgpr_32, %subreg.sub1
; CHECK-NEXT:   [[Reg47:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_USHORT [[Reg46]]:vreg_64, 0, 0, implicit $exec :: (load (s16) from %ir.gep2, addrspace 1)
; CHECK-NEXT:   [[Reg48:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_USHORT killed [[Reg46]]:vreg_64, 2, 0, implicit $exec :: (load (s16) from %ir.gep2 + 2, addrspace 1)
; CHECK-NEXT:   [[Reg49:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg48]]:vgpr_32, 16, killed [[Reg47]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg27]]:sreg_32 = S_ADD_I32 [[Reg26]]:sreg_32, 1, implicit-def dead $scc
; CHECK-NEXT:   [[Reg25]]:sreg_32 = S_ADD_I32 [[Reg24]]:sreg_32, -1, implicit-def dead $scc
; CHECK-NEXT:   [[Reg33]]:vgpr_32 = V_ADD_U32_e64 killed [[Reg24]]:sreg_32, killed [[Reg49]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg50:%[0-9]+]]:sreg_32 = V_CMP_GE_U32_e64 killed [[Reg26]]:sreg_32, [[Reg7]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg23]]:sreg_32 = SI_IF_BREAK killed [[Reg50]]:sreg_32, killed [[Reg22]]:sreg_32, implicit-def dead $scc
; CHECK-NEXT:   [[Reg51:%[0-9]+]]:vgpr_32 = COPY [[Reg27]]:sreg_32, implicit $exec
; CHECK-NEXT:   SI_LOOP [[Reg23]]:sreg_32, %bb.1, implicit-def dead $exec, implicit-def dead $scc, implicit $exec
; CHECK-NEXT:   S_BRANCH %bb.2
; EMPTY:
; CHECK: bb.2.bb:
; CHECK-NEXT: ; predecessors: %bb.1
; CHECK-NEXT:   successors: %bb.3(0x80000000); %bb.3(100.00%)
; EMPTY:
; CHECK:   SI_END_CF killed [[Reg23]]:sreg_32, implicit-def dead $exec, implicit-def dead $scc, implicit $exec
; CHECK-NEXT:   [[Reg52:%[0-9]+]]:vgpr_32 = V_ADD_U32_e64 -2, killed [[Reg51]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg53:%[0-9]+]]:vgpr_32 = V_MUL_LO_U32_e64 killed [[Reg31]]:vgpr_32, killed [[Reg52]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg54:%[0-9]+]]:vgpr_32 = V_MUL_LO_U32_e64 100, killed [[Reg28]]:vgpr_32, implicit $exec
; CHECK-NEXT:   GLOBAL_STORE_DWORD [[Reg9]]:vreg_64, [[Reg54]]:vgpr_32, 0, 0, implicit $exec :: (store (s32) into %ir.p3, addrspace 1)
; CHECK-NEXT:   [[Reg55:%[0-9]+]]:sreg_32 = S_MOV_B32 0
; EMPTY:
; CHECK: bb.3.loop2:
; CHECK-NEXT: ; predecessors: %bb.2, %bb.3
; CHECK-NEXT:   successors: %bb.4(0x04000000), %bb.3(0x7c000000); %bb.4(3.12%), %bb.3(96.88%)
; EMPTY:
; CHECK:   [[Reg56:%[0-9]+]]:sreg_32 = PHI [[Reg55]]:sreg_32, %bb.2, [[Reg57:%[0-9]+]]:sreg_32, %bb.3
; CHECK-NEXT:   [[Reg58:%[0-9]+]]:sreg_32 = PHI [[Reg55]]:sreg_32, %bb.2, [[Reg59:%[0-9]+]]:sreg_32, %bb.3
; CHECK-NEXT:   [[Reg60:%[0-9]+]]:vgpr_32 = PHI [[Reg32]]:vgpr_32, %bb.2, [[Reg53]]:vgpr_32, %bb.3
; CHECK-NEXT:   [[Reg59]]:sreg_32 = S_ADD_I32 killed [[Reg58]]:sreg_32, 2, implicit-def dead $scc
; CHECK-NEXT:   [[Reg61:%[0-9]+]]:sreg_32 = V_CMP_GE_U32_e64 [[Reg59]]:sreg_32, [[Reg8]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg57]]:sreg_32 = SI_IF_BREAK killed [[Reg61]]:sreg_32, killed [[Reg56]]:sreg_32, implicit-def dead $scc
; CHECK-NEXT:   [[Reg62:%[0-9]+]]:vgpr_32 = COPY [[Reg59]]:sreg_32, implicit $exec
; CHECK-NEXT:   SI_LOOP [[Reg57]]:sreg_32, %bb.3, implicit-def dead $exec, implicit-def dead $scc, implicit $exec
; CHECK-NEXT:   S_BRANCH %bb.4
; EMPTY:
; CHECK: bb.4.exit:
; CHECK-NEXT: ; predecessors: %bb.3
; EMPTY:
; CHECK:   SI_END_CF killed [[Reg57]]:sreg_32, implicit-def dead $exec, implicit-def dead $scc, implicit $exec
; CHECK-NEXT:   GLOBAL_STORE_DWORD killed [[Reg9]]:vreg_64, [[Reg60]]:vgpr_32, 0, 0, implicit $exec :: (store (s32) into %ir.p3, addrspace 1)
; CHECK-NEXT:   [[Reg63:%[0-9]+]]:vgpr_32 = V_ADD3_U32_e64 killed [[Reg60]]:vgpr_32, killed [[Reg62]]:vgpr_32, killed [[Reg30]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg64:%[0-9]+]]:vgpr_32 = V_ADD3_U32_e64 killed [[Reg63]]:vgpr_32, killed [[Reg54]]:vgpr_32, killed [[Reg42]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg65:%[0-9]+]]:vgpr_32 = V_ADD3_U32_e64 killed [[Reg18]]:vgpr_32, killed [[Reg64]]:vgpr_32, 100, implicit $exec
; CHECK-NEXT:   [[Reg66:%[0-9]+]]:sreg_32_xm0 = V_READFIRSTLANE_B32 killed [[Reg65]]:vgpr_32, implicit $exec
; CHECK-NEXT:   $sgpr0 = COPY killed [[Reg66]]:sreg_32_xm0
; CHECK-NEXT:   SI_RETURN_TO_EPILOG killed $sgpr0
; EMPTY:
; CHECK: # End machine code for function test6.
; EMPTY:
; CHECK: Next-use distance of Register [[Reg8]] = 32031.0
; CHECK-NEXT: Next-use distance of Register [[Reg7]] = 47.0
; CHECK-NEXT: Next-use distance of Register [[Reg6]] = 6.0
; CHECK-NEXT: Next-use distance of Register [[Reg5]] = 5.0
; CHECK-NEXT: Next-use distance of Register [[Reg4]] = 5.0
; CHECK-NEXT: Next-use distance of Register [[Reg3]] = 4.0
; CHECK-NEXT: Next-use distance of Register [[Reg2]] = 4.0
; CHECK-NEXT: Next-use distance of Register [[Reg1]] = 3.0
; CHECK-NEXT: Next-use distance of Register [[Reg9]] = 31.0
; CHECK-NEXT: Next-use distance of Register [[Reg10]] = 22.0
; CHECK-NEXT: Next-use distance of Register [[Reg11]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg12]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg13]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg14]] = 4.0
; CHECK-NEXT: Next-use distance of Register [[Reg15]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg16]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg17]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg18]] = 7.0
; CHECK-NEXT: Next-use distance of Register [[Reg19]] = 7.0
; CHECK-NEXT: Next-use distance of Register [[Reg20]] = 4.0
; CHECK-NEXT: Next-use distance of Register [[Reg21]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg22]] = 28.0
; CHECK-NEXT: Next-use distance of Register [[Reg24]] = 24.0
; CHECK-NEXT: Next-use distance of Register [[Reg26]] = 4.0
; CHECK-NEXT: Next-use distance of Register [[Reg28]] = 32004.0
; CHECK-NEXT: Next-use distance of Register [[Reg30]] = 41009.0
; CHECK-NEXT: Next-use distance of Register [[Reg32]] = 32009.0
; CHECK-NEXT: Next-use distance of Register [[Reg34]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg35]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg36]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg37]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg38]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg39]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg40]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg41]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg42]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg29]] = 21.0
; CHECK-NEXT: Next-use distance of Register [[Reg31]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg43]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg44]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg45]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg46]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg47]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg48]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg49]] = 3.0
; CHECK-NEXT: Next-use distance of Register [[Reg27]] = 5.0
; CHECK-NEXT: Next-use distance of Register [[Reg25]] = 8.0
; CHECK-NEXT: Next-use distance of Register [[Reg33]] = 11.0
; CHECK-NEXT: Next-use distance of Register [[Reg50]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg23]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg51]] = 32002.0
; CHECK-NEXT: Next-use distance of Register [[Reg52]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg53]] = 6.0
; CHECK-NEXT: Next-use distance of Register [[Reg54]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg55]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg56]] = 5.0
; CHECK-NEXT: Next-use distance of Register [[Reg58]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg60]] = 9002.0
; CHECK-NEXT: Next-use distance of Register [[Reg59]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg61]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg57]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg62]] = 9003.0
; CHECK-NEXT: Next-use distance of Register [[Reg63]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg64]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg65]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg66]] = 1.0
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
   %ld1 = load i32, ptr addrspace(1) %p1, align 1
   %add1 = add i32 %ld1, 100
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
   store i32 %mul3, ptr addrspace(1) %p3
   br label %loop2

loop2:
   %phi.inc2 = phi i32 [ 0, %bb ], [ %inc2, %loop2 ]
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

