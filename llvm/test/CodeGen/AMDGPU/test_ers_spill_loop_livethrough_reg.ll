; RUN: llc -mtriple=amdgcn -mcpu=gfx1200 -enable-next-use-analysis=true -verify-machineinstrs -dump-next-use-distance < %s 2>&1 | FileCheck %s

;
;       bb.0.entry
;           |
;    bb.1.loop.header<--+
;       /   |           |
;  bb.2.bb1 |           |
;       \   |           |
;      bb.5.Flow        |
;       /   |           |
;  bb.6.bb3 |           |
;       \   |           |
;      bb.3.Flow1       |
;       /   |           |
;  bb.4.bb2 |           |
;       \   |           |
;    bb.7.loop.latch----+
;           |
;       bb.8.exit
;
define amdgpu_ps void @test(ptr addrspace(1) %p1, ptr addrspace(1) %p2, ptr addrspace(1) %p3, i32 %TC) {
; CHECK-LABEL: # Machine code for function test: IsSSA, TracksLiveness
; CHECK-NEXT: Function Live Ins: $vgpr0 in [[Reg1:%[0-9]+]], $vgpr1 in [[Reg2:%[0-9]+]], $vgpr2 in [[Reg3:%[0-9]+]], $vgpr3 in [[Reg4:%[0-9]+]], $vgpr4 in [[Reg5:%[0-9]+]], $vgpr5 in [[Reg6:%[0-9]+]], $vgpr6 in [[Reg7:%[0-9]+]]
; EMPTY:
; CHECK: bb.0.entry:
; CHECK-NEXT:   successors: %bb.1(0x80000000); %bb.1(100.00%)
; CHECK-NEXT:   liveins: $vgpr0, $vgpr1, $vgpr2, $vgpr3, $vgpr4, $vgpr5, $vgpr6
; CHECK-NEXT:   [[Reg7]]:vgpr_32 = COPY killed $vgpr6
; CHECK-NEXT:   [[Reg6]]:vgpr_32 = COPY killed $vgpr5
; CHECK-NEXT:   [[Reg5]]:vgpr_32 = COPY killed $vgpr4
; CHECK-NEXT:   [[Reg4]]:vgpr_32 = COPY killed $vgpr3
; CHECK-NEXT:   [[Reg3]]:vgpr_32 = COPY killed $vgpr2
; CHECK-NEXT:   [[Reg2]]:vgpr_32 = COPY killed $vgpr1
; CHECK-NEXT:   [[Reg1]]:vgpr_32 = COPY killed $vgpr0
; CHECK-NEXT:   [[Reg8:%[0-9]+]]:vreg_64 = REG_SEQUENCE killed [[Reg5]]:vgpr_32, %subreg.sub0, killed [[Reg6]]:vgpr_32, %subreg.sub1
; CHECK-NEXT:   [[Reg9:%[0-9]+]]:vreg_64 = REG_SEQUENCE killed [[Reg3]]:vgpr_32, %subreg.sub0, killed [[Reg4]]:vgpr_32, %subreg.sub1
; CHECK-NEXT:   [[Reg10:%[0-9]+]]:vreg_64 = REG_SEQUENCE killed [[Reg1]]:vgpr_32, %subreg.sub0, killed [[Reg2]]:vgpr_32, %subreg.sub1
; CHECK-NEXT:   [[Reg11:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg10]]:vreg_64, 0, 0, implicit $exec :: (load (s8) from %ir.p1, addrspace 1)
; CHECK-NEXT:   [[Reg12:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg10]]:vreg_64, 1, 0, implicit $exec :: (load (s8) from %ir.p1 + 1, addrspace 1)
; CHECK-NEXT:   [[Reg13:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg12]]:vgpr_32, 8, killed [[Reg11]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg14:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg10]]:vreg_64, 2, 0, implicit $exec :: (load (s8) from %ir.p1 + 2, addrspace 1)
; CHECK-NEXT:   [[Reg15:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE killed [[Reg10]]:vreg_64, 3, 0, implicit $exec :: (load (s8) from %ir.p1 + 3, addrspace 1)
; CHECK-NEXT:   [[Reg16:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg15]]:vgpr_32, 8, killed [[Reg14]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg17:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg16]]:vgpr_32, 16, killed [[Reg13]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg18:%[0-9]+]]:sreg_32 = S_MOV_B32 0
; EMPTY:
; CHECK: bb.1.loop.header:
; CHECK-NEXT: ; predecessors: %bb.0, %bb.7
; CHECK-NEXT:   successors: %bb.2(0x40000000), %bb.5(0x40000000); %bb.2(50.00%), %bb.5(50.00%)
; EMPTY:
; CHECK:   [[Reg19:%[0-9]+]]:sreg_32 = PHI [[Reg18]]:sreg_32, %bb.0, [[Reg20:%[0-9]+]]:sreg_32, %bb.7
; CHECK-NEXT:   [[Reg21:%[0-9]+]]:vreg_64 = PHI undef [[Reg22:%[0-9]+]]:vreg_64, %bb.0, [[Reg23:%[0-9]+]]:vreg_64, %bb.7
; CHECK-NEXT:   [[Reg24:%[0-9]+]]:sreg_32 = PHI [[Reg18]]:sreg_32, %bb.0, [[Reg25:%[0-9]+]]:sreg_32, %bb.7
; CHECK-NEXT:   [[Reg26:%[0-9]+]]:vgpr_32 = PHI [[Reg17]]:vgpr_32, %bb.0, [[Reg27:%[0-9]+]]:vgpr_32, %bb.7
; CHECK-NEXT:   [[Reg28:%[0-9]+]]:sreg_32 = V_CMP_GE_I32_e64 [[Reg24]]:sreg_32, [[Reg17]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg29:%[0-9]+]]:sreg_32 = V_CMP_LT_I32_e64 [[Reg24]]:sreg_32, [[Reg17]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg30:%[0-9]+]]:sreg_32 = SI_IF killed [[Reg29]]:sreg_32, %bb.5, implicit-def dead $exec, implicit-def dead $scc, implicit $exec
; CHECK-NEXT:   S_BRANCH %bb.2
; EMPTY:
; CHECK: bb.2.bb1:
; CHECK-NEXT: ; predecessors: %bb.1
; CHECK-NEXT:   successors: %bb.5(0x80000000); %bb.5(100.00%)
; EMPTY:
; CHECK:   [[Reg31:%[0-9]+]]:sreg_32_xm0 = S_ASHR_I32 [[Reg24]]:sreg_32, 31, implicit-def dead $scc
; CHECK-NEXT:   [[Reg32:%[0-9]+]]:sreg_64 = REG_SEQUENCE [[Reg24]]:sreg_32, %subreg.sub0, killed [[Reg31]]:sreg_32_xm0, %subreg.sub1
; CHECK-NEXT:   [[Reg33:%[0-9]+]]:sreg_64 = nsw S_LSHL_B64 killed [[Reg32]]:sreg_64, 2, implicit-def dead $scc
; CHECK-NEXT:   [[Reg34:%[0-9]+]]:vgpr_32, [[Reg35:%[0-9]+]]:sreg_32_xm0_xexec = V_ADD_CO_U32_e64 [[Reg9]].sub0:vreg_64, [[Reg33]].sub0:sreg_64, 0, implicit $exec
; CHECK-NEXT:   [[Reg36:%[0-9]+]]:vgpr_32, dead $sgpr_null = V_ADDC_U32_e64 killed [[Reg33]].sub1:sreg_64, [[Reg9]].sub1:vreg_64, killed [[Reg35]]:sreg_32_xm0_xexec, 0, implicit $exec
; CHECK-NEXT:   [[Reg37:%[0-9]+]]:vreg_64 = REG_SEQUENCE killed [[Reg34]]:vgpr_32, %subreg.sub0, killed [[Reg36]]:vgpr_32, %subreg.sub1
; CHECK-NEXT:   [[Reg38:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_DWORD [[Reg37]]:vreg_64, 0, 0, implicit $exec :: (load (s32) from %ir.gep, addrspace 1)
; CHECK-NEXT:   [[Reg39:%[0-9]+]]:sreg_32 = V_CMP_LE_I32_e64 killed [[Reg38]]:vgpr_32, [[Reg17]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg40:%[0-9]+]]:sreg_32 = COPY $exec_lo
; CHECK-NEXT:   [[Reg41:%[0-9]+]]:sreg_32 = S_ANDN2_B32 killed [[Reg28]]:sreg_32, $exec_lo, implicit-def dead $scc
; CHECK-NEXT:   [[Reg42:%[0-9]+]]:sreg_32 = S_AND_B32 killed [[Reg39]]:sreg_32, $exec_lo, implicit-def dead $scc
; CHECK-NEXT:   [[Reg43:%[0-9]+]]:sreg_32 = S_OR_B32 killed [[Reg41]]:sreg_32, killed [[Reg42]]:sreg_32, implicit-def dead $scc
; CHECK-NEXT:   S_BRANCH %bb.5
; EMPTY:
; CHECK: bb.3.Flow1:
; CHECK-NEXT: ; predecessors: %bb.5, %bb.6
; CHECK-NEXT:   successors: %bb.4(0x40000000), %bb.7(0x40000000); %bb.4(50.00%), %bb.7(50.00%)
; EMPTY:
; CHECK:   [[Reg44:%[0-9]+]]:sreg_32 = PHI [[Reg45:%[0-9]+]]:sreg_32, %bb.5, [[Reg46:%[0-9]+]]:sreg_32, %bb.6
; CHECK-NEXT:   [[Reg47:%[0-9]+]]:vgpr_32 = PHI undef [[Reg48:%[0-9]+]]:vgpr_32, %bb.5, [[Reg49:%[0-9]+]]:vgpr_32, %bb.6
; CHECK-NEXT:   SI_END_CF killed [[Reg50:%[0-9]+]]:sreg_32, implicit-def dead $exec, implicit-def dead $scc, implicit $exec
; CHECK-NEXT:   [[Reg51:%[0-9]+]]:sreg_32 = SI_IF killed [[Reg44]]:sreg_32, %bb.7, implicit-def dead $exec, implicit-def dead $scc, implicit $exec
; CHECK-NEXT:   S_BRANCH %bb.4
; EMPTY:
; CHECK: bb.4.bb2:
; CHECK-NEXT: ; predecessors: %bb.3
; CHECK-NEXT:   successors: %bb.7(0x80000000); %bb.7(100.00%)
; EMPTY:
; CHECK:   GLOBAL_STORE_DWORD [[Reg23]]:vreg_64, killed [[Reg26]]:vgpr_32, 0, 0, implicit $exec :: (store (s32) into %ir.9, addrspace 1)
; CHECK-NEXT:   [[Reg52:%[0-9]+]]:vgpr_32 = V_MOV_B32_e32 1, implicit $exec
; CHECK-NEXT:   S_BRANCH %bb.7
; EMPTY:
; CHECK: bb.5.Flow:
; CHECK-NEXT: ; predecessors: %bb.1, %bb.2
; CHECK-NEXT:   successors: %bb.6(0x40000000), %bb.3(0x40000000); %bb.6(50.00%), %bb.3(50.00%)
; EMPTY:
; CHECK:   [[Reg53:%[0-9]+]]:sreg_32 = PHI [[Reg28]]:sreg_32, %bb.1, [[Reg43]]:sreg_32, %bb.2
; CHECK-NEXT:   [[Reg45]]:sreg_32 = PHI [[Reg18]]:sreg_32, %bb.1, [[Reg40]]:sreg_32, %bb.2
; CHECK-NEXT:   [[Reg23]]:vreg_64 = PHI [[Reg21]]:vreg_64, %bb.1, [[Reg37]]:vreg_64, %bb.2
; CHECK-NEXT:   SI_END_CF killed [[Reg30]]:sreg_32, implicit-def dead $exec, implicit-def dead $scc, implicit $exec
; CHECK-NEXT:   [[Reg50]]:sreg_32 = SI_IF killed [[Reg53]]:sreg_32, %bb.3, implicit-def dead $exec, implicit-def dead $scc, implicit $exec
; CHECK-NEXT:   S_BRANCH %bb.6
; EMPTY:
; CHECK: bb.6.bb3:
; CHECK-NEXT: ; predecessors: %bb.5
; CHECK-NEXT:   successors: %bb.3(0x80000000); %bb.3(100.00%)
; EMPTY:
; CHECK:   [[Reg54:%[0-9]+]]:vgpr_32 = V_LSHRREV_B32_e64 31, [[Reg26]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg55:%[0-9]+]]:vgpr_32 = V_ADD_U32_e64 [[Reg26]]:vgpr_32, killed [[Reg54]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg49]]:vgpr_32 = V_ASHRREV_I32_e64 1, killed [[Reg55]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg56:%[0-9]+]]:sreg_32 = S_ANDN2_B32 killed [[Reg45]]:sreg_32, $exec_lo, implicit-def dead $scc
; CHECK-NEXT:   [[Reg46]]:sreg_32 = COPY killed [[Reg56]]:sreg_32
; CHECK-NEXT:   S_BRANCH %bb.3
; EMPTY:
; CHECK: bb.7.loop.latch:
; CHECK-NEXT: ; predecessors: %bb.3, %bb.4
; CHECK-NEXT:   successors: %bb.8(0x04000000), %bb.1(0x7c000000); %bb.8(3.12%), %bb.1(96.88%)
; EMPTY:
; CHECK:   [[Reg27]]:vgpr_32 = PHI [[Reg47]]:vgpr_32, %bb.3, [[Reg52]]:vgpr_32, %bb.4
; CHECK-NEXT:   SI_END_CF killed [[Reg51]]:sreg_32, implicit-def dead $exec, implicit-def dead $scc, implicit $exec
; CHECK-NEXT:   [[Reg25]]:sreg_32 = S_ADD_I32 killed [[Reg24]]:sreg_32, 1, implicit-def dead $scc
; CHECK-NEXT:   [[Reg57:%[0-9]+]]:sreg_32 = V_CMP_GE_U32_e64 [[Reg25]]:sreg_32, [[Reg7]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg20]]:sreg_32 = SI_IF_BREAK killed [[Reg57]]:sreg_32, killed [[Reg19]]:sreg_32, implicit-def dead $scc
; CHECK-NEXT:   SI_LOOP [[Reg20]]:sreg_32, %bb.1, implicit-def dead $exec, implicit-def dead $scc, implicit $exec
; CHECK-NEXT:   S_BRANCH %bb.8
; EMPTY:
; CHECK: bb.8.exit:
; CHECK-NEXT: ; predecessors: %bb.7
; EMPTY:
; CHECK:   SI_END_CF killed [[Reg20]]:sreg_32, implicit-def dead $exec, implicit-def dead $scc, implicit $exec
; CHECK-NEXT:   GLOBAL_STORE_DWORD killed [[Reg8]]:vreg_64, killed [[Reg27]]:vgpr_32, 0, 0, implicit $exec :: (store (s32) into %ir.p3, addrspace 1)
; CHECK-NEXT:   S_ENDPGM 0
; EMPTY:
; CHECK: # End machine code for function test.
; EMPTY:
; CHECK: Next-use distance of Register [[Reg7]] = 40.0
; CHECK-NEXT: Next-use distance of Register [[Reg6]] = 6.0
; CHECK-NEXT: Next-use distance of Register [[Reg5]] = 5.0
; CHECK-NEXT: Next-use distance of Register [[Reg4]] = 5.0
; CHECK-NEXT: Next-use distance of Register [[Reg3]] = 4.0
; CHECK-NEXT: Next-use distance of Register [[Reg2]] = 4.0
; CHECK-NEXT: Next-use distance of Register [[Reg1]] = 3.0
; CHECK-NEXT: Next-use distance of Register [[Reg8]] = 26012.0
; CHECK-NEXT: Next-use distance of Register [[Reg9]] = 21.0
; CHECK-NEXT: Next-use distance of Register [[Reg10]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg11]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg12]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg13]] = 4.0
; CHECK-NEXT: Next-use distance of Register [[Reg14]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg15]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg16]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg17]] = 5.0
; CHECK-NEXT: Next-use distance of Register [[Reg18]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg19]] = 23.0
; CHECK-NEXT: Next-use distance of Register [[Reg21]] = 9.0
; CHECK-NEXT: Next-use distance of Register [[Reg24]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg26]] = 11.0
; CHECK-NEXT: Next-use distance of Register [[Reg28]] = 4.0
; CHECK-NEXT: Next-use distance of Register [[Reg29]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg30]] = 5.0
; CHECK-NEXT: Next-use distance of Register [[Reg31]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg32]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg33]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg34]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg35]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg36]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg37]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg38]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg39]] = 3.0
; CHECK-NEXT: Next-use distance of Register [[Reg40]] = 6.0
; CHECK-NEXT: Next-use distance of Register [[Reg41]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg42]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg43]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg44]] = 3.0
; CHECK-NEXT: Next-use distance of Register [[Reg47]] = 4.0
; CHECK-NEXT: Next-use distance of Register [[Reg51]] = 3.0
; CHECK-NEXT: Next-use distance of Register [[Reg52]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg53]] = 4.0
; CHECK-NEXT: Next-use distance of Register [[Reg45]] = 5.0
; CHECK-NEXT: Next-use distance of Register [[Reg23]] = 9.0
; CHECK-NEXT: Next-use distance of Register [[Reg50]] = 4.0
; CHECK-NEXT: Next-use distance of Register [[Reg54]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg55]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg49]] = 5.0
; CHECK-NEXT: Next-use distance of Register [[Reg56]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg46]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg27]] = 10.0
; CHECK-NEXT: Next-use distance of Register [[Reg25]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg57]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg20]] = 1.0
entry:
;     entry
;       |
;   loop.header<-+
;    |    |      |
;   bb1   |      |
;    | \  |      |
;   bb2 bb3      |
;    |   |       |
;   loop.latch---+
;       |
;      exit
   %ld1 = load i32, ptr addrspace(1) %p1, align 1
   br label %loop.header

loop.header:
   %phi.inc = phi i32 [ 0, %entry ], [ %inc, %loop.latch ]
   %phi1 = phi i32 [ %ld1, %entry ], [ %phi2, %loop.latch ]
   %cond1 = icmp slt i32 %phi.inc, %ld1
   br i1 %cond1, label %bb1, label %bb3

bb1:
   %sext = sext i32 %phi.inc to i64
   %gep = getelementptr inbounds i32, ptr addrspace(1) %p2, i64 %sext
   %ld2 = load i32, ptr addrspace(1) %gep, align 4
   %cond2 = icmp sgt i32 %ld2, %ld1
   br i1 %cond2, label %bb2, label %bb3

bb2:
   store i32 %phi1, ptr addrspace(1) %gep, align 4
   br label %loop.latch

bb3:
   %div = sdiv i32 %phi1, 2
   br label %loop.latch

loop.latch:
   %phi2 = phi i32 [ 1, %bb2 ], [ %div, %bb3 ]
   %inc = add i32 %phi.inc, 1
   %cond3 = icmp ult i32 %inc, %TC
   br i1 %cond3, label %loop.header, label %exit

exit:
   store i32 %phi2, ptr addrspace(1) %p3, align 4
   ret void
}
