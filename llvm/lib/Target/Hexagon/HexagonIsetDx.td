//=- HexagonIsetDx.td - Target Desc. for Hexagon Target -*- tablegen -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the Hexagon duplex instructions.
//
//===----------------------------------------------------------------------===//

// SA1_combine1i: Combines.
let isCodeGenOnly = 1, hasSideEffects = 0 in
def V4_SA1_combine1i: SUBInst <
  (outs DoubleRegs:$Rdd),
  (ins u2Imm:$u2),
  "$Rdd = combine(#1, #$u2)"> {
    bits<3> Rdd;
    bits<2> u2;

    let Inst{12-10} = 0b111;
    let Inst{8} = 0b0;
    let Inst{4-3} = 0b01;
    let Inst{2-0} = Rdd;
    let Inst{6-5} = u2;
  }

// SL2_jumpr31_f: Indirect conditional jump if false.
// SL2_jumpr31_f -> SL2_jumpr31_fnew
let Defs = [PC], Uses = [P0, R31], isCodeGenOnly = 1, isPredicated = 1, isPredicatedFalse = 1, isBranch = 1, isIndirectBranch = 1, hasSideEffects = 0 in
def V4_SL2_jumpr31_f: SUBInst <
  (outs ),
  (ins ),
  "if (!p0) jumpr r31"> {
    let Inst{12-6} = 0b1111111;
    let Inst{2-0} = 0b101;
  }

// SL2_deallocframe: Deallocate stack frame.
let Defs = [R31, R29, R30], Uses = [R30], isCodeGenOnly = 1, mayLoad = 1, accessSize = DoubleWordAccess in
def V4_SL2_deallocframe: SUBInst <
  (outs ),
  (ins ),
  "deallocframe"> {
    let Inst{12-6} = 0b1111100;
    let Inst{2} = 0b0;
  }

// SL2_return_f: Deallocate stack frame and return.
// SL2_return_f -> SL2_return_fnew
let Defs = [PC, R31, R29, R30], Uses = [R30, P0], isCodeGenOnly = 1, isPredicated = 1, isPredicatedFalse = 1, mayLoad = 1, accessSize = DoubleWordAccess, isBranch = 1, isIndirectBranch = 1 in
def V4_SL2_return_f: SUBInst <
  (outs ),
  (ins ),
  "if (!p0) dealloc_return"> {
    let Inst{12-6} = 0b1111101;
    let Inst{2-0} = 0b101;
  }

// SA1_combine3i: Combines.
let isCodeGenOnly = 1, hasSideEffects = 0 in
def V4_SA1_combine3i: SUBInst <
  (outs DoubleRegs:$Rdd),
  (ins u2Imm:$u2),
  "$Rdd = combine(#3, #$u2)"> {
    bits<3> Rdd;
    bits<2> u2;

    let Inst{12-10} = 0b111;
    let Inst{8} = 0b0;
    let Inst{4-3} = 0b11;
    let Inst{2-0} = Rdd;
    let Inst{6-5} = u2;
  }

// SS2_storebi0: Store byte.
let isCodeGenOnly = 1, mayStore = 1, accessSize = ByteAccess in
def V4_SS2_storebi0: SUBInst <
  (outs ),
  (ins IntRegs:$Rs, u4_0Imm:$u4_0),
  "memb($Rs + #$u4_0)=#0"> {
    bits<4> Rs;
    bits<4> u4_0;

    let Inst{12-8} = 0b10010;
    let Inst{7-4} = Rs;
    let Inst{3-0} = u4_0;
  }

// SA1_clrtnew: Clear if true.
let Uses = [P0], isCodeGenOnly = 1, isPredicated = 1, isPredicatedNew = 1, hasSideEffects = 0, hasNewValue = 1, opNewValue = 0 in
def V4_SA1_clrtnew: SUBInst <
  (outs IntRegs:$Rd),
  (ins PredRegs:$Pu),
  "if ($Pu.new) $Rd = #0"> {
    bits<4> Rd;

    let Inst{12-9} = 0b1101;
    let Inst{6-4} = 0b100;
    let Inst{3-0} = Rd;
  }

// SL2_loadruh_io: Load half.
let isCodeGenOnly = 1, mayLoad = 1, accessSize = HalfWordAccess, hasNewValue = 1, opNewValue = 0 in
def V4_SL2_loadruh_io: SUBInst <
  (outs IntRegs:$Rd),
  (ins IntRegs:$Rs, u3_1Imm:$u3_1),
  "$Rd = memuh($Rs + #$u3_1)"> {
    bits<4> Rd;
    bits<4> Rs;
    bits<4> u3_1;

    let Inst{12-11} = 0b01;
    let Inst{3-0} = Rd;
    let Inst{7-4} = Rs;
    let Inst{10-8} = u3_1{3-1};
  }

// SL2_jumpr31_tnew: Indirect conditional jump if true.
let Defs = [PC], Uses = [P0, R31], isCodeGenOnly = 1, isPredicated = 1, isPredicatedNew = 1, isBranch = 1, isIndirectBranch = 1, hasSideEffects = 0 in
def V4_SL2_jumpr31_tnew: SUBInst <
  (outs ),
  (ins ),
  "if (p0.new) jumpr:nt r31"> {
    let Inst{12-6} = 0b1111111;
    let Inst{2-0} = 0b110;
  }

// SA1_addi: Add.
let isCodeGenOnly = 1, hasSideEffects = 0, hasNewValue = 1, opNewValue = 0, isExtendable = 1, isExtentSigned = 1, opExtentBits = 7, opExtendable = 2 in
def V4_SA1_addi: SUBInst <
  (outs IntRegs:$Rx),
  (ins IntRegs:$_src_, s7Ext:$s7),
  "$Rx = add($_src_, #$s7)" ,
  [] ,
  "$_src_ = $Rx"> {
    bits<4> Rx;
    bits<7> s7;

    let Inst{12-11} = 0b00;
    let Inst{3-0} = Rx;
    let Inst{10-4} = s7;
  }

// SL1_loadrub_io: Load byte.
let isCodeGenOnly = 1, mayLoad = 1, accessSize = ByteAccess, hasNewValue = 1, opNewValue = 0 in
def V4_SL1_loadrub_io: SUBInst <
  (outs IntRegs:$Rd),
  (ins IntRegs:$Rs, u4_0Imm:$u4_0),
  "$Rd = memub($Rs + #$u4_0)"> {
    bits<4> Rd;
    bits<4> Rs;
    bits<4> u4_0;

    let Inst{12} = 0b1;
    let Inst{3-0} = Rd;
    let Inst{7-4} = Rs;
    let Inst{11-8} = u4_0;
  }

// SL1_loadri_io: Load word.
let isCodeGenOnly = 1, mayLoad = 1, accessSize = WordAccess, hasNewValue = 1, opNewValue = 0 in
def V4_SL1_loadri_io: SUBInst <
  (outs IntRegs:$Rd),
  (ins IntRegs:$Rs, u4_2Imm:$u4_2),
  "$Rd = memw($Rs + #$u4_2)"> {
    bits<4> Rd;
    bits<4> Rs;
    bits<6> u4_2;

    let Inst{12} = 0b0;
    let Inst{3-0} = Rd;
    let Inst{7-4} = Rs;
    let Inst{11-8} = u4_2{5-2};
  }

// SA1_cmpeqi: Compareimmed.
let Defs = [P0], isCodeGenOnly = 1, hasSideEffects = 0 in
def V4_SA1_cmpeqi: SUBInst <
  (outs ),
  (ins IntRegs:$Rs, u2Imm:$u2),
  "p0 = cmp.eq($Rs, #$u2)"> {
    bits<4> Rs;
    bits<2> u2;

    let Inst{12-8} = 0b11001;
    let Inst{7-4} = Rs;
    let Inst{1-0} = u2;
  }

// SA1_combinerz: Combines.
let isCodeGenOnly = 1, hasSideEffects = 0 in
def V4_SA1_combinerz: SUBInst <
  (outs DoubleRegs:$Rdd),
  (ins IntRegs:$Rs),
  "$Rdd = combine($Rs, #0)"> {
    bits<3> Rdd;
    bits<4> Rs;

    let Inst{12-10} = 0b111;
    let Inst{8} = 0b1;
    let Inst{3} = 0b1;
    let Inst{2-0} = Rdd;
    let Inst{7-4} = Rs;
  }

// SL2_return_t: Deallocate stack frame and return.
// SL2_return_t -> SL2_return_tnew
let Defs = [PC, R31, R29, R30], Uses = [R30, P0], isCodeGenOnly = 1, isPredicated = 1, mayLoad = 1, accessSize = DoubleWordAccess, isBranch = 1, isIndirectBranch = 1 in
def V4_SL2_return_t: SUBInst <
  (outs ),
  (ins ),
  "if (p0) dealloc_return"> {
    let Inst{12-6} = 0b1111101;
    let Inst{2-0} = 0b100;
  }

// SS2_allocframe: Allocate stack frame.
let Defs = [R29, R30], Uses = [R30, R31, R29], isCodeGenOnly = 1, mayStore = 1, accessSize = DoubleWordAccess in
def V4_SS2_allocframe: SUBInst <
  (outs ),
  (ins u5_3Imm:$u5_3),
  "allocframe(#$u5_3)"> {
    bits<8> u5_3;

    let Inst{12-9} = 0b1110;
    let Inst{8-4} = u5_3{7-3};
  }

// SS2_storeh_io: Store half.
let isCodeGenOnly = 1, mayStore = 1, accessSize = HalfWordAccess in
def V4_SS2_storeh_io: SUBInst <
  (outs ),
  (ins IntRegs:$Rs, u3_1Imm:$u3_1, IntRegs:$Rt),
  "memh($Rs + #$u3_1) = $Rt"> {
    bits<4> Rs;
    bits<4> u3_1;
    bits<4> Rt;

    let Inst{12-11} = 0b00;
    let Inst{7-4} = Rs;
    let Inst{10-8} = u3_1{3-1};
    let Inst{3-0} = Rt;
  }

// SS2_storewi0: Store word.
let isCodeGenOnly = 1, mayStore = 1, accessSize = WordAccess in
def V4_SS2_storewi0: SUBInst <
  (outs ),
  (ins IntRegs:$Rs, u4_2Imm:$u4_2),
  "memw($Rs + #$u4_2)=#0"> {
    bits<4> Rs;
    bits<6> u4_2;

    let Inst{12-8} = 0b10000;
    let Inst{7-4} = Rs;
    let Inst{3-0} = u4_2{5-2};
  }

// SS2_storewi1: Store word.
let isCodeGenOnly = 1, mayStore = 1, accessSize = WordAccess in
def V4_SS2_storewi1: SUBInst <
  (outs ),
  (ins IntRegs:$Rs, u4_2Imm:$u4_2),
  "memw($Rs + #$u4_2)=#1"> {
    bits<4> Rs;
    bits<6> u4_2;

    let Inst{12-8} = 0b10001;
    let Inst{7-4} = Rs;
    let Inst{3-0} = u4_2{5-2};
  }

// SL2_jumpr31: Indirect conditional jump if true.
let Defs = [PC], Uses = [R31], isCodeGenOnly = 1, isBranch = 1, isIndirectBranch = 1, hasSideEffects = 0 in
def V4_SL2_jumpr31: SUBInst <
  (outs ),
  (ins ),
  "jumpr r31"> {
    let Inst{12-6} = 0b1111111;
    let Inst{2} = 0b0;
  }

// SA1_combinezr: Combines.
let isCodeGenOnly = 1, hasSideEffects = 0 in
def V4_SA1_combinezr: SUBInst <
  (outs DoubleRegs:$Rdd),
  (ins IntRegs:$Rs),
  "$Rdd = combine(#0, $Rs)"> {
    bits<3> Rdd;
    bits<4> Rs;

    let Inst{12-10} = 0b111;
    let Inst{8} = 0b1;
    let Inst{3} = 0b0;
    let Inst{2-0} = Rdd;
    let Inst{7-4} = Rs;
  }

// SL2_loadrh_io: Load half.
let isCodeGenOnly = 1, mayLoad = 1, accessSize = HalfWordAccess, hasNewValue = 1, opNewValue = 0 in
def V4_SL2_loadrh_io: SUBInst <
  (outs IntRegs:$Rd),
  (ins IntRegs:$Rs, u3_1Imm:$u3_1),
  "$Rd = memh($Rs + #$u3_1)"> {
    bits<4> Rd;
    bits<4> Rs;
    bits<4> u3_1;

    let Inst{12-11} = 0b00;
    let Inst{3-0} = Rd;
    let Inst{7-4} = Rs;
    let Inst{10-8} = u3_1{3-1};
  }

// SA1_addrx: Add.
let isCodeGenOnly = 1, hasSideEffects = 0, hasNewValue = 1, opNewValue = 0 in
def V4_SA1_addrx: SUBInst <
  (outs IntRegs:$Rx),
  (ins IntRegs:$_src_, IntRegs:$Rs),
  "$Rx = add($_src_, $Rs)" ,
  [] ,
  "$_src_ = $Rx"> {
    bits<4> Rx;
    bits<4> Rs;

    let Inst{12-8} = 0b11000;
    let Inst{3-0} = Rx;
    let Inst{7-4} = Rs;
  }

// SA1_setin1: Set to -1.
let isCodeGenOnly = 1, hasSideEffects = 0, hasNewValue = 1, opNewValue = 0 in
def V4_SA1_setin1: SUBInst <
  (outs IntRegs:$Rd),
  (ins ),
  "$Rd = #{-1}"> {
    bits<4> Rd;

    let Inst{12-9} = 0b1101;
    let Inst{6} = 0b0;
    let Inst{3-0} = Rd;
  }

// SA1_sxth: Sxth.
let isCodeGenOnly = 1, hasSideEffects = 0, hasNewValue = 1, opNewValue = 0 in
def V4_SA1_sxth: SUBInst <
  (outs IntRegs:$Rd),
  (ins IntRegs:$Rs),
  "$Rd = sxth($Rs)"> {
    bits<4> Rd;
    bits<4> Rs;

    let Inst{12-8} = 0b10100;
    let Inst{3-0} = Rd;
    let Inst{7-4} = Rs;
  }

// SA1_combine0i: Combines.
let isCodeGenOnly = 1, hasSideEffects = 0 in
def V4_SA1_combine0i: SUBInst <
  (outs DoubleRegs:$Rdd),
  (ins u2Imm:$u2),
  "$Rdd = combine(#0, #$u2)"> {
    bits<3> Rdd;
    bits<2> u2;

    let Inst{12-10} = 0b111;
    let Inst{8} = 0b0;
    let Inst{4-3} = 0b00;
    let Inst{2-0} = Rdd;
    let Inst{6-5} = u2;
  }

// SA1_combine2i: Combines.
let isCodeGenOnly = 1, hasSideEffects = 0 in
def V4_SA1_combine2i: SUBInst <
  (outs DoubleRegs:$Rdd),
  (ins u2Imm:$u2),
  "$Rdd = combine(#2, #$u2)"> {
    bits<3> Rdd;
    bits<2> u2;

    let Inst{12-10} = 0b111;
    let Inst{8} = 0b0;
    let Inst{4-3} = 0b10;
    let Inst{2-0} = Rdd;
    let Inst{6-5} = u2;
  }

// SA1_sxtb: Sxtb.
let isCodeGenOnly = 1, hasSideEffects = 0, hasNewValue = 1, opNewValue = 0 in
def V4_SA1_sxtb: SUBInst <
  (outs IntRegs:$Rd),
  (ins IntRegs:$Rs),
  "$Rd = sxtb($Rs)"> {
    bits<4> Rd;
    bits<4> Rs;

    let Inst{12-8} = 0b10101;
    let Inst{3-0} = Rd;
    let Inst{7-4} = Rs;
  }

// SA1_clrf: Clear if false.
// SA1_clrf -> SA1_clrfnew
let Uses = [P0], isCodeGenOnly = 1, isPredicated = 1, isPredicatedFalse = 1, hasSideEffects = 0, hasNewValue = 1, opNewValue = 0 in
def V4_SA1_clrf: SUBInst <
  (outs IntRegs:$Rd),
  (ins PredRegs:$Pu),
  "if (!$Pu) $Rd = #0"> {
    bits<4> Rd;

    let Inst{12-9} = 0b1101;
    let Inst{6-4} = 0b111;
    let Inst{3-0} = Rd;
  }

// SL2_loadrb_io: Load byte.
let isCodeGenOnly = 1, mayLoad = 1, accessSize = ByteAccess, hasNewValue = 1, opNewValue = 0 in
def V4_SL2_loadrb_io: SUBInst <
  (outs IntRegs:$Rd),
  (ins IntRegs:$Rs, u3_0Imm:$u3_0),
  "$Rd = memb($Rs + #$u3_0)"> {
    bits<4> Rd;
    bits<4> Rs;
    bits<3> u3_0;

    let Inst{12-11} = 0b10;
    let Inst{3-0} = Rd;
    let Inst{7-4} = Rs;
    let Inst{10-8} = u3_0;
  }

// SA1_tfr: Tfr.
let isCodeGenOnly = 1, hasSideEffects = 0, hasNewValue = 1, opNewValue = 0 in
def V4_SA1_tfr: SUBInst <
  (outs IntRegs:$Rd),
  (ins IntRegs:$Rs),
  "$Rd = $Rs"> {
    bits<4> Rd;
    bits<4> Rs;

    let Inst{12-8} = 0b10000;
    let Inst{3-0} = Rd;
    let Inst{7-4} = Rs;
  }

// SL2_loadrd_sp: Load dword.
let Uses = [R29], isCodeGenOnly = 1, mayLoad = 1, accessSize = DoubleWordAccess in
def V4_SL2_loadrd_sp: SUBInst <
  (outs DoubleRegs:$Rdd),
  (ins u5_3Imm:$u5_3),
  "$Rdd = memd(r29 + #$u5_3)"> {
    bits<3> Rdd;
    bits<8> u5_3;

    let Inst{12-8} = 0b11110;
    let Inst{2-0} = Rdd;
    let Inst{7-3} = u5_3{7-3};
  }

// SA1_and1: And #1.
let isCodeGenOnly = 1, hasSideEffects = 0, hasNewValue = 1, opNewValue = 0 in
def V4_SA1_and1: SUBInst <
  (outs IntRegs:$Rd),
  (ins IntRegs:$Rs),
  "$Rd = and($Rs, #1)"> {
    bits<4> Rd;
    bits<4> Rs;

    let Inst{12-8} = 0b10010;
    let Inst{3-0} = Rd;
    let Inst{7-4} = Rs;
  }

// SS2_storebi1: Store byte.
let isCodeGenOnly = 1, mayStore = 1, accessSize = ByteAccess in
def V4_SS2_storebi1: SUBInst <
  (outs ),
  (ins IntRegs:$Rs, u4_0Imm:$u4_0),
  "memb($Rs + #$u4_0)=#1"> {
    bits<4> Rs;
    bits<4> u4_0;

    let Inst{12-8} = 0b10011;
    let Inst{7-4} = Rs;
    let Inst{3-0} = u4_0;
  }

// SA1_inc: Inc.
let isCodeGenOnly = 1, hasSideEffects = 0, hasNewValue = 1, opNewValue = 0 in
def V4_SA1_inc: SUBInst <
  (outs IntRegs:$Rd),
  (ins IntRegs:$Rs),
  "$Rd = add($Rs, #1)"> {
    bits<4> Rd;
    bits<4> Rs;

    let Inst{12-8} = 0b10001;
    let Inst{3-0} = Rd;
    let Inst{7-4} = Rs;
  }

// SS2_stored_sp: Store dword.
let Uses = [R29], isCodeGenOnly = 1, mayStore = 1, accessSize = DoubleWordAccess in
def V4_SS2_stored_sp: SUBInst <
  (outs ),
  (ins s6_3Imm:$s6_3, DoubleRegs:$Rtt),
  "memd(r29 + #$s6_3) = $Rtt"> {
    bits<9> s6_3;
    bits<3> Rtt;

    let Inst{12-9} = 0b0101;
    let Inst{8-3} = s6_3{8-3};
    let Inst{2-0} = Rtt;
  }

// SS2_storew_sp: Store word.
let Uses = [R29], isCodeGenOnly = 1, mayStore = 1, accessSize = WordAccess in
def V4_SS2_storew_sp: SUBInst <
  (outs ),
  (ins u5_2Imm:$u5_2, IntRegs:$Rt),
  "memw(r29 + #$u5_2) = $Rt"> {
    bits<7> u5_2;
    bits<4> Rt;

    let Inst{12-9} = 0b0100;
    let Inst{8-4} = u5_2{6-2};
    let Inst{3-0} = Rt;
  }

// SL2_jumpr31_fnew: Indirect conditional jump if false.
let Defs = [PC], Uses = [P0, R31], isCodeGenOnly = 1, isPredicated = 1, isPredicatedFalse = 1, isPredicatedNew = 1, isBranch = 1, isIndirectBranch = 1, hasSideEffects = 0 in
def V4_SL2_jumpr31_fnew: SUBInst <
  (outs ),
  (ins ),
  "if (!p0.new) jumpr:nt r31"> {
    let Inst{12-6} = 0b1111111;
    let Inst{2-0} = 0b111;
  }

// SA1_clrt: Clear if true.
// SA1_clrt -> SA1_clrtnew
let Uses = [P0], isCodeGenOnly = 1, isPredicated = 1, hasSideEffects = 0, hasNewValue = 1, opNewValue = 0 in
def V4_SA1_clrt: SUBInst <
  (outs IntRegs:$Rd),
  (ins PredRegs:$Pu),
  "if ($Pu) $Rd = #0"> {
    bits<4> Rd;

    let Inst{12-9} = 0b1101;
    let Inst{6-4} = 0b110;
    let Inst{3-0} = Rd;
  }

// SL2_return: Deallocate stack frame and return.
let Defs = [PC, R31, R29, R30], Uses = [R30], isCodeGenOnly = 1, mayLoad = 1, accessSize = DoubleWordAccess, isBranch = 1, isIndirectBranch = 1 in
def V4_SL2_return: SUBInst <
  (outs ),
  (ins ),
  "dealloc_return"> {
    let Inst{12-6} = 0b1111101;
    let Inst{2} = 0b0;
  }

// SA1_dec: Dec.
let isCodeGenOnly = 1, hasSideEffects = 0, hasNewValue = 1, opNewValue = 0 in
def V4_SA1_dec: SUBInst <
  (outs IntRegs:$Rd),
  (ins IntRegs:$Rs),
  "$Rd = add($Rs,#{-1})"> {
    bits<4> Rd;
    bits<4> Rs;

    let Inst{12-8} = 0b10011;
    let Inst{3-0} = Rd;
    let Inst{7-4} = Rs;
  }

// SA1_seti: Set immed.
let isCodeGenOnly = 1, hasSideEffects = 0, hasNewValue = 1, opNewValue = 0, isExtendable = 1, isExtentSigned = 0, opExtentBits = 6, opExtendable = 1 in
def V4_SA1_seti: SUBInst <
  (outs IntRegs:$Rd),
  (ins u6Ext:$u6),
  "$Rd = #$u6"> {
    bits<4> Rd;
    bits<6> u6;

    let Inst{12-10} = 0b010;
    let Inst{3-0} = Rd;
    let Inst{9-4} = u6;
  }

// SL2_jumpr31_t: Indirect conditional jump if true.
// SL2_jumpr31_t -> SL2_jumpr31_tnew
let Defs = [PC], Uses = [P0, R31], isCodeGenOnly = 1, isPredicated = 1, isBranch = 1, isIndirectBranch = 1, hasSideEffects = 0 in
def V4_SL2_jumpr31_t: SUBInst <
  (outs ),
  (ins ),
  "if (p0) jumpr r31"> {
    let Inst{12-6} = 0b1111111;
    let Inst{2-0} = 0b100;
  }

// SA1_clrfnew: Clear if false.
let Uses = [P0], isCodeGenOnly = 1, isPredicated = 1, isPredicatedFalse = 1, isPredicatedNew = 1, hasSideEffects = 0, hasNewValue = 1, opNewValue = 0 in
def V4_SA1_clrfnew: SUBInst <
  (outs IntRegs:$Rd),
  (ins PredRegs:$Pu),
  "if (!$Pu.new) $Rd = #0"> {
    bits<4> Rd;

    let Inst{12-9} = 0b1101;
    let Inst{6-4} = 0b101;
    let Inst{3-0} = Rd;
  }

// SS1_storew_io: Store word.
let isCodeGenOnly = 1, mayStore = 1, accessSize = WordAccess in
def V4_SS1_storew_io: SUBInst <
  (outs ),
  (ins IntRegs:$Rs, u4_2Imm:$u4_2, IntRegs:$Rt),
  "memw($Rs + #$u4_2) = $Rt"> {
    bits<4> Rs;
    bits<6> u4_2;
    bits<4> Rt;

    let Inst{12} = 0b0;
    let Inst{7-4} = Rs;
    let Inst{11-8} = u4_2{5-2};
    let Inst{3-0} = Rt;
  }

// SA1_zxtb: Zxtb.
let isCodeGenOnly = 1, hasSideEffects = 0, hasNewValue = 1, opNewValue = 0 in
def V4_SA1_zxtb: SUBInst <
  (outs IntRegs:$Rd),
  (ins IntRegs:$Rs),
  "$Rd = and($Rs, #255)"> {
    bits<4> Rd;
    bits<4> Rs;

    let Inst{12-8} = 0b10111;
    let Inst{3-0} = Rd;
    let Inst{7-4} = Rs;
  }

// SA1_addsp: Add.
let Uses = [R29], isCodeGenOnly = 1, hasSideEffects = 0, hasNewValue = 1, opNewValue = 0 in
def V4_SA1_addsp: SUBInst <
  (outs IntRegs:$Rd),
  (ins u6_2Imm:$u6_2),
  "$Rd = add(r29, #$u6_2)"> {
    bits<4> Rd;
    bits<8> u6_2;

    let Inst{12-10} = 0b011;
    let Inst{3-0} = Rd;
    let Inst{9-4} = u6_2{7-2};
  }

// SL2_loadri_sp: Load word.
let Uses = [R29], isCodeGenOnly = 1, mayLoad = 1, accessSize = WordAccess, hasNewValue = 1, opNewValue = 0 in
def V4_SL2_loadri_sp: SUBInst <
  (outs IntRegs:$Rd),
  (ins u5_2Imm:$u5_2),
  "$Rd = memw(r29 + #$u5_2)"> {
    bits<4> Rd;
    bits<7> u5_2;

    let Inst{12-9} = 0b1110;
    let Inst{3-0} = Rd;
    let Inst{8-4} = u5_2{6-2};
  }

// SS1_storeb_io: Store byte.
let isCodeGenOnly = 1, mayStore = 1, accessSize = ByteAccess in
def V4_SS1_storeb_io: SUBInst <
  (outs ),
  (ins IntRegs:$Rs, u4_0Imm:$u4_0, IntRegs:$Rt),
  "memb($Rs + #$u4_0) = $Rt"> {
    bits<4> Rs;
    bits<4> u4_0;
    bits<4> Rt;

    let Inst{12} = 0b1;
    let Inst{7-4} = Rs;
    let Inst{11-8} = u4_0;
    let Inst{3-0} = Rt;
  }

// SL2_return_tnew: Deallocate stack frame and return.
let Defs = [PC, R31, R29, R30], Uses = [R30, P0], isCodeGenOnly = 1, isPredicated = 1, isPredicatedNew = 1, mayLoad = 1, accessSize = DoubleWordAccess, isBranch = 1, isIndirectBranch = 1 in
def V4_SL2_return_tnew: SUBInst <
  (outs ),
  (ins ),
  "if (p0.new) dealloc_return:nt"> {
    let Inst{12-6} = 0b1111101;
    let Inst{2-0} = 0b110;
  }

// SL2_return_fnew: Deallocate stack frame and return.
let Defs = [PC, R31, R29, R30], Uses = [R30, P0], isCodeGenOnly = 1, isPredicated = 1, isPredicatedFalse = 1, isPredicatedNew = 1, mayLoad = 1, accessSize = DoubleWordAccess, isBranch = 1, isIndirectBranch = 1 in
def V4_SL2_return_fnew: SUBInst <
  (outs ),
  (ins ),
  "if (!p0.new) dealloc_return:nt"> {
    let Inst{12-6} = 0b1111101;
    let Inst{2-0} = 0b111;
  }

// SA1_zxth: Zxth.
let isCodeGenOnly = 1, hasSideEffects = 0, hasNewValue = 1, opNewValue = 0 in
def V4_SA1_zxth: SUBInst <
  (outs IntRegs:$Rd),
  (ins IntRegs:$Rs),
  "$Rd = zxth($Rs)"> {
    bits<4> Rd;
    bits<4> Rs;

    let Inst{12-8} = 0b10110;
    let Inst{3-0} = Rd;
    let Inst{7-4} = Rs;
  }

