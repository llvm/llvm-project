//==- HexagonSystemInst.td - System Instructions for Hexagon -*- tablegen -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the Hexagon instructions in TableGen format.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//                     Cache manipulation instructions.
//===----------------------------------------------------------------------===//
let mayStore = 1 in
class ST_MISC_CACHEOP<dag outs, dag ins,
              string asmstr, list<dag> pattern = [],
              bits<3> amode, bits<3> type, bits<1> un>
  : ST0Inst<outs, ins, asmstr, pattern, "", ST_tc_ld_SLOT0> {

    bits<5> Rs;
    bits<5> Rt;
    bits<5> Rd;
    let Inst{31-28} = 0b1010;
    let Inst{27-25} = amode;
    let Inst{24-22} = type;
    let Inst{21}    = un;
    let Inst{20-16} = Rs;
    let Inst{12-8}  = Rt;
    let Inst{4-0}   = Rd;
}

let mayStore = 1 in
class ST_MISC_CACHEOP_SYS<dag outs, dag ins,
              string asmstr, list<dag> pattern = [],
              bits<3> amode, bits<3> type, bits<1> un>
  : SYSInst<outs, ins, asmstr, pattern, ""> {

    bits<5> Rs;
    bits<5> Rt;
    bits<5> Rd;
    let Inst{31-28} = 0b1010;
    let Inst{27-25} = amode;
    let Inst{24-22} = type;
    let Inst{21}    = un;
    let Inst{20-16} = Rs;
    let Inst{12-8}  = Rt;
    let Inst{4-0}   = Rd;
}


let isSolo = 1, Rs = 0, Rt = 0, Rd = 0 in {
def Y2_syncht: ST_MISC_CACHEOP <(outs), (ins),
    "syncht" , [], 0b100, 0b001, 0b0>;
}

let Rt = 0, Rd = 0 in {
let isSoloAin1 = 1 in {
  def Y2_dccleana: ST_MISC_CACHEOP <(outs), (ins IntRegs:$Rs),
      "dccleana($Rs)", [], 0b000, 0b000, 0b0>;
  def Y2_dcinva: ST_MISC_CACHEOP <(outs), (ins IntRegs:$Rs),
      "dcinva($Rs)", [], 0b000, 0b000, 0b1>;
  def Y2_dccleaninva: ST_MISC_CACHEOP <(outs), (ins IntRegs:$Rs),
      "dccleaninva($Rs)", [], 0b000, 0b001, 0b0>;
  }
}

let isSoloAX = 1, hasSideEffects = 1, Rd = 0 in {
  def Y4_l2fetch: ST_MISC_CACHEOP_SYS<(outs), (ins IntRegs:$Rs, IntRegs:$Rt),
      "l2fetch($Rs, $Rt)", [], 0b011, 0b000, 0b0>;
  def Y5_l2fetch: ST_MISC_CACHEOP_SYS<(outs), (ins IntRegs:$Rs, DoubleRegs:$Rt),
      "l2fetch($Rs, $Rt)", [], 0b011, 0b010, 0b0>;
}

let hasSideEffects = 0, isSolo = 1 in
class Y2_INVALIDATE_CACHE<string mnemonic, bit MajOp>
  : JRInst <
  (outs), (ins IntRegs:$Rs),
  #mnemonic#"($Rs)" > {
    bits<5> Rs;

    let IClass = 0b0101;
    let Inst{27-21} = 0b0110110;
    let Inst{20-16} = Rs;
    let Inst{13-12} = 0b00;
    let Inst{11} = MajOp;
  }
// Instruction cache invalidate
def Y2_icinva : Y2_INVALIDATE_CACHE<"icinva", 0b0>;

// Zero an aligned 32-byte cacheline.
let isSoloAin1 = 1 in
def Y2_dczeroa: ST0Inst <(outs), (ins IntRegs:$Rs),
  "dczeroa($Rs)"> {
    bits<5> Rs;
    let IClass = 0b1010;
    let Inst{27-21} = 0b0000110;
    let Inst{13} = 0b0;
    let Inst{20-16} = Rs;
  }

// Memory synchronization.
let hasSideEffects = 0, isSolo = 1 in
def Y2_isync: JRInst <(outs), (ins),
  "isync"> {
    let IClass = 0b0101;
    let Inst{27-16} = 0b011111000000;
    let Inst{13} = 0b0;
    let Inst{9-0} = 0b0000000010;
  }

