//===- IA64InstrInfo.td - Describe the IA64 Instruction Set -----*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file describes the IA64 instruction set, defining the instructions, and
// properties of the instructions which are needed for code generation, machine
// code emission, and analysis.
//
//===----------------------------------------------------------------------===//

include "IA64InstrFormats.td"

//===----------------------------------------------------------------------===//
// IA-64 specific DAG Nodes.
//

def IA64getfd : SDNode<"IA64ISD::GETFD", SDTFPToIntOp, []>;

def retflag         : SDNode<"IA64ISD::RET_FLAG", SDTNone,
                           [SDNPHasChain, SDNPOptInFlag]>;

//===---------
// Instruction types

class isA { bit A=1; } // I or M unit
class isM { bit M=1; } // M unit
class isI { bit I=1; } // I unit
class isB { bit B=1; } // B unit
class isF { bit F=1; } // F unit
class isLX { bit LX=1; } // I/B

//===---------

def u2imm : Operand<i8>;
def u6imm : Operand<i8>;
def s8imm : Operand<i8> {
  let PrintMethod = "printS8ImmOperand";
}
def s14imm  : Operand<i64> {
  let PrintMethod = "printS14ImmOperand";
}
def s22imm  : Operand<i64> {
  let PrintMethod = "printS22ImmOperand";
}
def u64imm  : Operand<i64> {
  let PrintMethod = "printU64ImmOperand";
}
def s64imm  : Operand<i64> {
  let PrintMethod = "printS64ImmOperand";
}

let PrintMethod = "printGlobalOperand" in
  def globaladdress : Operand<i64>;

// the asmprinter needs to know about calls
let PrintMethod = "printCallOperand" in
  def calltarget : Operand<i64>;
  
/* new daggy action!!! */

def is32ones : PatLeaf<(i64 imm), [{
  // is32ones predicate - True if the immediate is 0x00000000FFFFFFFF 
  // Used to create ZXT4s appropriately 
  uint64_t v = (uint64_t)N->getZExtValue();
  return (v == 0x00000000FFFFFFFFLL);
}]>;

// isMIXable predicates - True if the immediate is
// 0xFF00FF00FF00FF00, 0x00FF00FF00FF00FF
// etc, through 0x00000000FFFFFFFF
// Used to test for the suitability of mix* 
def isMIX1Lable: PatLeaf<(i64 imm), [{
  return((uint64_t)N->getZExtValue()==0xFF00FF00FF00FF00LL);
}]>;
def isMIX1Rable: PatLeaf<(i64 imm), [{
  return((uint64_t)N->getZExtValue()==0x00FF00FF00FF00FFLL);
}]>;
def isMIX2Lable: PatLeaf<(i64 imm), [{
  return((uint64_t)N->getZExtValue()==0xFFFF0000FFFF0000LL);
}]>;
def isMIX2Rable: PatLeaf<(i64 imm), [{
  return((uint64_t)N->getZExtValue()==0x0000FFFF0000FFFFLL);
}]>;
def isMIX4Lable: PatLeaf<(i64 imm), [{
  return((uint64_t)N->getZExtValue()==0xFFFFFFFF00000000LL);
}]>;
def isMIX4Rable: PatLeaf<(i64 imm), [{
  return((uint64_t)N->getZExtValue()==0x00000000FFFFFFFFLL);
}]>;

def isSHLADDimm: PatLeaf<(i64 imm), [{
  // isSHLADDimm predicate - True if the immediate is exactly 1, 2, 3 or 4
  // - 0 is *not* okay.
  // Used to create shladd instructions appropriately
  int64_t v = (int64_t)N->getZExtValue();
  return (v >= 1 && v <= 4);
}]>;

def immSExt14  : PatLeaf<(i64 imm), [{
  // immSExt14 predicate - True if the immediate fits in a 14-bit sign extended
  // field.  Used by instructions like 'adds'.
  int64_t v = (int64_t)N->getZExtValue();
  return (v <= 8191 && v >= -8192);
}]>;

// imm64 predicate - True if the immediate fits in a 64-bit 
// field - i.e., true. used to keep movl happy
def imm64  : PatLeaf<(i64 imm)>; 

def ADD  : AForm_DAG<0x03, 0x0b, (outs GR:$dst), (ins GR:$src1, GR:$src2),
           "add $dst = $src1, $src2",
           [(set GR:$dst, (add GR:$src1, GR:$src2))]>, isA;

def ADD1 : AForm_DAG<0x03, 0x0b, (outs GR:$dst), (ins GR:$src1, GR:$src2),
           "add $dst = $src1, $src2, 1",
           [(set GR:$dst, (add (add GR:$src1, GR:$src2), 1))]>, isA;

def ADDS : AForm_DAG<0x03, 0x0b, (outs GR:$dst), (ins GR:$src1, s14imm:$imm),
           "adds $dst = $imm, $src1",
           [(set GR:$dst, (add GR:$src1, immSExt14:$imm))]>, isA;
 
def MOVL : AForm_DAG<0x03, 0x0b, (outs GR:$dst), (ins s64imm:$imm),
           "movl $dst = $imm",
           [(set GR:$dst, imm64:$imm)]>, isLX;

def ADDL_GA : AForm_DAG<0x03, 0x0b, (outs GR:$dst), (ins GR:$src1, globaladdress:$imm),
           "addl $dst = $imm, $src1",
           []>, isA;

// hmm 
def ADDL_EA : AForm_DAG<0x03, 0x0b, (outs GR:$dst), (ins GR:$src1, calltarget:$imm),
           "addl $dst = $imm, $src1",
           []>, isA;
 
def SUB  : AForm_DAG<0x03, 0x0b, (outs GR:$dst), (ins GR:$src1, GR:$src2),
           "sub $dst = $src1, $src2",
           [(set GR:$dst, (sub GR:$src1, GR:$src2))]>, isA;

def SUB1 : AForm_DAG<0x03, 0x0b, (outs GR:$dst), (ins GR:$src1, GR:$src2),
           "sub $dst = $src1, $src2, 1",
           [(set GR:$dst, (add (sub GR: $src1, GR:$src2), -1))]>, isA;

let isTwoAddress = 1 in {
def TPCADDIMM22 : AForm<0x03, 0x0b,
  (outs GR:$dst), (ins GR:$src1, s22imm:$imm, PR:$qp),
    "($qp) add $dst = $imm, $dst">, isA;
def TPCADDS : AForm_DAG<0x03, 0x0b,
  (outs GR:$dst), (ins GR:$src1, s14imm:$imm, PR:$qp),
    "($qp) adds $dst = $imm, $dst",
    []>, isA;
def TPCMPIMM8NE : AForm<0x03, 0x0b,
  (outs PR:$dst), (ins PR:$src1, s22imm:$imm, GR:$src2, PR:$qp),
    "($qp) cmp.ne $dst , p0 = $imm, $src2">, isA;
}

// zero extend a bool (predicate reg) into an integer reg
def ZXTb : Pat<(zext PR:$src),
          (TPCADDIMM22 (ADDS r0, 0), 1, PR:$src)>;
def AXTb : Pat<(anyext PR:$src),
          (TPCADDIMM22 (ADDS r0, 0), 1, PR:$src)>; 

// normal sign/zero-extends
def SXT1 : AForm_DAG<0x03, 0x0b, (outs GR:$dst), (ins GR:$src), "sxt1 $dst = $src",
           [(set GR:$dst, (sext_inreg GR:$src, i8))]>, isI;
def ZXT1 : AForm_DAG<0x03, 0x0b, (outs GR:$dst), (ins GR:$src), "zxt1 $dst = $src",
           [(set GR:$dst, (and GR:$src, 255))]>, isI;
def SXT2 : AForm_DAG<0x03, 0x0b, (outs GR:$dst), (ins GR:$src), "sxt2 $dst = $src",
           [(set GR:$dst, (sext_inreg GR:$src, i16))]>, isI;
def ZXT2 : AForm_DAG<0x03, 0x0b, (outs GR:$dst), (ins GR:$src), "zxt2 $dst = $src",
           [(set GR:$dst, (and GR:$src, 65535))]>, isI;
def SXT4 : AForm_DAG<0x03, 0x0b, (outs GR:$dst), (ins GR:$src), "sxt4 $dst = $src",
           [(set GR:$dst, (sext_inreg GR:$src, i32))]>, isI;
def ZXT4 : AForm_DAG<0x03, 0x0b, (outs GR:$dst), (ins GR:$src), "zxt4 $dst = $src",
           [(set GR:$dst, (and GR:$src, is32ones))]>, isI;

// fixme: shrs vs shru?
def MIX1L : AForm_DAG<0x03, 0x0b, (outs GR:$dst), (ins GR:$src1, GR:$src2),
          "mix1.l $dst = $src1, $src2",
          [(set GR:$dst, (or (and GR:$src1, isMIX1Lable),
                          (and (srl GR:$src2, (i64 8)), isMIX1Lable)))]>, isI;

def MIX2L : AForm_DAG<0x03, 0x0b, (outs GR:$dst), (ins GR:$src1, GR:$src2),
          "mix2.l $dst = $src1, $src2",
          [(set GR:$dst, (or (and GR:$src1, isMIX2Lable),
                          (and (srl GR:$src2, (i64 16)), isMIX2Lable)))]>, isI;

def MIX4L : AForm_DAG<0x03, 0x0b, (outs GR:$dst), (ins GR:$src1, GR:$src2),
          "mix4.l $dst = $src1, $src2",
          [(set GR:$dst, (or (and GR:$src1, isMIX4Lable),
                          (and (srl GR:$src2, (i64 32)), isMIX4Lable)))]>, isI;

def MIX1R : AForm_DAG<0x03, 0x0b, (outs GR:$dst), (ins GR:$src1, GR:$src2),
          "mix1.r $dst = $src1, $src2",
          [(set GR:$dst, (or (and (shl GR:$src1, (i64 8)), isMIX1Rable),
                          (and GR:$src2, isMIX1Rable)))]>, isI;

def MIX2R : AForm_DAG<0x03, 0x0b, (outs GR:$dst), (ins GR:$src1, GR:$src2),
          "mix2.r $dst = $src1, $src2",
          [(set GR:$dst, (or (and (shl GR:$src1, (i64 16)), isMIX2Rable),
                          (and GR:$src2, isMIX2Rable)))]>, isI;

def MIX4R : AForm_DAG<0x03, 0x0b, (outs GR:$dst), (ins GR:$src1, GR:$src2),
          "mix4.r $dst = $src1, $src2",
          [(set GR:$dst, (or (and (shl GR:$src1, (i64 32)), isMIX4Rable),
                          (and GR:$src2, isMIX4Rable)))]>, isI;

def GETFSIGD : AForm_DAG<0x03, 0x0b, (outs GR:$dst), (ins FP:$src),
  "getf.sig $dst = $src",
  []>, isM;

def SETFSIGD : AForm_DAG<0x03, 0x0b, (outs FP:$dst), (ins GR:$src),
  "setf.sig $dst = $src",
  []>, isM;

def XMALD : AForm_DAG<0x03, 0x0b, (outs FP:$dst), (ins FP:$src1, FP:$src2, FP:$src3),
  "xma.l $dst = $src1, $src2, $src3",
  []>, isF;
def XMAHD : AForm_DAG<0x03, 0x0b, (outs FP:$dst), (ins FP:$src1, FP:$src2, FP:$src3),
  "xma.h $dst = $src1, $src2, $src3",
  []>, isF;
def XMAHUD : AForm_DAG<0x03, 0x0b, (outs FP:$dst), (ins FP:$src1, FP:$src2, FP:$src3),
  "xma.hu $dst = $src1, $src2, $src3",
  []>, isF;

// pseudocode for integer multiplication 
def : Pat<(mul GR:$src1, GR:$src2),
           (GETFSIGD (XMALD (SETFSIGD GR:$src1), (SETFSIGD GR:$src2), F0))>;
def : Pat<(mulhs GR:$src1, GR:$src2),
           (GETFSIGD (XMAHD (SETFSIGD GR:$src1), (SETFSIGD GR:$src2), F0))>;
def : Pat<(mulhu GR:$src1, GR:$src2),
           (GETFSIGD (XMAHUD (SETFSIGD GR:$src1), (SETFSIGD GR:$src2), F0))>;

// TODO: addp4 (addp4 dst = src, r0 is a 32-bit add)
// has imm form, too

// def ADDS : AForm<0x03, 0x0b, (outs GR:$dst), (ins GR:$src1, s14imm:$imm),
//   "adds $dst = $imm, $src1">;

def AND   : AForm_DAG<0x03, 0x0b, (outs GR:$dst), (ins GR:$src1, GR:$src2),
          "and $dst = $src1, $src2",
          [(set GR:$dst, (and GR:$src1, GR:$src2))]>, isA;
def ANDCM : AForm_DAG<0x03, 0x0b, (outs GR:$dst), (ins GR:$src1, GR:$src2),
          "andcm $dst = $src1, $src2",
          [(set GR:$dst, (and GR:$src1, (not GR:$src2)))]>, isA;
// TODO: and/andcm/or/xor/add/sub/shift immediate forms
def OR    : AForm_DAG<0x03, 0x0b, (outs GR:$dst), (ins GR:$src1, GR:$src2),
          "or $dst = $src1, $src2",
          [(set GR:$dst, (or GR:$src1, GR:$src2))]>, isA;

def pOR   : AForm<0x03, 0x0b, (outs GR:$dst), (ins GR:$src1, GR:$src2, PR:$qp),
          "($qp) or $dst = $src1, $src2">, isA;

// the following are all a bit unfortunate: we throw away the complement
// of the compare!
def CMPEQ : AForm_DAG<0x03, 0x0b, (outs PR:$dst), (ins GR:$src1, GR:$src2),
          "cmp.eq $dst, p0 = $src1, $src2",
          [(set PR:$dst, (seteq GR:$src1, GR:$src2))]>, isA;
def CMPGT : AForm_DAG<0x03, 0x0b, (outs PR:$dst), (ins GR:$src1, GR:$src2),
          "cmp.gt $dst, p0 = $src1, $src2",
          [(set PR:$dst, (setgt GR:$src1, GR:$src2))]>, isA;
def CMPGE : AForm_DAG<0x03, 0x0b, (outs PR:$dst), (ins GR:$src1, GR:$src2),
          "cmp.ge $dst, p0 = $src1, $src2",
          [(set PR:$dst, (setge GR:$src1, GR:$src2))]>, isA;
def CMPLT : AForm_DAG<0x03, 0x0b, (outs PR:$dst), (ins GR:$src1, GR:$src2),
          "cmp.lt $dst, p0 = $src1, $src2",
          [(set PR:$dst, (setlt GR:$src1, GR:$src2))]>, isA;
def CMPLE : AForm_DAG<0x03, 0x0b, (outs PR:$dst), (ins GR:$src1, GR:$src2),
          "cmp.le $dst, p0 = $src1, $src2",
          [(set PR:$dst, (setle GR:$src1, GR:$src2))]>, isA;
def CMPNE : AForm_DAG<0x03, 0x0b, (outs PR:$dst), (ins GR:$src1, GR:$src2),
          "cmp.ne $dst, p0 = $src1, $src2",
          [(set PR:$dst, (setne GR:$src1, GR:$src2))]>, isA;
def CMPLTU: AForm_DAG<0x03, 0x0b, (outs PR:$dst), (ins GR:$src1, GR:$src2),
          "cmp.ltu $dst, p0 = $src1, $src2",
          [(set PR:$dst, (setult GR:$src1, GR:$src2))]>, isA;
def CMPGTU: AForm_DAG<0x03, 0x0b, (outs PR:$dst), (ins GR:$src1, GR:$src2),
          "cmp.gtu $dst, p0 = $src1, $src2",
          [(set PR:$dst, (setugt GR:$src1, GR:$src2))]>, isA;
def CMPLEU: AForm_DAG<0x03, 0x0b, (outs PR:$dst), (ins GR:$src1, GR:$src2),
          "cmp.leu $dst, p0 = $src1, $src2",
          [(set PR:$dst, (setule GR:$src1, GR:$src2))]>, isA;
def CMPGEU: AForm_DAG<0x03, 0x0b, (outs PR:$dst), (ins GR:$src1, GR:$src2),
          "cmp.geu $dst, p0 = $src1, $src2",
          [(set PR:$dst, (setuge GR:$src1, GR:$src2))]>, isA;

// and we do the whole thing again for FP compares!
def FCMPEQ : AForm_DAG<0x03, 0x0b, (outs PR:$dst), (ins FP:$src1, FP:$src2),
          "fcmp.eq $dst, p0 = $src1, $src2",
          [(set PR:$dst, (seteq FP:$src1, FP:$src2))]>, isF;
def FCMPGT : AForm_DAG<0x03, 0x0b, (outs PR:$dst), (ins FP:$src1, FP:$src2),
          "fcmp.gt $dst, p0 = $src1, $src2",
          [(set PR:$dst, (setgt FP:$src1, FP:$src2))]>, isF;
def FCMPGE : AForm_DAG<0x03, 0x0b, (outs PR:$dst), (ins FP:$src1, FP:$src2),
          "fcmp.ge $dst, p0 = $src1, $src2",
          [(set PR:$dst, (setge FP:$src1, FP:$src2))]>, isF;
def FCMPLT : AForm_DAG<0x03, 0x0b, (outs PR:$dst), (ins FP:$src1, FP:$src2),
          "fcmp.lt $dst, p0 = $src1, $src2",
          [(set PR:$dst, (setlt FP:$src1, FP:$src2))]>, isF;
def FCMPLE : AForm_DAG<0x03, 0x0b, (outs PR:$dst), (ins FP:$src1, FP:$src2),
          "fcmp.le $dst, p0 = $src1, $src2",
          [(set PR:$dst, (setle FP:$src1, FP:$src2))]>, isF;
def FCMPNE : AForm_DAG<0x03, 0x0b, (outs PR:$dst), (ins FP:$src1, FP:$src2),
          "fcmp.neq $dst, p0 = $src1, $src2",
          [(set PR:$dst, (setne FP:$src1, FP:$src2))]>, isF;
def FCMPLTU: AForm_DAG<0x03, 0x0b, (outs PR:$dst), (ins FP:$src1, FP:$src2),
          "fcmp.lt $dst, p0 = $src1, $src2",
          [(set PR:$dst, (setult FP:$src1, FP:$src2))]>, isF;
def FCMPGTU: AForm_DAG<0x03, 0x0b, (outs PR:$dst), (ins FP:$src1, FP:$src2),
          "fcmp.gt $dst, p0 = $src1, $src2",
          [(set PR:$dst, (setugt FP:$src1, FP:$src2))]>, isF;
def FCMPLEU: AForm_DAG<0x03, 0x0b, (outs PR:$dst), (ins FP:$src1, FP:$src2),
          "fcmp.le $dst, p0 = $src1, $src2",
          [(set PR:$dst, (setule FP:$src1, FP:$src2))]>, isF;
def FCMPGEU: AForm_DAG<0x03, 0x0b, (outs PR:$dst), (ins FP:$src1, FP:$src2),
          "fcmp.ge $dst, p0 = $src1, $src2",
          [(set PR:$dst, (setuge FP:$src1, FP:$src2))]>, isF;

def PCMPEQUNCR0R0 : AForm<0x03, 0x0b, (outs PR:$dst), (ins PR:$qp),
    "($qp) cmp.eq.unc $dst, p0 = r0, r0">, isA;

def : Pat<(trunc GR:$src),  // truncate i64 to i1
          (CMPNE GR:$src, r0)>; // $src!=0? If so, PR:$dst=true

let isTwoAddress=1 in {
  def TPCMPEQR0R0 : AForm<0x03, 0x0b, (outs PR:$dst), (ins PR:$bogus, PR:$qp),
    "($qp) cmp.eq $dst, p0 = r0, r0">, isA;
  def TPCMPNER0R0 : AForm<0x03, 0x0b, (outs PR:$dst), (ins PR:$bogus, PR:$qp),
    "($qp) cmp.ne $dst, p0 = r0, r0">, isA;
}

/* our pseudocode for OR on predicates is:
pC = pA OR pB
-------------
(pA) cmp.eq.unc pC,p0 = r0,r0  // pC = pA
 ;;
(pB) cmp.eq pC,p0 = r0,r0 // if (pB) pC = 1 */

def bOR   : Pat<(or PR:$src1, PR:$src2),
          (TPCMPEQR0R0 (PCMPEQUNCR0R0 PR:$src1), PR:$src2)>;

/* our pseudocode for AND on predicates is:
 *
(pA) cmp.eq.unc pC,p0 = r0,r0   // pC = pA
     cmp.eq pTemp,p0 = r0,r0    // pTemp = NOT pB
     ;;
(pB) cmp.ne pTemp,p0 = r0,r0
     ;;
(pTemp)cmp.ne pC,p0 = r0,r0    // if (NOT pB) pC = 0  */

def bAND  : Pat<(and PR:$src1, PR:$src2),
          ( TPCMPNER0R0 (PCMPEQUNCR0R0 PR:$src1),
            (TPCMPNER0R0 (CMPEQ r0, r0), PR:$src2) )>;

/* one possible routine for XOR on predicates is:

      // Compute px = py ^ pz
        // using sum of products: px = (py & !pz) | (pz & !py)
        // Uses 5 instructions in 3 cycles.
        // cycle 1
(pz)    cmp.eq.unc      px = r0, r0     // px = pz
(py)    cmp.eq.unc      pt = r0, r0     // pt = py
        ;;
        // cycle 2
(pt)    cmp.ne.and      px = r0, r0     // px = px & !pt (px = pz & !pt)
(pz)    cmp.ne.and      pt = r0, r0     // pt = pt & !pz
        ;;
        } { .mmi
        // cycle 3
(pt)    cmp.eq.or       px = r0, r0     // px = px | pt

*** Another, which we use here, requires one scratch GR. it is:

        mov             rt = 0          // initialize rt off critical path
        ;;

        // cycle 1
(pz)    cmp.eq.unc      px = r0, r0     // px = pz
(pz)    mov             rt = 1          // rt = pz
        ;;
        // cycle 2
(py)    cmp.ne          px = 1, rt      // if (py) px = !pz

.. these routines kindly provided by Jim Hull
*/
  
def bXOR  : Pat<(xor PR:$src1, PR:$src2),
          (TPCMPIMM8NE (PCMPEQUNCR0R0 PR:$src2), 1,
                        (TPCADDS (ADDS r0, 0), 1, PR:$src2),
                         PR:$src1)>;

def XOR   : AForm_DAG<0x03, 0x0b, (outs GR:$dst), (ins GR:$src1, GR:$src2),
          "xor $dst = $src1, $src2",
          [(set GR:$dst, (xor GR:$src1, GR:$src2))]>, isA;

def SHLADD: AForm_DAG<0x03, 0x0b, (outs GR:$dst), (ins GR:$src1,s64imm:$imm,GR:$src2),
          "shladd $dst = $src1, $imm, $src2",
          [(set GR:$dst, (add GR:$src2, (shl GR:$src1, isSHLADDimm:$imm)))]>, isA;

def SHL   : AForm_DAG<0x03, 0x0b, (outs GR:$dst), (ins GR:$src1, GR:$src2),
          "shl $dst = $src1, $src2",
          [(set GR:$dst, (shl GR:$src1, GR:$src2))]>, isI;

def SHRU  : AForm_DAG<0x03, 0x0b, (outs GR:$dst), (ins GR:$src1, GR:$src2),
          "shr.u $dst = $src1, $src2",
          [(set GR:$dst, (srl GR:$src1, GR:$src2))]>, isI;

def SHRS  : AForm_DAG<0x03, 0x0b, (outs GR:$dst), (ins GR:$src1, GR:$src2),
          "shr $dst = $src1, $src2",
          [(set GR:$dst, (sra GR:$src1, GR:$src2))]>, isI;

def MOV : AForm<0x03, 0x0b, (outs GR:$dst), (ins GR:$src), "mov $dst = $src">, isA;
def FMOV : AForm<0x03, 0x0b, (outs FP:$dst), (ins FP:$src),
  "mov $dst = $src">, isF; // XXX: there _is_ no fmov
def PMOV : AForm<0x03, 0x0b, (outs GR:$dst), (ins GR:$src, PR:$qp),
  "($qp) mov $dst = $src">, isA;

def SPILL_ALL_PREDICATES_TO_GR : AForm<0x03, 0x0b, (outs GR:$dst), (ins),
  "mov $dst = pr">, isI;
def FILL_ALL_PREDICATES_FROM_GR : AForm<0x03, 0x0b, (outs), (ins GR:$src),
  "mov pr = $src">, isI;

let isTwoAddress = 1 in {
  def CMOV : AForm<0x03, 0x0b, (outs GR:$dst), (ins GR:$src2, GR:$src, PR:$qp),
    "($qp) mov $dst = $src">, isA;
}

def PFMOV : AForm<0x03, 0x0b, (outs FP:$dst), (ins FP:$src, PR:$qp),
  "($qp) mov $dst = $src">, isF;

let isTwoAddress = 1 in {
  def CFMOV : AForm<0x03, 0x0b, (outs FP:$dst), (ins FP:$src2, FP:$src, PR:$qp),
    "($qp) mov $dst = $src">, isF;
}

def SELECTINT : Pat<(select PR:$which, GR:$src1, GR:$src2),
          (CMOV (MOV GR:$src2), GR:$src1, PR:$which)>; // note order!
def SELECTFP : Pat<(select PR:$which, FP:$src1, FP:$src2),
          (CFMOV (FMOV FP:$src2), FP:$src1, PR:$which)>; // note order!
// TODO: can do this faster, w/o using any integer regs (see pattern isel)
def SELECTBOOL : Pat<(select PR:$which, PR:$src1, PR:$src2), // note order!
          (CMPNE (CMOV
            (MOV (TPCADDIMM22 (ADDS r0, 0), 1, PR:$src2)),
            (TPCADDIMM22 (ADDS r0, 0), 1, PR:$src1), PR:$which), r0)>;

// load constants of various sizes // FIXME: prettyprint -ve constants
def : Pat<(i64 immSExt14:$imm), (ADDS r0, immSExt14:$imm)>;
def : Pat<(i1 -1), (CMPEQ r0, r0)>; // TODO: this should just be a ref to p0
def : Pat<(i1  0), (CMPNE r0, r0)>; // TODO: any instruction actually *using*
                                    //       this predicate should be killed!

// TODO: support postincrement (reg, imm9) loads+stores - this needs more
// tablegen support

def IUSE : PseudoInstIA64<(outs), (ins variable_ops), "// IUSE">;
def ADJUSTCALLSTACKUP : PseudoInstIA64<(outs), (ins variable_ops),
                                        "// ADJUSTCALLSTACKUP">;
def ADJUSTCALLSTACKDOWN : PseudoInstIA64<(outs), (ins variable_ops),
                                         "// ADJUSTCALLSTACKDOWN">;
def PSEUDO_ALLOC : PseudoInstIA64<(outs), (ins GR:$foo), "// PSEUDO_ALLOC">;

def ALLOC : AForm<0x03, 0x0b,
  (outs GR:$dst), (ins i8imm:$inputs, i8imm:$locals, i8imm:$outputs, i8imm:$rotating),
    "alloc $dst = ar.pfs,$inputs,$locals,$outputs,$rotating">, isM;

let isTwoAddress = 1 in {
  def TCMPNE : AForm<0x03, 0x0b,
  (outs PR:$dst), (ins PR:$src2, GR:$src3, GR:$src4),
    "cmp.ne $dst, p0 = $src3, $src4">, isA;
  
  def TPCMPEQOR : AForm<0x03, 0x0b,
  (outs PR:$dst), (ins PR:$src2, GR:$src3, GR:$src4, PR:$qp),
    "($qp) cmp.eq.or $dst, p0 = $src3, $src4">, isA;
  
  def TPCMPNE : AForm<0x03, 0x0b,
  (outs PR:$dst), (ins PR:$src2, GR:$src3, GR:$src4, PR:$qp),
    "($qp) cmp.ne $dst, p0 = $src3, $src4">, isA;
  
  def TPCMPEQ : AForm<0x03, 0x0b,
  (outs PR:$dst), (ins PR:$src2, GR:$src3, GR:$src4, PR:$qp),
    "($qp) cmp.eq $dst, p0 = $src3, $src4">, isA;
}

def MOVSIMM14 : AForm<0x03, 0x0b, (outs GR:$dst), (ins s14imm:$imm),
  "mov $dst = $imm">, isA;
def MOVSIMM22 : AForm<0x03, 0x0b, (outs GR:$dst), (ins s22imm:$imm),
  "mov $dst = $imm">, isA;
def MOVLIMM64 : AForm<0x03, 0x0b, (outs GR:$dst), (ins s64imm:$imm),
  "movl $dst = $imm">, isLX;

def SHLI : AForm<0x03, 0x0b, (outs GR:$dst), (ins GR:$src1, u6imm:$imm), 
  "shl $dst = $src1, $imm">, isI;
def SHRUI : AForm<0x03, 0x0b, (outs GR:$dst), (ins GR:$src1, u6imm:$imm),
  "shr.u $dst = $src1, $imm">, isI;
def SHRSI : AForm<0x03, 0x0b, (outs GR:$dst), (ins GR:$src1, u6imm:$imm),
  "shr $dst = $src1, $imm">, isI;

def EXTRU : AForm<0x03, 0x0b,
  (outs GR:$dst), (ins GR:$src1, u6imm:$imm1, u6imm:$imm2),
  "extr.u $dst = $src1, $imm1, $imm2">, isI;

def DEPZ : AForm<0x03, 0x0b,
  (outs GR:$dst), (ins GR:$src1, u6imm:$imm1, u6imm:$imm2),
  "dep.z $dst = $src1, $imm1, $imm2">, isI;

def PCMPEQOR : AForm<0x03, 0x0b, (outs PR:$dst), (ins GR:$src1, GR:$src2, PR:$qp),
  "($qp) cmp.eq.or $dst, p0 = $src1, $src2">, isA;
def PCMPEQUNC : AForm<0x03, 0x0b, (outs PR:$dst), (ins GR:$src1, GR:$src2, PR:$qp),
  "($qp) cmp.eq.unc $dst, p0 = $src1, $src2">, isA;
def PCMPNE : AForm<0x03, 0x0b, (outs PR:$dst), (ins GR:$src1, GR:$src2, PR:$qp),
  "($qp) cmp.ne $dst, p0 = $src1, $src2">, isA;

// two destinations! 
def BCMPEQ : AForm<0x03, 0x0b, (outs PR:$dst1, PR:$dst2), (ins GR:$src1, GR:$src2),
  "cmp.eq $dst1, dst2 = $src1, $src2">, isA;

def ADDIMM14 : AForm<0x03, 0x0b, (outs GR:$dst), (ins GR:$src1, s14imm:$imm),
  "adds $dst = $imm, $src1">, isA;

def ADDIMM22 : AForm<0x03, 0x0b, (outs GR:$dst), (ins GR:$src1, s22imm:$imm),
  "add $dst = $imm, $src1">, isA;
def CADDIMM22 : AForm<0x03, 0x0b, (outs GR:$dst), (ins GR:$src1, s22imm:$imm, PR:$qp),
  "($qp) add $dst = $imm, $src1">, isA;

def SUBIMM8 : AForm<0x03, 0x0b, (outs GR:$dst), (ins s8imm:$imm, GR:$src2),
  "sub $dst = $imm, $src2">, isA;

let mayStore = 1 in {
  def ST1 : AForm<0x03, 0x0b, (outs), (ins GR:$dstPtr, GR:$value),
    "st1 [$dstPtr] = $value">, isM;
  def ST2 : AForm<0x03, 0x0b, (outs), (ins GR:$dstPtr, GR:$value),
    "st2 [$dstPtr] = $value">, isM;
  def ST4 : AForm<0x03, 0x0b, (outs), (ins GR:$dstPtr, GR:$value),
    "st4 [$dstPtr] = $value">, isM;
  def ST8 : AForm<0x03, 0x0b, (outs), (ins GR:$dstPtr, GR:$value),
    "st8 [$dstPtr] = $value">, isM;
  def STF4 : AForm<0x03, 0x0b, (outs), (ins GR:$dstPtr, FP:$value),
    "stfs [$dstPtr] = $value">, isM;
  def STF8 : AForm<0x03, 0x0b, (outs), (ins GR:$dstPtr, FP:$value),
    "stfd [$dstPtr] = $value">, isM;
  def STF_SPILL : AForm<0x03, 0x0b, (outs), (ins GR:$dstPtr, FP:$value),
    "stf.spill [$dstPtr] = $value">, isM;
}

let canFoldAsLoad = 1 in {
  def LD1 : AForm<0x03, 0x0b, (outs GR:$dst), (ins GR:$srcPtr),
    "ld1 $dst = [$srcPtr]">, isM;
  def LD2 : AForm<0x03, 0x0b, (outs GR:$dst), (ins GR:$srcPtr),
    "ld2 $dst = [$srcPtr]">, isM;
  def LD4 : AForm<0x03, 0x0b, (outs GR:$dst), (ins GR:$srcPtr),
    "ld4 $dst = [$srcPtr]">, isM;
  def LD8 : AForm<0x03, 0x0b, (outs GR:$dst), (ins GR:$srcPtr),
    "ld8 $dst = [$srcPtr]">, isM;
  def LDF4 : AForm<0x03, 0x0b, (outs FP:$dst), (ins GR:$srcPtr),
    "ldfs $dst = [$srcPtr]">, isM;
  def LDF8 : AForm<0x03, 0x0b, (outs FP:$dst), (ins GR:$srcPtr),
    "ldfd $dst = [$srcPtr]">, isM;
  def LDF_FILL : AForm<0x03, 0x0b, (outs FP:$dst), (ins GR:$srcPtr),
    "ldf.fill $dst = [$srcPtr]">, isM;
}

def POPCNT : AForm_DAG<0x03, 0x0b, (outs GR:$dst), (ins GR:$src),
  "popcnt $dst = $src",
  [(set GR:$dst, (ctpop GR:$src))]>, isI;

// some FP stuff:  // TODO: single-precision stuff?
def FADD : AForm_DAG<0x03, 0x0b, (outs FP:$dst), (ins FP:$src1, FP:$src2),
  "fadd $dst = $src1, $src2",
  [(set FP:$dst, (fadd FP:$src1, FP:$src2))]>, isF;
def FADDS: AForm<0x03, 0x0b, (outs FP:$dst), (ins FP:$src1, FP:$src2),
  "fadd.s $dst = $src1, $src2">, isF;
def FSUB : AForm_DAG<0x03, 0x0b, (outs FP:$dst), (ins FP:$src1, FP:$src2),
  "fsub $dst = $src1, $src2",
  [(set FP:$dst, (fsub FP:$src1, FP:$src2))]>, isF;
def FMPY : AForm_DAG<0x03, 0x0b, (outs FP:$dst), (ins FP:$src1, FP:$src2),
  "fmpy $dst = $src1, $src2",
  [(set FP:$dst, (fmul FP:$src1, FP:$src2))]>, isF;
def FMA : AForm_DAG<0x03, 0x0b, (outs FP:$dst), (ins FP:$src1, FP:$src2, FP:$src3),
  "fma $dst = $src1, $src2, $src3",
  [(set FP:$dst, (fadd (fmul FP:$src1, FP:$src2), FP:$src3))]>, isF;
def FMS : AForm_DAG<0x03, 0x0b, (outs FP:$dst), (ins FP:$src1, FP:$src2, FP:$src3),
  "fms $dst = $src1, $src2, $src3",
  [(set FP:$dst, (fsub (fmul FP:$src1, FP:$src2), FP:$src3))]>, isF;
def FNMA : AForm_DAG<0x03, 0x0b, (outs FP:$dst), (ins FP:$src1, FP:$src2, FP:$src3),
  "fnma $dst = $src1, $src2, $src3",
  [(set FP:$dst, (fneg (fadd (fmul FP:$src1, FP:$src2), FP:$src3)))]>, isF;
def FABS : AForm_DAG<0x03, 0x0b, (outs FP:$dst), (ins FP:$src),
  "fabs $dst = $src",
  [(set FP:$dst, (fabs FP:$src))]>, isF;
def FNEG : AForm_DAG<0x03, 0x0b, (outs FP:$dst), (ins FP:$src),
  "fneg $dst = $src",
  [(set FP:$dst, (fneg FP:$src))]>, isF;
def FNEGABS : AForm_DAG<0x03, 0x0b, (outs FP:$dst), (ins FP:$src),
  "fnegabs $dst = $src",
  [(set FP:$dst, (fneg (fabs FP:$src)))]>, isF;

let isTwoAddress=1 in {
def TCFMAS1 : AForm<0x03, 0x0b,
  (outs FP:$dst), (ins FP:$bogussrc, FP:$src1, FP:$src2, FP:$src3, PR:$qp),
    "($qp) fma.s1 $dst = $src1, $src2, $src3">, isF;
def TCFMADS0 : AForm<0x03, 0x0b,
  (outs FP:$dst), (ins FP:$bogussrc, FP:$src1, FP:$src2, FP:$src3, PR:$qp),
    "($qp) fma.d.s0 $dst = $src1, $src2, $src3">, isF;
}

def CFMAS1 : AForm<0x03, 0x0b,
  (outs FP:$dst), (ins FP:$src1, FP:$src2, FP:$src3, PR:$qp),
    "($qp) fma.s1 $dst = $src1, $src2, $src3">, isF;
def CFNMAS1 : AForm<0x03, 0x0b,
  (outs FP:$dst), (ins FP:$src1, FP:$src2, FP:$src3, PR:$qp),
    "($qp) fnma.s1 $dst = $src1, $src2, $src3">, isF;

def CFMADS1 : AForm<0x03, 0x0b,
  (outs FP:$dst), (ins FP:$src1, FP:$src2, FP:$src3, PR:$qp),
    "($qp) fma.d.s1 $dst = $src1, $src2, $src3">, isF;
def CFMADS0 : AForm<0x03, 0x0b,
  (outs FP:$dst), (ins FP:$src1, FP:$src2, FP:$src3, PR:$qp),
    "($qp) fma.d.s0 $dst = $src1, $src2, $src3">, isF;
def CFNMADS1 : AForm<0x03, 0x0b,
  (outs FP:$dst), (ins FP:$src1, FP:$src2, FP:$src3, PR:$qp),
    "($qp) fnma.d.s1 $dst = $src1, $src2, $src3">, isF;

def FRCPAS0 : AForm<0x03, 0x0b, (outs FP:$dstFR, PR:$dstPR), (ins FP:$src1, FP:$src2),
  "frcpa.s0 $dstFR, $dstPR = $src1, $src2">, isF;
def FRCPAS1 : AForm<0x03, 0x0b, (outs FP:$dstFR, PR:$dstPR), (ins FP:$src1, FP:$src2),
  "frcpa.s1 $dstFR, $dstPR = $src1, $src2">, isF;

def XMAL : AForm<0x03, 0x0b, (outs FP:$dst), (ins FP:$src1, FP:$src2, FP:$src3),
  "xma.l $dst = $src1, $src2, $src3">, isF;

def FCVTXF : AForm<0x03, 0x0b, (outs FP:$dst), (ins FP:$src),
  "fcvt.xf $dst = $src">, isF;
def FCVTXUF : AForm<0x03, 0x0b, (outs FP:$dst), (ins FP:$src),
  "fcvt.xuf $dst = $src">, isF;
def FCVTXUFS1 : AForm<0x03, 0x0b, (outs FP:$dst), (ins FP:$src),
  "fcvt.xuf.s1 $dst = $src">, isF;
def FCVTFX : AForm<0x03, 0x0b, (outs FP:$dst), (ins FP:$src),
  "fcvt.fx $dst = $src">, isF;
def FCVTFXU : AForm<0x03, 0x0b, (outs FP:$dst), (ins FP:$src),
  "fcvt.fxu $dst = $src">, isF;

def FCVTFXTRUNC : AForm<0x03, 0x0b, (outs FP:$dst), (ins FP:$src),
  "fcvt.fx.trunc $dst = $src">, isF;
def FCVTFXUTRUNC : AForm<0x03, 0x0b, (outs FP:$dst), (ins FP:$src),
  "fcvt.fxu.trunc $dst = $src">, isF;

def FCVTFXTRUNCS1 : AForm<0x03, 0x0b, (outs FP:$dst), (ins FP:$src),
  "fcvt.fx.trunc.s1 $dst = $src">, isF;
def FCVTFXUTRUNCS1 : AForm<0x03, 0x0b, (outs FP:$dst), (ins FP:$src),
  "fcvt.fxu.trunc.s1 $dst = $src">, isF;

def FNORMD : AForm<0x03, 0x0b, (outs FP:$dst), (ins FP:$src),
  "fnorm.d $dst = $src">, isF;

def GETFD : AForm<0x03, 0x0b, (outs GR:$dst), (ins FP:$src),
  "getf.d $dst = $src">, isM;
def SETFD : AForm<0x03, 0x0b, (outs FP:$dst), (ins GR:$src),
  "setf.d $dst = $src">, isM;

def GETFSIG : AForm<0x03, 0x0b, (outs GR:$dst), (ins FP:$src),
  "getf.sig $dst = $src">, isM;
def SETFSIG : AForm<0x03, 0x0b, (outs FP:$dst), (ins GR:$src),
  "setf.sig $dst = $src">, isM;

// these four FP<->int conversion patterns need checking/cleaning
def SINT_TO_FP : Pat<(sint_to_fp GR:$src),
  (FNORMD (FCVTXF (SETFSIG GR:$src)))>;
def UINT_TO_FP : Pat<(uint_to_fp GR:$src),
  (FNORMD (FCVTXUF (SETFSIG GR:$src)))>;
def FP_TO_SINT : Pat<(i64 (fp_to_sint FP:$src)),
  (GETFSIG (FCVTFXTRUNC FP:$src))>;
def FP_TO_UINT : Pat<(i64 (fp_to_uint FP:$src)),
  (GETFSIG (FCVTFXUTRUNC FP:$src))>;

def fpimm0 : PatLeaf<(fpimm), [{
  return N->isExactlyValue(+0.0);
}]>;
def fpimm1 : PatLeaf<(fpimm), [{
  return N->isExactlyValue(+1.0);
}]>;
def fpimmn0 : PatLeaf<(fpimm), [{
  return N->isExactlyValue(-0.0);
}]>;
def fpimmn1 : PatLeaf<(fpimm), [{
  return N->isExactlyValue(-1.0);
}]>;

def : Pat<(f64 fpimm0), (FMOV F0)>;
def : Pat<(f64 fpimm1), (FMOV F1)>;
def : Pat<(f64 fpimmn0), (FNEG F0)>;
def : Pat<(f64 fpimmn1), (FNEG F1)>;

let isTerminator = 1, isBranch = 1 in {
  def BRL_NOTCALL : RawForm<0x03, 0xb0, (outs), (ins i64imm:$dst),
    "(p0) brl.cond.sptk $dst">, isB;
  def BRLCOND_NOTCALL : RawForm<0x03, 0xb0, (outs), (ins PR:$qp, i64imm:$dst),
    "($qp) brl.cond.sptk $dst">, isB;
  def BRCOND_NOTCALL : RawForm<0x03, 0xb0, (outs), (ins PR:$qp, GR:$dst),
    "($qp) br.cond.sptk $dst">, isB;
}

let isCall = 1, /* isTerminator = 1, isBranch = 1, */
  Uses = [out0,out1,out2,out3,out4,out5,out6,out7],
// all calls clobber non-callee-saved registers, and for now, they are these:
  Defs = [r2,r3,r8,r9,r10,r11,r14,r15,r16,r17,r18,r19,r20,r21,r22,r23,r24,
  r25,r26,r27,r28,r29,r30,r31,
  p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,
  F6,F7,F8,F9,F10,F11,F12,F13,F14,F15,
  F32,F33,F34,F35,F36,F37,F38,F39,F40,F41,F42,F43,F44,F45,F46,F47,F48,F49,
  F50,F51,F52,F53,F54,F55,F56,
  F57,F58,F59,F60,F61,F62,F63,F64,F65,F66,F67,F68,F69,F70,F71,F72,F73,F74,
  F75,F76,F77,F78,F79,F80,F81,
  F82,F83,F84,F85,F86,F87,F88,F89,F90,F91,F92,F93,F94,F95,F96,F97,F98,F99,
  F100,F101,F102,F103,F104,F105,
  F106,F107,F108,F109,F110,F111,F112,F113,F114,F115,F116,F117,F118,F119,
  F120,F121,F122,F123,F124,F125,F126,F127,
  out0,out1,out2,out3,out4,out5,out6,out7] in {
// old pattern call
  def BRCALL: RawForm<0x03, 0xb0, (outs), (ins calltarget:$dst),
  "br.call.sptk rp = $dst">, isB;   // FIXME: teach llvm about branch regs?
// new daggy stuff!  

// calls a globaladdress
  def BRCALL_IPREL_GA : RawForm<0x03, 0xb0, (outs), (ins calltarget:$dst),
  "br.call.sptk rp = $dst">, isB;       // FIXME: teach llvm about branch regs?
// calls an externalsymbol
  def BRCALL_IPREL_ES : RawForm<0x03, 0xb0, (outs), (ins calltarget:$dst),
  "br.call.sptk rp = $dst">, isB;       // FIXME: teach llvm about branch regs?
// calls through a function descriptor
  def BRCALL_INDIRECT : RawForm<0x03, 0xb0, (outs), (ins GR:$branchreg),
  "br.call.sptk rp = $branchreg">, isB; // FIXME: teach llvm about branch regs?
  def BRLCOND_CALL : RawForm<0x03, 0xb0, (outs), (ins PR:$qp, i64imm:$dst),
    "($qp) brl.cond.call.sptk $dst">, isB;
  def BRCOND_CALL : RawForm<0x03, 0xb0, (outs), (ins PR:$qp, GR:$dst),
    "($qp) br.cond.call.sptk $dst">, isB;
}

// Return branch:
let isTerminator = 1, isReturn = 1 in
  def RET : AForm_DAG<0x03, 0x0b, (outs), (ins),
            "br.ret.sptk.many rp",
            [(retflag)]>, isB; // return
def : Pat<(ret), (RET)>;

// the evil stop bit of despair
def STOP : PseudoInstIA64<(outs), (ins variable_ops), ";;">;

