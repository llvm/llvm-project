//===- ARMInstrThumb.td - Thumb support for ARM ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the Thumb instruction set.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Thumb specific DAG Nodes.
//

def ARMtcall : SDNode<"ARMISD::tCALL", SDT_ARMcall,
                      [SDNPHasChain, SDNPOptInFlag, SDNPOutFlag]>;

def imm_neg_XFORM : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(-(int)N->getZExtValue(), MVT::i32);
}]>;
def imm_comp_XFORM : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(~((uint32_t)N->getZExtValue()), MVT::i32);
}]>;


/// imm0_7 predicate - True if the 32-bit immediate is in the range [0,7].
def imm0_7 : PatLeaf<(i32 imm), [{
  return (uint32_t)N->getZExtValue() < 8;
}]>;
def imm0_7_neg : PatLeaf<(i32 imm), [{
  return (uint32_t)-N->getZExtValue() < 8;
}], imm_neg_XFORM>;

def imm0_255 : PatLeaf<(i32 imm), [{
  return (uint32_t)N->getZExtValue() < 256;
}]>;
def imm0_255_comp : PatLeaf<(i32 imm), [{
  return ~((uint32_t)N->getZExtValue()) < 256;
}]>;

def imm8_255 : PatLeaf<(i32 imm), [{
  return (uint32_t)N->getZExtValue() >= 8 && (uint32_t)N->getZExtValue() < 256;
}]>;
def imm8_255_neg : PatLeaf<(i32 imm), [{
  unsigned Val = -N->getZExtValue();
  return Val >= 8 && Val < 256;
}], imm_neg_XFORM>;

// Break imm's up into two pieces: an immediate + a left shift.
// This uses thumb_immshifted to match and thumb_immshifted_val and
// thumb_immshifted_shamt to get the val/shift pieces.
def thumb_immshifted : PatLeaf<(imm), [{
  return ARM_AM::isThumbImmShiftedVal((unsigned)N->getZExtValue());
}]>;

def thumb_immshifted_val : SDNodeXForm<imm, [{
  unsigned V = ARM_AM::getThumbImmNonShiftedVal((unsigned)N->getZExtValue());
  return CurDAG->getTargetConstant(V, MVT::i32);
}]>;

def thumb_immshifted_shamt : SDNodeXForm<imm, [{
  unsigned V = ARM_AM::getThumbImmValShift((unsigned)N->getZExtValue());
  return CurDAG->getTargetConstant(V, MVT::i32);
}]>;

// Define Thumb specific addressing modes.

// t_addrmode_rr := reg + reg
//
def t_addrmode_rr : Operand<i32>,
                    ComplexPattern<i32, 2, "SelectThumbAddrModeRR", []> {
  let PrintMethod = "printThumbAddrModeRROperand";
  let MIOperandInfo = (ops GPR:$base, GPR:$offsreg);
}

// t_addrmode_s4 := reg + reg
//                  reg + imm5 * 4
//
def t_addrmode_s4 : Operand<i32>,
                    ComplexPattern<i32, 3, "SelectThumbAddrModeS4", []> {
  let PrintMethod = "printThumbAddrModeS4Operand";
  let MIOperandInfo = (ops GPR:$base, i32imm:$offsimm, GPR:$offsreg);
}

// t_addrmode_s2 := reg + reg
//                  reg + imm5 * 2
//
def t_addrmode_s2 : Operand<i32>,
                    ComplexPattern<i32, 3, "SelectThumbAddrModeS2", []> {
  let PrintMethod = "printThumbAddrModeS2Operand";
  let MIOperandInfo = (ops GPR:$base, i32imm:$offsimm, GPR:$offsreg);
}

// t_addrmode_s1 := reg + reg
//                  reg + imm5
//
def t_addrmode_s1 : Operand<i32>,
                    ComplexPattern<i32, 3, "SelectThumbAddrModeS1", []> {
  let PrintMethod = "printThumbAddrModeS1Operand";
  let MIOperandInfo = (ops GPR:$base, i32imm:$offsimm, GPR:$offsreg);
}

// t_addrmode_sp := sp + imm8 * 4
//
def t_addrmode_sp : Operand<i32>,
                    ComplexPattern<i32, 2, "SelectThumbAddrModeSP", []> {
  let PrintMethod = "printThumbAddrModeSPOperand";
  let MIOperandInfo = (ops GPR:$base, i32imm:$offsimm);
}

//===----------------------------------------------------------------------===//
//  Miscellaneous Instructions.
//

let Defs = [SP], Uses = [SP] in {
def tADJCALLSTACKUP :
PseudoInst<(outs), (ins i32imm:$amt1, i32imm:$amt2),
           "@ tADJCALLSTACKUP $amt1",
           [(ARMcallseq_end imm:$amt1, imm:$amt2)]>, Requires<[IsThumb]>;

def tADJCALLSTACKDOWN : 
PseudoInst<(outs), (ins i32imm:$amt),
           "@ tADJCALLSTACKDOWN $amt",
           [(ARMcallseq_start imm:$amt)]>, Requires<[IsThumb]>;
}

let isNotDuplicable = 1 in
def tPICADD : TIt<(outs GPR:$dst), (ins GPR:$lhs, pclabel:$cp),
                  "$cp:\n\tadd $dst, pc",
                  [(set GPR:$dst, (ARMpic_add GPR:$lhs, imm:$cp))]>;

//===----------------------------------------------------------------------===//
//  Control Flow Instructions.
//

let isReturn = 1, isTerminator = 1 in {
  def tBX_RET : TI<(outs), (ins), "bx lr", [(ARMretflag)]>;
  // Alternative return instruction used by vararg functions.
  def tBX_RET_vararg : TI<(outs), (ins GPR:$target), "bx $target", []>;
}

// FIXME: remove when we have a way to marking a MI with these properties.
let isReturn = 1, isTerminator = 1 in
def tPOP_RET : TI<(outs reglist:$dst1, variable_ops), (ins),
                   "pop $dst1", []>;

let isCall = 1, 
  Defs = [R0, R1, R2, R3, LR,
          D0, D1, D2, D3, D4, D5, D6, D7] in {
  def tBL  : TIx2<(outs), (ins i32imm:$func, variable_ops),
                   "bl ${func:call}",
                   [(ARMtcall tglobaladdr:$func)]>;
  // ARMv5T and above
  def tBLXi : TIx2<(outs), (ins i32imm:$func, variable_ops),
                    "blx ${func:call}",
                    [(ARMcall tglobaladdr:$func)]>, Requires<[HasV5T]>;
  def tBLXr : TI<(outs), (ins GPR:$func, variable_ops),
                  "blx $func",
                  [(ARMtcall GPR:$func)]>, Requires<[HasV5T]>;
  // ARMv4T
  def tBX : TIx2<(outs), (ins GPR:$func, variable_ops),
                  "cpy lr, pc\n\tbx $func",
                  [(ARMcall_nolink GPR:$func)]>;
}

let isBranch = 1, isTerminator = 1 in {
  let isBarrier = 1 in {
    let isPredicable = 1 in
    def tB   : TI<(outs), (ins brtarget:$target), "b $target",
                  [(br bb:$target)]>;

  // Far jump
  def tBfar : TIx2<(outs), (ins brtarget:$target), "bl $target\t@ far jump",[]>;

  def tBR_JTr : TJTI<(outs),
                     (ins GPR:$target, jtblock_operand:$jt, i32imm:$id),
                     "cpy pc, $target \n\t.align\t2\n$jt",
                     [(ARMbrjt GPR:$target, tjumptable:$jt, imm:$id)]>;
  }
}

// FIXME: should be able to write a pattern for ARMBrcond, but can't use
// a two-value operand where a dag node expects two operands. :( 
let isBranch = 1, isTerminator = 1 in
  def tBcc : TI<(outs), (ins brtarget:$target, pred:$cc), "b$cc $target",
                 [/*(ARMbrcond bb:$target, imm:$cc)*/]>;

//===----------------------------------------------------------------------===//
//  Load Store Instructions.
//

let canFoldAsLoad = 1 in
def tLDR : TI4<(outs GPR:$dst), (ins t_addrmode_s4:$addr),
               "ldr $dst, $addr",
               [(set GPR:$dst, (load t_addrmode_s4:$addr))]>;

def tLDRB : TI1<(outs GPR:$dst), (ins t_addrmode_s1:$addr),
                "ldrb $dst, $addr",
                [(set GPR:$dst, (zextloadi8 t_addrmode_s1:$addr))]>;

def tLDRH : TI2<(outs GPR:$dst), (ins t_addrmode_s2:$addr),
                "ldrh $dst, $addr",
                [(set GPR:$dst, (zextloadi16 t_addrmode_s2:$addr))]>;

def tLDRSB : TI1<(outs GPR:$dst), (ins t_addrmode_rr:$addr),
                 "ldrsb $dst, $addr",
                 [(set GPR:$dst, (sextloadi8 t_addrmode_rr:$addr))]>;

def tLDRSH : TI2<(outs GPR:$dst), (ins t_addrmode_rr:$addr),
                 "ldrsh $dst, $addr",
                 [(set GPR:$dst, (sextloadi16 t_addrmode_rr:$addr))]>;

let canFoldAsLoad = 1 in
def tLDRspi : TIs<(outs GPR:$dst), (ins t_addrmode_sp:$addr),
                  "ldr $dst, $addr",
                  [(set GPR:$dst, (load t_addrmode_sp:$addr))]>;

// Special instruction for restore. It cannot clobber condition register
// when it's expanded by eliminateCallFramePseudoInstr().
let canFoldAsLoad = 1, mayLoad = 1 in
def tRestore : TIs<(outs GPR:$dst), (ins t_addrmode_sp:$addr),
                    "ldr $dst, $addr", []>;

// Load tconstpool
let canFoldAsLoad = 1 in
def tLDRpci : TIs<(outs GPR:$dst), (ins i32imm:$addr),
                  "ldr $dst, $addr",
                  [(set GPR:$dst, (load (ARMWrapper tconstpool:$addr)))]>;

// Special LDR for loads from non-pc-relative constpools.
let canFoldAsLoad = 1, mayLoad = 1, isReMaterializable = 1 in
def tLDRcp  : TIs<(outs GPR:$dst), (ins i32imm:$addr),
                  "ldr $dst, $addr", []>;

def tSTR : TI4<(outs), (ins GPR:$src, t_addrmode_s4:$addr),
               "str $src, $addr",
               [(store GPR:$src, t_addrmode_s4:$addr)]>;

def tSTRB : TI1<(outs), (ins GPR:$src, t_addrmode_s1:$addr),
                 "strb $src, $addr",
                 [(truncstorei8 GPR:$src, t_addrmode_s1:$addr)]>;

def tSTRH : TI2<(outs), (ins GPR:$src, t_addrmode_s2:$addr),
                 "strh $src, $addr",
                 [(truncstorei16 GPR:$src, t_addrmode_s2:$addr)]>;

def tSTRspi : TIs<(outs), (ins GPR:$src, t_addrmode_sp:$addr),
                   "str $src, $addr",
                   [(store GPR:$src, t_addrmode_sp:$addr)]>;

let mayStore = 1 in {
// Special instruction for spill. It cannot clobber condition register
// when it's expanded by eliminateCallFramePseudoInstr().
def tSpill : TIs<(outs), (ins GPR:$src, t_addrmode_sp:$addr),
                  "str $src, $addr", []>;
}

//===----------------------------------------------------------------------===//
//  Load / store multiple Instructions.
//

// TODO: A7-44: LDMIA - load multiple

let mayLoad = 1 in
def tPOP : TI<(outs reglist:$dst1, variable_ops), (ins),
               "pop $dst1", []>;

let mayStore = 1 in
def tPUSH : TI<(outs), (ins reglist:$src1, variable_ops),
                "push $src1", []>;

//===----------------------------------------------------------------------===//
//  Arithmetic Instructions.
//

// Add with carry
def tADC : TIt<(outs GPR:$dst), (ins GPR:$lhs, GPR:$rhs),
               "adc $dst, $rhs",
               [(set GPR:$dst, (adde GPR:$lhs, GPR:$rhs))]>;

def tADDS : TI<(outs GPR:$dst), (ins GPR:$lhs, GPR:$rhs),
               "add $dst, $lhs, $rhs",
               [(set GPR:$dst, (addc GPR:$lhs, GPR:$rhs))]>;


def tADDi3 : TI<(outs GPR:$dst), (ins GPR:$lhs, i32imm:$rhs),
                "add $dst, $lhs, $rhs",
                [(set GPR:$dst, (add GPR:$lhs, imm0_7:$rhs))]>;

def tADDi8 : TIt<(outs GPR:$dst), (ins GPR:$lhs, i32imm:$rhs),
                 "add $dst, $rhs",
                 [(set GPR:$dst, (add GPR:$lhs, imm8_255:$rhs))]>;

def tADDrr : TI<(outs GPR:$dst), (ins GPR:$lhs, GPR:$rhs),
                "add $dst, $lhs, $rhs",
                [(set GPR:$dst, (add GPR:$lhs, GPR:$rhs))]>;

def tADDhirr : TIt<(outs GPR:$dst), (ins GPR:$lhs, GPR:$rhs),
                   "add $dst, $rhs", []>;

def tADDrPCi : TI<(outs GPR:$dst), (ins i32imm:$rhs),
                  "add $dst, pc, $rhs * 4", []>;
def tADDrSPi : TI<(outs GPR:$dst), (ins GPR:$sp, i32imm:$rhs),
                  "add $dst, $sp, $rhs * 4", []>;
def tADDspi : TIt<(outs GPR:$dst), (ins GPR:$lhs, i32imm:$rhs),
                  "add $dst, $rhs * 4", []>;

def tAND : TIt<(outs GPR:$dst), (ins GPR:$lhs, GPR:$rhs),
                "and $dst, $rhs",
                [(set GPR:$dst, (and GPR:$lhs, GPR:$rhs))]>;

def tASRri : TI<(outs GPR:$dst), (ins GPR:$lhs, i32imm:$rhs),
                "asr $dst, $lhs, $rhs",
                [(set GPR:$dst, (sra GPR:$lhs, imm:$rhs))]>;

def tASRrr : TIt<(outs GPR:$dst), (ins GPR:$lhs, GPR:$rhs),
                 "asr $dst, $rhs",
                 [(set GPR:$dst, (sra GPR:$lhs, GPR:$rhs))]>;

def tBIC : TIt<(outs GPR:$dst), (ins GPR:$lhs, GPR:$rhs),
               "bic $dst, $rhs",
               [(set GPR:$dst, (and GPR:$lhs, (not GPR:$rhs)))]>;


def tCMN : TI<(outs), (ins GPR:$lhs, GPR:$rhs),
              "cmn $lhs, $rhs",
              [(ARMcmp GPR:$lhs, (ineg GPR:$rhs))]>;

def tCMPi8 : TI<(outs), (ins GPR:$lhs, i32imm:$rhs),
               "cmp $lhs, $rhs",
               [(ARMcmp GPR:$lhs, imm0_255:$rhs)]>;

def tCMPr : TI<(outs), (ins GPR:$lhs, GPR:$rhs),
               "cmp $lhs, $rhs",
               [(ARMcmp GPR:$lhs, GPR:$rhs)]>;

def tTST  : TI<(outs), (ins GPR:$lhs, GPR:$rhs),
               "tst $lhs, $rhs",
               [(ARMcmpNZ (and GPR:$lhs, GPR:$rhs), 0)]>;

def tCMNNZ : TI<(outs), (ins GPR:$lhs, GPR:$rhs),
                "cmn $lhs, $rhs",
                [(ARMcmpNZ GPR:$lhs, (ineg GPR:$rhs))]>;

def tCMPNZi8 : TI<(outs), (ins GPR:$lhs, i32imm:$rhs),
                 "cmp $lhs, $rhs",
                 [(ARMcmpNZ GPR:$lhs, imm0_255:$rhs)]>;

def tCMPNZr : TI<(outs), (ins GPR:$lhs, GPR:$rhs),
                 "cmp $lhs, $rhs",
                 [(ARMcmpNZ GPR:$lhs, GPR:$rhs)]>;

// TODO: A7-37: CMP(3) - cmp hi regs

def tEOR : TIt<(outs GPR:$dst), (ins GPR:$lhs, GPR:$rhs),
               "eor $dst, $rhs",
               [(set GPR:$dst, (xor GPR:$lhs, GPR:$rhs))]>;

def tLSLri : TI<(outs GPR:$dst), (ins GPR:$lhs, i32imm:$rhs),
                "lsl $dst, $lhs, $rhs",
                [(set GPR:$dst, (shl GPR:$lhs, imm:$rhs))]>;

def tLSLrr : TIt<(outs GPR:$dst), (ins GPR:$lhs, GPR:$rhs),
                 "lsl $dst, $rhs",
                 [(set GPR:$dst, (shl GPR:$lhs, GPR:$rhs))]>;

def tLSRri : TI<(outs GPR:$dst), (ins GPR:$lhs, i32imm:$rhs),
                "lsr $dst, $lhs, $rhs",
                [(set GPR:$dst, (srl GPR:$lhs, imm:$rhs))]>;

def tLSRrr : TIt<(outs GPR:$dst), (ins GPR:$lhs, GPR:$rhs),
                 "lsr $dst, $rhs",
                 [(set GPR:$dst, (srl GPR:$lhs, GPR:$rhs))]>;

// FIXME: This is not rematerializable because mov changes the condition code.
def tMOVi8 : TI<(outs GPR:$dst), (ins i32imm:$src),
                 "mov $dst, $src",
                 [(set GPR:$dst, imm0_255:$src)]>;

// TODO: A7-73: MOV(2) - mov setting flag.


// Note: MOV(2) of two low regs updates the flags, so we emit this as 'cpy',
// which is MOV(3).  This also supports high registers.
def tMOVr  : TI<(outs GPR:$dst), (ins GPR:$src),
                 "cpy $dst, $src", []>;

def tMUL : TIt<(outs GPR:$dst), (ins GPR:$lhs, GPR:$rhs),
               "mul $dst, $rhs",
               [(set GPR:$dst, (mul GPR:$lhs, GPR:$rhs))]>;

def tMVN : TI<(outs GPR:$dst), (ins GPR:$src),
              "mvn $dst, $src",
              [(set GPR:$dst, (not GPR:$src))]>;

def tNEG : TI<(outs GPR:$dst), (ins GPR:$src),
              "neg $dst, $src",
              [(set GPR:$dst, (ineg GPR:$src))]>;

def tORR : TIt<(outs GPR:$dst), (ins GPR:$lhs, GPR:$rhs),
               "orr $dst, $rhs",
               [(set GPR:$dst, (or GPR:$lhs, GPR:$rhs))]>;


def tREV : TI<(outs GPR:$dst), (ins GPR:$src),
              "rev $dst, $src",
              [(set GPR:$dst, (bswap GPR:$src))]>, 
              Requires<[IsThumb, HasV6]>;

def tREV16 : TI<(outs GPR:$dst), (ins GPR:$src),
                "rev16 $dst, $src",
                [(set GPR:$dst,
                    (or (and (srl GPR:$src, 8), 0xFF),
                        (or (and (shl GPR:$src, 8), 0xFF00),
                            (or (and (srl GPR:$src, 8), 0xFF0000),
                                (and (shl GPR:$src, 8), 0xFF000000)))))]>,
                Requires<[IsThumb, HasV6]>;

def tREVSH : TI<(outs GPR:$dst), (ins GPR:$src),
                "revsh $dst, $src",
                [(set GPR:$dst,
                   (sext_inreg
                     (or (srl (and GPR:$src, 0xFFFF), 8),
                         (shl GPR:$src, 8)), i16))]>,
                Requires<[IsThumb, HasV6]>;

def tROR : TIt<(outs GPR:$dst), (ins GPR:$lhs, GPR:$rhs),
                "ror $dst, $rhs",
                [(set GPR:$dst, (rotr GPR:$lhs, GPR:$rhs))]>;


// Subtract with carry
def tSBC : TIt<(outs GPR:$dst), (ins GPR:$lhs, GPR:$rhs),
                "sbc $dst, $rhs",
                [(set GPR:$dst, (sube GPR:$lhs, GPR:$rhs))]>;

def tSUBS : TI<(outs GPR:$dst), (ins GPR:$lhs, GPR:$rhs),
                "sub $dst, $lhs, $rhs",
               [(set GPR:$dst, (subc GPR:$lhs, GPR:$rhs))]>;


// TODO: A7-96: STMIA - store multiple.

def tSUBi3 : TI<(outs GPR:$dst), (ins GPR:$lhs, i32imm:$rhs),
                "sub $dst, $lhs, $rhs",
                [(set GPR:$dst, (add GPR:$lhs, imm0_7_neg:$rhs))]>;
                
def tSUBi8 : TIt<(outs GPR:$dst), (ins GPR:$lhs, i32imm:$rhs),
                  "sub $dst, $rhs",
                  [(set GPR:$dst, (add GPR:$lhs, imm8_255_neg:$rhs))]>;
                
def tSUBrr : TI<(outs GPR:$dst), (ins GPR:$lhs, GPR:$rhs),
                "sub $dst, $lhs, $rhs",
                [(set GPR:$dst, (sub GPR:$lhs, GPR:$rhs))]>;

def tSUBspi : TIt<(outs GPR:$dst), (ins GPR:$lhs, i32imm:$rhs),
                  "sub $dst, $rhs * 4", []>;

def tSXTB  : TI<(outs GPR:$dst), (ins GPR:$src),
                "sxtb $dst, $src",
                [(set GPR:$dst, (sext_inreg GPR:$src, i8))]>,
                Requires<[IsThumb, HasV6]>;
def tSXTH  : TI<(outs GPR:$dst), (ins GPR:$src),
                "sxth $dst, $src",
                [(set GPR:$dst, (sext_inreg GPR:$src, i16))]>,
                Requires<[IsThumb, HasV6]>;


def tUXTB  : TI<(outs GPR:$dst), (ins GPR:$src),
                "uxtb $dst, $src",
                [(set GPR:$dst, (and GPR:$src, 0xFF))]>,
                Requires<[IsThumb, HasV6]>;
def tUXTH  : TI<(outs GPR:$dst), (ins GPR:$src),
                "uxth $dst, $src",
                [(set GPR:$dst, (and GPR:$src, 0xFFFF))]>, 
                Requires<[IsThumb, HasV6]>;


// Conditional move tMOVCCr - Used to implement the Thumb SELECT_CC DAG operation.
// Expanded by the scheduler into a branch sequence.
let usesCustomDAGSchedInserter = 1 in  // Expanded by the scheduler.
  def tMOVCCr :
  PseudoInst<(outs GPR:$dst), (ins GPR:$false, GPR:$true, pred:$cc),
              "@ tMOVCCr $cc",
              [/*(set GPR:$dst, (ARMcmov GPR:$false, GPR:$true, imm:$cc))*/]>;

// tLEApcrel - Load a pc-relative address into a register without offending the
// assembler.
def tLEApcrel : TIx2<(outs GPR:$dst), (ins i32imm:$label),
                    !strconcat(!strconcat(".set PCRELV${:uid}, ($label-(",
                                          "${:private}PCRELL${:uid}+4))\n"),
                               !strconcat("\tmov $dst, #PCRELV${:uid}\n",
                                  "${:private}PCRELL${:uid}:\n\tadd $dst, pc")),
                    []>;

def tLEApcrelJT : TIx2<(outs GPR:$dst), (ins i32imm:$label, i32imm:$id),
          !strconcat(!strconcat(".set PCRELV${:uid}, (${label}_${id:no_hash}-(",
                                         "${:private}PCRELL${:uid}+4))\n"),
                     !strconcat("\tmov $dst, #PCRELV${:uid}\n",
                                "${:private}PCRELL${:uid}:\n\tadd $dst, pc")),
                    []>;

//===----------------------------------------------------------------------===//
// TLS Instructions
//

// __aeabi_read_tp preserves the registers r1-r3.
let isCall = 1,
  Defs = [R0, LR] in {
  def tTPsoft  : TIx2<(outs), (ins),
               "bl __aeabi_read_tp",
               [(set R0, ARMthread_pointer)]>;
}

//===----------------------------------------------------------------------===//
// Non-Instruction Patterns
//

// ConstantPool, GlobalAddress
def : ThumbPat<(ARMWrapper  tglobaladdr :$dst), (tLEApcrel tglobaladdr :$dst)>;
def : ThumbPat<(ARMWrapper  tconstpool  :$dst), (tLEApcrel tconstpool  :$dst)>;

// JumpTable
def : ThumbPat<(ARMWrapperJT tjumptable:$dst, imm:$id),
               (tLEApcrelJT tjumptable:$dst, imm:$id)>;

// Direct calls
def : ThumbPat<(ARMtcall texternalsym:$func), (tBL texternalsym:$func)>;
def : ThumbV5Pat<(ARMcall texternalsym:$func), (tBLXi texternalsym:$func)>;

// Indirect calls to ARM routines
def : ThumbV5Pat<(ARMcall GPR:$dst), (tBLXr GPR:$dst)>;

// zextload i1 -> zextload i8
def : ThumbPat<(zextloadi1 t_addrmode_s1:$addr),
               (tLDRB t_addrmode_s1:$addr)>;
                  
// extload -> zextload
def : ThumbPat<(extloadi1  t_addrmode_s1:$addr),  (tLDRB t_addrmode_s1:$addr)>;
def : ThumbPat<(extloadi8  t_addrmode_s1:$addr),  (tLDRB t_addrmode_s1:$addr)>;
def : ThumbPat<(extloadi16 t_addrmode_s2:$addr),  (tLDRH t_addrmode_s2:$addr)>;

// Large immediate handling.

// Two piece imms.
def : ThumbPat<(i32 thumb_immshifted:$src),
               (tLSLri (tMOVi8 (thumb_immshifted_val imm:$src)),
                       (thumb_immshifted_shamt imm:$src))>;

def : ThumbPat<(i32 imm0_255_comp:$src),
               (tMVN (tMOVi8 (imm_comp_XFORM imm:$src)))>;
