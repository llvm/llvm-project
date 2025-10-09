//===-- RISCVInstrInfo.td - Target Description for RISC-V --*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the RISC-V instructions in TableGen format.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// RISC-V specific DAG Nodes.
//===----------------------------------------------------------------------===//

class RVSDNode<string opcode, SDTypeProfile type_profile,
               list<SDNodeProperty> properties = []>
    : SDNode<"RISCVISD::" # opcode, type_profile, properties> {
  bit HasPassthruOp = false;
  bit HasMaskOp = false;
  let TSFlags{0} = HasPassthruOp;
  let TSFlags{1} = HasMaskOp;
}

// Target-independent type requirements, but with target-specific formats.
def SDT_CallSeqStart : SDCallSeqStart<[SDTCisVT<0, i32>,
                                       SDTCisVT<1, i32>]>;
def SDT_CallSeqEnd   : SDCallSeqEnd<[SDTCisVT<0, i32>,
                                     SDTCisVT<1, i32>]>;

// Target-dependent type requirements.
def SDT_RISCVCall     : SDTypeProfile<0, -1, [SDTCisVT<0, XLenVT>]>;
def SDT_RISCVSelectCC : SDTypeProfile<1, 5, [SDTCisSameAs<1, 2>,
                                             SDTCisVT<3, OtherVT>,
                                             SDTCisSameAs<0, 4>,
                                             SDTCisSameAs<4, 5>]>;
def SDT_RISCVBrCC : SDTypeProfile<0, 4, [SDTCisSameAs<0, 1>,
                                         SDTCisVT<2, OtherVT>,
                                         SDTCisVT<3, OtherVT>]>;
def SDT_RISCVIntUnaryOpW : SDTypeProfile<1, 1, [
  SDTCisSameAs<0, 1>, SDTCisVT<0, i64>
]>;
def SDT_RISCVIntBinOpW : SDTypeProfile<1, 2, [
  SDTCisSameAs<0, 1>, SDTCisSameAs<0, 2>, SDTCisVT<0, i64>
]>;

// Target-independent nodes, but with target-specific formats.
def callseq_start : SDNode<"ISD::CALLSEQ_START", SDT_CallSeqStart,
                           [SDNPHasChain, SDNPOutGlue]>;
def callseq_end   : SDNode<"ISD::CALLSEQ_END", SDT_CallSeqEnd,
                           [SDNPHasChain, SDNPOptInGlue, SDNPOutGlue]>;

def riscv_call      : RVSDNode<"CALL", SDT_RISCVCall,
                               [SDNPHasChain, SDNPOptInGlue, SDNPOutGlue,
                                SDNPVariadic]>;
def riscv_tail      : RVSDNode<"TAIL", SDT_RISCVCall,
                               [SDNPHasChain, SDNPOptInGlue, SDNPOutGlue,
                                SDNPVariadic]>;

// Software guarded calls for large code model
def riscv_sw_guarded_call : RVSDNode<"SW_GUARDED_CALL", SDT_RISCVCall,
                                     [SDNPHasChain, SDNPOptInGlue, SDNPOutGlue,
                                      SDNPVariadic]>;
def riscv_sw_guarded_tail : RVSDNode<"SW_GUARDED_TAIL", SDT_RISCVCall,
                                     [SDNPHasChain, SDNPOptInGlue, SDNPOutGlue,
                                      SDNPVariadic]>;

def riscv_ret_glue  : RVSDNode<"RET_GLUE", SDTNone,
                               [SDNPHasChain, SDNPOptInGlue, SDNPVariadic]>;
def riscv_sret_glue : RVSDNode<"SRET_GLUE", SDTNone,
                               [SDNPHasChain, SDNPOptInGlue]>;
def riscv_mret_glue : RVSDNode<"MRET_GLUE", SDTNone,
                               [SDNPHasChain, SDNPOptInGlue]>;
def riscv_mnret_glue : RVSDNode<"MNRET_GLUE", SDTNone,
                                [SDNPHasChain, SDNPOptInGlue]>;
def riscv_mileaveret_glue : RVSDNode<"QC_C_MILEAVERET_GLUE", SDTNone,
                                     [SDNPHasChain, SDNPOptInGlue]>;

// Select with condition operator - This selects between a true value and
// a false value (ops #3 and #4) based on the boolean result of comparing
// the lhs and rhs (ops #0 and #1) of a conditional expression with the
// condition code in op #2, a XLenVT constant from the ISD::CondCode enum.
// The lhs and rhs are XLenVT integers. The true and false values can be
// integer or floating point.
def riscv_selectcc  : RVSDNode<"SELECT_CC", SDT_RISCVSelectCC>;
def riscv_brcc      : RVSDNode<"BR_CC", SDT_RISCVBrCC,
                               [SDNPHasChain]>;

// Software guarded BRIND node. Operand is the target address.
def riscv_sw_guarded_brind : RVSDNode<"SW_GUARDED_BRIND",
                                      SDTBrind, [SDNPHasChain]>;

// RV64I shifts, directly matching the semantics of the named RISC-V
// instructions.
def riscv_sllw      : RVSDNode<"SLLW", SDT_RISCVIntBinOpW>;
def riscv_sraw      : RVSDNode<"SRAW", SDT_RISCVIntBinOpW>;
def riscv_srlw      : RVSDNode<"SRLW", SDT_RISCVIntBinOpW>;

// Reads value of CSR. The first operand is the address of the required CSR.
// The result is the read value.
def riscv_read_csr  : RVSDNode<"READ_CSR",
                               SDTypeProfile<1, 1, [SDTCisInt<0>,
                                                    SDTCisInt<1>]>,
                               [SDNPHasChain]>;
// Write value to CSR. The first operand is the address of the required CSR,
// the second is the value to write.
def riscv_write_csr : RVSDNode<"WRITE_CSR",
                               SDTypeProfile<0, 2, [SDTCisInt<0>,
                                                    SDTCisInt<1>]>,
                               [SDNPHasChain]>;

// Read and write value of CSR. The first operand is the address of the
// required CSR, the second is the value to write, and the result is the
// read value (before modification).
def riscv_swap_csr  : RVSDNode<"SWAP_CSR",
                               SDTypeProfile<1, 2, [SDTCisInt<0>,
                                                    SDTCisInt<1>,
                                                    SDTCisInt<2>]>,
                               [SDNPHasChain]>;

// Clear bits of CSR. The first operand is the address of the required CSR,
// the second is the bitmask of cleared bits.
def riscv_clear_csr  : RVSDNode<"CLEAR_CSR",
                               SDTypeProfile<0, 2, [SDTCisInt<0>,
                                                    SDTCisInt<1>]>,
                               [SDNPHasChain]>;

// Set bits of CSR. The first operand is the address of the required CSR,
// the second is the bitmask of bits to set.
def riscv_set_csr  : RVSDNode<"SET_CSR",
                               SDTypeProfile<0, 2, [SDTCisInt<0>,
                                                    SDTCisInt<1>]>,
                               [SDNPHasChain]>;

// A read of the 64-bit counter CSR on a 32-bit target (returns (Lo, Hi)).
// It takes a chain operand and another two target constant operands (the
// CSR numbers of the low and high parts of the counter).
def riscv_read_counter_wide : RVSDNode<"READ_COUNTER_WIDE",
                                       SDTypeProfile<2, 2, [SDTCisVT<0, i32>,
                                                            SDTCisVT<1, i32>,
                                                            SDTCisInt<2>,
                                                            SDTCisInt<3>]>,
                                       [SDNPHasChain, SDNPSideEffect]>;

// Add the Lo 12 bits from an address. Selected to ADDI.
def riscv_add_lo : RVSDNode<"ADD_LO", SDTIntBinOp>;

// Get the Hi 20 bits from an address. Selected to LUI.
def riscv_hi : RVSDNode<"HI", SDTIntUnaryOp>;

// Represents an AUIPC+ADDI pair. Selected to PseudoLLA.
def riscv_lla : RVSDNode<"LLA", SDTIntUnaryOp>;

// Selected as PseudoAddTPRel. Used to emit a TP-relative relocation.
def riscv_add_tprel : RVSDNode<"ADD_TPREL",
                               SDTypeProfile<1, 3, [SDTCisSameAs<0, 1>,
                                                    SDTCisSameAs<0, 2>,
                                                    SDTCisSameAs<0, 3>,
                                                    SDTCisInt<0>]>>;

// To avoid stack clash, allocation is performed by block and each block is
// probed.
def riscv_probed_alloca : RVSDNode<"PROBED_ALLOCA",
                                   SDTypeProfile<0, 1, [SDTCisPtrTy<0>]>,
                                   [SDNPHasChain, SDNPMayStore]>;


defvar GPRPairVT = untyped;

/// Turn a pair of `i<xlen>`s into an even-odd register pair (`untyped`).
/// - Output: `untyped` even-odd register pair
/// - Input 0: `i<xlen>` low-order bits, for even register.
/// - Input 1: `i<xlen>` high-order bits, for odd register.
def riscv_build_gpr_pair : RVSDNode<"BuildGPRPair",
                                    SDTypeProfile<1, 2, [SDTCisVT<0, GPRPairVT>,
                                                         SDTCisVT<1, XLenVT>,
                                                         SDTCisVT<2, XLenVT>]>>;

/// Turn an even-odd register pair (`untyped`) into a pair of `i<xlen>`s.
/// - Output 0: `i<xlen>` low-order bits, from even register.
/// - Output 1: `i<xlen>` high-order bits, from odd register.
/// - Input: `untyped` even-odd register pair
def riscv_split_gpr_pair : RVSDNode<"SplitGPRPair",
                                    SDTypeProfile<2, 1, [SDTCisVT<0, XLenVT>,
                                                         SDTCisVT<1, XLenVT>,
                                                         SDTCisVT<2, GPRPairVT>]>>;

//===----------------------------------------------------------------------===//
// Operand and SDNode transformation definitions.
//===----------------------------------------------------------------------===//

class ImmXLenAsmOperand<string prefix, string suffix = ""> : AsmOperandClass {
  let Name = prefix # "ImmXLen" # suffix;
  let RenderMethod = "addImmOperands";
  let DiagnosticType = !strconcat("Invalid", Name);
}

class ImmAsmOperand<string prefix, int width, string suffix> : AsmOperandClass {
  let Name = prefix # "Imm" # width # suffix;
  let RenderMethod = "addImmOperands";
  let DiagnosticType = !strconcat("Invalid", Name);
}

def ImmZeroAsmOperand : AsmOperandClass {
  let Name = "ImmZero";
  let RenderMethod = "addImmOperands";
  let DiagnosticType = !strconcat("Invalid", Name);
  let DiagnosticString = "immediate must be zero";
}

// A parse method for (${gpr}) or 0(${gpr}), where the 0 is be silently ignored.
def ZeroOffsetMemOpOperand : AsmOperandClass {
  let Name = "ZeroOffsetMemOpOperand";
  let RenderMethod = "addRegOperands";
  let PredicateMethod = "isGPR";
  let ParserMethod = "parseZeroOffsetMemOp";
}

class MemOperand<RegisterClass regClass> : RegisterOperand<regClass>{
  let OperandType = "OPERAND_MEMORY";
}

def GPRMemZeroOffset : MemOperand<GPR> {
  let ParserMatchClass = ZeroOffsetMemOpOperand;
  let PrintMethod = "printZeroOffsetMemOp";
}

def GPRMem : MemOperand<GPR>;

def SPMem : MemOperand<SP>;

def GPRCMem : MemOperand<GPRC>;

class SImmAsmOperand<int width, string suffix = "">
    : ImmAsmOperand<"S", width, suffix> {
}

class UImmAsmOperand<int width, string suffix = "">
    : ImmAsmOperand<"U", width, suffix> {
}

class BareSImmNLsb0AsmOperand<int width>
    : ImmAsmOperand<"BareS", width, "Lsb0"> {
  let PredicateMethod = "isBareSimmNLsb0<" # width # ">";
}

class RISCVOp<ValueType vt = XLenVT> : Operand<vt> {
  let OperandNamespace = "RISCVOp";
}

class RISCVUImmOp<int bitsNum> : RISCVOp {
  let ParserMatchClass = UImmAsmOperand<bitsNum>;
  let EncoderMethod = "getImmOpValue";
  let DecoderMethod = "decodeUImmOperand<" # bitsNum # ">";
  let OperandType = "OPERAND_UIMM" # bitsNum;
  let MCOperandPredicate = [{
    int64_t Imm;
    if (!MCOp.evaluateAsConstantImm(Imm))
      return false;
    return isUInt<}]# bitsNum #[{>(Imm);
  }];
}

class RISCVUImmLeafOp<int bitsNum> :
  RISCVUImmOp<bitsNum>, ImmLeaf<XLenVT, "return isUInt<" # bitsNum # ">(Imm);">;

class RISCVSImmOp<int bitsNum> : RISCVOp {
  let ParserMatchClass = SImmAsmOperand<bitsNum>;
  let EncoderMethod = "getImmOpValue";
  let DecoderMethod = "decodeSImmOperand<" # bitsNum # ">";
  let OperandType = "OPERAND_SIMM" # bitsNum;
  let MCOperandPredicate = [{
    int64_t Imm;
    if (!MCOp.evaluateAsConstantImm(Imm))
      return false;
    return isInt<}] # bitsNum # [{>(Imm);
  }];
}

class RISCVSImmLeafOp<int bitsNum> :
  RISCVSImmOp<bitsNum>, ImmLeaf<XLenVT, "return isInt<" # bitsNum # ">(Imm);">;

def FenceArg : AsmOperandClass {
  let Name = "FenceArg";
  let RenderMethod = "addFenceArgOperands";
  let ParserMethod = "parseFenceArg";
}

def fencearg : RISCVOp {
  let ParserMatchClass = FenceArg;
  let PrintMethod = "printFenceArg";
  let DecoderMethod = "decodeUImmOperand<4>";
  let OperandType = "OPERAND_UIMM4";
}

def UImmLog2XLenAsmOperand : AsmOperandClass {
  let Name = "UImmLog2XLen";
  let RenderMethod = "addImmOperands";
  let DiagnosticType = "InvalidUImmLog2XLen";
}

def uimmlog2xlen : RISCVOp, ImmLeaf<XLenVT, [{
  if (Subtarget->is64Bit())
    return isUInt<6>(Imm);
  return isUInt<5>(Imm);
}]> {
  let ParserMatchClass = UImmLog2XLenAsmOperand;
  let DecoderMethod = "decodeUImmLog2XLenOperand";
  let MCOperandPredicate = [{
    int64_t Imm;
    if (!MCOp.evaluateAsConstantImm(Imm))
      return false;
    if (STI.getTargetTriple().isArch64Bit())
      return isUInt<6>(Imm);
    return isUInt<5>(Imm);
  }];
  let OperandType = "OPERAND_UIMMLOG2XLEN";
}

def InsnDirectiveOpcode : AsmOperandClass {
  let Name = "InsnDirectiveOpcode";
  let ParserMethod = "parseInsnDirectiveOpcode";
  let RenderMethod = "addImmOperands";
  let PredicateMethod = "isImm";
}

def uimm1 : RISCVUImmLeafOp<1>;
def uimm2 : RISCVUImmLeafOp<2>;
def uimm3 : RISCVUImmLeafOp<3>;
def uimm4 : RISCVUImmLeafOp<4>;
def uimm5 : RISCVUImmLeafOp<5>;
def uimm6 : RISCVUImmLeafOp<6>;
def uimm7_opcode : RISCVUImmOp<7> {
  let ParserMatchClass = InsnDirectiveOpcode;
}
def uimm7 : RISCVUImmLeafOp<7>;
def uimm8 : RISCVUImmOp<8>;
def uimm16 : RISCVUImmOp<16>;
def uimm32 : RISCVUImmOp<32>;
def uimm48 : RISCVUImmOp<48>;
def uimm64 : RISCVUImmOp<64>;

def simm12_lo : RISCVSImmLeafOp<12> {
  let ParserMatchClass = SImmAsmOperand<12, "LO">;
  let MCOperandPredicate = [{
    int64_t Imm;
    if (MCOp.evaluateAsConstantImm(Imm))
      return isInt<12>(Imm);
    return MCOp.isBareSymbolRef();
  }];
}

// A 12-bit signed immediate which cannot fit in 6-bit signed immediate,
// but even negative value fit in 12-bit.
def simm12_no6 : ImmLeaf<XLenVT, [{
  return isInt<12>(Imm) && !isInt<6>(Imm) && isInt<12>(-Imm);}]>;

class BareSImm13Lsb0MaybeSym : Operand<OtherVT> {
  let MCOperandPredicate = [{
    int64_t Imm;
    if (MCOp.evaluateAsConstantImm(Imm))
      return isShiftedInt<12, 1>(Imm);
    return MCOp.isBareSymbolRef();
  }];
}

// A 13-bit signed immediate where the least significant bit is zero. The ImmLeaf
// is needed so that the CompressInstEmitter can correctly add checks for the
// compress patterns that involve instructions that use this operand. Similar to
// bare_simm9_lsb0 in RISCVInstrInfoC.td.
def bare_simm13_lsb0 : BareSImm13Lsb0MaybeSym,
                       ImmLeaf<XLenVT, [{return isShiftedInt<12, 1>(Imm);}]> {
  let ParserMatchClass = BareSImmNLsb0AsmOperand<13>;
  let PrintMethod = "printBranchOperand";
  let EncoderMethod = "getImmOpValueAsrN<1>";
  let DecoderMethod = "decodeSImmOperandAndLslN<13, 1>";
  let OperandType = "OPERAND_PCREL";
}

// We need this (sort of) duplicate definition since adding ImmLeaf to
// bare_simm13_lsb0 above makes it not sit well with codegen patterns where it
// is used to match with a basic block (eg. BccPat<>).
def bare_simm13_lsb0_bb : BareSImm13Lsb0MaybeSym;

class UImm20OperandMaybeSym : RISCVUImmOp<20> {
  let MCOperandPredicate = [{
    int64_t Imm;
    if (MCOp.evaluateAsConstantImm(Imm))
      return isUInt<20>(Imm);
    return MCOp.isBareSymbolRef();
  }];
}

def uimm20_lui : UImm20OperandMaybeSym {
  let ParserMatchClass = UImmAsmOperand<20, "LUI">;
}
def uimm20_auipc : UImm20OperandMaybeSym {
  let ParserMatchClass = UImmAsmOperand<20, "AUIPC">;
}

def uimm20 : RISCVUImmOp<20>;

def Simm21Lsb0JALAsmOperand : BareSImmNLsb0AsmOperand<21> {
  let ParserMethod = "parseJALOffset";
}

// A 21-bit signed immediate where the least significant bit is zero.
def simm21_lsb0_jal : Operand<OtherVT>,
                      ImmLeaf<XLenVT, [{return isShiftedInt<20, 1>(Imm);}]>  {
  let ParserMatchClass = Simm21Lsb0JALAsmOperand;
  let PrintMethod = "printBranchOperand";
  let EncoderMethod = "getImmOpValueAsrN<1>";
  let DecoderMethod = "decodeSImmOperandAndLslN<21, 1>";
  let MCOperandPredicate = [{
    int64_t Imm;
    if (MCOp.evaluateAsConstantImm(Imm))
      return isShiftedInt<20, 1>(Imm);
    return MCOp.isBareSymbolRef();
  }];
  let OperandType = "OPERAND_PCREL";
}

def BareSymbol : AsmOperandClass {
  let Name = "BareSymbol";
  let RenderMethod = "addImmOperands";
  let DiagnosticType = "InvalidBareSymbol";
  let DiagnosticString = "operand must be a bare symbol name";
  let ParserMethod = "parseBareSymbol";
}

// A bare symbol.
def bare_symbol : Operand<XLenVT> {
  let ParserMatchClass = BareSymbol;
}

def CallSymbol : AsmOperandClass {
  let Name = "CallSymbol";
  let RenderMethod = "addImmOperands";
  let DiagnosticType = "InvalidCallSymbol";
  let DiagnosticString = "operand must be a bare symbol name";
  let ParserMethod = "parseCallSymbol";
}

// A bare symbol used in call/tail only.
def call_symbol : Operand<XLenVT> {
  let ParserMatchClass = CallSymbol;
}

def PseudoJumpSymbol : AsmOperandClass {
  let Name = "PseudoJumpSymbol";
  let RenderMethod = "addImmOperands";
  let DiagnosticType = "InvalidPseudoJumpSymbol";
  let DiagnosticString = "operand must be a valid jump target";
  let ParserMethod = "parsePseudoJumpSymbol";
}

// A bare symbol used for pseudo jumps only.
def pseudo_jump_symbol : Operand<XLenVT> {
  let ParserMatchClass = PseudoJumpSymbol;
}

def TPRelAddSymbol : AsmOperandClass {
  let Name = "TPRelAddSymbol";
  let RenderMethod = "addImmOperands";
  let DiagnosticType = "InvalidTPRelAddSymbol";
  let DiagnosticString = "operand must be a symbol with %tprel_add specifier";
  let ParserMethod = "parseOperandWithSpecifier";
}

// A bare symbol with the %tprel_add variant.
def tprel_add_symbol : Operand<XLenVT> {
  let ParserMatchClass = TPRelAddSymbol;
}

def CSRSystemRegister : AsmOperandClass {
  let Name = "CSRSystemRegister";
  let ParserMethod = "parseCSRSystemRegister";
  let DiagnosticType = "InvalidCSRSystemRegister";
}

def csr_sysreg : RISCVOp, TImmLeaf<XLenVT, "return isUInt<12>(Imm);"> {
  let ParserMatchClass = CSRSystemRegister;
  let PrintMethod = "printCSRSystemRegister";
  let DecoderMethod = "decodeUImmOperand<12>";
  let OperandType = "OPERAND_UIMM12";
}

// A parameterized register class alternative to i32imm/i64imm from Target.td.
def ixlenimm : Operand<XLenVT>;

// Condition code used by select and short forward branch pseudos.
def cond_code : RISCVOp {
  let OperandType = "OPERAND_COND_CODE";
}

def ixlenimm_li : Operand<XLenVT> {
  let ParserMatchClass = ImmXLenAsmOperand<"", "LI">;
}

// Accepts subset of LI operands, used by LAImm and LLAImm
def ixlenimm_li_restricted : Operand<XLenVT> {
  let ParserMatchClass = ImmXLenAsmOperand<"", "LI_Restricted">;
}

// Standalone (codegen-only) immleaf patterns.

// A 12-bit signed immediate plus one where the imm range will be -2047~2048.
def simm12_plus1 : ImmLeaf<XLenVT,
  [{return (isInt<12>(Imm) && Imm != -2048) || Imm == 2048;}]>;

// A 6-bit constant greater than 32.
def uimm6gt32 : ImmLeaf<XLenVT, [{
  return isUInt<6>(Imm) && Imm > 32;
}]>;

// Addressing modes.
def AddrRegImm : ComplexPattern<iPTR, 2, "SelectAddrRegImm">;

class AddrRegRegScale<int N>
    : ComplexPattern<iPTR, 3, "SelectAddrRegRegScale<"#N#">">;
class AddrRegZextRegScale<int N>
    : ComplexPattern<i64, 3, "SelectAddrRegZextRegScale<"#N#", 32>",
                     [], [], 10>;

// Return the negation of an immediate value.
def NegImm : SDNodeXForm<imm, [{
  return CurDAG->getSignedTargetConstant(-N->getSExtValue(), SDLoc(N),
                                         N->getValueType(0));
}]>;
def GINegImm : GICustomOperandRenderer<"renderNegImm">,
  GISDNodeXFormEquiv<NegImm>;

// Return an immediate value minus 32.
def ImmSub32 : SDNodeXForm<imm, [{
  return CurDAG->getSignedTargetConstant(N->getSExtValue() - 32, SDLoc(N),
                                         N->getValueType(0));
}]>;

// Return an immediate subtracted from XLen.
def ImmSubFromXLen : SDNodeXForm<imm, [{
  uint64_t XLen = Subtarget->getXLen();
  return CurDAG->getTargetConstant(XLen - N->getZExtValue(), SDLoc(N),
                                   N->getValueType(0));
}]>;
def GIImmSubFromXLen : GICustomOperandRenderer<"renderImmSubFromXLen">,
  GISDNodeXFormEquiv<ImmSubFromXLen>;

// Return an immediate subtracted from 32.
def ImmSubFrom32 : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(32 - N->getZExtValue(), SDLoc(N),
                                   N->getValueType(0));
}]>;
def GIImmSubFrom32 : GICustomOperandRenderer<"renderImmSubFrom32">,
  GISDNodeXFormEquiv<ImmSubFrom32>;

// Check if (add r, imm) can be optimized to (ADDI (ADDI r, imm0), imm1),
// in which imm = imm0 + imm1 and both imm0 and imm1 are simm12. We make imm0
// as large as possible and imm1 as small as possible so that we might be able
// to use c.addi for the small immediate.
def AddiPair : ImmLeaf<XLenVT, [{
  // The immediate operand must be in range [-4096,-2049] or [2048,4094].
  return (-4096 <= Imm && Imm <= -2049) || (2048 <= Imm && Imm <= 4094);
}]>;

// Return imm - (imm < 0 ? -2048 : 2047).
def AddiPairImmSmall : SDNodeXForm<imm, [{
  int64_t Imm = N->getSExtValue();
  int64_t Adj = N->getSExtValue() < 0 ? -2048 : 2047;
  return CurDAG->getSignedTargetConstant(Imm - Adj, SDLoc(N),
                                         N->getValueType(0));
}]>;
def GIAddiPairImmSmall : GICustomOperandRenderer<"renderAddiPairImmSmall">,
                         GISDNodeXFormEquiv<AddiPairImmSmall>;

// Return -2048 if immediate is negative or 2047 if positive. These are the
// largest simm12 values.
def AddiPairImmLarge : SDNodeXForm<imm, [{
  int64_t Imm = N->getSExtValue() < 0 ? -2048 : 2047;
  return CurDAG->getSignedTargetConstant(Imm, SDLoc(N), N->getValueType(0));
}]>;
def GIAddiPairImmLarge : GICustomOperandRenderer<"renderAddiPairImmLarge">,
                         GISDNodeXFormEquiv<AddiPairImmLarge>;

def TrailingZeros : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(llvm::countr_zero(N->getZExtValue()),
                                   SDLoc(N), N->getValueType(0));
}]>;
def GITrailingZeros : GICustomOperandRenderer<"renderTrailingZeros">,
  GISDNodeXFormEquiv<TrailingZeros>;

def XLenSubTrailingOnes : SDNodeXForm<imm, [{
  uint64_t XLen = Subtarget->getXLen();
  uint64_t TrailingOnes = llvm::countr_one(N->getZExtValue());
  return CurDAG->getTargetConstant(XLen - TrailingOnes, SDLoc(N),
                                   N->getValueType(0));
}]>;
def GIXLenSubTrailingOnes : GICustomOperandRenderer<"renderXLenSubTrailingOnes">,
  GISDNodeXFormEquiv<XLenSubTrailingOnes>;

// Checks if this mask is a non-empty sequence of ones starting at the
// most/least significant bit with the remainder zero and exceeds simm32/simm12.
def LeadingOnesMask : ImmLeaf<XLenVT, [{
  return !isInt<32>(Imm) && isMask_64(~Imm);
}], TrailingZeros>;

def TrailingOnesMask : IntImmLeaf<XLenVT, [{
  return !isInt<12>(Imm.getSExtValue()) && isMask_64(Imm.getZExtValue());
}], XLenSubTrailingOnes>;

// Similar to LeadingOnesMask, but only consider leading ones in the lower 32
// bits.
def LeadingOnesWMask : ImmLeaf<XLenVT, [{
  // If the value is a uint32 but not an int32, it must have bit 31 set and
  // bits 63:32 cleared. After that we're looking for a shifted mask but not
  // an all ones mask.
  return !isInt<32>(Imm) && isUInt<32>(Imm) && isShiftedMask_64(Imm) &&
         Imm != UINT64_C(0xffffffff);
}], TrailingZeros>;

//===----------------------------------------------------------------------===//
// Instruction Formats
//===----------------------------------------------------------------------===//

include "RISCVInstrFormats.td"
include "RISCVInstrFormatsC.td"
include "RISCVInstrFormatsV.td"

//===----------------------------------------------------------------------===//
// Instruction Class Templates
//===----------------------------------------------------------------------===//

class BranchCC_rri<bits<3> funct3, string opcodestr>
    : RVInstB<funct3, OPC_BRANCH, (outs),
              (ins GPR:$rs1, GPR:$rs2, bare_simm13_lsb0:$imm12),
              opcodestr, "$rs1, $rs2, $imm12">,
      Sched<[WriteJmp, ReadJmp, ReadJmp]> {
  let isBranch = 1;
  let isTerminator = 1;
  let hasSideEffects = 0;
  let mayLoad = 0;
  let mayStore = 0;
}

let hasSideEffects = 0, mayLoad = 1, mayStore = 0 in {
class Load_ri<bits<3> funct3, string opcodestr, DAGOperand rty = GPR>
    : RVInstI<funct3, OPC_LOAD, (outs rty:$rd), (ins GPRMem:$rs1, simm12_lo:$imm12),
              opcodestr, "$rd, ${imm12}(${rs1})">;

class HLoad_r<bits<7> funct7, bits<5> funct5, string opcodestr>
    : RVInstR<funct7, 0b100, OPC_SYSTEM, (outs GPR:$rd),
              (ins GPRMemZeroOffset:$rs1), opcodestr, "$rd, $rs1"> {
  let rs2 = funct5;
}
}

// Operands for stores are in the order srcreg, base, offset rather than
// reflecting the order these fields are specified in the instruction
// encoding.
let hasSideEffects = 0, mayLoad = 0, mayStore = 1 in {
class Store_rri<bits<3> funct3, string opcodestr, DAGOperand rty = GPR>
    : RVInstS<funct3, OPC_STORE, (outs),
              (ins rty:$rs2, GPRMem:$rs1, simm12_lo:$imm12),
              opcodestr, "$rs2, ${imm12}(${rs1})">;

class HStore_rr<bits<7> funct7, string opcodestr>
    : RVInstR<funct7, 0b100, OPC_SYSTEM, (outs),
              (ins GPR:$rs2, GPRMemZeroOffset:$rs1),
               opcodestr, "$rs2, $rs1"> {
  let rd = 0;
}
}

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class ALU_ri<bits<3> funct3, string opcodestr>
    : RVInstI<funct3, OPC_OP_IMM, (outs GPR:$rd), (ins GPR:$rs1, simm12_lo:$imm12),
              opcodestr, "$rd, $rs1, $imm12">,
      Sched<[WriteIALU, ReadIALU]>;

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class Shift_ri<bits<5> imm11_7, bits<3> funct3, string opcodestr>
    : RVInstIShift<imm11_7, funct3, OPC_OP_IMM, (outs GPR:$rd),
                   (ins GPR:$rs1, uimmlog2xlen:$shamt), opcodestr,
                   "$rd, $rs1, $shamt">;

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class ALU_rr<bits<7> funct7, bits<3> funct3, string opcodestr,
             bit Commutable = 0>
    : RVInstR<funct7, funct3, OPC_OP, (outs GPR:$rd), (ins GPR:$rs1, GPR:$rs2),
              opcodestr, "$rd, $rs1, $rs2"> {
  let isCommutable = Commutable;
}

let hasNoSchedulingInfo = 1,
    hasSideEffects = 1, mayLoad = 0, mayStore = 0 in
class CSR_ir<bits<3> funct3, string opcodestr>
    : RVInstI<funct3, OPC_SYSTEM, (outs GPR:$rd), (ins csr_sysreg:$imm12, GPR:$rs1),
              opcodestr, "$rd, $imm12, $rs1">, Sched<[WriteCSR, ReadCSR]>;

let hasNoSchedulingInfo = 1,
    hasSideEffects = 1, mayLoad = 0, mayStore = 0 in
class CSR_ii<bits<3> funct3, string opcodestr>
    : RVInstI<funct3, OPC_SYSTEM, (outs GPR:$rd),
              (ins csr_sysreg:$imm12, uimm5:$rs1),
              opcodestr, "$rd, $imm12, $rs1">, Sched<[WriteCSR]>;

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class ShiftW_ri<bits<7> imm11_5, bits<3> funct3, string opcodestr>
    : RVInstIShiftW<imm11_5, funct3, OPC_OP_IMM_32, (outs GPR:$rd),
                    (ins GPR:$rs1, uimm5:$shamt), opcodestr,
                    "$rd, $rs1, $shamt">;

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class ALUW_rr<bits<7> funct7, bits<3> funct3, string opcodestr,
              bit Commutable = 0>
    : RVInstR<funct7, funct3, OPC_OP_32, (outs GPR:$rd),
              (ins GPR:$rs1, GPR:$rs2), opcodestr, "$rd, $rs1, $rs2"> {
  let isCommutable = Commutable;
}

let hasSideEffects = 1, mayLoad = 0, mayStore = 0 in
class Priv<string opcodestr, bits<7> funct7>
    : RVInstR<funct7, 0b000, OPC_SYSTEM, (outs), (ins), opcodestr, "">;

let hasSideEffects = 1, mayLoad = 0, mayStore = 0 in
class Priv_rr<string opcodestr, bits<7> funct7>
    : RVInstR<funct7, 0b000, OPC_SYSTEM, (outs), (ins GPR:$rs1, GPR:$rs2),
              opcodestr, "$rs1, $rs2"> {
  let rd = 0;
}

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class Unary_r<bits<12> imm12, bits<3> funct3, string opcodestr>
    : RVInstIUnary<imm12, funct3, OPC_OP_IMM, (outs GPR:$rd), (ins GPR:$rs1),
                   opcodestr, "$rd, $rs1">;

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class UnaryW_r<bits<12> imm12, bits<3> funct3, string opcodestr>
    : RVInstIUnary<imm12, funct3, OPC_OP_IMM_32, (outs GPR:$rd), (ins GPR:$rs1),
                   opcodestr, "$rd, $rs1">;

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in {
let isReMaterializable = 1, isAsCheapAsAMove = 1,
    IsSignExtendingOpW = 1 in
def LUI : RVInstU<OPC_LUI, (outs GPR:$rd), (ins uimm20_lui:$imm20),
                  "lui", "$rd, $imm20">, Sched<[WriteIALU]>;

def AUIPC : RVInstU<OPC_AUIPC, (outs GPR:$rd), (ins uimm20_auipc:$imm20),
                    "auipc", "$rd, $imm20">, Sched<[WriteIALU]>;

def JAL : RVInstJ<OPC_JAL, (outs GPR:$rd), (ins simm21_lsb0_jal:$imm20),
                  "jal", "$rd, $imm20">, Sched<[WriteJal]>;

def JALR : RVInstI<0b000, OPC_JALR, (outs GPR:$rd),
                   (ins GPR:$rs1, simm12_lo:$imm12),
                   "jalr", "$rd, ${imm12}(${rs1})">,
           Sched<[WriteJalr, ReadJalr]>;
} // hasSideEffects = 0, mayLoad = 0, mayStore = 0

def BEQ  : BranchCC_rri<0b000, "beq">;
def BNE  : BranchCC_rri<0b001, "bne">;
def BLT  : BranchCC_rri<0b100, "blt">;
def BGE  : BranchCC_rri<0b101, "bge">;
def BLTU : BranchCC_rri<0b110, "bltu">;
def BGEU : BranchCC_rri<0b111, "bgeu">;

let IsSignExtendingOpW = 1 in {
def LB  : Load_ri<0b000, "lb">, Sched<[WriteLDB, ReadMemBase]>;
def LH  : Load_ri<0b001, "lh">, Sched<[WriteLDH, ReadMemBase]>;
def LW  : Load_ri<0b010, "lw">, Sched<[WriteLDW, ReadMemBase]>;
def LBU : Load_ri<0b100, "lbu">, Sched<[WriteLDB, ReadMemBase]>;
def LHU : Load_ri<0b101, "lhu">, Sched<[WriteLDH, ReadMemBase]>;
}

def SB : Store_rri<0b000, "sb">, Sched<[WriteSTB, ReadStoreData, ReadMemBase]>;
def SH : Store_rri<0b001, "sh">, Sched<[WriteSTH, ReadStoreData, ReadMemBase]>;
def SW : Store_rri<0b010, "sw">, Sched<[WriteSTW, ReadStoreData, ReadMemBase]>;

// ADDI isn't always rematerializable, but isReMaterializable will be used as
// a hint which is verified in isReMaterializableImpl.
let isReMaterializable = 1, isAsCheapAsAMove = 1 in
def ADDI  : ALU_ri<0b000, "addi">;

let IsSignExtendingOpW = 1 in {
def SLTI  : ALU_ri<0b010, "slti">;
def SLTIU : ALU_ri<0b011, "sltiu">;
}

let isReMaterializable = 1, isAsCheapAsAMove = 1 in {
def XORI  : ALU_ri<0b100, "xori">;
def ORI   : ALU_ri<0b110, "ori">;
}

def ANDI  : ALU_ri<0b111, "andi">;

def SLLI : Shift_ri<0b00000, 0b001, "slli">,
           Sched<[WriteShiftImm, ReadShiftImm]>;
def SRLI : Shift_ri<0b00000, 0b101, "srli">,
           Sched<[WriteShiftImm, ReadShiftImm]>;
def SRAI : Shift_ri<0b01000, 0b101, "srai">,
           Sched<[WriteShiftImm, ReadShiftImm]>;

def ADD  : ALU_rr<0b0000000, 0b000, "add", Commutable=1>,
           Sched<[WriteIALU, ReadIALU, ReadIALU]>;
def SUB  : ALU_rr<0b0100000, 0b000, "sub">,
           Sched<[WriteIALU, ReadIALU, ReadIALU]>;
def SLL  : ALU_rr<0b0000000, 0b001, "sll">,
           Sched<[WriteShiftReg, ReadShiftReg, ReadShiftReg]>;
let IsSignExtendingOpW = 1 in {
def SLT  : ALU_rr<0b0000000, 0b010, "slt">,
           Sched<[WriteIALU, ReadIALU, ReadIALU]>;
def SLTU : ALU_rr<0b0000000, 0b011, "sltu">,
           Sched<[WriteIALU, ReadIALU, ReadIALU]>;
}
def XOR  : ALU_rr<0b0000000, 0b100, "xor", Commutable=1>,
           Sched<[WriteIALU, ReadIALU, ReadIALU]>;
def SRL  : ALU_rr<0b0000000, 0b101, "srl">,
           Sched<[WriteShiftReg, ReadShiftReg, ReadShiftReg]>;
def SRA  : ALU_rr<0b0100000, 0b101, "sra">,
           Sched<[WriteShiftReg, ReadShiftReg, ReadShiftReg]>;
def OR   : ALU_rr<0b0000000, 0b110, "or", Commutable=1>,
           Sched<[WriteIALU, ReadIALU, ReadIALU]>;
def AND  : ALU_rr<0b0000000, 0b111, "and", Commutable=1>,
           Sched<[WriteIALU, ReadIALU, ReadIALU]>;

let hasSideEffects = 1, mayLoad = 0, mayStore = 0 in {
def FENCE : RVInstI<0b000, OPC_MISC_MEM, (outs),
                    (ins fencearg:$pred, fencearg:$succ),
                    "fence", "$pred, $succ">, Sched<[]> {
  bits<4> pred;
  bits<4> succ;

  let rs1 = 0;
  let rd = 0;
  let imm12 = {0b0000,pred,succ};
}

def FENCE_TSO : RVInstI<0b000, OPC_MISC_MEM, (outs), (ins), "fence.tso", "">, Sched<[]> {
  let rs1 = 0;
  let rd = 0;
  let imm12 = {0b1000,0b0011,0b0011};
}

def FENCE_I : RVInstI<0b001, OPC_MISC_MEM, (outs), (ins), "fence.i", "">, Sched<[]> {
  let rs1 = 0;
  let rd = 0;
  let imm12 = 0;
}

def ECALL : RVInstI<0b000, OPC_SYSTEM, (outs), (ins), "ecall", "">, Sched<[WriteJmp]> {
  let rs1 = 0;
  let rd = 0;
  let imm12 = 0;
}

let isTrap = 1 in
def EBREAK : RVInstI<0b000, OPC_SYSTEM, (outs), (ins), "ebreak", "">,
             Sched<[]> {
  let rs1 = 0;
  let rd = 0;
  let imm12 = 1;
}

// This is a de facto standard (as set by GNU binutils) 32-bit unimplemented
// instruction (i.e., it should always trap, if your implementation has invalid
// instruction traps).
let isTrap = 1 in
def UNIMP : RVInstI<0b001, OPC_SYSTEM, (outs), (ins), "unimp", "">,
            Sched<[]> {
  let rs1 = 0;
  let rd = 0;
  let imm12 = 0b110000000000;
}

} // hasSideEffects = 1, mayLoad = 0, mayStore = 0

def CSRRW : CSR_ir<0b001, "csrrw">;
def CSRRS : CSR_ir<0b010, "csrrs">;
def CSRRC : CSR_ir<0b011, "csrrc">;

def CSRRWI : CSR_ii<0b101, "csrrwi">;
def CSRRSI : CSR_ii<0b110, "csrrsi">;
def CSRRCI : CSR_ii<0b111, "csrrci">;

/// RV64I instructions

let Predicates = [IsRV64] in {
def LWU   : Load_ri<0b110, "lwu">, Sched<[WriteLDW, ReadMemBase]>;
def LD    : Load_ri<0b011, "ld">, Sched<[WriteLDD, ReadMemBase]>;
def SD    : Store_rri<0b011, "sd">, Sched<[WriteSTD, ReadStoreData, ReadMemBase]>;

let IsSignExtendingOpW = 1 in {
let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
def ADDIW : RVInstI<0b000, OPC_OP_IMM_32, (outs GPR:$rd),
                    (ins GPR:$rs1, simm12_lo:$imm12),
                    "addiw", "$rd, $rs1, $imm12">,
            Sched<[WriteIALU32, ReadIALU32]>;

def SLLIW : ShiftW_ri<0b0000000, 0b001, "slliw">,
            Sched<[WriteShiftImm32, ReadShiftImm32]>;
def SRLIW : ShiftW_ri<0b0000000, 0b101, "srliw">,
            Sched<[WriteShiftImm32, ReadShiftImm32]>;
def SRAIW : ShiftW_ri<0b0100000, 0b101, "sraiw">,
            Sched<[WriteShiftImm32, ReadShiftImm32]>;

def ADDW  : ALUW_rr<0b0000000, 0b000, "addw", Commutable=1>,
            Sched<[WriteIALU32, ReadIALU32, ReadIALU32]>;
def SUBW  : ALUW_rr<0b0100000, 0b000, "subw">,
            Sched<[WriteIALU32, ReadIALU32, ReadIALU32]>;
def SLLW  : ALUW_rr<0b0000000, 0b001, "sllw">,
            Sched<[WriteShiftReg32, ReadShiftReg32, ReadShiftReg32]>;
def SRLW  : ALUW_rr<0b0000000, 0b101, "srlw">,
            Sched<[WriteShiftReg32, ReadShiftReg32, ReadShiftReg32]>;
def SRAW  : ALUW_rr<0b0100000, 0b101, "sraw">,
            Sched<[WriteShiftReg32, ReadShiftReg32, ReadShiftReg32]>;
} // IsSignExtendingOpW = 1
} // Predicates = [IsRV64]

//===----------------------------------------------------------------------===//
// Privileged instructions
//===----------------------------------------------------------------------===//

let isBarrier = 1, isReturn = 1, isTerminator = 1 in {
def SRET : Priv<"sret", 0b0001000>, Sched<[]> {
  let rd = 0;
  let rs1 = 0;
  let rs2 = 0b00010;
}

def MRET : Priv<"mret", 0b0011000>, Sched<[]> {
  let rd = 0;
  let rs1 = 0;
  let rs2 = 0b00010;
}

let Predicates = [HasStdExtSmrnmi] in {
def MNRET : Priv<"mnret", 0b0111000>, Sched<[]> {
  let rd = 0;
  let rs1 = 0;
  let rs2 = 0b00010;
}
}// Predicates = [HasStdExtSmrnmi]
} // isBarrier = 1, isReturn = 1, isTerminator = 1


def WFI : Priv<"wfi", 0b0001000>, Sched<[]> {
  let rd = 0;
  let rs1 = 0;
  let rs2 = 0b00101;
}

let Predicates = [HasStdExtSvinval] in {
def SFENCE_W_INVAL : Priv<"sfence.w.inval", 0b0001100>, Sched<[]> {
  let rd = 0;
  let rs1 = 0;
  let rs2 = 0;
}

def SFENCE_INVAL_IR : Priv<"sfence.inval.ir", 0b0001100>, Sched<[]> {
  let rd = 0;
  let rs1 = 0;
  let rs2 = 0b00001;
}
def SINVAL_VMA  : Priv_rr<"sinval.vma", 0b0001011>, Sched<[]>;
def HINVAL_VVMA : Priv_rr<"hinval.vvma", 0b0010011>, Sched<[]>;
def HINVAL_GVMA : Priv_rr<"hinval.gvma", 0b0110011>, Sched<[]>;
} // Predicates = [HasStdExtSvinval]

def SFENCE_VMA  : Priv_rr<"sfence.vma", 0b0001001>, Sched<[]>;

let Predicates = [HasStdExtH] in {
def HFENCE_VVMA : Priv_rr<"hfence.vvma", 0b0010001>, Sched<[]>;
def HFENCE_GVMA : Priv_rr<"hfence.gvma", 0b0110001>, Sched<[]>;

def HLV_B   : HLoad_r<0b0110000, 0b00000, "hlv.b">, Sched<[]>;
def HLV_BU  : HLoad_r<0b0110000, 0b00001, "hlv.bu">, Sched<[]>;
def HLV_H   : HLoad_r<0b0110010, 0b00000, "hlv.h">, Sched<[]>;
def HLV_HU  : HLoad_r<0b0110010, 0b00001, "hlv.hu">, Sched<[]>;
def HLVX_HU : HLoad_r<0b0110010, 0b00011, "hlvx.hu">, Sched<[]>;
def HLV_W   : HLoad_r<0b0110100, 0b00000, "hlv.w">, Sched<[]>;
def HLVX_WU : HLoad_r<0b0110100, 0b00011, "hlvx.wu">, Sched<[]>;
def HSV_B   : HStore_rr<0b0110001, "hsv.b">, Sched<[]>;
def HSV_H   : HStore_rr<0b0110011, "hsv.h">, Sched<[]>;
def HSV_W   : HStore_rr<0b0110101, "hsv.w">, Sched<[]>;
}
let Predicates = [IsRV64, HasStdExtH] in {
def HLV_WU  : HLoad_r<0b0110100, 0b00001, "hlv.wu">, Sched<[]>;
def HLV_D   : HLoad_r<0b0110110, 0b00000, "hlv.d">, Sched<[]>;
def HSV_D   : HStore_rr<0b0110111, "hsv.d">, Sched<[]>;
}

let Predicates = [HasStdExtSmctrOrSsctr] in {
def SCTRCLR : Priv<"sctrclr", 0b0001000>, Sched<[]> {
  let rd = 0;
  let rs1 = 0;
  let rs2 = 0b00100;
}
}

//===----------------------------------------------------------------------===//
// Debug instructions
//===----------------------------------------------------------------------===//

let isBarrier = 1, isReturn = 1, isTerminator = 1 in {
def DRET : Priv<"dret", 0b0111101>, Sched<[]> {
  let rd = 0;
  let rs1 = 0;
  let rs2 = 0b10010;
}
} // isBarrier = 1, isReturn = 1, isTerminator = 1

//===----------------------------------------------------------------------===//
// Assembler Pseudo Instructions (User-Level ISA, Version 2.2, Chapter 20)
//===----------------------------------------------------------------------===//

// Note that the size is 32 because up to 8 32-bit instructions are needed to
// generate an arbitrary 64-bit immediate. However, the size does not really
// matter since PseudoLI is currently only used in the AsmParser where it gets
// expanded to real instructions immediately.
let hasSideEffects = 0, mayLoad = 0, mayStore = 0, Size = 32,
    isCodeGenOnly = 0, isAsmParserOnly = 1 in
def PseudoLI : Pseudo<(outs GPR:$rd), (ins ixlenimm_li:$imm), [],
                      "li", "$rd, $imm">;

def PseudoLB  : PseudoLoad<"lb">;
def PseudoLBU : PseudoLoad<"lbu">;
def PseudoLH  : PseudoLoad<"lh">;
def PseudoLHU : PseudoLoad<"lhu">;
def PseudoLW  : PseudoLoad<"lw">;

def PseudoSB  : PseudoStore<"sb">;
def PseudoSH  : PseudoStore<"sh">;
def PseudoSW  : PseudoStore<"sw">;

let Predicates = [IsRV64] in {
def PseudoLWU : PseudoLoad<"lwu">;
def PseudoLD  : PseudoLoad<"ld">;
def PseudoSD  : PseudoStore<"sd">;
} // Predicates = [IsRV64]

def : InstAlias<"nop",           (ADDI      X0,      X0,           0), 3>;
def : InstAlias<"li $rd, $imm",  (ADDI GPR:$rd,      X0, simm12_lo:$imm), 2>;
def : InstAlias<"mv $rd, $rs",   (ADDI GPR:$rd, GPR:$rs,           0)>;

def : InstAlias<"not $rd, $rs",  (XORI GPR:$rd, GPR:$rs,      -1)>;
def : InstAlias<"neg $rd, $rs",  (SUB  GPR:$rd,      X0, GPR:$rs)>;

let Predicates = [IsRV64] in {
def : InstAlias<"negw $rd, $rs",   (SUBW  GPR:$rd,      X0, GPR:$rs)>;
def : InstAlias<"sext.w $rd, $rs", (ADDIW GPR:$rd, GPR:$rs,       0)>;
} // Predicates = [IsRV64]

def : InstAlias<"seqz $rd, $rs", (SLTIU GPR:$rd, GPR:$rs,       1)>;
def : InstAlias<"snez $rd, $rs", (SLTU  GPR:$rd,      X0, GPR:$rs)>;
def : InstAlias<"sltz $rd, $rs", (SLT   GPR:$rd, GPR:$rs,      X0)>;
def : InstAlias<"sgtz $rd, $rs", (SLT   GPR:$rd,      X0, GPR:$rs)>;

// sgt/sgtu are recognised by the GNU assembler but the canonical slt/sltu
// form will always be printed. Therefore, set a zero weight.
def : InstAlias<"sgt $rd, $rs, $rt", (SLT GPR:$rd, GPR:$rt, GPR:$rs), 0>;
def : InstAlias<"sgtu $rd, $rs, $rt", (SLTU GPR:$rd, GPR:$rt, GPR:$rs), 0>;

def : InstAlias<"beqz $rs, $offset",
                (BEQ GPR:$rs,      X0, bare_simm13_lsb0:$offset)>;
def : InstAlias<"bnez $rs, $offset",
                (BNE GPR:$rs,      X0, bare_simm13_lsb0:$offset)>;
def : InstAlias<"blez $rs, $offset",
                (BGE      X0, GPR:$rs, bare_simm13_lsb0:$offset)>;
def : InstAlias<"bgez $rs, $offset",
                (BGE GPR:$rs,      X0, bare_simm13_lsb0:$offset)>;
def : InstAlias<"bltz $rs, $offset",
                (BLT GPR:$rs,      X0, bare_simm13_lsb0:$offset)>;
def : InstAlias<"bgtz $rs, $offset",
                (BLT      X0, GPR:$rs, bare_simm13_lsb0:$offset)>;

// Always output the canonical mnemonic for the pseudo branch instructions.
// The GNU tools emit the canonical mnemonic for the branch pseudo instructions
// as well (e.g. "bgt" will be recognised by the assembler but never printed by
// objdump). Match this behaviour by setting a zero weight.
def : InstAlias<"bgt $rs, $rt, $offset",
                (BLT  GPR:$rt, GPR:$rs, bare_simm13_lsb0:$offset), 0>;
def : InstAlias<"ble $rs, $rt, $offset",
                (BGE  GPR:$rt, GPR:$rs, bare_simm13_lsb0:$offset), 0>;
def : InstAlias<"bgtu $rs, $rt, $offset",
                (BLTU GPR:$rt, GPR:$rs, bare_simm13_lsb0:$offset), 0>;
def : InstAlias<"bleu $rs, $rt, $offset",
                (BGEU GPR:$rt, GPR:$rs, bare_simm13_lsb0:$offset), 0>;

def : InstAlias<"j $offset",   (JAL X0, simm21_lsb0_jal:$offset)>;
def : InstAlias<"jal $offset", (JAL X1, simm21_lsb0_jal:$offset)>;

// Non-zero offset aliases of "jalr" are the lowest weight, followed by the
// two-register form, then the one-register forms and finally "ret".
def : InstAlias<"jr $rs",                (JALR      X0, GPR:$rs, 0), 3>;
def : InstAlias<"jr ${offset}(${rs})",   (JALR      X0, GPR:$rs, simm12_lo:$offset)>;
def : InstAlias<"jalr $rs",              (JALR      X1, GPR:$rs, 0), 3>;
def : InstAlias<"jalr ${offset}(${rs})", (JALR      X1, GPR:$rs, simm12_lo:$offset)>;
def : InstAlias<"jalr $rd, $rs",         (JALR GPR:$rd, GPR:$rs, 0), 2>;
def : InstAlias<"ret",                   (JALR      X0,      X1, 0), 4>;

// Non-canonical forms for jump targets also accepted by the assembler.
def : InstAlias<"jr $rs, $offset",        (JALR      X0, GPR:$rs, simm12_lo:$offset), 0>;
def : InstAlias<"jalr $rs, $offset",      (JALR      X1, GPR:$rs, simm12_lo:$offset), 0>;
def : InstAlias<"jalr $rd, $rs, $offset", (JALR GPR:$rd, GPR:$rs, simm12_lo:$offset), 0>;
def : InstAlias<"jr (${rs})",             (JALR      X0, GPR:$rs, 0), 0>;
def : InstAlias<"jalr (${rs})",           (JALR      X1, GPR:$rs, 0), 0>;
def : InstAlias<"jalr $rd, (${rs})",      (JALR GPR:$rd, GPR:$rs, 0), 0>;

def : InstAlias<"fence", (FENCE 0xF, 0xF)>; // 0xF == iorw

let Predicates = [HasStdExtZihintpause] in
def : InstAlias<"pause", (FENCE 0x1, 0x0)>; // 0x1 == w

def : InstAlias<"rdinstret $rd", (CSRRS GPR:$rd, INSTRET.Encoding, X0), 2>;
def : InstAlias<"rdcycle $rd",   (CSRRS GPR:$rd, CYCLE.Encoding, X0), 2>;
def : InstAlias<"rdtime $rd",    (CSRRS GPR:$rd, TIME.Encoding, X0), 2>;

let Predicates = [IsRV32] in {
def : InstAlias<"rdinstreth $rd", (CSRRS GPR:$rd, INSTRETH.Encoding, X0), 2>;
def : InstAlias<"rdcycleh $rd",   (CSRRS GPR:$rd, CYCLEH.Encoding, X0), 2>;
def : InstAlias<"rdtimeh $rd",    (CSRRS GPR:$rd, TIMEH.Encoding, X0), 2>;
} // Predicates = [IsRV32]

def : InstAlias<"csrr $rd, $csr", (CSRRS GPR:$rd, csr_sysreg:$csr,      X0)>;
def : InstAlias<"csrw $csr, $rs", (CSRRW      X0, csr_sysreg:$csr, GPR:$rs)>;
def : InstAlias<"csrs $csr, $rs", (CSRRS      X0, csr_sysreg:$csr, GPR:$rs)>;
def : InstAlias<"csrc $csr, $rs", (CSRRC      X0, csr_sysreg:$csr, GPR:$rs)>;

def : InstAlias<"csrwi $csr, $imm", (CSRRWI X0, csr_sysreg:$csr, uimm5:$imm)>;
def : InstAlias<"csrsi $csr, $imm", (CSRRSI X0, csr_sysreg:$csr, uimm5:$imm)>;
def : InstAlias<"csrci $csr, $imm", (CSRRCI X0, csr_sysreg:$csr, uimm5:$imm)>;

let EmitPriority = 0 in {
def : InstAlias<"csrw $csr, $imm", (CSRRWI X0, csr_sysreg:$csr, uimm5:$imm)>;
def : InstAlias<"csrs $csr, $imm", (CSRRSI X0, csr_sysreg:$csr, uimm5:$imm)>;
def : InstAlias<"csrc $csr, $imm", (CSRRCI X0, csr_sysreg:$csr, uimm5:$imm)>;

def : InstAlias<"csrrw $rd, $csr, $imm", (CSRRWI GPR:$rd, csr_sysreg:$csr, uimm5:$imm)>;
def : InstAlias<"csrrs $rd, $csr, $imm", (CSRRSI GPR:$rd, csr_sysreg:$csr, uimm5:$imm)>;
def : InstAlias<"csrrc $rd, $csr, $imm", (CSRRCI GPR:$rd, csr_sysreg:$csr, uimm5:$imm)>;
}

def : InstAlias<"sfence.vma",     (SFENCE_VMA      X0, X0), 2>;
def : InstAlias<"sfence.vma $rs", (SFENCE_VMA GPR:$rs, X0)>;

def : InstAlias<"hfence.gvma",     (HFENCE_GVMA      X0, X0), 2>;
def : InstAlias<"hfence.gvma $rs", (HFENCE_GVMA GPR:$rs, X0)>;

def : InstAlias<"hfence.vvma",     (HFENCE_VVMA      X0, X0), 2>;
def : InstAlias<"hfence.vvma $rs", (HFENCE_VVMA GPR:$rs, X0)>;

let Predicates = [HasStdExtZihintntl] in {
  def : InstAlias<"ntl.p1",     (ADD   X0, X0, X2)>;
  def : InstAlias<"ntl.pall",   (ADD   X0, X0, X3)>;
  def : InstAlias<"ntl.s1",     (ADD   X0, X0, X4)>;
  def : InstAlias<"ntl.all",    (ADD   X0, X0, X5)>;
} // Predicates = [HasStdExtZihintntl]

let EmitPriority = 0 in {
def : InstAlias<"lb $rd, (${rs1})",
                (LB  GPR:$rd, GPR:$rs1, 0)>;
def : InstAlias<"lh $rd, (${rs1})",
                (LH  GPR:$rd, GPR:$rs1, 0)>;
def : InstAlias<"lw $rd, (${rs1})",
                (LW  GPR:$rd, GPR:$rs1, 0)>;
def : InstAlias<"lbu $rd, (${rs1})",
                (LBU  GPR:$rd, GPR:$rs1, 0)>;
def : InstAlias<"lhu $rd, (${rs1})",
                (LHU  GPR:$rd, GPR:$rs1, 0)>;

def : InstAlias<"sb $rs2, (${rs1})",
                (SB  GPR:$rs2, GPR:$rs1, 0)>;
def : InstAlias<"sh $rs2, (${rs1})",
                (SH  GPR:$rs2, GPR:$rs1, 0)>;
def : InstAlias<"sw $rs2, (${rs1})",
                (SW  GPR:$rs2, GPR:$rs1, 0)>;

def : InstAlias<"add $rd, $rs1, $imm12",
                (ADDI  GPR:$rd, GPR:$rs1, simm12_lo:$imm12)>;
def : InstAlias<"and $rd, $rs1, $imm12",
                (ANDI  GPR:$rd, GPR:$rs1, simm12_lo:$imm12)>;
def : InstAlias<"xor $rd, $rs1, $imm12",
                (XORI  GPR:$rd, GPR:$rs1, simm12_lo:$imm12)>;
def : InstAlias<"or $rd, $rs1, $imm12",
                (ORI  GPR:$rd, GPR:$rs1, simm12_lo:$imm12)>;
def : InstAlias<"sll $rd, $rs1, $shamt",
                (SLLI  GPR:$rd, GPR:$rs1, uimmlog2xlen:$shamt)>;
def : InstAlias<"srl $rd, $rs1, $shamt",
                (SRLI  GPR:$rd, GPR:$rs1, uimmlog2xlen:$shamt)>;
def : InstAlias<"sra $rd, $rs1, $shamt",
                (SRAI  GPR:$rd, GPR:$rs1, uimmlog2xlen:$shamt)>;
let Predicates = [IsRV64] in {
def : InstAlias<"lwu $rd, (${rs1})",
                (LWU  GPR:$rd, GPR:$rs1, 0)>;
def : InstAlias<"ld $rd, (${rs1})",
                (LD  GPR:$rd, GPR:$rs1, 0)>;
def : InstAlias<"sd $rs2, (${rs1})",
                (SD  GPR:$rs2, GPR:$rs1, 0)>;

def : InstAlias<"addw $rd, $rs1, $imm12",
                (ADDIW  GPR:$rd, GPR:$rs1, simm12_lo:$imm12)>;
def : InstAlias<"sllw $rd, $rs1, $shamt",
                (SLLIW  GPR:$rd, GPR:$rs1, uimm5:$shamt)>;
def : InstAlias<"srlw $rd, $rs1, $shamt",
                (SRLIW  GPR:$rd, GPR:$rs1, uimm5:$shamt)>;
def : InstAlias<"sraw $rd, $rs1, $shamt",
                (SRAIW  GPR:$rd, GPR:$rs1, uimm5:$shamt)>;
} // Predicates = [IsRV64]
def : InstAlias<"slt $rd, $rs1, $imm12",
                (SLTI  GPR:$rd, GPR:$rs1, simm12_lo:$imm12)>;
def : InstAlias<"sltu $rd, $rs1, $imm12",
                (SLTIU  GPR:$rd, GPR:$rs1, simm12_lo:$imm12)>;
}

def : MnemonicAlias<"move", "mv">;

// The SCALL and SBREAK instructions were renamed to ECALL and EBREAK in
// version 2.1 of the user-level ISA. Like the GNU toolchain, we still accept
// the old name for backwards compatibility.
def : MnemonicAlias<"scall", "ecall">;
def : MnemonicAlias<"sbreak", "ebreak">;

def : InstAlias<"zext.b $rd, $rs", (ANDI GPR:$rd, GPR:$rs, 0xFF)>;

let Predicates = [HasStdExtZicfilp] in {
def : InstAlias<"lpad $imm20", (AUIPC X0, uimm20:$imm20)>;
}

//===----------------------------------------------------------------------===//
// .insn directive instructions
//===----------------------------------------------------------------------===//

def AnyRegOperand : AsmOperandClass {
  let Name = "AnyRegOperand";
  let RenderMethod = "addRegOperands";
  let PredicateMethod = "isAnyReg";
}

def AnyReg : Operand<XLenVT> {
  let OperandType = "OPERAND_REGISTER";
  let ParserMatchClass = AnyRegOperand;
}

// isCodeGenOnly = 1 to hide them from the tablegened assembly parser.
let isCodeGenOnly = 1, hasSideEffects = 1, mayLoad = 1, mayStore = 1,
    hasNoSchedulingInfo = 1 in {
def Insn16 : RVInst16<(outs), (ins uimm16:$value), "", "", [], InstFormatOther> {
  bits<16> value;

  let Inst{15-0} = value;
  let AsmString = ".insn 0x2, $value";
}
def Insn32 : RVInst<(outs), (ins uimm32:$value), "", "", [], InstFormatOther> {
  bits<32> value;

  let Inst{31-0} = value;
  let AsmString = ".insn 0x4, $value";
}
def Insn48 : RVInst48<(outs), (ins uimm48:$value), "", "", [], InstFormatOther> {
  bits<48> value;
  let Inst{47-0} = value;
  let AsmString = ".insn 0x6, $value";
}
def Insn64 : RVInst64<(outs), (ins uimm64:$value), "", "", [], InstFormatOther> {
  bits<64> value;
  let Inst{63-0} = value;
  let AsmString = ".insn 0x8, $value";
}
} // isCodeGenOnly, hasSideEffects, mayLoad, mayStore, hasNoSchedulingInfo

// isCodeGenOnly = 1 to hide them from the tablegened assembly parser.
let isCodeGenOnly = 1, hasSideEffects = 1, mayLoad = 1, mayStore = 1,
    hasNoSchedulingInfo = 1 in {
def InsnR : DirectiveInsnR<(outs AnyReg:$rd), (ins uimm7_opcode:$opcode, uimm3:$funct3,
                                                   uimm7:$funct7, AnyReg:$rs1,
                                                   AnyReg:$rs2),
                           "$opcode, $funct3, $funct7, $rd, $rs1, $rs2">;
def InsnR4 : DirectiveInsnR4<(outs AnyReg:$rd), (ins uimm7_opcode:$opcode,
                                                     uimm3:$funct3,
                                                     uimm2:$funct2,
                                                     AnyReg:$rs1, AnyReg:$rs2,
                                                     AnyReg:$rs3),
                            "$opcode, $funct3, $funct2, $rd, $rs1, $rs2, $rs3">;
def InsnI : DirectiveInsnI<(outs AnyReg:$rd), (ins uimm7_opcode:$opcode, uimm3:$funct3,
                                                   AnyReg:$rs1, simm12_lo:$imm12),
                           "$opcode, $funct3, $rd, $rs1, $imm12">;
def InsnI_Mem : DirectiveInsnI<(outs AnyReg:$rd), (ins uimm7_opcode:$opcode,
                                                       uimm3:$funct3,
                                                       AnyReg:$rs1,
                                                       simm12_lo:$imm12),
                               "$opcode, $funct3, $rd, ${imm12}(${rs1})">;
def InsnB : DirectiveInsnB<(outs), (ins uimm7_opcode:$opcode, uimm3:$funct3,
                                        AnyReg:$rs1, AnyReg:$rs2,
                                        bare_simm13_lsb0:$imm12),
                           "$opcode, $funct3, $rs1, $rs2, $imm12">;
def InsnU : DirectiveInsnU<(outs AnyReg:$rd), (ins uimm7_opcode:$opcode,
                                                   uimm20_lui:$imm20),
                           "$opcode, $rd, $imm20">;
def InsnJ : DirectiveInsnJ<(outs AnyReg:$rd), (ins uimm7_opcode:$opcode,
                                                   simm21_lsb0_jal:$imm20),
                           "$opcode, $rd, $imm20">;
def InsnS : DirectiveInsnS<(outs), (ins uimm7_opcode:$opcode, uimm3:$funct3,
                                        AnyReg:$rs2, AnyReg:$rs1,
                                        simm12_lo:$imm12),
                           "$opcode, $funct3, $rs2, ${imm12}(${rs1})">;
} // isCodeGenOnly, hasSideEffects, mayLoad, mayStore, hasNoSchedulingInfo

// Use InstAliases to match these so that we can combine the insn and format
// into a mnemonic to use as the key for the tablegened asm matcher table. The
// parser will take care of creating these fake mnemonics and will only do it
// for known formats.
let EmitPriority = 0 in {
def : InstAlias<".insn_r $opcode, $funct3, $funct7, $rd, $rs1, $rs2",
                (InsnR AnyReg:$rd, uimm7_opcode:$opcode, uimm3:$funct3, uimm7:$funct7,
                       AnyReg:$rs1, AnyReg:$rs2)>;
// Accept 4 register form of ".insn r" as alias for ".insn r4".
def : InstAlias<".insn_r $opcode, $funct3, $funct2, $rd, $rs1, $rs2, $rs3",
                (InsnR4 AnyReg:$rd, uimm7_opcode:$opcode, uimm3:$funct3, uimm2:$funct2,
                        AnyReg:$rs1, AnyReg:$rs2, AnyReg:$rs3)>;
def : InstAlias<".insn_r4 $opcode, $funct3, $funct2, $rd, $rs1, $rs2, $rs3",
                (InsnR4 AnyReg:$rd, uimm7_opcode:$opcode, uimm3:$funct3, uimm2:$funct2,
                        AnyReg:$rs1, AnyReg:$rs2, AnyReg:$rs3)>;
def : InstAlias<".insn_i $opcode, $funct3, $rd, $rs1, $imm12",
                (InsnI AnyReg:$rd, uimm7_opcode:$opcode, uimm3:$funct3, AnyReg:$rs1,
                       simm12_lo:$imm12)>;
def : InstAlias<".insn_i $opcode, $funct3, $rd, ${imm12}(${rs1})",
                (InsnI_Mem AnyReg:$rd, uimm7_opcode:$opcode, uimm3:$funct3,
                           AnyReg:$rs1, simm12_lo:$imm12)>;
def : InstAlias<".insn_i $opcode, $funct3, $rd, (${rs1})",
                (InsnI_Mem AnyReg:$rd, uimm7_opcode:$opcode, uimm3:$funct3,
                           AnyReg:$rs1, 0)>;
def : InstAlias<".insn_b $opcode, $funct3, $rs1, $rs2, $imm12",
                (InsnB uimm7_opcode:$opcode, uimm3:$funct3, AnyReg:$rs1,
                       AnyReg:$rs2, bare_simm13_lsb0:$imm12)>;
// Accept sb as an alias for b.
def : InstAlias<".insn_sb $opcode, $funct3, $rs1, $rs2, $imm12",
                (InsnB uimm7_opcode:$opcode, uimm3:$funct3, AnyReg:$rs1,
                       AnyReg:$rs2, bare_simm13_lsb0:$imm12)>;
def : InstAlias<".insn_u $opcode, $rd, $imm20",
                (InsnU AnyReg:$rd, uimm7_opcode:$opcode, uimm20_lui:$imm20)>;
def : InstAlias<".insn_j $opcode, $rd, $imm20",
                (InsnJ AnyReg:$rd, uimm7_opcode:$opcode, simm21_lsb0_jal:$imm20)>;
// Accept uj as an alias for j.
def : InstAlias<".insn_uj $opcode, $rd, $imm20",
                (InsnJ AnyReg:$rd, uimm7_opcode:$opcode, simm21_lsb0_jal:$imm20)>;
def : InstAlias<".insn_s $opcode, $funct3, $rs2, ${imm12}(${rs1})",
                (InsnS uimm7_opcode:$opcode, uimm3:$funct3, AnyReg:$rs2,
                       AnyReg:$rs1, simm12_lo:$imm12)>;
def : InstAlias<".insn_s $opcode, $funct3, $rs2, (${rs1})",
                (InsnS uimm7_opcode:$opcode, uimm3:$funct3, AnyReg:$rs2,
                       AnyReg:$rs1, 0)>;
}

//===----------------------------------------------------------------------===//
// Pseudo-instructions and codegen patterns
//
// Naming convention: For 'generic' pattern classes, we use the naming
// convention PatTy1Ty2. For pattern classes which offer a more complex
// expansion, prefix the class name, e.g. BccPat.
//===----------------------------------------------------------------------===//

/// Generic pattern classes

class PatGpr<SDPatternOperator OpNode, RVInst Inst, ValueType vt = XLenVT>
    : Pat<(vt (OpNode (vt GPR:$rs1))), (Inst GPR:$rs1)>;
class PatGprGpr<SDPatternOperator OpNode, RVInst Inst, ValueType vt1 = XLenVT,
                ValueType vt2 = XLenVT>
    : Pat<(vt1 (OpNode (vt1 GPR:$rs1), (vt2 GPR:$rs2))), (Inst GPR:$rs1, GPR:$rs2)>;

class PatGprImm<SDPatternOperator OpNode, RVInst Inst, ImmLeaf ImmType,
                ValueType vt = XLenVT>
    : Pat<(vt (OpNode (vt GPR:$rs1), ImmType:$imm)),
          (Inst GPR:$rs1, ImmType:$imm)>;
class PatGprSimm12<SDPatternOperator OpNode, RVInstI Inst>
    : PatGprImm<OpNode, Inst, simm12_lo>;
class PatGprUimmLog2XLen<SDPatternOperator OpNode, RVInstIShift Inst>
    : PatGprImm<OpNode, Inst, uimmlog2xlen>;

/// Predicates

def assertsexti32 : PatFrag<(ops node:$src), (assertsext node:$src), [{
  return cast<VTSDNode>(N->getOperand(1))->getVT().bitsLE(MVT::i32);
}]>;
def sexti16 : ComplexPattern<XLenVT, 1, "selectSExtBits<16>">;

def sexti32 : ComplexPattern<i64, 1, "selectSExtBits<32>">;
def gi_sexti32 : GIComplexOperandMatcher<s64, "selectSExtBits<32>">,
                 GIComplexPatternEquiv<sexti32>;

def assertzexti32 : PatFrag<(ops node:$src), (assertzext node:$src), [{
  return cast<VTSDNode>(N->getOperand(1))->getVT().bitsLE(MVT::i32);
}]>;

def zexti32 : ComplexPattern<i64, 1, "selectZExtBits<32>">;
def gi_zexti32 : GIComplexOperandMatcher<s64, "selectZExtBits<32>">,
                 GIComplexPatternEquiv<zexti32>;

def zexti16 : ComplexPattern<XLenVT, 1, "selectZExtBits<16>">;
def gi_zexti16 : GIComplexOperandMatcher<s32, "selectZExtBits<16>">,
                 GIComplexPatternEquiv<zexti16>;

def zexti8 : ComplexPattern<XLenVT, 1, "selectZExtBits<8>">;
def gi_zexti8  : GIComplexOperandMatcher<s32, "selectZExtBits<8>">,
                 GIComplexPatternEquiv<zexti8>;

def ext : PatFrags<(ops node:$A), [(sext node:$A), (zext node:$A)]>;

class binop_oneuse<SDPatternOperator operator>
    : PatFrag<(ops node:$A, node:$B),
              (operator node:$A, node:$B)> {
  let HasOneUse = 1;
}

def and_oneuse : binop_oneuse<and>;
def mul_oneuse : binop_oneuse<mul>;

def mul_const_oneuse : PatFrag<(ops node:$A, node:$B),
                               (mul node:$A, node:$B), [{
  if (auto *N1C = dyn_cast<ConstantSDNode>(N->getOperand(1)))
    return N1C->hasOneUse();
  return false;
}]>;

class unop_oneuse<SDPatternOperator operator>
    : PatFrag<(ops node:$A),
              (operator node:$A)> {
  let HasOneUse = 1;
}

def sext_oneuse   : unop_oneuse<sext>;
def zext_oneuse   : unop_oneuse<zext>;
def anyext_oneuse : unop_oneuse<anyext>;
def ext_oneuse    : unop_oneuse<ext>;
def fpext_oneuse  : unop_oneuse<any_fpextend>;

def 33signbits_node : PatLeaf<(i64 GPR:$src), [{
  return CurDAG->ComputeNumSignBits(Op) > 32;
}]>;

class immop_oneuse<ImmLeaf leaf> : PatLeaf<(leaf), [{
  return N->hasOneUse();
}]> {
  let GISelPredicateCode = [{
    return MRI.hasOneNonDBGUse(MI.getOperand(0).getReg());
  }];
}

/// Simple arithmetic operations

def : PatGprGpr<add, ADD>;
def : PatGprSimm12<add, ADDI>;
def : PatGprGpr<sub, SUB>;
def : PatGprGpr<or, OR>;
def : PatGprSimm12<or, ORI>;
def : PatGprGpr<and, AND>;
def : PatGprSimm12<and, ANDI>;
def : PatGprGpr<xor, XOR>;
def : PatGprSimm12<xor, XORI>;
def : PatGprUimmLog2XLen<shl, SLLI>;
def : PatGprUimmLog2XLen<srl, SRLI>;
def : PatGprUimmLog2XLen<sra, SRAI>;

// Select 'or' as ADDI if the immediate bits are known to be 0 in $rs1. This
// can improve compressibility.
def riscv_or_disjoint : PatFrag<(ops node:$lhs, node:$rhs), (or node:$lhs, node:$rhs),[{
  return orDisjoint(N);
}]>;
def : PatGprSimm12<riscv_or_disjoint, ADDI>;

def add_like : PatFrags<(ops node:$lhs, node:$rhs),
                        [(riscv_or_disjoint node:$lhs, node:$rhs),
                         (add  node:$lhs, node:$rhs)]>;

def riscv_xor_like : PatFrags<(ops node:$lhs, node:$rhs),
                              [(riscv_or_disjoint node:$lhs, node:$rhs),
                               (xor  node:$lhs, node:$rhs)]>;

// negate of low bit can be done via two (compressible) shifts.  The negate
// is never compressible since rs1 and rd can't be the same register.
def : Pat<(i32 (sub 0, (and_oneuse GPR:$rs, 1))),
          (SRAI (i32 (SLLI $rs, 31)), 31)>, Requires<[IsRV32]>;
def : Pat<(i64 (sub 0, (and_oneuse GPR:$rs, 1))),
          (SRAI (i64 (SLLI $rs, 63)), 63)>, Requires<[IsRV64]>;

// AND with leading/trailing ones mask exceeding simm32/simm12.
def : Pat<(i64 (and GPR:$rs, immop_oneuse<LeadingOnesMask>:$mask)),
          (SLLI (i64 (SRLI $rs, (TrailingZeros imm:$mask))),
                (TrailingZeros imm:$mask))>;
def : Pat<(XLenVT (and GPR:$rs, immop_oneuse<TrailingOnesMask>:$mask)),
          (SRLI (XLenVT (SLLI $rs, (XLenSubTrailingOnes imm:$mask))),
                (XLenSubTrailingOnes imm:$mask))>;

// Match both a plain shift and one where the shift amount is masked (this is
// typically introduced when the legalizer promotes the shift amount and
// zero-extends it). For RISC-V, the mask is unnecessary as shifts in the base
// ISA only read the least significant 5 bits (RV32I) or 6 bits (RV64I).
def shiftMaskXLen : ComplexPattern<XLenVT, 1, "selectShiftMaskXLen", [], [], 0>;
def shiftMask32   : ComplexPattern<i64, 1, "selectShiftMask32", [], [], 0>;
// FIXME: This is labelled as handling 's32', however the ComplexPattern it
// refers to handles both i32 and i64 based on the HwMode. Currently this LLT
// parameter appears to be ignored so this pattern works for both, however we
// should add a LowLevelTypeByHwMode, and use that to define our XLenLLT instead
// here.
def GIShiftMaskXLen :
    GIComplexOperandMatcher<s32, "selectShiftMaskXLen">,
    GIComplexPatternEquiv<shiftMaskXLen>;
def GIShiftMask32 :
    GIComplexOperandMatcher<s64, "selectShiftMask32">,
    GIComplexPatternEquiv<shiftMask32>;

class shiftop<SDPatternOperator operator>
    : PatFrag<(ops node:$val, node:$count),
              (operator node:$val, (XLenVT (shiftMaskXLen node:$count)))>;
class shiftopw<SDPatternOperator operator>
    : PatFrag<(ops node:$val, node:$count),
              (operator node:$val, (i64 (shiftMask32 node:$count)))>;

def : PatGprGpr<shiftop<shl>, SLL>;
def : PatGprGpr<shiftop<srl>, SRL>;
def : PatGprGpr<shiftop<sra>, SRA>;

// This is a special case of the ADD instruction used to facilitate the use of a
// fourth operand to emit a relocation on a symbol relating to this instruction.
// The relocation does not affect any bits of the instruction itself but is used
// as a hint to the linker.
let hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 0 in
def PseudoAddTPRel : Pseudo<(outs GPR:$rd),
                            (ins GPR:$rs1, GPR:$rs2, tprel_add_symbol:$src), [],
                            "add", "$rd, $rs1, $rs2, $src">;

/// FrameIndex calculations

// Transforms frameindex -> tframeindex.
def to_tframeindex : SDNodeXForm<frameindex, [{
  return CurDAG->getTargetFrameIndex(N->getIndex(), N->getValueType(0));
}]>;

def : GICustomOperandRenderer<"renderFrameIndex">,
      GISDNodeXFormEquiv<to_tframeindex>;

def : Pat<(frameindex:$fi), (ADDI (iPTR (to_tframeindex $fi)), 0)>;

def : Pat<(add_like frameindex:$fi, simm12_lo:$offset),
          (ADDI (iPTR (to_tframeindex $fi)), simm12_lo:$offset)>;

def GIAddrRegImm :
  GIComplexOperandMatcher<s32, "selectAddrRegImm">,
  GIComplexPatternEquiv<AddrRegImm>;

/// Stack probing

let hasSideEffects = 1, mayLoad = 1, mayStore = 1, isCodeGenOnly = 1,
    Defs = [X2], Uses = [X2] in {
// Probed stack allocation of a constant size, used in function prologues when
// stack-clash protection is enabled.
def PROBED_STACKALLOC : Pseudo<(outs),
                               (ins GPR:$target),
                               []>,
                               Sched<[]>;
def PROBED_STACKALLOC_RVV : Pseudo<(outs),
                                   (ins GPR:$target),
                                   []>,
                                   Sched<[]>;
let usesCustomInserter = 1 in
def PROBED_STACKALLOC_DYN : Pseudo<(outs),
                               (ins GPR:$target),
                               [(riscv_probed_alloca GPR:$target)]>,
                               Sched<[]>;
}

/// HI and ADD_LO address nodes.

// Pseudo for a rematerializable LUI+ADDI sequence for loading an address.
// It will be expanded after register allocation.
// FIXME: The scheduling information does not reflect the multiple instructions.
let Size = 8, isReMaterializable = 1 in
def PseudoMovAddr : Pseudo<(outs GPR:$dst), (ins uimm20_lui:$hi, simm12_lo:$lo), []>,
                    Sched<[WriteIALU]>;

def riscv_hi_oneuse : unop_oneuse<riscv_hi>;
def addr_hi_lo : PatFrag<(ops node:$hi, node:$lo),
                         (riscv_add_lo (riscv_hi_oneuse node:$hi), node:$lo)>;

def : Pat<(addr_hi_lo tglobaladdr:$hi, tglobaladdr:$lo),
          (PseudoMovAddr tglobaladdr:$hi, tglobaladdr:$lo)>;
def : Pat<(addr_hi_lo tblockaddress:$hi, tblockaddress:$lo),
          (PseudoMovAddr tblockaddress:$hi, tblockaddress:$lo)>;
def : Pat<(addr_hi_lo tjumptable:$hi, tjumptable:$lo),
          (PseudoMovAddr tjumptable:$hi, tjumptable:$lo)>;
def : Pat<(addr_hi_lo tconstpool:$hi, tconstpool:$lo),
          (PseudoMovAddr tconstpool:$hi, tconstpool:$lo)>;

def : Pat<(riscv_hi tglobaladdr:$in), (LUI tglobaladdr:$in)>;
def : Pat<(riscv_hi tblockaddress:$in), (LUI tblockaddress:$in)>;
def : Pat<(riscv_hi tjumptable:$in), (LUI tjumptable:$in)>;
def : Pat<(riscv_hi tconstpool:$in), (LUI tconstpool:$in)>;

def : Pat<(riscv_add_lo GPR:$hi, tglobaladdr:$lo),
          (ADDI GPR:$hi, tglobaladdr:$lo)>;
def : Pat<(riscv_add_lo GPR:$hi, tblockaddress:$lo),
          (ADDI GPR:$hi, tblockaddress:$lo)>;
def : Pat<(riscv_add_lo GPR:$hi, tjumptable:$lo),
          (ADDI GPR:$hi, tjumptable:$lo)>;
def : Pat<(riscv_add_lo GPR:$hi, tconstpool:$lo),
          (ADDI GPR:$hi, tconstpool:$lo)>;

/// TLS address nodes.

def : Pat<(riscv_hi tglobaltlsaddr:$in), (LUI tglobaltlsaddr:$in)>;
def : Pat<(riscv_add_tprel GPR:$rs1, GPR:$rs2, tglobaltlsaddr:$src),
          (PseudoAddTPRel GPR:$rs1, GPR:$rs2, tglobaltlsaddr:$src)>;
def : Pat<(riscv_add_lo GPR:$src, tglobaltlsaddr:$lo),
          (ADDI GPR:$src, tglobaltlsaddr:$lo)>;

/// Setcc

def : PatGprGpr<setlt, SLT>;
def : PatGprSimm12<setlt, SLTI>;
def : PatGprGpr<setult, SLTU>;
def : PatGprSimm12<setult, SLTIU>;

// RISC-V doesn't have general instructions for integer setne/seteq, but we can
// check for equality with 0. These ComplexPatterns rewrite the setne/seteq into
// something that can be compared with 0.
// These ComplexPatterns must be used in pairs.
def riscv_setne : ComplexPattern<XLenVT, 1, "selectSETNE", [setcc]>;
def riscv_seteq : ComplexPattern<XLenVT, 1, "selectSETEQ", [setcc]>;

// Define pattern expansions for setcc operations that aren't directly
// handled by a RISC-V instruction.
def : Pat<(riscv_seteq (XLenVT GPR:$rs1)), (SLTIU GPR:$rs1, 1)>;
def : Pat<(riscv_setne (XLenVT GPR:$rs1)), (SLTU (XLenVT X0), GPR:$rs1)>;
def : Pat<(XLenVT (setne (XLenVT GPR:$rs1), -1)), (SLTIU GPR:$rs1, -1)>;
def : Pat<(XLenVT (seteq (XLenVT (and GPR:$rs, immop_oneuse<TrailingOnesMask>:$mask)), 0)),
          (SLTIU (XLenVT (SLLI GPR:$rs, (XLenSubTrailingOnes imm:$mask))), 1)>;
def : Pat<(XLenVT (setne (XLenVT (and GPR:$rs, immop_oneuse<TrailingOnesMask>:$mask)), 0)),
          (SLTU (XLenVT X0), (XLenVT (SLLI GPR:$rs, (XLenSubTrailingOnes imm:$mask))))>;

def IntCCtoRISCVCC : SDNodeXForm<riscv_selectcc, [{
  ISD::CondCode CC = cast<CondCodeSDNode>(N->getOperand(2))->get();
  RISCVCC::CondCode BrCC = getRISCVCCForIntCC(CC);
  return CurDAG->getTargetConstant(BrCC, SDLoc(N), Subtarget->getXLenVT());
}]>;

def riscv_selectcc_frag : PatFrag<(ops node:$lhs, node:$rhs, node:$cc,
                                       node:$truev, node:$falsev),
                                  (riscv_selectcc node:$lhs, node:$rhs,
                                                  node:$cc, node:$truev,
                                                  node:$falsev), [{}],
                                  IntCCtoRISCVCC>;

multiclass SelectCC_GPR_rrirr<DAGOperand valty, ValueType vt> {
  let usesCustomInserter = 1 in
  def _Using_CC_GPR : Pseudo<(outs valty:$dst),
                             (ins GPR:$lhs, GPR:$rhs, cond_code:$cc,
                              valty:$truev, valty:$falsev),
                             [(set valty:$dst,
                               (riscv_selectcc_frag:$cc (XLenVT GPR:$lhs), GPR:$rhs, cond,
                                                        (vt valty:$truev), valty:$falsev))]>;
  // Explicitly select 0 in the condition to X0. The register coalescer doesn't
  // always do it.
  def : Pat<(riscv_selectcc_frag:$cc (XLenVT GPR:$lhs), 0, cond, (vt valty:$truev),
                                     valty:$falsev),
            (!cast<Instruction>(NAME#"_Using_CC_GPR") GPR:$lhs, (XLenVT X0),
             (IntCCtoRISCVCC $cc), valty:$truev, valty:$falsev)>;
}

let Predicates = [NoConditionalMoveFusion] in
defm Select_GPR : SelectCC_GPR_rrirr<GPR, XLenVT>;

class SelectCompressOpt<CondCode Cond>
    : Pat<(riscv_selectcc_frag:$select (XLenVT GPR:$lhs), simm12_no6:$Constant, Cond,
                                       (XLenVT GPR:$truev), GPR:$falsev),
    (Select_GPR_Using_CC_GPR (XLenVT (ADDI GPR:$lhs, (NegImm simm12_lo:$Constant))), (XLenVT X0),
                          (IntCCtoRISCVCC $select), GPR:$truev, GPR:$falsev)>;

def OptForMinSize : Predicate<"MF ? MF->getFunction().hasMinSize() : false">;

let Predicates = [HasStdExtC, OptForMinSize] in {
  def : SelectCompressOpt<SETEQ>;
  def : SelectCompressOpt<SETNE>;
}

multiclass SelectCC_GPR_riirr<DAGOperand valty, DAGOperand imm> {
  let usesCustomInserter = 1 in
  def Select_# valty #_Using_ # NAME
      : Pseudo<(outs valty:$dst),
               (ins valty:$lhs, imm:$imm, cond_code:$cc,
                valty:$truev, valty:$falsev), []>;
}

let Predicates = [IsRV32] in {
def : Pat<(i32 (setlt (i32 GPR:$rs1), 0)), (SRLI GPR:$rs1, 31)>; // compressible
}
let Predicates = [IsRV64] in {
def : Pat<(i64 (setlt (i64 GPR:$rs1), 0)), (SRLI GPR:$rs1, 63)>; // compressible
def : Pat<(i64 (setlt (sext_inreg GPR:$rs1, i32), 0)), (SRLIW GPR:$rs1, 31)>;
}

/// Branches and jumps

// Match `riscv_brcc` and lower to the appropriate RISC-V branch instruction.
multiclass BccPat<CondCode Cond, RVInstB Inst> {
  def : Pat<(riscv_brcc (XLenVT GPR:$rs1), GPR:$rs2, Cond, bb:$imm12),
            (Inst GPR:$rs1, GPR:$rs2, bare_simm13_lsb0_bb:$imm12)>;
  // Explicitly select 0 to X0. The register coalescer doesn't always do it.
  def : Pat<(riscv_brcc (XLenVT GPR:$rs1), 0, Cond, bb:$imm12),
            (Inst GPR:$rs1, (XLenVT X0), bare_simm13_lsb0_bb:$imm12)>;
}

class BrccCompressOpt<CondCode Cond, RVInstB Inst>
    : Pat<(riscv_brcc GPR:$lhs, simm12_no6:$Constant, Cond, bb:$place),
          (Inst (XLenVT (ADDI GPR:$lhs, (NegImm simm12_lo:$Constant))),
                (XLenVT X0), bb:$place)>;

defm : BccPat<SETEQ, BEQ>;
defm : BccPat<SETNE, BNE>;
defm : BccPat<SETLT, BLT>;
defm : BccPat<SETGE, BGE>;
defm : BccPat<SETULT, BLTU>;
defm : BccPat<SETUGE, BGEU>;

let Predicates = [HasStdExtZca, OptForMinSize] in {
  def : BrccCompressOpt<SETEQ, BEQ>;
  def : BrccCompressOpt<SETNE, BNE>;
}

class LongBccPseudo : Pseudo<(outs),
                             (ins GPR:$rs1, GPR:$rs2, simm21_lsb0_jal:$imm20),
                             []> {
  let Size = 8;
  let isBarrier = 1;
  let isBranch = 1;
  let hasSideEffects = 0;
  let mayStore = 0;
  let mayLoad = 0;
  let isAsmParserOnly = 1;
  let hasNoSchedulingInfo = 1;
}

def PseudoLongBEQ : LongBccPseudo;
def PseudoLongBNE : LongBccPseudo;
def PseudoLongBLT : LongBccPseudo;
def PseudoLongBGE : LongBccPseudo;
def PseudoLongBLTU : LongBccPseudo;
def PseudoLongBGEU : LongBccPseudo;

let isBarrier = 1, isBranch = 1, isTerminator = 1 in
def PseudoBR : Pseudo<(outs), (ins simm21_lsb0_jal:$imm20), [(br bb:$imm20)]>,
               PseudoInstExpansion<(JAL X0, simm21_lsb0_jal:$imm20)>;

let Predicates = [NoStdExtZicfilp],
    isBarrier = 1, isBranch = 1, isIndirectBranch = 1, isTerminator = 1 in
def PseudoBRIND : Pseudo<(outs), (ins GPRJALR:$rs1, simm12_lo:$imm12), []>,
                  PseudoInstExpansion<(JALR X0, GPR:$rs1, simm12_lo:$imm12)>;

let Predicates = [HasStdExtZicfilp],
    isBarrier = 1, isBranch = 1, isIndirectBranch = 1, isTerminator = 1 in {
def PseudoBRINDNonX7 : Pseudo<(outs), (ins GPRJALRNonX7:$rs1, simm12_lo:$imm12), []>,
                       PseudoInstExpansion<(JALR X0, GPR:$rs1, simm12_lo:$imm12)>;
def PseudoBRINDX7 : Pseudo<(outs), (ins GPRX7:$rs1, simm12_lo:$imm12), []>,
                    PseudoInstExpansion<(JALR X0, GPR:$rs1, simm12_lo:$imm12)>;
}

// For Zicfilp, need to avoid using X7/T2 for indirect branches which need
// landing pad.
let Predicates = [HasStdExtZicfilp] in {
def : Pat<(brind GPRJALRNonX7:$rs1), (PseudoBRINDNonX7 GPRJALRNonX7:$rs1, 0)>;
def : Pat<(brind (add GPRJALRNonX7:$rs1, simm12_lo:$imm12)),
          (PseudoBRINDNonX7 GPRJALRNonX7:$rs1, simm12_lo:$imm12)>;

def : Pat<(riscv_sw_guarded_brind GPRX7:$rs1), (PseudoBRINDX7 GPRX7:$rs1, 0)>;
def : Pat<(riscv_sw_guarded_brind (add GPRX7:$rs1, simm12_lo:$imm12)),
          (PseudoBRINDX7 GPRX7:$rs1, simm12_lo:$imm12)>;
}

let Predicates = [NoStdExtZicfilp] in {
def : Pat<(brind GPRJALR:$rs1), (PseudoBRIND GPRJALR:$rs1, 0)>;
def : Pat<(brind (add GPRJALR:$rs1, simm12_lo:$imm12)),
          (PseudoBRIND GPRJALR:$rs1, simm12_lo:$imm12)>;
}

// PseudoCALLReg is a generic pseudo instruction for calls which will eventually
// expand to auipc and jalr while encoding, with any given register used as the
// destination.
// Define AsmString to print "call" when compile with -S flag.
// Define isCodeGenOnly = 0 to support parsing assembly "call" instruction.
let isCall = 1, isBarrier = 1, isCodeGenOnly = 0, Size = 8, hasSideEffects = 0,
    mayStore = 0, mayLoad = 0 in
def PseudoCALLReg : Pseudo<(outs GPR:$rd), (ins call_symbol:$func), [],
                           "call", "$rd, $func">,
                    Sched<[WriteIALU, WriteJalr, ReadJalr]>;

// PseudoCALL is a pseudo instruction which will eventually expand to auipc
// and jalr while encoding. This is desirable, as an auipc+jalr pair with
// R_RISCV_CALL and R_RISCV_RELAX relocations can be be relaxed by the linker
// if the offset fits in a signed 21-bit immediate.
// Define AsmString to print "call" when compile with -S flag.
// Define isCodeGenOnly = 0 to support parsing assembly "call" instruction.
let isCall = 1, Defs = [X1], isCodeGenOnly = 0, Size = 8 in
def PseudoCALL : Pseudo<(outs), (ins call_symbol:$func), [],
                        "call", "$func">,
                 Sched<[WriteIALU, WriteJalr, ReadJalr]>;

def : Pat<(riscv_call tglobaladdr:$func), (PseudoCALL tglobaladdr:$func)>;
def : Pat<(riscv_call texternalsym:$func), (PseudoCALL texternalsym:$func)>;

def : Pat<(riscv_sret_glue), (SRET)>;
def : Pat<(riscv_mret_glue), (MRET)>;
let Predicates = [HasStdExtSmrnmi] in
def : Pat<(riscv_mnret_glue), (MNRET)>;

let isCall = 1, Defs = [X1] in {
let Predicates = [NoStdExtZicfilp] in
def PseudoCALLIndirect : Pseudo<(outs), (ins GPRJALR:$rs1),
                                [(riscv_call GPRJALR:$rs1)]>,
                         PseudoInstExpansion<(JALR X1, GPR:$rs1, 0)>;
let Predicates = [HasStdExtZicfilp] in {
def PseudoCALLIndirectNonX7 : Pseudo<(outs), (ins GPRJALRNonX7:$rs1),
                                     [(riscv_call GPRJALRNonX7:$rs1)]>,
                              PseudoInstExpansion<(JALR X1, GPR:$rs1, 0)>;
// For large code model, non-indirect calls could be software-guarded
def PseudoCALLIndirectX7 : Pseudo<(outs), (ins GPRX7:$rs1),
                                  [(riscv_sw_guarded_call GPRX7:$rs1)]>,
                           PseudoInstExpansion<(JALR X1, GPR:$rs1, 0)>;
}
}

let isBarrier = 1, isReturn = 1, isTerminator = 1 in
def PseudoRET : Pseudo<(outs), (ins), [(riscv_ret_glue)]>,
                PseudoInstExpansion<(JALR X0, X1, 0)>;

// PseudoTAIL is a pseudo instruction similar to PseudoCALL and will eventually
// expand to auipc and jalr while encoding.
// Define AsmString to print "tail" when compile with -S flag.
let isCall = 1, isTerminator = 1, isReturn = 1, isBarrier = 1, Uses = [X2],
    Size = 8, isCodeGenOnly = 0 in
def PseudoTAIL : Pseudo<(outs), (ins call_symbol:$dst), [],
                        "tail", "$dst">,
                 Sched<[WriteIALU, WriteJalr, ReadJalr]>;

let isCall = 1, isTerminator = 1, isReturn = 1, isBarrier = 1, Uses = [X2] in {
let Predicates = [NoStdExtZicfilp] in
def PseudoTAILIndirect : Pseudo<(outs), (ins GPRTC:$rs1),
                                [(riscv_tail GPRTC:$rs1)]>,
                         PseudoInstExpansion<(JALR X0, GPR:$rs1, 0)>;
let Predicates = [HasStdExtZicfilp] in {
def PseudoTAILIndirectNonX7 : Pseudo<(outs), (ins GPRTCNonX7:$rs1),
                                     [(riscv_tail GPRTCNonX7:$rs1)]>,
                              PseudoInstExpansion<(JALR X0, GPR:$rs1, 0)>;
// For large code model, non-indirect calls could be software-guarded
def PseudoTAILIndirectX7 : Pseudo<(outs), (ins GPRX7:$rs1),
                                  [(riscv_sw_guarded_tail GPRX7:$rs1)]>,
                           PseudoInstExpansion<(JALR X0, GPR:$rs1, 0)>;
}
}

def : Pat<(riscv_tail (iPTR tglobaladdr:$dst)),
          (PseudoTAIL tglobaladdr:$dst)>;
def : Pat<(riscv_tail (iPTR texternalsym:$dst)),
          (PseudoTAIL texternalsym:$dst)>;

let isCall = 0, isBarrier = 1, isBranch = 1, isTerminator = 1, Size = 8,
    isCodeGenOnly = 0, hasSideEffects = 0, mayStore = 0, mayLoad = 0 in
def PseudoJump : Pseudo<(outs GPR:$rd), (ins pseudo_jump_symbol:$target), [],
                        "jump", "$target, $rd">,
                 Sched<[WriteIALU, WriteJalr, ReadJalr]>;

// Pseudo for a rematerializable constant materialization sequence.
// This is an experimental feature enabled by
// -riscv-use-rematerializable-movimm in RISCVISelDAGToDAG.cpp
// It will be expanded after register allocation.
// FIXME: The scheduling information does not reflect the multiple instructions.
let hasSideEffects = 0, mayLoad = 0, mayStore = 0, Size = 8,
    isReMaterializable = 1 in
def PseudoMovImm : Pseudo<(outs GPR:$dst), (ins i32imm:$imm), []>,
                   Sched<[WriteIALU]>;

let hasSideEffects = 0, mayLoad = 0, mayStore = 0, Size = 8, isCodeGenOnly = 0,
    isAsmParserOnly = 1 in
def PseudoLLA : Pseudo<(outs GPR:$dst), (ins bare_symbol:$src), [],
                       "lla", "$dst, $src">;

// Refer to comment on PseudoLI for explanation of Size=32
let hasSideEffects = 0, mayLoad = 0, mayStore = 0, Size = 8, isCodeGenOnly = 0,
    isAsmParserOnly = 1 in
def PseudoLLAImm : Pseudo<(outs GPR:$dst), (ins ixlenimm_li_restricted:$imm), [],
                          "lla", "$dst, $imm">;
def : Pat<(riscv_lla tglobaladdr:$in), (PseudoLLA tglobaladdr:$in)>;
def : Pat<(riscv_lla tblockaddress:$in), (PseudoLLA tblockaddress:$in)>;
def : Pat<(riscv_lla tjumptable:$in), (PseudoLLA tjumptable:$in)>;
def : Pat<(riscv_lla tconstpool:$in), (PseudoLLA tconstpool:$in)>;

let hasSideEffects = 0, mayLoad = 1, mayStore = 0, Size = 8, isCodeGenOnly = 0,
    isAsmParserOnly = 1 in
def PseudoLGA : Pseudo<(outs GPR:$dst), (ins bare_symbol:$src), [],
                       "lga", "$dst, $src">;

let hasSideEffects = 0, mayLoad = 1, mayStore = 0, Size = 8, isCodeGenOnly = 0,
    isAsmParserOnly = 1 in
def PseudoLA : Pseudo<(outs GPR:$dst), (ins bare_symbol:$src), [],
                      "la", "$dst, $src">;

// Refer to comment on PseudoLI for explanation of Size=32
let hasSideEffects = 0, mayLoad = 0, mayStore = 0, Size = 32,
    isCodeGenOnly = 0, isAsmParserOnly = 1 in
def PseudoLAImm : Pseudo<(outs GPR:$rd), (ins ixlenimm_li_restricted:$imm), [],
                         "la", "$rd, $imm">;

let hasSideEffects = 0, mayLoad = 1, mayStore = 0, Size = 8, isCodeGenOnly = 0,
    isAsmParserOnly = 1 in
def PseudoLA_TLS_IE : Pseudo<(outs GPR:$dst), (ins bare_symbol:$src), [],
                             "la.tls.ie", "$dst, $src">;

let hasSideEffects = 0, mayLoad = 0, mayStore = 0, Size = 8, isCodeGenOnly = 0,
    isAsmParserOnly = 1 in
def PseudoLA_TLS_GD : Pseudo<(outs GPR:$dst), (ins bare_symbol:$src), [],
                             "la.tls.gd", "$dst, $src">;

let hasSideEffects = 0, mayLoad = 1, mayStore = 0, Size = 32, isCodeGenOnly = 0 in
def PseudoLA_TLSDESC : Pseudo<(outs GPR:$dst), (ins bare_symbol:$src), [],
                             "la.tlsdesc", "$dst, $src">;

def TLSDESCCallSymbol : AsmOperandClass {
  let Name = "TLSDESCCallSymbol";
  let RenderMethod = "addImmOperands";
  let DiagnosticType = "InvalidTLSDESCCallSymbol";
  let DiagnosticString = "operand must be a symbol with %tlsdesc_call specifier";
  let ParserMethod = "parseOperandWithSpecifier";
}

// A bare symbol with the %tlsdesc_call variant.
def tlsdesc_call_symbol : Operand<XLenVT> {
  let ParserMatchClass = TLSDESCCallSymbol;
}
// This is a special case of the JALR instruction used to facilitate the use of a
// fourth operand to emit a relocation on a symbol relating to this instruction.
// The relocation does not affect any bits of the instruction itself but is used
// as a hint to the linker.
let isCall = 1, isBarrier = 1, isCodeGenOnly = 0, Size = 8, hasSideEffects = 0,
    mayStore = 0, mayLoad = 0 in
def PseudoTLSDESCCall : Pseudo<(outs GPR:$rd),
                         (ins GPR:$rs1, simm12_lo:$imm12, tlsdesc_call_symbol:$src), [],
                         "jalr", "$rd, ${imm12}(${rs1}), $src">,
                         Sched<[WriteJalr, ReadJalr]> {
  let Defs = [X10];
  let Uses = [X10];
}


/// Sign/Zero Extends

// There are single-instruction versions of these in Zbb, so disable these
// Pseudos if that extension is present.
let hasSideEffects = 0, mayLoad = 0,
    mayStore = 0, isCodeGenOnly = 0, isAsmParserOnly = 1 in {
def PseudoSEXT_B : Pseudo<(outs GPR:$rd), (ins GPR:$rs), [], "sext.b", "$rd, $rs">;
def PseudoSEXT_H : Pseudo<(outs GPR:$rd), (ins GPR:$rs), [], "sext.h", "$rd, $rs">;
// rv64's sext.w is defined above, using InstAlias<"sext.w ...
// zext.b is defined above, using InstAlias<"zext.b ...
def PseudoZEXT_H : Pseudo<(outs GPR:$rd), (ins GPR:$rs), [], "zext.h", "$rd, $rs">;
} // hasSideEffects = 0, ...

let Predicates = [IsRV64], hasSideEffects = 0, mayLoad = 0, mayStore = 0,
  isCodeGenOnly = 0, isAsmParserOnly = 1 in {
def PseudoZEXT_W : Pseudo<(outs GPR:$rd), (ins GPR:$rs), [], "zext.w", "$rd, $rs">;
} // Predicates = [IsRV64], ...

/// Loads

class LdPat<PatFrag LoadOp, RVInst Inst, ValueType vt = XLenVT>
    : Pat<(vt (LoadOp (AddrRegImm (XLenVT GPRMem:$rs1), simm12_lo:$imm12))),
          (Inst GPRMem:$rs1, simm12_lo:$imm12)>;

def : LdPat<sextloadi8, LB>;
def : LdPat<extloadi8, LBU>; // Prefer unsigned due to no c.lb in Zcb.
def : LdPat<sextloadi16, LH>;
def : LdPat<extloadi16, LH>;
def : LdPat<load, LW, i32>, Requires<[IsRV32]>;
def : LdPat<zextloadi8, LBU>;
def : LdPat<zextloadi16, LHU>;

/// Stores

class StPat<PatFrag StoreOp, RVInst Inst, RegisterClass StTy,
            ValueType vt>
    : Pat<(StoreOp (vt StTy:$rs2), (AddrRegImm (XLenVT GPRMem:$rs1),
                   simm12_lo:$imm12)),
          (Inst StTy:$rs2, GPRMem:$rs1, simm12_lo:$imm12)>;

def : StPat<truncstorei8, SB, GPR, XLenVT>;
def : StPat<truncstorei16, SH, GPR, XLenVT>;
def : StPat<store, SW, GPR, i32>, Requires<[IsRV32]>;

/// Fences

// Refer to Table A.6 in the version 2.3 draft of the RISC-V Instruction Set
// Manual: Volume I.

// fence acquire -> fence r, rw
def : Pat<(atomic_fence (XLenVT 4), (timm)), (FENCE 0b10, 0b11)>;
// fence release -> fence rw, w
def : Pat<(atomic_fence (XLenVT 5), (timm)), (FENCE 0b11, 0b1)>;
// fence acq_rel -> fence.tso
def : Pat<(atomic_fence (XLenVT 6), (timm)), (FENCE_TSO)>;
// fence seq_cst -> fence rw, rw
def : Pat<(atomic_fence (XLenVT 7), (timm)), (FENCE 0b11, 0b11)>;

// Lowering for atomic load and store is defined in RISCVInstrInfoA.td.
// Although these are lowered to fence+load/store instructions defined in the
// base RV32I/RV64I ISA, this lowering is only used when the A extension is
// present. This is necessary as it isn't valid to mix __atomic_* libcalls
// with inline atomic operations for the same object.

/// Access to system registers

// Helpers for defining specific operations. They are defined for each system
// register separately. Side effect is not used because dependencies are
// expressed via use-def properties.

class ReadSysReg<SysReg SR, list<Register> Regs>
  : Pseudo<(outs GPR:$rd), (ins),
           [(set GPR:$rd, (XLenVT (riscv_read_csr (XLenVT SR.Encoding))))]>,
    PseudoInstExpansion<(CSRRS GPR:$rd, SR.Encoding, X0)> {
  let hasSideEffects = 0;
  let Uses = Regs;
}

class WriteSysReg<SysReg SR, list<Register> Regs>
  : Pseudo<(outs), (ins GPR:$val),
           [(riscv_write_csr (XLenVT SR.Encoding), (XLenVT GPR:$val))]>,
    PseudoInstExpansion<(CSRRW X0, SR.Encoding, GPR:$val)> {
  let hasSideEffects = 0;
  let Defs = Regs;
}

class WriteSysRegImm<SysReg SR, list<Register> Regs>
  : Pseudo<(outs), (ins uimm5:$val),
           [(riscv_write_csr (XLenVT SR.Encoding), uimm5:$val)]>,
    PseudoInstExpansion<(CSRRWI X0, SR.Encoding, uimm5:$val)> {
  let hasSideEffects = 0;
  let Defs = Regs;
}

class SwapSysReg<SysReg SR, list<Register> Regs>
  : Pseudo<(outs GPR:$rd), (ins GPR:$val),
           [(set GPR:$rd, (riscv_swap_csr (XLenVT SR.Encoding), (XLenVT GPR:$val)))]>,
    PseudoInstExpansion<(CSRRW GPR:$rd, SR.Encoding, GPR:$val)> {
  let hasSideEffects = 0;
  let Uses = Regs;
  let Defs = Regs;
}

class SwapSysRegImm<SysReg SR, list<Register> Regs>
  : Pseudo<(outs GPR:$rd), (ins uimm5:$val),
           [(set GPR:$rd, (XLenVT (riscv_swap_csr (XLenVT SR.Encoding), uimm5:$val)))]>,
    PseudoInstExpansion<(CSRRWI GPR:$rd, SR.Encoding, uimm5:$val)> {
  let hasSideEffects = 0;
  let Uses = Regs;
  let Defs = Regs;
}

class ClearSysReg<SysReg SR, list<Register> Regs>
  : Pseudo<(outs), (ins GPR:$val),
           [(riscv_clear_csr (XLenVT SR.Encoding), (XLenVT GPR:$val))]>,
    PseudoInstExpansion<(CSRRC X0, SR.Encoding, GPR:$val)> {
  let hasSideEffects = 0;
  let Uses = Regs;
  let Defs = Regs;
}

class ClearSysRegImm<SysReg SR, list<Register> Regs>
  : Pseudo<(outs), (ins uimm5:$val),
           [(riscv_clear_csr (XLenVT SR.Encoding), uimm5:$val)]>,
    PseudoInstExpansion<(CSRRCI X0, SR.Encoding, uimm5:$val)> {
  let hasSideEffects = 0;
  let Uses = Regs;
  let Defs = Regs;
}

class SetSysReg<SysReg SR, list<Register> Regs>
  : Pseudo<(outs), (ins GPR:$val),
           [(riscv_set_csr (XLenVT SR.Encoding), (XLenVT GPR:$val))]>,
    PseudoInstExpansion<(CSRRS X0, SR.Encoding, GPR:$val)> {
  let hasSideEffects = 0;
  let Uses = Regs;
  let Defs = Regs;
}

class SetSysRegImm<SysReg SR, list<Register> Regs>
  : Pseudo<(outs), (ins uimm5:$val),
           [(riscv_set_csr (XLenVT SR.Encoding), uimm5:$val)]>,
    PseudoInstExpansion<(CSRRSI X0, SR.Encoding, uimm5:$val)> {
  let hasSideEffects = 0;
  let Uses = Regs;
  let Defs = Regs;
}

def ReadFRM : ReadSysReg<SysRegFRM, [FRM]>;
let hasPostISelHook = 1 in {
def WriteFRM : WriteSysReg<SysRegFRM, [FRM]>;
def WriteFRMImm : WriteSysRegImm<SysRegFRM, [FRM]>;
def SwapFRMImm : SwapSysRegImm<SysRegFRM, [FRM]>;
}

def WriteVXRMImm : WriteSysRegImm<SysRegVXRM, [VXRM]>;

let hasSideEffects = true in {
def ReadFFLAGS : ReadSysReg<SysRegFFLAGS, [FFLAGS]>;
def WriteFFLAGS : WriteSysReg<SysRegFFLAGS, [FFLAGS]>;
}

let hasPostISelHook = 1 in {
def ReadFCSR : ReadSysReg<SysRegFCSR, [FRM, FFLAGS]>;
def WriteFCSR : WriteSysReg<SysRegFCSR, [FRM, FFLAGS]>;
def WriteFCSRImm : WriteSysRegImm<SysRegFCSR, [FRM, FFLAGS]>;
def ClearFCSR : ClearSysReg<SysRegFCSR, [FRM, FFLAGS]>;
def ClearFCSRImm : ClearSysRegImm<SysRegFCSR, [FRM, FFLAGS]>;
def SetFCSR : SetSysReg<SysRegFCSR, [FRM, FFLAGS]>;
def SetFCSRImm : SetSysRegImm<SysRegFCSR, [FRM, FFLAGS]>;
}

/// Other pseudo-instructions

// Pessimistically assume the stack pointer will be clobbered
let Defs = [X2], Uses = [X2] in {
def ADJCALLSTACKDOWN : Pseudo<(outs), (ins i32imm:$amt1, i32imm:$amt2),
                              [(callseq_start timm:$amt1, timm:$amt2)]>;
def ADJCALLSTACKUP   : Pseudo<(outs), (ins i32imm:$amt1, i32imm:$amt2),
                              [(callseq_end timm:$amt1, timm:$amt2)]>;
} // Defs = [X2], Uses = [X2]

/// RV64 patterns

let Predicates = [IsRV64, NoStdExtZba] in {
def : Pat<(i64 (and GPR:$rs1, 0xffffffff)), (SRLI (i64 (SLLI GPR:$rs1, 32)), 32)>;

// If we're shifting a 32-bit zero extended value left by 0-31 bits, use 2
// shifts instead of 3. This can occur when unsigned is used to index an array.
def : Pat<(i64 (shl (and GPR:$rs1, 0xffffffff), uimm5:$shamt)),
          (SRLI (i64 (SLLI GPR:$rs1, 32)), (ImmSubFrom32 uimm5:$shamt))>;
} // Predicates = [IsRV64, NoStdExtZba]

class binop_allhusers<SDPatternOperator operator>
    : PatFrag<(ops node:$lhs, node:$rhs),
              (XLenVT (operator node:$lhs, node:$rhs)), [{
  return hasAllHUsers(N);
}]> {
  let GISelPredicateCode = [{ return hasAllHUsers(MI); }];
}

// PatFrag to allow ADDW/SUBW/MULW/SLLW to be selected from i64 add/sub/mul/shl
// if only the lower 32 bits of their result is used.
class binop_allwusers<SDPatternOperator operator>
    : PatFrag<(ops node:$lhs, node:$rhs), (i64 (operator node:$lhs, node:$rhs)),
              [{
  return hasAllWUsers(N);
}]> {
  let GISelPredicateCode = [{ return hasAllWUsers(MI); }];
}

def sexti32_allwusers : PatFrag<(ops node:$src),
                                (sext_inreg node:$src, i32), [{
  return hasAllWUsers(N);
}]>;

def ImmSExt32 : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(SignExtend64<32>(N->getSExtValue()),
                                   SDLoc(N), N->getValueType(0));
}]>;
// Look for constants where the upper 32 bits are 0, but sign extending bit 31
// would be an simm12.
def u32simm12 : ImmLeaf<XLenVT, [{
  return isUInt<32>(Imm) && isInt<12>(SignExtend64<32>(Imm));
}], ImmSExt32>;

let Predicates = [IsRV64] in {

def : Pat<(i64 (and GPR:$rs, immop_oneuse<LeadingOnesWMask>:$mask)),
          (SLLI (i64 (SRLIW $rs, (TrailingZeros imm:$mask))),
                (TrailingZeros imm:$mask))>;

/// sext and zext

// Sign extend is not needed if all users are W instructions.
def : Pat<(sexti32_allwusers GPR:$rs1), (XLenVT GPR:$rs1)>;

def : Pat<(sext_inreg GPR:$rs1, i32), (ADDIW GPR:$rs1, 0)>;

/// ALU operations

def : Pat<(i64 (srl (and GPR:$rs1, 0xffffffff), uimm5:$shamt)),
          (SRLIW GPR:$rs1, uimm5:$shamt)>;
def : Pat<(i64 (srl (shl GPR:$rs1, (i64 32)), uimm6gt32:$shamt)),
          (SRLIW GPR:$rs1, (ImmSub32 uimm6gt32:$shamt))>;
def : Pat<(sra (sext_inreg GPR:$rs1, i32), uimm5:$shamt),
          (SRAIW GPR:$rs1, uimm5:$shamt)>;
def : Pat<(i64 (sra (shl GPR:$rs1, (i64 32)), uimm6gt32:$shamt)),
          (SRAIW GPR:$rs1, (ImmSub32 uimm6gt32:$shamt))>;

def : PatGprGpr<shiftopw<riscv_sllw>, SLLW>;
def : PatGprGpr<shiftopw<riscv_srlw>, SRLW>;
def : PatGprGpr<shiftopw<riscv_sraw>, SRAW>;

// Select W instructions if only the lower 32 bits of the result are used.
def : PatGprGpr<binop_allwusers<add>, ADDW>;
def : PatGprSimm12<binop_allwusers<add>, ADDIW>;
def : PatGprImm<binop_allwusers<add>, ADDIW, u32simm12>;
def : PatGprGpr<binop_allwusers<sub>, SUBW>;
def : PatGprImm<binop_allwusers<shl>, SLLIW, uimm5>;

// If this is a shr of a value sign extended from i32, and all the users only
// use the lower 32 bits, we can use an sraiw to remove the sext_inreg. This
// occurs because SimplifyDemandedBits prefers srl over sra.
def : Pat<(binop_allwusers<srl> (sext_inreg GPR:$rs1, i32), uimm5:$shamt),
          (SRAIW GPR:$rs1, uimm5:$shamt)>;

// Use binop_allwusers to recover immediates that may have been broken by
// SimplifyDemandedBits.
def : Pat<(binop_allwusers<and> GPR:$rs1, 0xffffffff),
          (COPY GPR:$rs1)>;
def : PatGprImm<binop_allwusers<and>, ANDI, u32simm12>;
def : PatGprImm<binop_allwusers<or>,  ORI,  u32simm12>;
def : PatGprImm<binop_allwusers<xor>, XORI, u32simm12>;

// Select 'or' as ADDIW if the immediate bits are known to be 0 in $rs1 and
// $rs1 is sign extended. This can improve compressibility. Using ADDIW gives
// more power to RISCVOptWInstrs.
def : Pat<(riscv_or_disjoint 33signbits_node:$rs1, simm12_lo:$imm),
          (ADDIW $rs1, simm12_lo:$imm)>;

/// Loads

def : LdPat<sextloadi32, LW, i64>;
def : LdPat<extloadi32, LW, i64>;
def : LdPat<zextloadi32, LWU, i64>;
def : LdPat<load, LD, i64>;

/// Stores

def : StPat<truncstorei32, SW, GPR, i64>;
def : StPat<store, SD, GPR, i64>;
} // Predicates = [IsRV64]

// On RV64, we can directly read these 64-bit counter CSRs.
let Predicates = [IsRV64] in {
/// readcyclecounter
def : Pat<(i64 (readcyclecounter)), (CSRRS CYCLE.Encoding, (XLenVT X0))>;
/// readsteadycounter
def : Pat<(i64 (readsteadycounter)), (CSRRS TIME.Encoding, (XLenVT X0))>;
}

// On RV32, ReadCounterWide will be expanded to the suggested loop reading both
// halves of 64-bit counter CSRs.
let Predicates = [IsRV32], usesCustomInserter = 1, hasNoSchedulingInfo = 1 in
def ReadCounterWide : Pseudo<(outs GPR:$lo, GPR:$hi), (ins i32imm:$csr_lo, i32imm:$csr_hi),
                             [(set GPR:$lo, GPR:$hi,
                              (riscv_read_counter_wide csr_sysreg:$csr_lo, csr_sysreg:$csr_hi))],
                             "", "">;

/// traps

// We lower `trap` to `unimp`, as this causes a hard exception on nearly all
// systems.
def : Pat<(trap), (UNIMP)>;

// We lower `debugtrap` to `ebreak`, as this will get the attention of the
// debugger if possible.
def : Pat<(debugtrap), (EBREAK)>;

let Predicates = [IsRV64], Uses = [X5],
    Defs = [X1, X6, X7, X28, X29, X30, X31], Size = 8 in
def HWASAN_CHECK_MEMACCESS_SHORTGRANULES
  : Pseudo<(outs), (ins GPRJALR:$ptr, i32imm:$accessinfo),
           [(int_hwasan_check_memaccess_shortgranules (i64 X5), GPRJALR:$ptr,
                                                      (i32 timm:$accessinfo))]>;

// This gets lowered into a 20-byte instruction sequence (at most)
let hasSideEffects = 0, mayLoad = 1, mayStore = 0,
    Defs = [ X6, X7, X28, X29, X30, X31 ], Size = 20 in {
def KCFI_CHECK
  : Pseudo<(outs), (ins GPRJALR:$ptr, i32imm:$type), []>, Sched<[]>;
}

/// Simple optimization
def : Pat<(XLenVT (add GPR:$rs1, immop_oneuse<AddiPair>:$rs2)),
          (ADDI (XLenVT (ADDI GPR:$rs1, (AddiPairImmLarge imm:$rs2))),
                (AddiPairImmSmall imm:$rs2))>;

def negImm : ComplexPattern<i64, 1, "selectNegImm", [], [], 0>;

let Predicates = [IsRV64] in {
// Select W instructions if only the lower 32-bits of the result are used.
def : Pat<(binop_allwusers<add> GPR:$rs1, immop_oneuse<AddiPair>:$rs2),
          (ADDIW (i64 (ADDIW GPR:$rs1, (AddiPairImmLarge imm:$rs2))),
                 (AddiPairImmSmall imm:$rs2))>;

// Select SUB if the negated constant is cheaper to materialize.
def : Pat<(i64 (add GPR:$rs1, negImm:$rs2)), (SUB GPR:$rs1, negImm:$rs2)>;
}

//===----------------------------------------------------------------------===//
// Zihintpause
//===----------------------------------------------------------------------===//

// Zihintpause
let Predicates = [HasStdExtZihintpause] in
def : Pat<(int_riscv_pause), (FENCE 0x1, 0x0)>;

//===----------------------------------------------------------------------===//
// Standard extensions
//===----------------------------------------------------------------------===//

// Multiply and Division
include "RISCVInstrInfoM.td"

// Atomic
include "RISCVInstrInfoA.td"
include "RISCVInstrInfoZa.td"
include "RISCVInstrInfoZalasr.td"

// Integer
include "RISCVInstrInfoZimop.td"
include "RISCVInstrInfoZicbo.td"
include "RISCVInstrInfoZicond.td"
include "RISCVInstrInfoZilsd.td"
include "RISCVInstrInfoZibi.td"

// Scalar FP
include "RISCVInstrInfoF.td"
include "RISCVInstrInfoD.td"
include "RISCVInstrInfoQ.td"
include "RISCVInstrInfoZfh.td"
include "RISCVInstrInfoZfbfmin.td"
include "RISCVInstrInfoZfa.td"

// Scalar bitmanip and cryptography
include "RISCVInstrInfoZb.td"
include "RISCVInstrInfoZk.td"

// Vector
include "RISCVInstrInfoV.td"
include "RISCVInstrInfoZvk.td"
include "RISCVInstrInfoZvqdotq.td"
include "RISCVInstrInfoZvfofp8min.td"

// Packed SIMD
include "RISCVInstrInfoP.td"

// Compressed
include "RISCVInstrInfoC.td"
include "RISCVInstrInfoZc.td"
include "RISCVInstrInfoZcmop.td"
include "RISCVInstrInfoZclsd.td"

// Control Flow Integriy, this requires Zimop/Zcmop
include "RISCVInstrInfoZicfiss.td"

// Short Forward Branch
include "RISCVInstrInfoSFB.td"

//===----------------------------------------------------------------------===//
// Vendor extensions
//===----------------------------------------------------------------------===//

include "RISCVInstrInfoXVentana.td"
include "RISCVInstrInfoXTHead.td"
include "RISCVInstrInfoXSf.td"
include "RISCVInstrInfoXSfmm.td"
include "RISCVInstrInfoXCV.td"
include "RISCVInstrInfoXwch.td"
include "RISCVInstrInfoXqci.td"
include "RISCVInstrInfoXqccmp.td"
include "RISCVInstrInfoXMips.td"
include "RISCVInstrInfoXRivos.td"
include "RISCVInstrInfoXAndes.td"
include "RISCVInstrInfoXSpacemiT.td"

//===----------------------------------------------------------------------===//
// Global ISel
//===----------------------------------------------------------------------===//

include "RISCVInstrGISel.td"
