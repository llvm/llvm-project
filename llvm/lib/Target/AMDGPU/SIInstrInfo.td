//===-- SIInstrInfo.td - SI Instruction Infos -------------*- tablegen -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
def isCI : Predicate<"Subtarget->getGeneration() "
                      ">= SISubtarget::SEA_ISLANDS">;
def isCIOnly : Predicate<"Subtarget->getGeneration() =="
                         "SISubtarget::SEA_ISLANDS">,
  AssemblerPredicate <"FeatureSeaIslands">;

def DisableInst : Predicate <"false">, AssemblerPredicate<"FeatureDisable">;

// Execpt for the NONE field, this must be kept in sync with the
// SIEncodingFamily enum in AMDGPUInstrInfo.cpp
def SIEncodingFamily {
  int NONE = -1;
  int SI = 0;
  int VI = 1;
}

//===----------------------------------------------------------------------===//
// SI DAG Nodes
//===----------------------------------------------------------------------===//

def SIload_constant : SDNode<"AMDGPUISD::LOAD_CONSTANT",
  SDTypeProfile<1, 2, [SDTCisVT<0, f32>, SDTCisVT<1, v4i32>, SDTCisVT<2, i32>]>,
                      [SDNPMayLoad, SDNPMemOperand]
>;

def SIatomic_inc : SDNode<"AMDGPUISD::ATOMIC_INC", SDTAtomic2,
  [SDNPMayLoad, SDNPMayStore, SDNPMemOperand, SDNPHasChain]
>;

def SIatomic_dec : SDNode<"AMDGPUISD::ATOMIC_DEC", SDTAtomic2,
  [SDNPMayLoad, SDNPMayStore, SDNPMemOperand, SDNPHasChain]
>;

def SItbuffer_store : SDNode<"AMDGPUISD::TBUFFER_STORE_FORMAT",
  SDTypeProfile<0, 13,
    [SDTCisVT<0, v4i32>,   // rsrc(SGPR)
     SDTCisVT<1, iAny>,   // vdata(VGPR)
     SDTCisVT<2, i32>,    // num_channels(imm)
     SDTCisVT<3, i32>,    // vaddr(VGPR)
     SDTCisVT<4, i32>,    // soffset(SGPR)
     SDTCisVT<5, i32>,    // inst_offset(imm)
     SDTCisVT<6, i32>,    // dfmt(imm)
     SDTCisVT<7, i32>,    // nfmt(imm)
     SDTCisVT<8, i32>,    // offen(imm)
     SDTCisVT<9, i32>,    // idxen(imm)
     SDTCisVT<10, i32>,   // glc(imm)
     SDTCisVT<11, i32>,   // slc(imm)
     SDTCisVT<12, i32>    // tfe(imm)
    ]>,
  [SDNPMayStore, SDNPMemOperand, SDNPHasChain]
>;

def SDTBufferLoad : SDTypeProfile<1, 5,
    [                    // vdata
     SDTCisVT<1, v4i32>, // rsrc
     SDTCisVT<2, i32>,   // vindex
     SDTCisVT<3, i32>,   // offset
     SDTCisVT<4, i1>,    // glc
     SDTCisVT<5, i1>]>;  // slc

def SIbuffer_load : SDNode <"AMDGPUISD::BUFFER_LOAD", SDTBufferLoad,
                            [SDNPMemOperand, SDNPHasChain, SDNPMayLoad]>;
def SIbuffer_load_format : SDNode <"AMDGPUISD::BUFFER_LOAD_FORMAT", SDTBufferLoad,
                            [SDNPMemOperand, SDNPHasChain, SDNPMayLoad]>;

class SDSample<string opcode> : SDNode <opcode,
  SDTypeProfile<1, 4, [SDTCisVT<0, v4f32>, SDTCisVT<2, v8i32>,
                       SDTCisVT<3, v4i32>, SDTCisVT<4, i32>]>
>;

def SIsample : SDSample<"AMDGPUISD::SAMPLE">;
def SIsampleb : SDSample<"AMDGPUISD::SAMPLEB">;
def SIsampled : SDSample<"AMDGPUISD::SAMPLED">;
def SIsamplel : SDSample<"AMDGPUISD::SAMPLEL">;

def SIpc_add_rel_offset : SDNode<"AMDGPUISD::PC_ADD_REL_OFFSET",
  SDTypeProfile<1, 2, [SDTCisVT<0, iPTR>, SDTCisSameAs<0,1>, SDTCisSameAs<0,2>]>
>;

//===----------------------------------------------------------------------===//
// PatFrags for global memory operations
//===----------------------------------------------------------------------===//

defm atomic_inc_global : global_binary_atomic_op<SIatomic_inc>;
defm atomic_dec_global : global_binary_atomic_op<SIatomic_dec>;

//===----------------------------------------------------------------------===//
// SDNodes and PatFrag for local loads and stores to enable s_mov_b32 m0, -1
// to be glued to the memory instructions.
//===----------------------------------------------------------------------===//

def SIld_local : SDNode <"ISD::LOAD", SDTLoad,
  [SDNPHasChain, SDNPMayLoad, SDNPMemOperand, SDNPInGlue]
>;

def si_ld_local : PatFrag <(ops node:$ptr), (SIld_local node:$ptr), [{
  return cast<LoadSDNode>(N)->getAddressSpace() == AMDGPUASI.LOCAL_ADDRESS;
}]>;

def si_load_local : PatFrag <(ops node:$ptr), (si_ld_local node:$ptr), [{
  return cast<LoadSDNode>(N)->getAddressingMode() == ISD::UNINDEXED &&
         cast<LoadSDNode>(N)->getExtensionType() == ISD::NON_EXTLOAD;
}]>;

def si_load_local_align8 : Aligned8Bytes <
  (ops node:$ptr), (si_load_local node:$ptr)
>;

def si_sextload_local : PatFrag <(ops node:$ptr), (si_ld_local node:$ptr), [{
  return cast<LoadSDNode>(N)->getExtensionType() == ISD::SEXTLOAD;
}]>;
def si_az_extload_local : AZExtLoadBase <si_ld_local>;

multiclass SIExtLoadLocal <PatFrag ld_node> {

  def _i8 : PatFrag <(ops node:$ptr), (ld_node node:$ptr),
                     [{return cast<LoadSDNode>(N)->getMemoryVT() == MVT::i8;}]
  >;

  def _i16 : PatFrag <(ops node:$ptr), (ld_node node:$ptr),
                     [{return cast<LoadSDNode>(N)->getMemoryVT() == MVT::i16;}]
  >;
}

defm si_sextload_local : SIExtLoadLocal <si_sextload_local>;
defm si_az_extload_local : SIExtLoadLocal <si_az_extload_local>;

def SIst_local : SDNode <"ISD::STORE", SDTStore,
  [SDNPHasChain, SDNPMayStore, SDNPMemOperand, SDNPInGlue]
>;

def si_st_local : PatFrag <
  (ops node:$val, node:$ptr), (SIst_local node:$val, node:$ptr), [{
  return cast<StoreSDNode>(N)->getAddressSpace() == AMDGPUASI.LOCAL_ADDRESS;
}]>;

def si_store_local : PatFrag <
  (ops node:$val, node:$ptr), (si_st_local node:$val, node:$ptr), [{
  return cast<StoreSDNode>(N)->getAddressingMode() == ISD::UNINDEXED &&
         !cast<StoreSDNode>(N)->isTruncatingStore();
}]>;

def si_store_local_align8 : Aligned8Bytes <
  (ops node:$val, node:$ptr), (si_store_local node:$val, node:$ptr)
>;

def si_truncstore_local : PatFrag <
  (ops node:$val, node:$ptr), (si_st_local node:$val, node:$ptr), [{
  return cast<StoreSDNode>(N)->isTruncatingStore();
}]>;

def si_truncstore_local_i8 : PatFrag <
  (ops node:$val, node:$ptr), (si_truncstore_local node:$val, node:$ptr), [{
  return cast<StoreSDNode>(N)->getMemoryVT() == MVT::i8;
}]>;

def si_truncstore_local_i16 : PatFrag <
  (ops node:$val, node:$ptr), (si_truncstore_local node:$val, node:$ptr), [{
  return cast<StoreSDNode>(N)->getMemoryVT() == MVT::i16;
}]>;

def si_setcc_uniform : PatFrag <
  (ops node:$lhs, node:$rhs, node:$cond),
  (setcc node:$lhs, node:$rhs, node:$cond), [{
  for (SDNode *Use : N->uses()) {
    if (Use->isMachineOpcode() || Use->getOpcode() != ISD::CopyToReg)
      return false;

    unsigned Reg = cast<RegisterSDNode>(Use->getOperand(1))->getReg();
    if (Reg != AMDGPU::SCC)
      return false;
  }
  return true;
}]>;

def si_uniform_br : PatFrag <
  (ops node:$cond, node:$bb), (brcond node:$cond, node:$bb), [{
  return isUniformBr(N);
}]>;

def si_uniform_br_scc : PatFrag <
  (ops node:$cond, node:$bb), (si_uniform_br node:$cond, node:$bb), [{
  return isCBranchSCC(N);
}]>;

def lshr_rev : PatFrag <
  (ops node:$src1, node:$src0),
  (srl $src0, $src1)
>;

def ashr_rev : PatFrag <
  (ops node:$src1, node:$src0),
  (sra $src0, $src1)
>;

def lshl_rev : PatFrag <
  (ops node:$src1, node:$src0),
  (shl $src0, $src1)
>;

multiclass SIAtomicM0Glue2 <string op_name, bit is_amdgpu = 0> {

  def _glue : SDNode <
    !if(is_amdgpu, "AMDGPUISD", "ISD")#"::ATOMIC_"#op_name, SDTAtomic2,
    [SDNPHasChain, SDNPMayStore, SDNPMayLoad, SDNPMemOperand, SDNPInGlue]
  >;

  def _local : local_binary_atomic_op <!cast<SDNode>(NAME#"_glue")>;
}

defm si_atomic_load_add : SIAtomicM0Glue2 <"LOAD_ADD">;
defm si_atomic_load_sub : SIAtomicM0Glue2 <"LOAD_SUB">;
defm si_atomic_inc : SIAtomicM0Glue2 <"INC", 1>;
defm si_atomic_dec : SIAtomicM0Glue2 <"DEC", 1>;
defm si_atomic_load_and : SIAtomicM0Glue2 <"LOAD_AND">;
defm si_atomic_load_min : SIAtomicM0Glue2 <"LOAD_MIN">;
defm si_atomic_load_max : SIAtomicM0Glue2 <"LOAD_MAX">;
defm si_atomic_load_or : SIAtomicM0Glue2 <"LOAD_OR">;
defm si_atomic_load_xor : SIAtomicM0Glue2 <"LOAD_XOR">;
defm si_atomic_load_umin : SIAtomicM0Glue2 <"LOAD_UMIN">;
defm si_atomic_load_umax : SIAtomicM0Glue2 <"LOAD_UMAX">;
defm si_atomic_swap : SIAtomicM0Glue2 <"SWAP">;

def si_atomic_cmp_swap_glue : SDNode <"ISD::ATOMIC_CMP_SWAP", SDTAtomic3,
  [SDNPHasChain, SDNPMayStore, SDNPMayLoad, SDNPMemOperand, SDNPInGlue]
>;

defm si_atomic_cmp_swap : AtomicCmpSwapLocal <si_atomic_cmp_swap_glue>;

def as_i1imm : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(N->getZExtValue(), SDLoc(N), MVT::i1);
}]>;

def as_i8imm : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(N->getZExtValue(), SDLoc(N), MVT::i8);
}]>;

def as_i16imm : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(N->getSExtValue(), SDLoc(N), MVT::i16);
}]>;

def as_i32imm: SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(N->getSExtValue(), SDLoc(N), MVT::i32);
}]>;

def as_i64imm: SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(N->getSExtValue(), SDLoc(N), MVT::i64);
}]>;

// Copied from the AArch64 backend:
def bitcast_fpimm_to_i32 : SDNodeXForm<fpimm, [{
return CurDAG->getTargetConstant(
  N->getValueAPF().bitcastToAPInt().getZExtValue(), SDLoc(N), MVT::i32);
}]>;

def frameindex_to_targetframeindex : SDNodeXForm<frameindex, [{
  auto FI = cast<FrameIndexSDNode>(N);
  return CurDAG->getTargetFrameIndex(FI->getIndex(), MVT::i32);
}]>;

// Copied from the AArch64 backend:
def bitcast_fpimm_to_i64 : SDNodeXForm<fpimm, [{
return CurDAG->getTargetConstant(
  N->getValueAPF().bitcastToAPInt().getZExtValue(), SDLoc(N), MVT::i64);
}]>;

def SIMM16bit : PatLeaf <(imm),
  [{return isInt<16>(N->getSExtValue());}]
>;

class InlineImm <ValueType vt> : PatLeaf <(vt imm), [{
  return isInlineImmediate(N);
}]>;

class InlineFPImm <ValueType vt> : PatLeaf <(vt fpimm), [{
  return isInlineImmediate(N);
}]>;

class VGPRImm <dag frag> : PatLeaf<frag, [{
  if (Subtarget->getGeneration() < SISubtarget::SOUTHERN_ISLANDS) {
    return false;
  }
  const SIRegisterInfo *SIRI =
      static_cast<const SIRegisterInfo *>(Subtarget->getRegisterInfo());
  unsigned Limit = 0;
  for (SDNode::use_iterator U = N->use_begin(), E = SDNode::use_end();
         Limit < 10 && U != E; ++U, ++Limit) {
    const TargetRegisterClass *RC = getOperandRegClass(*U, U.getOperandNo());

    // If the register class is unknown, it could be an unknown
    // register class that needs to be an SGPR, e.g. an inline asm
    // constraint
    if (!RC || SIRI->isSGPRClass(RC))
      return false;
  }

  return Limit < 10;
}]>;

def NegateImm : SDNodeXForm<imm, [{
  return CurDAG->getConstant(-N->getSExtValue(), SDLoc(N), MVT::i32);
}]>;

// TODO: When FP inline imm values work?
def NegSubInlineConst32 : ImmLeaf<i32, [{
  return Imm < -16 && Imm >= -64;
}], NegateImm>;

def NegSubInlineConst16 : ImmLeaf<i16, [{
  return Imm < -16 && Imm >= -64;
}], NegateImm>;

//===----------------------------------------------------------------------===//
// Custom Operands
//===----------------------------------------------------------------------===//

def SoppBrTarget : AsmOperandClass {
  let Name = "SoppBrTarget";
  let ParserMethod = "parseSOppBrTarget";
}

def sopp_brtarget : Operand<OtherVT> {
  let EncoderMethod = "getSOPPBrEncoding";
  let DecoderMethod = "decodeSoppBrTarget";
  let OperandType = "OPERAND_PCREL";
  let ParserMatchClass = SoppBrTarget;
}

def si_ga : Operand<iPTR>;

def InterpSlotMatchClass : AsmOperandClass {
  let Name = "InterpSlot";
  let PredicateMethod = "isInterpSlot";
  let ParserMethod = "parseInterpSlot";
  let RenderMethod = "addImmOperands";
}

def InterpSlot : Operand<i32> {
  let PrintMethod = "printInterpSlot";
  let ParserMatchClass = InterpSlotMatchClass;
  let OperandType = "OPERAND_IMMEDIATE";
}

def AttrMatchClass : AsmOperandClass {
  let Name = "Attr";
  let PredicateMethod = "isInterpAttr";
  let ParserMethod = "parseInterpAttr";
  let RenderMethod = "addImmOperands";
}

// It appears to be necessary to create a separate operand for this to
// be able to parse attr<num> with no space.
def Attr : Operand<i32> {
  let PrintMethod = "printInterpAttr";
  let ParserMatchClass = AttrMatchClass;
  let OperandType = "OPERAND_IMMEDIATE";
}

def AttrChanMatchClass : AsmOperandClass {
  let Name = "AttrChan";
  let PredicateMethod = "isAttrChan";
  let RenderMethod = "addImmOperands";
}

def AttrChan : Operand<i32> {
  let PrintMethod = "printInterpAttrChan";
  let ParserMatchClass = AttrChanMatchClass;
  let OperandType = "OPERAND_IMMEDIATE";
}

def SendMsgMatchClass : AsmOperandClass {
  let Name = "SendMsg";
  let PredicateMethod = "isSendMsg";
  let ParserMethod = "parseSendMsgOp";
  let RenderMethod = "addImmOperands";
}

def SwizzleMatchClass : AsmOperandClass {
  let Name = "Swizzle";
  let PredicateMethod = "isSwizzle";
  let ParserMethod = "parseSwizzleOp";
  let RenderMethod = "addImmOperands";
  let IsOptional = 1;
}

def ExpTgtMatchClass : AsmOperandClass {
  let Name = "ExpTgt";
  let PredicateMethod = "isExpTgt";
  let ParserMethod = "parseExpTgt";
  let RenderMethod = "printExpTgt";
}

def SendMsgImm : Operand<i32> {
  let PrintMethod = "printSendMsg";
  let ParserMatchClass = SendMsgMatchClass;
}

def SwizzleImm : Operand<i16> {
  let PrintMethod = "printSwizzle";
  let ParserMatchClass = SwizzleMatchClass;
}

def SWaitMatchClass : AsmOperandClass {
  let Name = "SWaitCnt";
  let RenderMethod = "addImmOperands";
  let ParserMethod = "parseSWaitCntOps";
}

def VReg32OrOffClass : AsmOperandClass {
  let Name = "VReg32OrOff";
  let ParserMethod = "parseVReg32OrOff";
}

def WAIT_FLAG : Operand <i32> {
  let ParserMatchClass = SWaitMatchClass;
  let PrintMethod = "printWaitFlag";
}

include "SIInstrFormats.td"
include "VIInstrFormats.td"

// ===----------------------------------------------------------------------===//
// ExpSrc* Special cases for exp src operands which are printed as
// "off" depending on en operand.
// ===----------------------------------------------------------------------===//

def ExpSrc0 : RegisterOperand<VGPR_32> {
  let PrintMethod = "printExpSrc0";
  let ParserMatchClass = VReg32OrOffClass;
}

def ExpSrc1 : RegisterOperand<VGPR_32> {
  let PrintMethod = "printExpSrc1";
  let ParserMatchClass = VReg32OrOffClass;
}

def ExpSrc2 : RegisterOperand<VGPR_32> {
  let PrintMethod = "printExpSrc2";
  let ParserMatchClass = VReg32OrOffClass;
}

def ExpSrc3 : RegisterOperand<VGPR_32> {
  let PrintMethod = "printExpSrc3";
  let ParserMatchClass = VReg32OrOffClass;
}

class SDWA9Src : RegisterOperand<VS_32> {
  let OperandNamespace = "AMDGPU";
  let OperandType = "OPERAND_SDWA9_SRC";
  let EncoderMethod = "getSDWA9SrcEncoding";
}

def SDWA9Src32 : SDWA9Src {
  let DecoderMethod = "decodeSDWA9Src32";
}

def SDWA9Src16 : SDWA9Src {
  let DecoderMethod = "decodeSDWA9Src16";
}

def SDWA9VopcDst : VOPDstOperand<SReg_64> {
  let OperandNamespace = "AMDGPU";
  let OperandType = "OPERAND_SDWA9_VOPC_DST";
  let EncoderMethod = "getSDWA9VopcDstEncoding";
  let DecoderMethod = "decodeSDWA9VopcDst";
}

class NamedMatchClass<string CName, bit Optional = 1> : AsmOperandClass {
  let Name = "Imm"#CName;
  let PredicateMethod = "is"#CName;
  let ParserMethod = !if(Optional, "parseOptionalOperand", "parse"#CName);
  let RenderMethod = "addImmOperands";
  let IsOptional = Optional;
  let DefaultMethod = !if(Optional, "default"#CName, ?);
}

class NamedOperandBit<string Name, AsmOperandClass MatchClass> : Operand<i1> {
  let PrintMethod = "print"#Name;
  let ParserMatchClass = MatchClass;
}

class NamedOperandU8<string Name, AsmOperandClass MatchClass> : Operand<i8> {
  let PrintMethod = "print"#Name;
  let ParserMatchClass = MatchClass;
}

class NamedOperandU12<string Name, AsmOperandClass MatchClass> : Operand<i16> {
  let PrintMethod = "print"#Name;
  let ParserMatchClass = MatchClass;
}

class NamedOperandU16<string Name, AsmOperandClass MatchClass> : Operand<i16> {
  let PrintMethod = "print"#Name;
  let ParserMatchClass = MatchClass;
}

class NamedOperandS13<string Name, AsmOperandClass MatchClass> : Operand<i16> {
  let PrintMethod = "print"#Name;
  let ParserMatchClass = MatchClass;
}

class NamedOperandU32<string Name, AsmOperandClass MatchClass> : Operand<i32> {
  let PrintMethod = "print"#Name;
  let ParserMatchClass = MatchClass;
}

class NamedOperandU32Default0<string Name, AsmOperandClass MatchClass> :
  OperandWithDefaultOps<i32, (ops (i32 0))> {
  let PrintMethod = "print"#Name;
  let ParserMatchClass = MatchClass;
}

let OperandType = "OPERAND_IMMEDIATE" in {

def offen : NamedOperandBit<"Offen", NamedMatchClass<"Offen">>;
def idxen : NamedOperandBit<"Idxen", NamedMatchClass<"Idxen">>;
def addr64 : NamedOperandBit<"Addr64", NamedMatchClass<"Addr64">>;

def offset_u12 : NamedOperandU12<"Offset", NamedMatchClass<"OffsetU12">>;
def offset_s13 : NamedOperandS13<"OffsetS13", NamedMatchClass<"OffsetS13">>;
def offset : NamedOperandU16<"Offset", NamedMatchClass<"Offset">>;
def offset0 : NamedOperandU8<"Offset0", NamedMatchClass<"Offset0">>;
def offset1 : NamedOperandU8<"Offset1", NamedMatchClass<"Offset1">>;

def gds : NamedOperandBit<"GDS", NamedMatchClass<"GDS">>;

def omod : NamedOperandU32<"OModSI", NamedMatchClass<"OModSI">>;
def clampmod : NamedOperandBit<"ClampSI", NamedMatchClass<"ClampSI">>;

def GLC : NamedOperandBit<"GLC", NamedMatchClass<"GLC">>;
def slc : NamedOperandBit<"SLC", NamedMatchClass<"SLC">>;
def tfe : NamedOperandBit<"TFE", NamedMatchClass<"TFE">>;
def unorm : NamedOperandBit<"UNorm", NamedMatchClass<"UNorm">>;
def da : NamedOperandBit<"DA", NamedMatchClass<"DA">>;
def r128 : NamedOperandBit<"R128", NamedMatchClass<"R128">>;
def lwe : NamedOperandBit<"LWE", NamedMatchClass<"LWE">>;
def exp_compr : NamedOperandBit<"ExpCompr", NamedMatchClass<"ExpCompr">>;
def exp_vm : NamedOperandBit<"ExpVM", NamedMatchClass<"ExpVM">>;

def dmask : NamedOperandU16<"DMask", NamedMatchClass<"DMask">>;

def dpp_ctrl : NamedOperandU32<"DPPCtrl", NamedMatchClass<"DPPCtrl", 0>>;
def row_mask : NamedOperandU32<"RowMask", NamedMatchClass<"RowMask">>;
def bank_mask : NamedOperandU32<"BankMask", NamedMatchClass<"BankMask">>;
def bound_ctrl : NamedOperandBit<"BoundCtrl", NamedMatchClass<"BoundCtrl">>;

def dst_sel : NamedOperandU32<"SDWADstSel", NamedMatchClass<"SDWADstSel">>;
def src0_sel : NamedOperandU32<"SDWASrc0Sel", NamedMatchClass<"SDWASrc0Sel">>;
def src1_sel : NamedOperandU32<"SDWASrc1Sel", NamedMatchClass<"SDWASrc1Sel">>;
def dst_unused : NamedOperandU32<"SDWADstUnused", NamedMatchClass<"SDWADstUnused">>;

def op_sel : NamedOperandU32Default0<"OpSel", NamedMatchClass<"OpSel">>;
def op_sel_hi : NamedOperandU32Default0<"OpSelHi", NamedMatchClass<"OpSelHi">>;
def neg_lo : NamedOperandU32Default0<"NegLo", NamedMatchClass<"NegLo">>;
def neg_hi : NamedOperandU32Default0<"NegHi", NamedMatchClass<"NegHi">>;

def hwreg : NamedOperandU16<"Hwreg", NamedMatchClass<"Hwreg", 0>>;

def exp_tgt : NamedOperandU8<"ExpTgt", NamedMatchClass<"ExpTgt", 0>> {

}

} // End OperandType = "OPERAND_IMMEDIATE"

class KImmMatchClass<int size> : AsmOperandClass {
  let Name = "KImmFP"#size;
  let PredicateMethod = "isKImmFP"#size;
  let ParserMethod = "parseImm";
  let RenderMethod = "addKImmFP"#size#"Operands";
}

class kimmOperand<ValueType vt> : Operand<vt> {
  let OperandNamespace = "AMDGPU";
  let OperandType = "OPERAND_KIMM"#vt.Size;
  let PrintMethod = "printU"#vt.Size#"ImmOperand";
  let ParserMatchClass = !cast<AsmOperandClass>("KImmFP"#vt.Size#"MatchClass");
}

// 32-bit VALU immediate operand that uses the constant bus.
def KImmFP32MatchClass : KImmMatchClass<32>;
def f32kimm : kimmOperand<i32>;

// 32-bit VALU immediate operand with a 16-bit value that uses the
// constant bus.
def KImmFP16MatchClass : KImmMatchClass<16>;
def f16kimm : kimmOperand<i16>;


def VOPDstS64 : VOPDstOperand <SReg_64>;

class FPInputModsMatchClass <int opSize> : AsmOperandClass {
  let Name = "RegOrImmWithFP"#opSize#"InputMods";
  let ParserMethod = "parseRegOrImmWithFPInputMods";
  let PredicateMethod = "isRegOrImmWithFP"#opSize#"InputMods";
}

def FP16InputModsMatchClass : FPInputModsMatchClass<16>;
def FP32InputModsMatchClass : FPInputModsMatchClass<32>;
def FP64InputModsMatchClass : FPInputModsMatchClass<64>;

class InputMods <AsmOperandClass matchClass> : Operand <i32> {
  let OperandNamespace = "AMDGPU";
  let OperandType = "OPERAND_INPUT_MODS";
  let ParserMatchClass = matchClass;
}

class FPInputMods <FPInputModsMatchClass matchClass> : InputMods <matchClass> {
  let PrintMethod = "printOperandAndFPInputMods";
}

def FP16InputMods : FPInputMods<FP16InputModsMatchClass>;
def FP32InputMods : FPInputMods<FP32InputModsMatchClass>;
def FP64InputMods : FPInputMods<FP64InputModsMatchClass>;

class IntInputModsMatchClass <int opSize> : AsmOperandClass {
  let Name = "RegOrImmWithInt"#opSize#"InputMods";
  let ParserMethod = "parseRegOrImmWithIntInputMods";
  let PredicateMethod = "isRegOrImmWithInt"#opSize#"InputMods";
}
def Int32InputModsMatchClass : IntInputModsMatchClass<32>;
def Int64InputModsMatchClass : IntInputModsMatchClass<64>;

class IntInputMods <IntInputModsMatchClass matchClass> : InputMods <matchClass> {
  let PrintMethod = "printOperandAndIntInputMods";
}
def Int32InputMods : IntInputMods<Int32InputModsMatchClass>;
def Int64InputMods : IntInputMods<Int64InputModsMatchClass>;

def FPRegInputModsMatchClass : AsmOperandClass {
  let Name = "RegWithFPInputMods";
  let ParserMethod = "parseRegWithFPInputMods";
  let PredicateMethod = "isRegKind";
}

def FPRegInputMods : InputMods <FPRegInputModsMatchClass> {
  let PrintMethod = "printOperandAndFPInputMods";
}

def FPVRegInputModsMatchClass : AsmOperandClass {
  let Name = "VRegWithFPInputMods";
  let ParserMethod = "parseRegWithFPInputMods";
  let PredicateMethod = "isVReg";
}

def FPVRegInputMods : InputMods <FPVRegInputModsMatchClass> {
  let PrintMethod = "printOperandAndFPInputMods";
}


def IntRegInputModsMatchClass : AsmOperandClass {
  let Name = "RegWithIntInputMods";
  let ParserMethod = "parseRegWithIntInputMods";
  let PredicateMethod = "isRegKind";
}

def IntRegInputMods : InputMods <IntRegInputModsMatchClass> {
  let PrintMethod = "printOperandAndIntInputMods";
}

def IntVRegInputModsMatchClass : AsmOperandClass {
  let Name = "VRegWithIntInputMods";
  let ParserMethod = "parseRegWithIntInputMods";
  let PredicateMethod = "isVReg";
}

def IntVRegInputMods : InputMods <IntVRegInputModsMatchClass> {
  let PrintMethod = "printOperandAndIntInputMods";
}

class PackedFPInputModsMatchClass <int opSize> : AsmOperandClass {
  let Name = "PackedFP"#opSize#"InputMods";
  let ParserMethod = "parseRegOrImm";
  let PredicateMethod = "isRegOrImm";
//  let PredicateMethod = "isPackedFP"#opSize#"InputMods";
}

class PackedIntInputModsMatchClass <int opSize> : AsmOperandClass {
  let Name = "PackedInt"#opSize#"InputMods";
  let ParserMethod = "parseRegOrImm";
  let PredicateMethod = "isRegOrImm";
//  let PredicateMethod = "isPackedInt"#opSize#"InputMods";
}

def PackedF16InputModsMatchClass : PackedFPInputModsMatchClass<16>;
def PackedI16InputModsMatchClass : PackedIntInputModsMatchClass<16>;

class PackedFPInputMods <PackedFPInputModsMatchClass matchClass> : InputMods <matchClass> {
//  let PrintMethod = "printPackedFPInputMods";
}

class PackedIntInputMods <PackedIntInputModsMatchClass matchClass> : InputMods <matchClass> {
  //let PrintMethod = "printPackedIntInputMods";
}

def PackedF16InputMods : PackedFPInputMods<PackedF16InputModsMatchClass>;
def PackedI16InputMods : PackedIntInputMods<PackedI16InputModsMatchClass>;

//===----------------------------------------------------------------------===//
// Complex patterns
//===----------------------------------------------------------------------===//

def DS1Addr1Offset : ComplexPattern<i32, 2, "SelectDS1Addr1Offset">;
def DS64Bit4ByteAligned : ComplexPattern<i32, 3, "SelectDS64Bit4ByteAligned">;

def MOVRELOffset : ComplexPattern<i32, 2, "SelectMOVRELOffset">;

def VOP3Mods0 : ComplexPattern<untyped, 4, "SelectVOP3Mods0">;
def VOP3Mods0Clamp : ComplexPattern<untyped, 3, "SelectVOP3Mods0Clamp">;
def VOP3Mods0Clamp0OMod : ComplexPattern<untyped, 4, "SelectVOP3Mods0Clamp0OMod">;
def VOP3Mods  : ComplexPattern<untyped, 2, "SelectVOP3Mods">;
def VOP3NoMods : ComplexPattern<untyped, 1, "SelectVOP3NoMods">;
// VOP3Mods, but the input source is known to never be NaN.
def VOP3Mods_nnan : ComplexPattern<fAny, 2, "SelectVOP3Mods_NNaN">;

def VOP3OMods : ComplexPattern<untyped, 3, "SelectVOP3OMods">;

def VOP3PMods  : ComplexPattern<untyped, 2, "SelectVOP3PMods">;
def VOP3PMods0 : ComplexPattern<untyped, 3, "SelectVOP3PMods0">;


//===----------------------------------------------------------------------===//
// SI assembler operands
//===----------------------------------------------------------------------===//

def SIOperand {
  int ZERO = 0x80;
  int VCC = 0x6A;
  int FLAT_SCR = 0x68;
}

// This should be kept in sync with SISrcMods enum
def SRCMODS {
  int NONE = 0;
  int NEG = 1;
  int ABS = 2;
  int NEG_ABS = 3;

  int NEG_HI = ABS;
  int OP_SEL_0 = 4;
  int OP_SEL_1 = 8;
}

def DSTCLAMP {
  int NONE = 0;
  int ENABLE = 1;
}

def DSTOMOD {
  int NONE = 0;
}

def TRAPID{
  int LLVM_TRAP = 2;
  int LLVM_DEBUG_TRAP = 3;
}

//===----------------------------------------------------------------------===//
//
// SI Instruction multiclass helpers.
//
// Instructions with _32 take 32-bit operands.
// Instructions with _64 take 64-bit operands.
//
// VOP_* instructions can use either a 32-bit or 64-bit encoding.  The 32-bit
// encoding is the standard encoding, but instruction that make use of
// any of the instruction modifiers must use the 64-bit encoding.
//
// Instructions with _e32 use the 32-bit encoding.
// Instructions with _e64 use the 64-bit encoding.
//
//===----------------------------------------------------------------------===//

class SIMCInstr <string pseudo, int subtarget> {
  string PseudoInstr = pseudo;
  int Subtarget = subtarget;
}

//===----------------------------------------------------------------------===//
// EXP classes
//===----------------------------------------------------------------------===//

class EXP_Helper<bit done, SDPatternOperator node = null_frag> : EXPCommon<
  (outs),
  (ins exp_tgt:$tgt,
       ExpSrc0:$src0, ExpSrc1:$src1, ExpSrc2:$src2, ExpSrc3:$src3,
       exp_vm:$vm, exp_compr:$compr, i8imm:$en),
  "exp$tgt $src0, $src1, $src2, $src3"#!if(done, " done", "")#"$compr$vm",
  [(node (i8 timm:$tgt), (i8 timm:$en),
         f32:$src0, f32:$src1, f32:$src2, f32:$src3,
         (i1 timm:$compr), (i1 timm:$vm))]> {
  let AsmMatchConverter = "cvtExp";
}

// Split EXP instruction into EXP and EXP_DONE so we can set
// mayLoad for done=1.
multiclass EXP_m<bit done, SDPatternOperator node> {
  let mayLoad = done in {
    let isPseudo = 1, isCodeGenOnly = 1 in {
      def "" : EXP_Helper<done, node>,
               SIMCInstr <"exp"#!if(done, "_done", ""), SIEncodingFamily.NONE>;
    }

    let done = done in {
      def _si : EXP_Helper<done>,
                SIMCInstr <"exp"#!if(done, "_done", ""), SIEncodingFamily.SI>,
                EXPe {
        let AssemblerPredicates = [isSICI];
        let DecoderNamespace = "SICI";
        let DisableDecoder = DisableSIDecoder;
      }

      def _vi : EXP_Helper<done>,
                SIMCInstr <"exp"#!if(done, "_done", ""), SIEncodingFamily.VI>,
                EXPe_vi {
        let AssemblerPredicates = [isVI];
        let DecoderNamespace = "VI";
        let DisableDecoder = DisableVIDecoder;
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// Vector ALU classes
//===----------------------------------------------------------------------===//

class getNumSrcArgs<ValueType Src0, ValueType Src1, ValueType Src2> {
  int ret =
    !if (!eq(Src0.Value, untyped.Value),      0,
      !if (!eq(Src1.Value, untyped.Value),    1,   // VOP1
         !if (!eq(Src2.Value, untyped.Value), 2,   // VOP2
                                              3))); // VOP3
}

// Returns the register class to use for the destination of VOP[123C]
// instructions for the given VT.
class getVALUDstForVT<ValueType VT> {
  RegisterOperand ret = !if(!eq(VT.Size, 32), VOPDstOperand<VGPR_32>,
                          !if(!eq(VT.Size, 128), VOPDstOperand<VReg_128>,
                            !if(!eq(VT.Size, 64), VOPDstOperand<VReg_64>,
                              !if(!eq(VT.Size, 16), VOPDstOperand<VGPR_32>,
                              VOPDstOperand<SReg_64>)))); // else VT == i1
}

// Returns the register class to use for the destination of VOP[12C]
// instructions with GFX9 SDWA extension
class getSDWA9DstForVT<ValueType VT> {
  RegisterOperand ret = !if(!eq(VT.Size, 1),
                            SDWA9VopcDst, // VOPC
                            VOPDstOperand<VGPR_32>); // VOP1/2 32-bit dst
}

// Returns the register class to use for source 0 of VOP[12C]
// instructions for the given VT.
class getVOPSrc0ForVT<ValueType VT> {
  bit isFP = !if(!eq(VT.Value, f16.Value), 1,
             !if(!eq(VT.Value, v2f16.Value), 1,
             !if(!eq(VT.Value, f32.Value), 1,
             !if(!eq(VT.Value, f64.Value), 1,
             0))));

  RegisterOperand ret =
    !if(isFP,
      !if(!eq(VT.Size, 64),
         VSrc_f64,
         !if(!eq(VT.Value, f16.Value),
            VSrc_f16,
            !if(!eq(VT.Value, v2f16.Value),
               VCSrc_v2f16,
               VSrc_f32
            )
         )
       ),
       !if(!eq(VT.Size, 64),
          VSrc_b64,
          !if(!eq(VT.Value, i16.Value),
             VSrc_b16,
             !if(!eq(VT.Value, v2i16.Value),
                VCSrc_v2b16,
                VSrc_b32
             )
          )
       )
    );
}

// Returns the vreg register class to use for source operand given VT
class getVregSrcForVT<ValueType VT> {
  RegisterClass ret = !if(!eq(VT.Size, 128), VReg_128,
                        !if(!eq(VT.Size, 64), VReg_64, VGPR_32));
}

class getSDWA9SrcForVT <ValueType VT> {
  RegisterOperand ret = !if(!eq(VT.Size, 16), SDWA9Src16, SDWA9Src32);
}

// Returns the register class to use for sources of VOP3 instructions for the
// given VT.
class getVOP3SrcForVT<ValueType VT> {
  bit isFP = !if(!eq(VT.Value, f16.Value), 1,
             !if(!eq(VT.Value, v2f16.Value), 1,
             !if(!eq(VT.Value, f32.Value), 1,
             !if(!eq(VT.Value, f64.Value), 1,
             0))));
  RegisterOperand ret =
  !if(!eq(VT.Size, 128),
     VSrc_128,
     !if(!eq(VT.Size, 64),
        !if(isFP,
           VCSrc_f64,
           VCSrc_b64),
        !if(!eq(VT.Value, i1.Value),
           SCSrc_b64,
           !if(isFP,
              !if(!eq(VT.Value, f16.Value),
                 VCSrc_f16,
                 !if(!eq(VT.Value, v2f16.Value),
                    VCSrc_v2f16,
                    VCSrc_f32
                 )
              ),
              !if(!eq(VT.Value, i16.Value),
                 VCSrc_b16,
                 !if(!eq(VT.Value, v2i16.Value),
                    VCSrc_v2b16,
                    VCSrc_b32
                 )
              )
           )
        )
     )
  );
}

// Returns 1 if the source arguments have modifiers, 0 if they do not.
// XXX - do f16 instructions?
class isFloatType<ValueType SrcVT> {
  bit ret =
    !if(!eq(SrcVT.Value, f16.Value), 1,
    !if(!eq(SrcVT.Value, f32.Value), 1,
    !if(!eq(SrcVT.Value, f64.Value), 1,
    !if(!eq(SrcVT.Value, v2f16.Value), 1,
    0))));
}

class isIntType<ValueType SrcVT> {
  bit ret =
    !if(!eq(SrcVT.Value, i16.Value), 1,
    !if(!eq(SrcVT.Value, i32.Value), 1,
    !if(!eq(SrcVT.Value, i64.Value), 1,
    0)));
}

class isPackedType<ValueType SrcVT> {
  bit ret =
    !if(!eq(SrcVT.Value, v2i16.Value), 1,
      !if(!eq(SrcVT.Value, v2f16.Value), 1, 0)
    );
}

// Float or packed int
class isModifierType<ValueType SrcVT> {
  bit ret =
    !if(!eq(SrcVT.Value, f16.Value), 1,
    !if(!eq(SrcVT.Value, f32.Value), 1,
    !if(!eq(SrcVT.Value, f64.Value), 1,
    !if(!eq(SrcVT.Value, v2f16.Value), 1,
    !if(!eq(SrcVT.Value, v2i16.Value), 1,
    0)))));
}

// Return type of input modifiers operand for specified input operand
class getSrcMod <ValueType VT> {
  bit isFP = !if(!eq(VT.Value, f16.Value), 1,
               !if(!eq(VT.Value, f32.Value), 1,
               !if(!eq(VT.Value, f64.Value), 1,
               0)));
  bit isPacked = isPackedType<VT>.ret;
  Operand ret =  !if(!eq(VT.Size, 64),
                     !if(isFP, FP64InputMods, Int64InputMods),
                       !if(isFP,
                         !if(!eq(VT.Value, f16.Value),
                            FP16InputMods,
                            FP32InputMods
                          ),
                         Int32InputMods)
                     );
}

// Return type of input modifiers operand specified input operand for SDWA/DPP
class getSrcModExt <ValueType VT> {
    bit isFP = !if(!eq(VT.Value, f16.Value), 1,
               !if(!eq(VT.Value, f32.Value), 1,
               !if(!eq(VT.Value, f64.Value), 1,
               0)));
  Operand ret = !if(isFP, FPVRegInputMods, IntVRegInputMods);
}

// Return type of input modifiers operand specified input operand for SDWA 9
class getSrcModSDWA9 <ValueType VT> {
    bit isFP = !if(!eq(VT.Value, f16.Value), 1,
               !if(!eq(VT.Value, f32.Value), 1,
               !if(!eq(VT.Value, f64.Value), 1,
               0)));
  Operand ret = !if(isFP, FPRegInputMods, IntRegInputMods);
}

// Returns the input arguments for VOP[12C] instructions for the given SrcVT.
class getIns32 <RegisterOperand Src0RC, RegisterClass Src1RC, int NumSrcArgs> {
  dag ret = !if(!eq(NumSrcArgs, 1), (ins Src0RC:$src0),               // VOP1
            !if(!eq(NumSrcArgs, 2), (ins Src0RC:$src0, Src1RC:$src1), // VOP2
                                    (ins)));
}

// Returns the input arguments for VOP3 instructions for the given SrcVT.
class getIns64 <RegisterOperand Src0RC, RegisterOperand Src1RC,
                RegisterOperand Src2RC, int NumSrcArgs,
                bit HasModifiers, bit HasOMod,
                Operand Src0Mod, Operand Src1Mod, Operand Src2Mod> {

  dag ret =
    !if (!eq(NumSrcArgs, 0),
      // VOP1 without input operands (V_NOP, V_CLREXCP)
      (ins),
      /* else */
    !if (!eq(NumSrcArgs, 1),
      !if (!eq(HasModifiers, 1),
        // VOP1 with modifiers
        (ins Src0Mod:$src0_modifiers, Src0RC:$src0,
             clampmod:$clamp, omod:$omod)
      /* else */,
        // VOP1 without modifiers
        (ins Src0RC:$src0)
      /* endif */ ),
    !if (!eq(NumSrcArgs, 2),
      !if (!eq(HasModifiers, 1),
        // VOP 2 with modifiers
        !if( !eq(HasOMod, 1),
          (ins Src0Mod:$src0_modifiers, Src0RC:$src0,
               Src1Mod:$src1_modifiers, Src1RC:$src1,
               clampmod:$clamp, omod:$omod),
           (ins Src0Mod:$src0_modifiers, Src0RC:$src0,
               Src1Mod:$src1_modifiers, Src1RC:$src1,
               clampmod:$clamp))
      /* else */,
        // VOP2 without modifiers
        (ins Src0RC:$src0, Src1RC:$src1)
      /* endif */ )
    /* NumSrcArgs == 3 */,
      !if (!eq(HasModifiers, 1),
        // VOP3 with modifiers
        !if (!eq(HasOMod, 1),
          (ins Src0Mod:$src0_modifiers, Src0RC:$src0,
               Src1Mod:$src1_modifiers, Src1RC:$src1,
               Src2Mod:$src2_modifiers, Src2RC:$src2,
               clampmod:$clamp, omod:$omod),
          (ins Src0Mod:$src0_modifiers, Src0RC:$src0,
               Src1Mod:$src1_modifiers, Src1RC:$src1,
               Src2Mod:$src2_modifiers, Src2RC:$src2,
               clampmod:$clamp))
      /* else */,
        // VOP3 without modifiers
        (ins Src0RC:$src0, Src1RC:$src1, Src2RC:$src2)
      /* endif */ ))));
}

/// XXX - src1 may only allow VGPRs?

// The modifiers (except clamp) are dummy operands for the benefit of
// printing and parsing. They defer their values to looking at the
// srcN_modifiers for what to print.
class getInsVOP3P <RegisterOperand Src0RC, RegisterOperand Src1RC,
                   RegisterOperand Src2RC, int NumSrcArgs,
                   bit HasClamp,
                   Operand Src0Mod, Operand Src1Mod, Operand Src2Mod> {
  dag ret = !if (!eq(NumSrcArgs, 2),
    !if (HasClamp,
      (ins Src0Mod:$src0_modifiers, Src0RC:$src0,
           Src1Mod:$src1_modifiers, Src1RC:$src1,
           clampmod:$clamp,
           op_sel:$op_sel, op_sel_hi:$op_sel_hi,
           neg_lo:$neg_lo, neg_hi:$neg_hi),
      (ins Src0Mod:$src0_modifiers, Src0RC:$src0,
           Src1Mod:$src1_modifiers, Src1RC:$src1,
           op_sel:$op_sel, op_sel_hi:$op_sel_hi,
           neg_lo:$neg_lo, neg_hi:$neg_hi)),
    // else NumSrcArgs == 3
    !if (HasClamp,
      (ins Src0Mod:$src0_modifiers, Src0RC:$src0,
           Src1Mod:$src1_modifiers, Src1RC:$src1,
           Src2Mod:$src2_modifiers, Src2RC:$src2,
           clampmod:$clamp,
           op_sel:$op_sel, op_sel_hi:$op_sel_hi,
           neg_lo:$neg_lo, neg_hi:$neg_hi),
      (ins Src0Mod:$src0_modifiers, Src0RC:$src0,
           Src1Mod:$src1_modifiers, Src1RC:$src1,
           Src2Mod:$src2_modifiers, Src2RC:$src2,
           op_sel:$op_sel, op_sel_hi:$op_sel_hi,
           neg_lo:$neg_lo, neg_hi:$neg_hi))
  );
}

class getInsDPP <RegisterClass Src0RC, RegisterClass Src1RC, int NumSrcArgs,
                 bit HasModifiers, Operand Src0Mod, Operand Src1Mod> {

  dag ret = !if (!eq(NumSrcArgs, 0),
                // VOP1 without input operands (V_NOP)
                (ins dpp_ctrl:$dpp_ctrl, row_mask:$row_mask,
                     bank_mask:$bank_mask, bound_ctrl:$bound_ctrl),
            !if (!eq(NumSrcArgs, 1),
              !if (!eq(HasModifiers, 1),
                // VOP1_DPP with modifiers
                (ins Src0Mod:$src0_modifiers, Src0RC:$src0,
                     dpp_ctrl:$dpp_ctrl, row_mask:$row_mask,
                     bank_mask:$bank_mask, bound_ctrl:$bound_ctrl)
              /* else */,
                // VOP1_DPP without modifiers
                (ins Src0RC:$src0, dpp_ctrl:$dpp_ctrl, row_mask:$row_mask,
                bank_mask:$bank_mask, bound_ctrl:$bound_ctrl)
              /* endif */)
              /* NumSrcArgs == 2 */,
              !if (!eq(HasModifiers, 1),
                // VOP2_DPP with modifiers
                (ins Src0Mod:$src0_modifiers, Src0RC:$src0,
                     Src1Mod:$src1_modifiers, Src1RC:$src1,
                     dpp_ctrl:$dpp_ctrl, row_mask:$row_mask,
                     bank_mask:$bank_mask, bound_ctrl:$bound_ctrl)
              /* else */,
                // VOP2_DPP without modifiers
                (ins Src0RC:$src0, Src1RC:$src1, dpp_ctrl:$dpp_ctrl,
                row_mask:$row_mask, bank_mask:$bank_mask,
                bound_ctrl:$bound_ctrl)
             /* endif */)));
}

class getInsSDWA <RegisterClass Src0RC, RegisterClass Src1RC, int NumSrcArgs,
                  bit HasFloatModifiers, Operand Src0Mod, Operand Src1Mod,
                  ValueType DstVT> {

  dag ret = !if(!eq(NumSrcArgs, 0),
               // VOP1 without input operands (V_NOP)
               (ins),
            !if(!eq(NumSrcArgs, 1),
               // VOP1_SDWA
               (ins Src0Mod:$src0_modifiers, Src0RC:$src0,
                    clampmod:$clamp, dst_sel:$dst_sel, dst_unused:$dst_unused,
                    src0_sel:$src0_sel),
            !if(!eq(NumSrcArgs, 2),
               !if(!eq(DstVT.Size, 1),
                  // VOPC_SDWA with modifiers
                  (ins Src0Mod:$src0_modifiers, Src0RC:$src0,
                       Src1Mod:$src1_modifiers, Src1RC:$src1,
                       clampmod:$clamp, src0_sel:$src0_sel, src1_sel:$src1_sel),
                  // VOP2_SDWA with modifiers
                  (ins Src0Mod:$src0_modifiers, Src0RC:$src0,
                       Src1Mod:$src1_modifiers, Src1RC:$src1,
                       clampmod:$clamp, dst_sel:$dst_sel, dst_unused:$dst_unused,
                       src0_sel:$src0_sel, src1_sel:$src1_sel)),
            (ins)/* endif */)));
}

// Ins for GFX9 SDWA
class getInsSDWA9 <RegisterOperand Src0RC, RegisterOperand Src1RC, int NumSrcArgs,
                   bit HasSDWAOMod, Operand Src0Mod, Operand Src1Mod,
                   ValueType DstVT> {

  dag ret = !if(!eq(NumSrcArgs, 0),
               // VOP1 without input operands (V_NOP)
               (ins),
            !if(!eq(NumSrcArgs, 1),
               // VOP1
               !if(!eq(HasSDWAOMod, 0),
                  // VOP1_SDWA9 without omod
                  (ins Src0Mod:$src0_modifiers, Src0RC:$src0,
                       clampmod:$clamp,
                       dst_sel:$dst_sel, dst_unused:$dst_unused,
                       src0_sel:$src0_sel),
                  // VOP1_SDWA9 with omod
                  (ins Src0Mod:$src0_modifiers, Src0RC:$src0,
                       clampmod:$clamp, omod:$omod,
                       dst_sel:$dst_sel, dst_unused:$dst_unused,
                       src0_sel:$src0_sel)),
            !if(!eq(NumSrcArgs, 2),
               !if(!eq(DstVT.Size, 1),
                  // VOPC_SDWA9
                  (ins Src0Mod:$src0_modifiers, Src0RC:$src0,
                       Src1Mod:$src1_modifiers, Src1RC:$src1,
                       src0_sel:$src0_sel, src1_sel:$src1_sel),
                  // VOP2_SDWA9
                  !if(!eq(HasSDWAOMod, 0),
                     // VOP2_SDWA9 without omod
                     (ins Src0Mod:$src0_modifiers, Src0RC:$src0,
                          Src1Mod:$src1_modifiers, Src1RC:$src1,
                          clampmod:$clamp,
                          dst_sel:$dst_sel, dst_unused:$dst_unused,
                          src0_sel:$src0_sel, src1_sel:$src1_sel),
                     // VOP1_SDWA9 with omod
                     (ins Src0Mod:$src0_modifiers, Src0RC:$src0,
                          Src1Mod:$src1_modifiers, Src1RC:$src1,
                          clampmod:$clamp, omod:$omod,
                          dst_sel:$dst_sel, dst_unused:$dst_unused,
                          src0_sel:$src0_sel, src1_sel:$src1_sel))),
            (ins)/* endif */)));
}

// Outs for DPP and SDWA
class getOutsExt <bit HasDst, ValueType DstVT, RegisterOperand DstRCExt> {
  dag ret = !if(HasDst,
                !if(!eq(DstVT.Size, 1),
                    (outs), // no dst for VOPC, we use "vcc"-token as dst in SDWA VOPC instructions
                    (outs DstRCExt:$vdst)),
                (outs)); // V_NOP
}

// Outs for GFX9 SDWA
class getOutsSDWA9 <bit HasDst, ValueType DstVT, RegisterOperand DstRCSDWA9> {
  dag ret = !if(HasDst,
                !if(!eq(DstVT.Size, 1),
                    (outs DstRCSDWA9:$sdst),
                    (outs DstRCSDWA9:$vdst)),
                (outs)); // V_NOP
}

// Returns the assembly string for the inputs and outputs of a VOP[12C]
// instruction.  This does not add the _e32 suffix, so it can be reused
// by getAsm64.
class getAsm32 <bit HasDst, int NumSrcArgs, ValueType DstVT = i32> {
  string dst = !if(!eq(DstVT.Size, 1), "$sdst", "$vdst"); // use $sdst for VOPC
  string src0 = ", $src0";
  string src1 = ", $src1";
  string src2 = ", $src2";
  string ret = !if(HasDst, dst, "") #
               !if(!eq(NumSrcArgs, 1), src0, "") #
               !if(!eq(NumSrcArgs, 2), src0#src1, "") #
               !if(!eq(NumSrcArgs, 3), src0#src1#src2, "");
}

// Returns the assembly string for the inputs and outputs of a VOP3
// instruction.
class getAsm64 <bit HasDst, int NumSrcArgs, bit HasModifiers,
                bit HasOMod, ValueType DstVT = i32> {
  string dst = !if(!eq(DstVT.Size, 1), "$sdst", "$vdst"); // use $sdst for VOPC
  string src0 = !if(!eq(NumSrcArgs, 1), "$src0_modifiers", "$src0_modifiers,");
  string src1 = !if(!eq(NumSrcArgs, 1), "",
                   !if(!eq(NumSrcArgs, 2), " $src1_modifiers",
                                           " $src1_modifiers,"));
  string src2 = !if(!eq(NumSrcArgs, 3), " $src2_modifiers", "");
  string ret =
  !if(!eq(HasModifiers, 0),
      getAsm32<HasDst, NumSrcArgs, DstVT>.ret,
      dst#", "#src0#src1#src2#"$clamp"#!if(HasOMod, "$omod", ""));
}

// Returns the assembly string for the inputs and outputs of a VOP3P
// instruction.
class getAsmVOP3P <bit HasDst, int NumSrcArgs, bit HasModifiers,
                   bit HasClamp, ValueType DstVT = i32> {
  string dst = " $vdst";
  string src0 = !if(!eq(NumSrcArgs, 1), "$src0", "$src0,");
  string src1 = !if(!eq(NumSrcArgs, 1), "",
                   !if(!eq(NumSrcArgs, 2), " $src1",
                                           " $src1,"));
  string src2 = !if(!eq(NumSrcArgs, 3), " $src2", "");

  string mods = !if(HasModifiers, "$neg_lo$neg_hi", "");
  string clamp = !if(HasClamp, "$clamp", "");

  // Each modifier is printed as an array of bits for each operand, so
  // all operands are printed as part of src0_modifiers.
  string ret = dst#", "#src0#src1#src2#"$op_sel$op_sel_hi"#mods#clamp;
}

class getAsmDPP <bit HasDst, int NumSrcArgs, bit HasModifiers, ValueType DstVT = i32> {
  string dst = !if(HasDst,
                   !if(!eq(DstVT.Size, 1),
                       "$sdst",
                       "$vdst"),
                    ""); // use $sdst for VOPC
  string src0 = !if(!eq(NumSrcArgs, 1), "$src0_modifiers", "$src0_modifiers,");
  string src1 = !if(!eq(NumSrcArgs, 1), "",
                   !if(!eq(NumSrcArgs, 2), " $src1_modifiers",
                                           " $src1_modifiers,"));
  string args = !if(!eq(HasModifiers, 0),
                     getAsm32<0, NumSrcArgs, DstVT>.ret,
                     ", "#src0#src1);
  string ret = dst#args#" $dpp_ctrl$row_mask$bank_mask$bound_ctrl";
}

class getAsmSDWA <bit HasDst, int NumSrcArgs, ValueType DstVT = i32> {
  string dst = !if(HasDst,
                   !if(!eq(DstVT.Size, 1),
                       " vcc", // use vcc token as dst for VOPC instructioins
                       "$vdst"),
                    "");
  string src0 = "$src0_modifiers";
  string src1 = "$src1_modifiers";
  string args = !if(!eq(NumSrcArgs, 0),
                    "",
                    !if(!eq(NumSrcArgs, 1),
                        ", "#src0#"$clamp",
                        ", "#src0#", "#src1#"$clamp"
                     )
                );
  string sdwa = !if(!eq(NumSrcArgs, 0),
                    "",
                    !if(!eq(NumSrcArgs, 1),
                        " $dst_sel $dst_unused $src0_sel",
                        !if(!eq(DstVT.Size, 1),
                            " $src0_sel $src1_sel", // No dst_sel and dst_unused for VOPC
                            " $dst_sel $dst_unused $src0_sel $src1_sel"
                        )
                    )
                );
  string ret = dst#args#sdwa;
}

class getAsmSDWA9 <bit HasDst, bit HasOMod, int NumSrcArgs,
                   ValueType DstVT = i32> {
  string dst = !if(HasDst,
                   !if(!eq(DstVT.Size, 1),
                       "$sdst", // VOPC
                       "$vdst"), // VOP1/2
                    "");
  string src0 = "$src0_modifiers";
  string src1 = "$src1_modifiers";
  string out_mods = !if(!eq(HasOMod, 0), "$clamp", "$clamp$omod");
  string args = !if(!eq(NumSrcArgs, 0), "",
                    !if(!eq(NumSrcArgs, 1),
                        ", "#src0,
                        ", "#src0#", "#src1
                     )
                );
  string sdwa = !if(!eq(NumSrcArgs, 0), "",
                    !if(!eq(NumSrcArgs, 1),
                        out_mods#" $dst_sel $dst_unused $src0_sel",
                        !if(!eq(DstVT.Size, 1),
                            " $src0_sel $src1_sel", // No dst_sel, dst_unused and output modifiers for VOPC
                            out_mods#" $dst_sel $dst_unused $src0_sel $src1_sel"
                        )
                    )
                );
  string ret = dst#args#sdwa;
}


// Function that checks if instruction supports DPP and SDWA
class getHasExt <int NumSrcArgs, ValueType DstVT = i32, ValueType Src0VT = i32,
                 ValueType Src1VT = i32> {
  bit ret = !if(!eq(NumSrcArgs, 3),
                0, // NumSrcArgs == 3 - No DPP or SDWA for VOP3
                !if(!eq(DstVT.Size, 64),
                    0, // 64-bit dst - No DPP or SDWA for 64-bit operands
                    !if(!eq(Src0VT.Size, 64),
                        0, // 64-bit src0
                        !if(!eq(Src0VT.Size, 64),
                            0, // 64-bit src2
                            1
                        )
                    )
                )
            );
}

class BitOr<bit a, bit b> {
  bit ret = !if(a, 1, !if(b, 1, 0));
}

class BitAnd<bit a, bit b> {
  bit ret = !if(a, !if(b, 1, 0), 0);
}

class VOPProfile <list<ValueType> _ArgVT> {

  field list<ValueType> ArgVT = _ArgVT;

  field ValueType DstVT = ArgVT[0];
  field ValueType Src0VT = ArgVT[1];
  field ValueType Src1VT = ArgVT[2];
  field ValueType Src2VT = ArgVT[3];
  field RegisterOperand DstRC = getVALUDstForVT<DstVT>.ret;
  field RegisterOperand DstRCDPP = getVALUDstForVT<DstVT>.ret;
  field RegisterOperand DstRCSDWA = getVALUDstForVT<DstVT>.ret;
  field RegisterOperand DstRCSDWA9 = getSDWA9DstForVT<DstVT>.ret;
  field RegisterOperand Src0RC32 = getVOPSrc0ForVT<Src0VT>.ret;
  field RegisterClass Src1RC32 = getVregSrcForVT<Src1VT>.ret;
  field RegisterOperand Src0RC64 = getVOP3SrcForVT<Src0VT>.ret;
  field RegisterOperand Src1RC64 = getVOP3SrcForVT<Src1VT>.ret;
  field RegisterOperand Src2RC64 = getVOP3SrcForVT<Src2VT>.ret;
  field RegisterClass Src0DPP = getVregSrcForVT<Src0VT>.ret;
  field RegisterClass Src1DPP = getVregSrcForVT<Src1VT>.ret;
  field RegisterClass Src0SDWA = getVregSrcForVT<Src0VT>.ret;
  field RegisterClass Src1SDWA = getVregSrcForVT<Src1VT>.ret;
  field RegisterOperand Src0SDWA9 = getSDWA9SrcForVT<Src0VT>.ret;
  field RegisterOperand Src1SDWA9 = getSDWA9SrcForVT<Src0VT>.ret;
  field Operand Src0Mod = getSrcMod<Src0VT>.ret;
  field Operand Src1Mod = getSrcMod<Src1VT>.ret;
  field Operand Src2Mod = getSrcMod<Src2VT>.ret;
  field Operand Src0ModDPP = getSrcModExt<Src0VT>.ret;
  field Operand Src1ModDPP = getSrcModExt<Src1VT>.ret;
  field Operand Src0ModSDWA = getSrcModExt<Src0VT>.ret;
  field Operand Src1ModSDWA = getSrcModExt<Src1VT>.ret;
  field Operand Src0ModSDWA9 = getSrcModSDWA9<Src0VT>.ret;
  field Operand Src1ModSDWA9 = getSrcModSDWA9<Src1VT>.ret;


  field bit HasDst = !if(!eq(DstVT.Value, untyped.Value), 0, 1);
  field bit HasDst32 = HasDst;
  field bit EmitDst = HasDst; // force dst encoding, see v_movreld_b32 special case
  field int NumSrcArgs = getNumSrcArgs<Src0VT, Src1VT, Src2VT>.ret;
  field bit HasSrc0 = !if(!eq(Src0VT.Value, untyped.Value), 0, 1);
  field bit HasSrc1 = !if(!eq(Src1VT.Value, untyped.Value), 0, 1);
  field bit HasSrc2 = !if(!eq(Src2VT.Value, untyped.Value), 0, 1);

  // TODO: Modifiers logic is somewhat adhoc here, to be refined later
  field bit HasModifiers = isModifierType<Src0VT>.ret;

  field bit HasSrc0FloatMods = isFloatType<Src0VT>.ret;
  field bit HasSrc1FloatMods = isFloatType<Src1VT>.ret;
  field bit HasSrc2FloatMods = isFloatType<Src2VT>.ret;

  field bit HasSrc0IntMods = isIntType<Src0VT>.ret;
  field bit HasSrc1IntMods = isIntType<Src1VT>.ret;
  field bit HasSrc2IntMods = isIntType<Src2VT>.ret;

  field bit HasSrc0Mods = HasModifiers;
  field bit HasSrc1Mods = !if(HasModifiers, BitOr<HasSrc1FloatMods, HasSrc1IntMods>.ret, 0);
  field bit HasSrc2Mods = !if(HasModifiers, BitOr<HasSrc2FloatMods, HasSrc2IntMods>.ret, 0);

  field bit HasClamp = HasModifiers;
  field bit HasSDWAClamp = EmitDst;
  field bit HasFPClamp = BitAnd<isFloatType<DstVT>.ret, HasClamp>.ret;

  field bit IsPacked = isPackedType<Src0VT>.ret;
  field bit HasOpSel = IsPacked;
  field bit HasOMod = !if(HasOpSel, 0, HasModifiers);
  field bit HasSDWAOMod = isFloatType<DstVT>.ret;

  field bit HasExt = getHasExt<NumSrcArgs, DstVT, Src0VT, Src1VT>.ret;
  field bit HasSDWA9 = HasExt;

  field Operand Src0PackedMod = !if(HasSrc0FloatMods, PackedF16InputMods, PackedI16InputMods);
  field Operand Src1PackedMod = !if(HasSrc1FloatMods, PackedF16InputMods, PackedI16InputMods);
  field Operand Src2PackedMod = !if(HasSrc2FloatMods, PackedF16InputMods, PackedI16InputMods);

  field dag Outs = !if(HasDst,(outs DstRC:$vdst),(outs));

  // VOP3b instructions are a special case with a second explicit
  // output. This is manually overridden for them.
  field dag Outs32 = Outs;
  field dag Outs64 = Outs;
  field dag OutsDPP = getOutsExt<HasDst, DstVT, DstRCDPP>.ret;
  field dag OutsSDWA = getOutsExt<HasDst, DstVT, DstRCSDWA>.ret;
  field dag OutsSDWA9 = getOutsSDWA9<HasDst, DstVT, DstRCSDWA9>.ret;

  field dag Ins32 = getIns32<Src0RC32, Src1RC32, NumSrcArgs>.ret;
  field dag Ins64 = getIns64<Src0RC64, Src1RC64, Src2RC64, NumSrcArgs,
                             HasModifiers, HasOMod, Src0Mod, Src1Mod,
                             Src2Mod>.ret;
  field dag InsVOP3P = getInsVOP3P<Src0RC64, Src1RC64, Src2RC64,
                                   NumSrcArgs, HasClamp,
                                   Src0PackedMod, Src1PackedMod, Src2PackedMod>.ret;

  field dag InsDPP = getInsDPP<Src0DPP, Src1DPP, NumSrcArgs,
                               HasModifiers, Src0ModDPP, Src1ModDPP>.ret;
  field dag InsSDWA = getInsSDWA<Src0SDWA, Src1SDWA, NumSrcArgs,
                                 HasModifiers, Src0ModSDWA, Src1ModSDWA,
                                 DstVT>.ret;
  field dag InsSDWA9 = getInsSDWA9<Src0SDWA9, Src1SDWA9, NumSrcArgs,
                                   HasSDWAOMod, Src0ModSDWA9, Src1ModSDWA9,
                                   DstVT>.ret;

  field string Asm32 = getAsm32<HasDst, NumSrcArgs, DstVT>.ret;
  field string Asm64 = getAsm64<HasDst, NumSrcArgs, HasModifiers, HasOMod, DstVT>.ret;
  field string AsmVOP3P = getAsmVOP3P<HasDst, NumSrcArgs, HasModifiers, HasClamp, DstVT>.ret;
  field string AsmDPP = getAsmDPP<HasDst, NumSrcArgs, HasModifiers, DstVT>.ret;
  field string AsmSDWA = getAsmSDWA<HasDst, NumSrcArgs, DstVT>.ret;
  field string AsmSDWA9 = getAsmSDWA9<HasDst, HasSDWAOMod, NumSrcArgs, DstVT>.ret;
}

class VOP_NO_EXT <VOPProfile p> : VOPProfile <p.ArgVT> {
  let HasExt = 0;
  let HasSDWA9 = 0;
}

def VOP_F16_F16 : VOPProfile <[f16, f16, untyped, untyped]>;
def VOP_F16_I16 : VOPProfile <[f16, i16, untyped, untyped]>;
def VOP_I16_F16 : VOPProfile <[i16, f16, untyped, untyped]>;

def VOP_F16_F16_F16 : VOPProfile <[f16, f16, f16, untyped]>;
def VOP_F16_F16_I16 : VOPProfile <[f16, f16, i16, untyped]>;
def VOP_F16_F16_I32 : VOPProfile <[f16, f16, i32, untyped]>;
def VOP_I16_I16_I16 : VOPProfile <[i16, i16, i16, untyped]>;

def VOP_I16_I16_I16_I16 : VOPProfile <[i16, i16, i16, i16, untyped]>;
def VOP_F16_F16_F16_F16 : VOPProfile <[f16, f16, f16, f16, untyped]>;

def VOP_V2F16_V2F16_V2F16 : VOPProfile <[v2f16, v2f16, v2f16, untyped]>;
def VOP_V2I16_V2I16_V2I16 : VOPProfile <[v2i16, v2i16, v2i16, untyped]>;
def VOP_B32_F16_F16 : VOPProfile <[i32, f16, f16, untyped]>;

def VOP_V2F16_V2F16_V2F16_V2F16 : VOPProfile <[v2f16, v2f16, v2f16, v2f16]>;
def VOP_V2I16_V2I16_V2I16_V2I16 : VOPProfile <[v2i16, v2i16, v2i16, v2i16]>;

def VOP_NONE : VOPProfile <[untyped, untyped, untyped, untyped]>;

def VOP_F32_F32 : VOPProfile <[f32, f32, untyped, untyped]>;
def VOP_F32_F64 : VOPProfile <[f32, f64, untyped, untyped]>;
def VOP_F32_I32 : VOPProfile <[f32, i32, untyped, untyped]>;
def VOP_F64_F32 : VOPProfile <[f64, f32, untyped, untyped]>;
def VOP_F64_F64 : VOPProfile <[f64, f64, untyped, untyped]>;
def VOP_F64_I32 : VOPProfile <[f64, i32, untyped, untyped]>;
def VOP_I32_F32 : VOPProfile <[i32, f32, untyped, untyped]>;
def VOP_I32_F64 : VOPProfile <[i32, f64, untyped, untyped]>;
def VOP_I32_I32 : VOPProfile <[i32, i32, untyped, untyped]>;
def VOP_F16_F32 : VOPProfile <[f16, f32, untyped, untyped]>;
def VOP_F32_F16 : VOPProfile <[f32, f16, untyped, untyped]>;

def VOP_F32_F32_F16 : VOPProfile <[f32, f32, f16, untyped]>;
def VOP_F32_F32_F32 : VOPProfile <[f32, f32, f32, untyped]>;
def VOP_F32_F32_I32 : VOPProfile <[f32, f32, i32, untyped]>;
def VOP_F64_F64_F64 : VOPProfile <[f64, f64, f64, untyped]>;
def VOP_F64_F64_I32 : VOPProfile <[f64, f64, i32, untyped]>;
def VOP_I32_F32_F32 : VOPProfile <[i32, f32, f32, untyped]>;
def VOP_I32_F32_I32 : VOPProfile <[i32, f32, i32, untyped]>;
def VOP_I32_I32_I32 : VOPProfile <[i32, i32, i32, untyped]>;
def VOP_V2F16_F32_F32 : VOPProfile <[v2f16, f32, f32, untyped]>;

def VOP_I64_I64_I32 : VOPProfile <[i64, i64, i32, untyped]>;
def VOP_I64_I32_I64 : VOPProfile <[i64, i32, i64, untyped]>;
def VOP_I64_I64_I64 : VOPProfile <[i64, i64, i64, untyped]>;

def VOP_F16_F32_F16_F32 : VOPProfile <[f16, f32, f16, f32]>;
def VOP_F32_F32_F16_F16 : VOPProfile <[f32, f32, f16, f16]>;
def VOP_F32_F32_F32_F32 : VOPProfile <[f32, f32, f32, f32]>;
def VOP_F64_F64_F64_F64 : VOPProfile <[f64, f64, f64, f64]>;
def VOP_I32_I32_I32_I32 : VOPProfile <[i32, i32, i32, i32]>;
def VOP_I64_I32_I32_I64 : VOPProfile <[i64, i32, i32, i64]>;
def VOP_I32_F32_I32_I32 : VOPProfile <[i32, f32, i32, i32]>;
def VOP_I64_I64_I32_I64 : VOPProfile <[i64, i64, i32, i64]>;
def VOP_V4I32_I64_I32_V4I32 : VOPProfile <[v4i32, i64, i32, v4i32]>;

class Commutable_REV <string revOp, bit isOrig> {
  string RevOp = revOp;
  bit IsOrig = isOrig;
}

class AtomicNoRet <string noRetOp, bit isRet> {
  string NoRetOp = noRetOp;
  bit IsRet = isRet;
}

//===----------------------------------------------------------------------===//
// Interpolation opcodes
//===----------------------------------------------------------------------===//

class VINTRP_Pseudo <string opName, dag outs, dag ins, list<dag> pattern> :
  VINTRPCommon <outs, ins, "", pattern>,
  SIMCInstr<opName, SIEncodingFamily.NONE> {
  let isPseudo = 1;
  let isCodeGenOnly = 1;
}

class VINTRP_Real_si <bits <2> op, string opName, dag outs, dag ins,
                      string asm> :
  VINTRPCommon <outs, ins, asm, []>,
  VINTRPe <op>,
  SIMCInstr<opName, SIEncodingFamily.SI> {
  let AssemblerPredicate = SIAssemblerPredicate;
  let DecoderNamespace = "SICI";
  let DisableDecoder = DisableSIDecoder;
}

class VINTRP_Real_vi <bits <2> op, string opName, dag outs, dag ins,
                      string asm> :
  VINTRPCommon <outs, ins, asm, []>,
  VINTRPe_vi <op>,
  SIMCInstr<opName, SIEncodingFamily.VI> {
  let AssemblerPredicate = VIAssemblerPredicate;
  let DecoderNamespace = "VI";
  let DisableDecoder = DisableVIDecoder;
}

multiclass VINTRP_m <bits <2> op, dag outs, dag ins, string asm,
                     list<dag> pattern = []> {
  def "" : VINTRP_Pseudo <NAME, outs, ins, pattern>;

  def _si : VINTRP_Real_si <op, NAME, outs, ins, asm>;

  def _vi : VINTRP_Real_vi <op, NAME, outs, ins, asm>;
}

//===----------------------------------------------------------------------===//
// Vector instruction mappings
//===----------------------------------------------------------------------===//

// Maps an opcode in e32 form to its e64 equivalent
def getVOPe64 : InstrMapping {
  let FilterClass = "VOP";
  let RowFields = ["OpName"];
  let ColFields = ["Size", "VOP3"];
  let KeyCol = ["4", "0"];
  let ValueCols = [["8", "1"]];
}

// Maps an opcode in e64 form to its e32 equivalent
def getVOPe32 : InstrMapping {
  let FilterClass = "VOP";
  let RowFields = ["OpName"];
  let ColFields = ["Size", "VOP3"];
  let KeyCol = ["8", "1"];
  let ValueCols = [["4", "0"]];
}

// Maps ordinary instructions to their SDWA counterparts
def getSDWAOp : InstrMapping {
  let FilterClass = "VOP";
  let RowFields = ["OpName"];
  let ColFields = ["AsmVariantName"];
  let KeyCol = ["Default"];
  let ValueCols = [["SDWA"]];
}

// Maps ordinary instructions to their SDWA GFX9 counterparts
def getSDWA9Op : InstrMapping {
  let FilterClass = "VOP";
  let RowFields = ["OpName"];
  let ColFields = ["AsmVariantName"];
  let KeyCol = ["Default"];
  let ValueCols = [["SDWA9"]];
}

def getMaskedMIMGOp : InstrMapping {
  let FilterClass = "MIMG_Mask";
  let RowFields = ["Op"];
  let ColFields = ["Channels"];
  let KeyCol = ["4"];
  let ValueCols = [["1"], ["2"], ["3"] ];
}

// Maps an commuted opcode to its original version
def getCommuteOrig : InstrMapping {
  let FilterClass = "Commutable_REV";
  let RowFields = ["RevOp"];
  let ColFields = ["IsOrig"];
  let KeyCol = ["0"];
  let ValueCols = [["1"]];
}

// Maps an original opcode to its commuted version
def getCommuteRev : InstrMapping {
  let FilterClass = "Commutable_REV";
  let RowFields = ["RevOp"];
  let ColFields = ["IsOrig"];
  let KeyCol = ["1"];
  let ValueCols = [["0"]];
}

def getMCOpcodeGen : InstrMapping {
  let FilterClass = "SIMCInstr";
  let RowFields = ["PseudoInstr"];
  let ColFields = ["Subtarget"];
  let KeyCol = [!cast<string>(SIEncodingFamily.NONE)];
  let ValueCols = [[!cast<string>(SIEncodingFamily.SI)],
                   [!cast<string>(SIEncodingFamily.VI)]];
}

// Get equivalent SOPK instruction.
def getSOPKOp : InstrMapping {
  let FilterClass = "SOPKInstTable";
  let RowFields = ["BaseCmpOp"];
  let ColFields = ["IsSOPK"];
  let KeyCol = ["0"];
  let ValueCols = [["1"]];
}

def getAddr64Inst : InstrMapping {
  let FilterClass = "MUBUFAddr64Table";
  let RowFields = ["OpName"];
  let ColFields = ["IsAddr64"];
  let KeyCol = ["0"];
  let ValueCols = [["1"]];
}

// Maps an atomic opcode to its version with a return value.
def getAtomicRetOp : InstrMapping {
  let FilterClass = "AtomicNoRet";
  let RowFields = ["NoRetOp"];
  let ColFields = ["IsRet"];
  let KeyCol = ["0"];
  let ValueCols = [["1"]];
}

// Maps an atomic opcode to its returnless version.
def getAtomicNoRetOp : InstrMapping {
  let FilterClass = "AtomicNoRet";
  let RowFields = ["NoRetOp"];
  let ColFields = ["IsRet"];
  let KeyCol = ["1"];
  let ValueCols = [["0"]];
}

include "SIInstructions.td"
include "CIInstructions.td"

include "DSInstructions.td"
include "MIMGInstructions.td"
