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

class vop {
  field bits<9> SI3;
  field bits<10> VI3;
}

class vopc <bits<8> si, bits<8> vi = !add(0x40, si)> : vop {
  field bits<8> SI = si;
  field bits<8> VI = vi;

  field bits<9>  SI3 = {0, si{7-0}};
  field bits<10> VI3 = {0, 0, vi{7-0}};
}

class vop1 <bits<8> si, bits<8> vi = si> : vop {
  field bits<8> SI = si;
  field bits<8> VI = vi;

  field bits<9>  SI3 = {1, 1, si{6-0}};
  field bits<10> VI3 = !add(0x140, vi);
}

class vop2 <bits<6> si, bits<6> vi = si> : vop {
  field bits<6> SI = si;
  field bits<6> VI = vi;

  field bits<9>  SI3 = {1, 0, 0, si{5-0}};
  field bits<10> VI3 = {0, 1, 0, 0, vi{5-0}};
}

// Specify a VOP2 opcode for SI and VOP3 opcode for VI
// that doesn't have VOP2 encoding on VI
class vop23 <bits<6> si, bits<10> vi> : vop2 <si> {
  let VI3 = vi;
}

class vop3 <bits<9> si, bits<10> vi = {0, si}> : vop {
  let SI3 = si;
  let VI3 = vi;
}

class sop1 <bits<8> si, bits<8> vi = si> {
  field bits<8> SI = si;
  field bits<8> VI = vi;
}

class sop2 <bits<7> si, bits<7> vi = si> {
  field bits<7> SI = si;
  field bits<7> VI = vi;
}

class sopk <bits<5> si, bits<5> vi = si> {
  field bits<5> SI = si;
  field bits<5> VI = vi;
}

class dsop <bits<8> si, bits<8> vi = si> {
  field bits<8> SI = si;
  field bits<8> VI = vi;
}

// Specify an SMRD opcode for SI and SMEM opcode for VI

// FIXME: This should really be bits<5> si, Tablegen crashes if
// parameter default value is other parameter with different bit size
class smrd<bits<8> si, bits<8> vi = si> {
  field bits<5> SI = si{4-0};
  field bits<8> VI = vi;
}

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

def SIload_input : SDNode<"AMDGPUISD::LOAD_INPUT",
  SDTypeProfile<1, 3, [SDTCisVT<0, v4f32>, SDTCisVT<1, v4i32>, SDTCisVT<2, i16>,
                       SDTCisVT<3, i32>]>
>;

class SDSample<string opcode> : SDNode <opcode,
  SDTypeProfile<1, 4, [SDTCisVT<0, v4f32>, SDTCisVT<2, v8i32>,
                       SDTCisVT<3, v4i32>, SDTCisVT<4, i32>]>
>;

def SIsample : SDSample<"AMDGPUISD::SAMPLE">;
def SIsampleb : SDSample<"AMDGPUISD::SAMPLEB">;
def SIsampled : SDSample<"AMDGPUISD::SAMPLED">;
def SIsamplel : SDSample<"AMDGPUISD::SAMPLEL">;

def SIpc_add_rel_offset : SDNode<"AMDGPUISD::PC_ADD_REL_OFFSET",
  SDTypeProfile<1, 1, [SDTCisVT<0, iPTR>, SDTCisSameAs<0,1>]>
>;

//===----------------------------------------------------------------------===//
// PatFrags for FLAT instructions
//===----------------------------------------------------------------------===//

class flat_ld <SDPatternOperator ld> : PatFrag<(ops node:$ptr),
                                               (ld node:$ptr), [{
  const MemSDNode *LD = cast<MemSDNode>(N);
  return LD->getAddressSpace() == AMDGPUAS::FLAT_ADDRESS ||
         LD->getAddressSpace() == AMDGPUAS::GLOBAL_ADDRESS ||
         LD->getAddressSpace() == AMDGPUAS::CONSTANT_ADDRESS;
}]>;

def flat_load : flat_ld <load>;
def atomic_flat_load : flat_ld<atomic_load>;
def flat_az_extloadi8 : flat_ld <az_extloadi8>;
def flat_sextloadi8 : flat_ld <sextloadi8>;
def flat_az_extloadi16 : flat_ld <az_extloadi16>;
def flat_sextloadi16 : flat_ld <sextloadi16>;

class flat_st <SDPatternOperator st> : PatFrag<(ops node:$val, node:$ptr),
                                               (st node:$val, node:$ptr), [{
  const MemSDNode *ST = cast<MemSDNode>(N);
  return ST->getAddressSpace() == AMDGPUAS::FLAT_ADDRESS ||
         ST->getAddressSpace() == AMDGPUAS::GLOBAL_ADDRESS;
}]>;

def flat_store: flat_st <store>;
def atomic_flat_store: flat_st <atomic_store>;
def flat_truncstorei8 : flat_st <truncstorei8>;
def flat_truncstorei16 : flat_st <truncstorei16>;

class MubufLoad <SDPatternOperator op> : PatFrag <
  (ops node:$ptr), (op node:$ptr), [{

  const MemSDNode *LD = cast<MemSDNode>(N);
  return LD->getAddressSpace() == AMDGPUAS::GLOBAL_ADDRESS ||
         LD->getAddressSpace() == AMDGPUAS::CONSTANT_ADDRESS;
}]>;

def mubuf_load : MubufLoad <load>;
def mubuf_az_extloadi8 : MubufLoad <az_extloadi8>;
def mubuf_sextloadi8 : MubufLoad <sextloadi8>;
def mubuf_az_extloadi16 : MubufLoad <az_extloadi16>;
def mubuf_sextloadi16 : MubufLoad <sextloadi16>;

def mubuf_load_atomic : MubufLoad <atomic_load>;

def smrd_load : PatFrag <(ops node:$ptr), (load node:$ptr), [{
  auto Ld = cast<LoadSDNode>(N);
  return Ld->getAlignment() >= 4  &&
    Ld->getAddressSpace() == AMDGPUAS::CONSTANT_ADDRESS &&
    static_cast<const SITargetLowering *>(getTargetLowering())->isMemOpUniform(N);
}]>;

//===----------------------------------------------------------------------===//
// PatFrags for global memory operations
//===----------------------------------------------------------------------===//

def atomic_inc_global : global_binary_atomic_op<SIatomic_inc>;
def atomic_dec_global : global_binary_atomic_op<SIatomic_dec>;

def atomic_inc_flat : flat_binary_atomic_op<SIatomic_inc>;
def atomic_dec_flat : flat_binary_atomic_op<SIatomic_dec>;

//===----------------------------------------------------------------------===//
// SDNodes and PatFrag for local loads and stores to enable s_mov_b32 m0, -1
// to be glued to the memory instructions.
//===----------------------------------------------------------------------===//

def SIld_local : SDNode <"ISD::LOAD", SDTLoad,
  [SDNPHasChain, SDNPMayLoad, SDNPMemOperand, SDNPInGlue]
>;

def si_ld_local : PatFrag <(ops node:$ptr), (SIld_local node:$ptr), [{
  return cast<LoadSDNode>(N)->getAddressSpace() == AMDGPUAS::LOCAL_ADDRESS;
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
  return cast<StoreSDNode>(N)->getAddressSpace() == AMDGPUAS::LOCAL_ADDRESS;
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

// Copied from the AArch64 backend:
def bitcast_fpimm_to_i64 : SDNodeXForm<fpimm, [{
return CurDAG->getTargetConstant(
  N->getValueAPF().bitcastToAPInt().getZExtValue(), SDLoc(N), MVT::i64);
}]>;

def SIMM16bit : PatLeaf <(imm),
  [{return isInt<16>(N->getSExtValue());}]
>;

def IMM20bit : PatLeaf <(imm),
  [{return isUInt<20>(N->getZExtValue());}]
>;

def mubuf_vaddr_offset : PatFrag<
  (ops node:$ptr, node:$offset, node:$imm_offset),
  (add (add node:$ptr, node:$offset), node:$imm_offset)
>;

class InlineImm <ValueType vt> : PatLeaf <(vt imm), [{
  return isInlineImmediate(N);
}]>;

class InlineFPImm <ValueType vt> : PatLeaf <(vt fpimm), [{
  return isInlineImmediate(N);
}]>;

class SGPRImm <dag frag> : PatLeaf<frag, [{
  if (Subtarget->getGeneration() < SISubtarget::SOUTHERN_ISLANDS) {
    return false;
  }
  const SIRegisterInfo *SIRI =
      static_cast<const SIRegisterInfo *>(Subtarget->getRegisterInfo());
  for (SDNode::use_iterator U = N->use_begin(), E = SDNode::use_end();
                                                U != E; ++U) {
    const TargetRegisterClass *RC = getOperandRegClass(*U, U.getOperandNo());
    if (RC && SIRI->isSGPRClass(RC))
      return true;
  }
  return false;
}]>;

//===----------------------------------------------------------------------===//
// Custom Operands
//===----------------------------------------------------------------------===//

def SoppBrTarget : AsmOperandClass {
  let Name = "SoppBrTarget";
  let ParserMethod = "parseSOppBrTarget";
}

def sopp_brtarget : Operand<OtherVT> {
  let EncoderMethod = "getSOPPBrEncoding";
  let OperandType = "OPERAND_PCREL";
  let ParserMatchClass = SoppBrTarget;
}

def si_ga : Operand<iPTR>;

def InterpSlot : Operand<i32> {
  let PrintMethod = "printInterpSlot";
}

def SendMsgMatchClass : AsmOperandClass {
  let Name = "SendMsg";
  let PredicateMethod = "isSendMsg";
  let ParserMethod = "parseSendMsgOp";
  let RenderMethod = "addImmOperands";
}

def SendMsgImm : Operand<i32> {
  let PrintMethod = "printSendMsg";
  let ParserMatchClass = SendMsgMatchClass;
}

def SWaitMatchClass : AsmOperandClass {
  let Name = "SWaitCnt";
  let RenderMethod = "addImmOperands";
  let ParserMethod = "parseSWaitCntOps";
}

def WAIT_FLAG : Operand <i32> {
  let ParserMatchClass = SWaitMatchClass;
  let PrintMethod = "printWaitFlag";
}

include "SIInstrFormats.td"
include "VIInstrFormats.td"

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

class NamedOperandU16<string Name, AsmOperandClass MatchClass> : Operand<i16> {
  let PrintMethod = "print"#Name;
  let ParserMatchClass = MatchClass;
}

class NamedOperandU32<string Name, AsmOperandClass MatchClass> : Operand<i32> {
  let PrintMethod = "print"#Name;
  let ParserMatchClass = MatchClass;
}

let OperandType = "OPERAND_IMMEDIATE" in {

def offen : NamedOperandBit<"Offen", NamedMatchClass<"Offen">>;
def idxen : NamedOperandBit<"Idxen", NamedMatchClass<"Idxen">>;
def addr64 : NamedOperandBit<"Addr64", NamedMatchClass<"Addr64">>;

def offset : NamedOperandU16<"Offset", NamedMatchClass<"Offset">>;
def offset0 : NamedOperandU8<"Offset0", NamedMatchClass<"Offset0">>;
def offset1 : NamedOperandU8<"Offset1", NamedMatchClass<"Offset1">>;

def gds : NamedOperandBit<"GDS", NamedMatchClass<"GDS">>;

def omod : NamedOperandU32<"OModSI", NamedMatchClass<"OModSI">>;
def clampmod : NamedOperandBit<"ClampSI", NamedMatchClass<"ClampSI">>;

def smrd_offset : NamedOperandU32<"SMRDOffset", NamedMatchClass<"SMRDOffset">>;
def smrd_literal_offset : NamedOperandU32<"SMRDLiteralOffset", NamedMatchClass<"SMRDLiteralOffset">>;

def glc : NamedOperandBit<"GLC", NamedMatchClass<"GLC">>;
def slc : NamedOperandBit<"SLC", NamedMatchClass<"SLC">>;
def tfe : NamedOperandBit<"TFE", NamedMatchClass<"TFE">>;
def unorm : NamedOperandBit<"UNorm", NamedMatchClass<"UNorm">>;
def da : NamedOperandBit<"DA", NamedMatchClass<"DA">>;
def r128 : NamedOperandBit<"R128", NamedMatchClass<"R128">>;
def lwe : NamedOperandBit<"LWE", NamedMatchClass<"LWE">>;

def dmask : NamedOperandU16<"DMask", NamedMatchClass<"DMask">>;

def dpp_ctrl : NamedOperandU32<"DPPCtrl", NamedMatchClass<"DPPCtrl", 0>>;
def row_mask : NamedOperandU32<"RowMask", NamedMatchClass<"RowMask">>;
def bank_mask : NamedOperandU32<"BankMask", NamedMatchClass<"BankMask">>;
def bound_ctrl : NamedOperandBit<"BoundCtrl", NamedMatchClass<"BoundCtrl">>;

def dst_sel : NamedOperandU32<"SDWADstSel", NamedMatchClass<"SDWADstSel">>;
def src0_sel : NamedOperandU32<"SDWASrc0Sel", NamedMatchClass<"SDWASrc0Sel">>;
def src1_sel : NamedOperandU32<"SDWASrc1Sel", NamedMatchClass<"SDWASrc1Sel">>;
def dst_unused : NamedOperandU32<"SDWADstUnused", NamedMatchClass<"SDWADstUnused">>;

def hwreg : NamedOperandU16<"Hwreg", NamedMatchClass<"Hwreg", 0>>;

} // End OperandType = "OPERAND_IMMEDIATE"


def VOPDstS64 : VOPDstOperand <SReg_64>;

def FPInputModsMatchClass : AsmOperandClass {
  let Name = "RegOrImmWithFPInputMods";
  let ParserMethod = "parseRegOrImmWithFPInputMods";
  let PredicateMethod = "isRegOrImmWithInputMods";
}

def FPInputMods : Operand <i32> {
  let PrintMethod = "printOperandAndFPInputMods";
  let ParserMatchClass = FPInputModsMatchClass;
}

def IntInputModsMatchClass : AsmOperandClass {
  let Name = "RegOrImmWithIntInputMods";
  let ParserMethod = "parseRegOrImmWithIntInputMods";
  let PredicateMethod = "isRegOrImmWithInputMods";
}

def IntInputMods: Operand <i32> {
  let PrintMethod = "printOperandAndIntInputMods";
  let ParserMatchClass = IntInputModsMatchClass;
}

//===----------------------------------------------------------------------===//
// Complex patterns
//===----------------------------------------------------------------------===//

def DS1Addr1Offset : ComplexPattern<i32, 2, "SelectDS1Addr1Offset">;
def DS64Bit4ByteAligned : ComplexPattern<i32, 3, "SelectDS64Bit4ByteAligned">;

def MUBUFAddr32 : ComplexPattern<i64, 9, "SelectMUBUFAddr32">;
def MUBUFAddr64 : ComplexPattern<i64, 7, "SelectMUBUFAddr64">;
def MUBUFAddr64Atomic : ComplexPattern<i64, 5, "SelectMUBUFAddr64">;
def FLATAtomic : ComplexPattern<i64, 3, "SelectFlat">;
def MUBUFScratch : ComplexPattern<i64, 4, "SelectMUBUFScratch">;
def MUBUFOffset : ComplexPattern<i64, 6, "SelectMUBUFOffset">;
def MUBUFOffsetNoGLC : ComplexPattern<i64, 3, "SelectMUBUFOffset">;
def MUBUFOffsetAtomic : ComplexPattern<i64, 4, "SelectMUBUFOffset">;
def MUBUFIntrinsicOffset : ComplexPattern<i32, 2, "SelectMUBUFIntrinsicOffset">;
def MUBUFIntrinsicVOffset : ComplexPattern<i32, 3, "SelectMUBUFIntrinsicVOffset">;

def SMRDImm   : ComplexPattern<i64, 2, "SelectSMRDImm">;
def SMRDImm32 : ComplexPattern<i64, 2, "SelectSMRDImm32">;
def SMRDSgpr  : ComplexPattern<i64, 2, "SelectSMRDSgpr">;
def SMRDBufferImm   : ComplexPattern<i32, 1, "SelectSMRDBufferImm">;
def SMRDBufferImm32 : ComplexPattern<i32, 1, "SelectSMRDBufferImm32">;
def SMRDBufferSgpr  : ComplexPattern<i32, 1, "SelectSMRDBufferSgpr">;

def MOVRELOffset : ComplexPattern<i32, 2, "SelectMOVRELOffset">;

def VOP3Mods0 : ComplexPattern<untyped, 4, "SelectVOP3Mods0">;
def VOP3NoMods0 : ComplexPattern<untyped, 4, "SelectVOP3NoMods0">;
def VOP3Mods0Clamp : ComplexPattern<untyped, 3, "SelectVOP3Mods0Clamp">;
def VOP3Mods0Clamp0OMod : ComplexPattern<untyped, 4, "SelectVOP3Mods0Clamp0OMod">;
def VOP3Mods  : ComplexPattern<untyped, 2, "SelectVOP3Mods">;
def VOP3NoMods : ComplexPattern<untyped, 2, "SelectVOP3NoMods">;

//===----------------------------------------------------------------------===//
// SI assembler operands
//===----------------------------------------------------------------------===//

def SIOperand {
  int ZERO = 0x80;
  int VCC = 0x6A;
  int FLAT_SCR = 0x68;
}

def SRCMODS {
  int NONE = 0;
  int NEG = 1;
}

def DSTCLAMP {
  int NONE = 0;
}

def DSTOMOD {
  int NONE = 0;
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

class EXPCommon : InstSI<
  (outs),
  (ins i32imm:$en, i32imm:$tgt, i32imm:$compr, i32imm:$done, i32imm:$vm,
       VGPR_32:$src0, VGPR_32:$src1, VGPR_32:$src2, VGPR_32:$src3),
  "exp $en, $tgt, $compr, $done, $vm, $src0, $src1, $src2, $src3",
  [] > {

  let EXP_CNT = 1;
  let Uses = [EXEC];
  let SchedRW = [WriteExport];
}

multiclass EXP_m {

  let isPseudo = 1, isCodeGenOnly = 1 in {
    def "" : EXPCommon, SIMCInstr <"exp", SIEncodingFamily.NONE> ;
  }

  def _si : EXPCommon, SIMCInstr <"exp", SIEncodingFamily.SI>, EXPe {
    let DecoderNamespace="SICI";
    let DisableDecoder = DisableSIDecoder;
  }

  def _vi : EXPCommon, SIMCInstr <"exp", SIEncodingFamily.VI>, EXPe_vi {
    let DecoderNamespace="VI";
    let DisableDecoder = DisableVIDecoder;
  }
}

//===----------------------------------------------------------------------===//
// Scalar classes
//===----------------------------------------------------------------------===//

class SOP1_Pseudo <string opName, dag outs, dag ins, list<dag> pattern> :
  SOP1 <outs, ins, "", pattern>,
  SIMCInstr<opName, SIEncodingFamily.NONE> {
  let isPseudo = 1;
  let isCodeGenOnly = 1;
}

class SOP1_Real_si <sop1 op, string opName, dag outs, dag ins, string asm> :
  SOP1 <outs, ins, asm, []>,
  SOP1e <op.SI>,
  SIMCInstr<opName, SIEncodingFamily.SI> {
  let isCodeGenOnly = 0;
  let AssemblerPredicates = [isSICI];
  let DecoderNamespace = "SICI";
  let DisableDecoder = DisableSIDecoder;
}

class SOP1_Real_vi <sop1 op, string opName, dag outs, dag ins, string asm> :
  SOP1 <outs, ins, asm, []>,
  SOP1e <op.VI>,
  SIMCInstr<opName, SIEncodingFamily.VI> {
  let isCodeGenOnly = 0;
  let AssemblerPredicates = [isVI];
  let DecoderNamespace = "VI";
  let DisableDecoder = DisableVIDecoder;
}

multiclass SOP1_m <sop1 op, string opName, dag outs, dag ins, string asm,
                   list<dag> pattern> {

  def "" : SOP1_Pseudo <opName, outs, ins, pattern>;

  def _si : SOP1_Real_si <op, opName, outs, ins, asm>;

  def _vi : SOP1_Real_vi <op, opName, outs, ins, asm>;

}

multiclass SOP1_32 <sop1 op, string opName, list<dag> pattern> : SOP1_m <
    op, opName, (outs SReg_32:$sdst), (ins SSrc_32:$src0),
    opName#" $sdst, $src0", pattern
>;

multiclass SOP1_64 <sop1 op, string opName, list<dag> pattern> : SOP1_m <
    op, opName, (outs SReg_64:$sdst), (ins SSrc_64:$src0),
    opName#" $sdst, $src0", pattern
>;

// no input, 64-bit output.
multiclass SOP1_64_0 <sop1 op, string opName, list<dag> pattern> {
  def "" : SOP1_Pseudo <opName, (outs SReg_64:$sdst), (ins), pattern>;

  def _si : SOP1_Real_si <op, opName, (outs SReg_64:$sdst), (ins),
    opName#" $sdst"> {
    let src0 = 0;
  }

  def _vi : SOP1_Real_vi <op, opName, (outs SReg_64:$sdst), (ins),
    opName#" $sdst"> {
    let src0 = 0;
  }
}

// 64-bit input, no output
multiclass SOP1_1 <sop1 op, string opName, list<dag> pattern> {
  def "" : SOP1_Pseudo <opName, (outs), (ins SReg_64:$src0), pattern>;

  def _si : SOP1_Real_si <op, opName, (outs), (ins SReg_64:$src0),
    opName#" $src0"> {
    let sdst = 0;
  }

  def _vi : SOP1_Real_vi <op, opName, (outs), (ins SReg_64:$src0),
    opName#" $src0"> {
    let sdst = 0;
  }
}

// 64-bit input, 32-bit output.
multiclass SOP1_32_64 <sop1 op, string opName, list<dag> pattern> : SOP1_m <
    op, opName, (outs SReg_32:$sdst), (ins SSrc_64:$src0),
    opName#" $sdst, $src0", pattern
>;

// 32-bit input, 64-bit output.
multiclass SOP1_64_32 <sop1 op, string opName, list<dag> pattern> : SOP1_m <
    op, opName, (outs SReg_64:$sdst), (ins SSrc_32:$src0),
    opName#" $sdst, $src0", pattern
>;

class SOP2_Pseudo<string opName, dag outs, dag ins, list<dag> pattern> :
  SOP2<outs, ins, "", pattern>,
  SIMCInstr<opName, SIEncodingFamily.NONE> {
  let isPseudo = 1;
  let isCodeGenOnly = 1;
  let Size = 4;

  // Pseudo instructions have no encodings, but adding this field here allows
  // us to do:
  // let sdst = xxx in {
  // for multiclasses that include both real and pseudo instructions.
  field bits<7> sdst = 0;
}

class SOP2_Real_si<sop2 op, string opName, dag outs, dag ins, string asm> :
  SOP2<outs, ins, asm, []>,
  SOP2e<op.SI>,
  SIMCInstr<opName, SIEncodingFamily.SI> {
  let AssemblerPredicates = [isSICI];
  let DecoderNamespace = "SICI";
  let DisableDecoder = DisableSIDecoder;
}

class SOP2_Real_vi<sop2 op, string opName, dag outs, dag ins, string asm> :
  SOP2<outs, ins, asm, []>,
  SOP2e<op.VI>,
  SIMCInstr<opName, SIEncodingFamily.VI> {
  let AssemblerPredicates = [isVI];
  let DecoderNamespace = "VI";
  let DisableDecoder = DisableVIDecoder;
}

multiclass SOP2_m <sop2 op, string opName, dag outs, dag ins, string asm,
                   list<dag> pattern> {

  def "" : SOP2_Pseudo <opName, outs, ins, pattern>;

  def _si : SOP2_Real_si <op, opName, outs, ins, asm>;

  def _vi : SOP2_Real_vi <op, opName, outs, ins, asm>;

}

multiclass SOP2_32 <sop2 op, string opName, list<dag> pattern> : SOP2_m <
    op, opName, (outs SReg_32:$sdst), (ins SSrc_32:$src0, SSrc_32:$src1),
    opName#" $sdst, $src0, $src1", pattern
>;

multiclass SOP2_64 <sop2 op, string opName, list<dag> pattern> : SOP2_m <
    op, opName, (outs SReg_64:$sdst), (ins SSrc_64:$src0, SSrc_64:$src1),
    opName#" $sdst, $src0, $src1", pattern
>;

multiclass SOP2_64_32 <sop2 op, string opName, list<dag> pattern> : SOP2_m <
    op, opName, (outs SReg_64:$sdst), (ins SSrc_64:$src0, SSrc_32:$src1),
    opName#" $sdst, $src0, $src1", pattern
>;

multiclass SOP2_64_32_32 <sop2 op, string opName, list<dag> pattern> : SOP2_m <
    op, opName, (outs SReg_64:$sdst), (ins SSrc_32:$src0, SSrc_32:$src1),
    opName#" $sdst, $src0, $src1", pattern
>;

class SOPC_Base <bits<7> op, RegisterOperand rc0, RegisterOperand rc1,
                 string opName, list<dag> pattern = []> : SOPC <
  op, (outs), (ins rc0:$src0, rc1:$src1),
  opName#" $src0, $src1", pattern > {
  let Defs = [SCC];
}
class SOPC_Helper <bits<7> op, RegisterOperand rc, ValueType vt,
                    string opName, PatLeaf cond> : SOPC_Base <
  op, rc, rc, opName,
  [(set SCC, (si_setcc_uniform vt:$src0, vt:$src1, cond))] > {
}

class SOPC_CMP_32<bits<7> op, string opName, PatLeaf cond = COND_NULL>
  : SOPC_Helper<op, SSrc_32, i32, opName, cond>;

class SOPC_32<bits<7> op, string opName, list<dag> pattern = []>
  : SOPC_Base<op, SSrc_32, SSrc_32, opName, pattern>;

class SOPC_64_32<bits<7> op, string opName, list<dag> pattern = []>
  : SOPC_Base<op, SSrc_64, SSrc_32, opName, pattern>;

class SOPK_Pseudo <string opName, dag outs, dag ins, list<dag> pattern> :
  SOPK <outs, ins, "", pattern>,
  SIMCInstr<opName, SIEncodingFamily.NONE> {
  let isPseudo = 1;
  let isCodeGenOnly = 1;
}

class SOPK_Real_si <sopk op, string opName, dag outs, dag ins, string asm> :
  SOPK <outs, ins, asm, []>,
  SOPKe <op.SI>,
  SIMCInstr<opName, SIEncodingFamily.SI> {
  let AssemblerPredicates = [isSICI];
  let DecoderNamespace = "SICI";
  let DisableDecoder = DisableSIDecoder;
  let isCodeGenOnly = 0;
}

class SOPK_Real_vi <sopk op, string opName, dag outs, dag ins, string asm> :
  SOPK <outs, ins, asm, []>,
  SOPKe <op.VI>,
  SIMCInstr<opName, SIEncodingFamily.VI> {
  let AssemblerPredicates = [isVI];
  let DecoderNamespace = "VI";
  let DisableDecoder = DisableVIDecoder;
  let isCodeGenOnly = 0;
}

multiclass SOPK_m <sopk op, string opName, dag outs, dag ins, string opAsm,
                   string asm = opName#opAsm> {
  def "" : SOPK_Pseudo <opName, outs, ins, []>;

  def _si : SOPK_Real_si <op, opName, outs, ins, asm>;

  def _vi : SOPK_Real_vi <op, opName, outs, ins, asm>;

}

multiclass SOPK_32 <sopk op, string opName, list<dag> pattern> {
  def "" : SOPK_Pseudo <opName, (outs SReg_32:$sdst), (ins u16imm:$simm16),
    pattern>;

  def _si : SOPK_Real_si <op, opName, (outs SReg_32:$sdst), (ins u16imm:$simm16),
    opName#" $sdst, $simm16">;

  def _vi : SOPK_Real_vi <op, opName, (outs SReg_32:$sdst), (ins u16imm:$simm16),
    opName#" $sdst, $simm16">;
}

multiclass SOPK_SCC <sopk op, string opName, list<dag> pattern> {
  def "" : SOPK_Pseudo <opName, (outs),
    (ins SReg_32:$src0, u16imm:$src1), pattern> {
    let Defs = [SCC];
  }


  def _si : SOPK_Real_si <op, opName, (outs),
    (ins SReg_32:$sdst, u16imm:$simm16), opName#" $sdst, $simm16"> {
    let Defs = [SCC];
  }

  def _vi : SOPK_Real_vi <op, opName, (outs),
    (ins SReg_32:$sdst, u16imm:$simm16), opName#" $sdst, $simm16"> {
    let Defs = [SCC];
  }
}

multiclass SOPK_32TIE <sopk op, string opName, list<dag> pattern> : SOPK_m <
  op, opName, (outs SReg_32:$sdst), (ins SReg_32:$src0, u16imm:$simm16),
  " $sdst, $simm16"
>;

multiclass SOPK_IMM32 <sopk op, string opName, dag outs, dag ins,
                       string argAsm, string asm = opName#argAsm> {

  def "" : SOPK_Pseudo <opName, outs, ins, []>;

  def _si : SOPK <outs, ins, asm, []>,
            SOPK64e <op.SI>,
            SIMCInstr<opName, SIEncodingFamily.SI> {
              let AssemblerPredicates = [isSICI];
              let DecoderNamespace = "SICI";
              let DisableDecoder = DisableSIDecoder;
              let isCodeGenOnly = 0;
            }

  def _vi : SOPK <outs, ins, asm, []>,
            SOPK64e <op.VI>,
            SIMCInstr<opName, SIEncodingFamily.VI> {
              let AssemblerPredicates = [isVI];
              let DecoderNamespace = "VI";
              let DisableDecoder = DisableVIDecoder;
              let isCodeGenOnly = 0;
            }
}
//===----------------------------------------------------------------------===//
// SMRD classes
//===----------------------------------------------------------------------===//

class SMRD_Pseudo <string opName, dag outs, dag ins, list<dag> pattern> :
  SMRD <outs, ins, "", pattern>,
  SIMCInstr<opName, SIEncodingFamily.NONE> {
  let isPseudo = 1;
  let isCodeGenOnly = 1;
}

class SMRD_IMM_Real_si <bits<5> op, string opName, dag outs, dag ins,
                        string asm> :
  SMRD <outs, ins, asm, []>,
  SMRD_IMMe <op>,
  SIMCInstr<opName, SIEncodingFamily.SI> {
  let AssemblerPredicates = [isSICI];
  let DecoderNamespace = "SICI";
  let DisableDecoder = DisableSIDecoder;
}

class SMRD_SOFF_Real_si <bits<5> op, string opName, dag outs, dag ins,
                         string asm> :
  SMRD <outs, ins, asm, []>,
  SMRD_SOFFe <op>,
  SIMCInstr<opName, SIEncodingFamily.SI> {
  let AssemblerPredicates = [isSICI];
  let DecoderNamespace = "SICI";
  let DisableDecoder = DisableSIDecoder;
}


class SMRD_IMM_Real_vi <bits<8> op, string opName, dag outs, dag ins,
                        string asm, list<dag> pattern = []> :
  SMRD <outs, ins, asm, pattern>,
  SMEM_IMMe_vi <op>,
  SIMCInstr<opName, SIEncodingFamily.VI> {
  let AssemblerPredicates = [isVI];
  let DecoderNamespace = "VI";
  let DisableDecoder = DisableVIDecoder;
}

class SMRD_SOFF_Real_vi <bits<8> op, string opName, dag outs, dag ins,
                         string asm, list<dag> pattern = []> :
  SMRD <outs, ins, asm, pattern>,
  SMEM_SOFFe_vi <op>,
  SIMCInstr<opName, SIEncodingFamily.VI> {
  let AssemblerPredicates = [isVI];
  let DecoderNamespace = "VI";
  let DisableDecoder = DisableVIDecoder;
}


multiclass SMRD_IMM_m <smrd op, string opName, dag outs, dag ins,
                   string asm, list<dag> pattern> {

  def "" : SMRD_Pseudo <opName, outs, ins, pattern>;

  def _si : SMRD_IMM_Real_si <op.SI, opName, outs, ins, asm>;

  // glc is only applicable to scalar stores, which are not yet
  // implemented.
  let glc = 0 in {
    def _vi : SMRD_IMM_Real_vi <op.VI, opName, outs, ins, asm>;
  }
}

multiclass SMRD_SOFF_m <smrd op, string opName, dag outs, dag ins,
                        string asm, list<dag> pattern> {

  def "" : SMRD_Pseudo <opName, outs, ins, pattern>;

  def _si : SMRD_SOFF_Real_si <op.SI, opName, outs, ins, asm>;

  // glc is only applicable to scalar stores, which are not yet
  // implemented.
  let glc = 0 in {
    def _vi : SMRD_SOFF_Real_vi <op.VI, opName, outs, ins, asm>;
  }
}

multiclass SMRD_Special <smrd op, string opName, dag outs,
                       int sdst_ = ?,
                       string opStr = "",
                       list<dag> pattern = []> {
  let hasSideEffects = 1 in {
    def "" : SMRD_Pseudo <opName, outs, (ins), pattern>;

    let sbase = 0, soff = 0, sdst = sdst_ in {
      def _si : SMRD_SOFF_Real_si <op.SI, opName, outs, (ins), opName#opStr>;

      let glc = 0 in {
        def _vi : SMRD_SOFF_Real_vi <op.VI, opName, outs, (ins), opName#opStr>;
      }
    }
  }
}

multiclass SMRD_Inval <smrd op, string opName,
                     SDPatternOperator node> {
  let mayStore = 1 in {
    defm : SMRD_Special<op, opName, (outs), 0, "", [(node)]>;
  }
}

class SMEM_Inval <bits<8> op, string opName, SDPatternOperator node> :
  SMRD_SOFF_Real_vi<op, opName, (outs), (ins), opName, [(node)]> {
  let hasSideEffects = 1;
  let mayStore = 1;
  let sbase = 0;
  let sdst = 0;
  let glc = 0;
  let soff = 0;
}

class SMEM_Ret <bits<8> op, string opName, SDPatternOperator node> :
  SMRD_SOFF_Real_vi<op, opName, (outs SReg_64:$sdst), (ins),
  opName#" $sdst", [(set i64:$sdst, (node))]> {
  let hasSideEffects = 1;
  let mayStore = ?;
  let mayLoad = ?;
  let sbase = 0;
  let glc = 0;
  let soff = 0;
}

multiclass SMRD_Helper <smrd op, string opName, RegisterClass baseClass,
                        RegisterClass dstClass> {
  defm _IMM : SMRD_IMM_m <
    op, opName#"_IMM", (outs dstClass:$sdst),
    (ins baseClass:$sbase, smrd_offset:$offset),
    opName#" $sdst, $sbase, $offset", []
  >;

  def _IMM_ci : SMRD <
    (outs dstClass:$sdst), (ins baseClass:$sbase, smrd_literal_offset:$offset),
    opName#" $sdst, $sbase, $offset", []>, SMRD_IMMe_ci <op.SI> {
    let AssemblerPredicates = [isCIOnly];
    let DecoderNamespace = "CI";
  }

  defm _SGPR : SMRD_SOFF_m <
    op, opName#"_SGPR", (outs dstClass:$sdst),
    (ins baseClass:$sbase, SReg_32:$soff),
    opName#" $sdst, $sbase, $soff", []
  >;
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
                          !if(!eq(VT.Size, 64), VOPDstOperand<VReg_64>,
                            !if(!eq(VT.Size, 16), VOPDstOperand<VGPR_32>,
                            VOPDstOperand<SReg_64>))); // else VT == i1
}

// Returns the register class to use for source 0 of VOP[12C]
// instructions for the given VT.
class getVOPSrc0ForVT<ValueType VT> {
  RegisterOperand ret = !if(!eq(VT.Size, 64), VSrc_64, VSrc_32);
}

// Returns the vreg register class to use for source operand given VT
class getVregSrcForVT<ValueType VT> {
  RegisterClass ret = !if(!eq(VT.Size, 64), VReg_64, VGPR_32);
}


// Returns the register class to use for sources of VOP3 instructions for the
// given VT.
class getVOP3SrcForVT<ValueType VT> {
  RegisterOperand ret =
  !if(!eq(VT.Size, 64),
      VCSrc_64,
      !if(!eq(VT.Value, i1.Value),
          SCSrc_64,
          VCSrc_32
       )
    );
}

// Returns 1 if the source arguments have modifiers, 0 if they do not.
// XXX - do f16 instructions?
class hasModifiers<ValueType SrcVT> {
  bit ret =
    !if(!eq(SrcVT.Value, f32.Value), 1,
    !if(!eq(SrcVT.Value, f64.Value), 1,
    0));
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
                bit HasModifiers> {

  dag ret =
    !if (!eq(NumSrcArgs, 0),
      // VOP1 without input operands (V_NOP, V_CLREXCP)
      (ins),
      /* else */
    !if (!eq(NumSrcArgs, 1),
      !if (!eq(HasModifiers, 1),
        // VOP1 with modifiers
        (ins FPInputMods:$src0_modifiers, Src0RC:$src0,
             clampmod:$clamp, omod:$omod)
      /* else */,
        // VOP1 without modifiers
        (ins Src0RC:$src0)
      /* endif */ ),
    !if (!eq(NumSrcArgs, 2),
      !if (!eq(HasModifiers, 1),
        // VOP 2 with modifiers
        (ins FPInputMods:$src0_modifiers, Src0RC:$src0,
             FPInputMods:$src1_modifiers, Src1RC:$src1,
             clampmod:$clamp, omod:$omod)
      /* else */,
        // VOP2 without modifiers
        (ins Src0RC:$src0, Src1RC:$src1)
      /* endif */ )
    /* NumSrcArgs == 3 */,
      !if (!eq(HasModifiers, 1),
        // VOP3 with modifiers
        (ins FPInputMods:$src0_modifiers, Src0RC:$src0,
             FPInputMods:$src1_modifiers, Src1RC:$src1,
             FPInputMods:$src2_modifiers, Src2RC:$src2,
             clampmod:$clamp, omod:$omod)
      /* else */,
        // VOP3 without modifiers
        (ins Src0RC:$src0, Src1RC:$src1, Src2RC:$src2)
      /* endif */ ))));
}

class getInsDPP <RegisterClass Src0RC, RegisterClass Src1RC, int NumSrcArgs,
                                                             bit HasModifiers> {

  dag ret = !if (!eq(NumSrcArgs, 0),
                // VOP1 without input operands (V_NOP)
                (ins dpp_ctrl:$dpp_ctrl, row_mask:$row_mask,
                     bank_mask:$bank_mask, bound_ctrl:$bound_ctrl),
            !if (!eq(NumSrcArgs, 1),
              !if (!eq(HasModifiers, 1),
                // VOP1_DPP with modifiers
                (ins FPInputMods:$src0_modifiers, Src0RC:$src0,
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
                (ins FPInputMods:$src0_modifiers, Src0RC:$src0,
                     FPInputMods:$src1_modifiers, Src1RC:$src1,
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
                  bit HasFloatModifiers, ValueType DstVT> {

  dag ret = !if(!eq(NumSrcArgs, 0),
               // VOP1 without input operands (V_NOP)
               (ins),
            !if(!eq(NumSrcArgs, 1),
                !if(HasFloatModifiers,
                    // VOP1_SDWA with float modifiers
                    (ins FPInputMods:$src0_fmodifiers, Src0RC:$src0,
                         clampmod:$clamp, dst_sel:$dst_sel, dst_unused:$dst_unused,
                         src0_sel:$src0_sel)
                /* else */,
                    // VOP1_SDWA with sext modifier
                    (ins IntInputMods:$src0_imodifiers, Src0RC:$src0,
                         clampmod:$clamp, dst_sel:$dst_sel, dst_unused:$dst_unused,
                         src0_sel:$src0_sel)
                /* endif */)
              /* NumSrcArgs == 2 */,
              !if(HasFloatModifiers,
                  !if(!eq(DstVT.Size, 1),
                      // VOPC_SDWA with float modifiers
                      (ins FPInputMods:$src0_fmodifiers, Src0RC:$src0,
                           FPInputMods:$src1_fmodifiers, Src1RC:$src1,
                           clampmod:$clamp, src0_sel:$src0_sel, src1_sel:$src1_sel),
                      // VOP2_SDWA or VOPC_SDWA with float modifiers
                      (ins FPInputMods:$src0_fmodifiers, Src0RC:$src0,
                           FPInputMods:$src1_fmodifiers, Src1RC:$src1,
                           clampmod:$clamp, dst_sel:$dst_sel, dst_unused:$dst_unused,
                           src0_sel:$src0_sel, src1_sel:$src1_sel)
                  ),
              /* else */
                !if(!eq(DstVT.Size, 1),
                    // VOPC_SDWA with sext modifiers
                    (ins IntInputMods:$src0_imodifiers, Src0RC:$src0,
                         IntInputMods:$src1_imodifiers, Src1RC:$src1,
                         clampmod:$clamp, src0_sel:$src0_sel, src1_sel:$src1_sel),
                    // VOP2_SDWA or VOPC_SDWA with sext modifier
                    (ins IntInputMods:$src0_imodifiers, Src0RC:$src0,
                         IntInputMods:$src1_imodifiers, Src1RC:$src1,
                         clampmod:$clamp, dst_sel:$dst_sel, dst_unused:$dst_unused,
                         src0_sel:$src0_sel, src1_sel:$src1_sel)
                )
             /* endif */)));
}

// Outs for DPP and SDWA
class getOutsExt <bit HasDst, ValueType DstVT, RegisterOperand DstRCDPP> {
  dag ret = !if(HasDst,
                !if(!eq(DstVT.Size, 1),
                    (outs), // no dst for VOPC, we use "vcc"-token as dst in SDWA VOPC instructions
                    (outs DstRCDPP:$vdst)),
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
class getAsm64 <bit HasDst, int NumSrcArgs, bit HasModifiers, ValueType DstVT = i32> {
  string dst = !if(!eq(DstVT.Size, 1), "$sdst", "$vdst"); // use $sdst for VOPC
  string src0 = !if(!eq(NumSrcArgs, 1), "$src0_modifiers", "$src0_modifiers,");
  string src1 = !if(!eq(NumSrcArgs, 1), "",
                   !if(!eq(NumSrcArgs, 2), " $src1_modifiers",
                                           " $src1_modifiers,"));
  string src2 = !if(!eq(NumSrcArgs, 3), " $src2_modifiers", "");
  string ret =
  !if(!eq(HasModifiers, 0),
      getAsm32<HasDst, NumSrcArgs, DstVT>.ret,
      dst#", "#src0#src1#src2#"$clamp"#"$omod");
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

class getAsmSDWA <bit HasDst, int NumSrcArgs, bit HasFloatModifiers,
                  ValueType DstVT = i32> {
  string dst = !if(HasDst,
                   !if(!eq(DstVT.Size, 1),
                       " vcc", // use vcc token as dst for VOPC instructioins
                       "$vdst"),
                    "");
  string src0 = !if(HasFloatModifiers, "$src0_fmodifiers", "$src0_imodifiers");
  string src1 = !if(HasFloatModifiers, "$src1_fmodifiers", "$src1_imodifiers");
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

class VOPProfile <list<ValueType> _ArgVT> {

  field list<ValueType> ArgVT = _ArgVT;

  field ValueType DstVT = ArgVT[0];
  field ValueType Src0VT = ArgVT[1];
  field ValueType Src1VT = ArgVT[2];
  field ValueType Src2VT = ArgVT[3];
  field RegisterOperand DstRC = getVALUDstForVT<DstVT>.ret;
  field RegisterOperand DstRCDPP = getVALUDstForVT<DstVT>.ret;
  field RegisterOperand DstRCSDWA = getVALUDstForVT<DstVT>.ret;
  field RegisterOperand Src0RC32 = getVOPSrc0ForVT<Src0VT>.ret;
  field RegisterClass Src1RC32 = getVregSrcForVT<Src1VT>.ret;
  field RegisterOperand Src0RC64 = getVOP3SrcForVT<Src0VT>.ret;
  field RegisterOperand Src1RC64 = getVOP3SrcForVT<Src1VT>.ret;
  field RegisterOperand Src2RC64 = getVOP3SrcForVT<Src2VT>.ret;
  field RegisterClass Src0DPP = getVregSrcForVT<Src0VT>.ret;
  field RegisterClass Src1DPP = getVregSrcForVT<Src1VT>.ret;
  field RegisterClass Src0SDWA = getVregSrcForVT<Src0VT>.ret;
  field RegisterClass Src1SDWA = getVregSrcForVT<Src1VT>.ret;

  field bit HasDst = !if(!eq(DstVT.Value, untyped.Value), 0, 1);
  field bit HasDst32 = HasDst;
  field int NumSrcArgs = getNumSrcArgs<Src0VT, Src1VT, Src2VT>.ret;
  field bit HasModifiers = hasModifiers<Src0VT>.ret;

  field bit HasExt = getHasExt<NumSrcArgs, DstVT, Src0VT, Src1VT>.ret;

  field dag Outs = !if(HasDst,(outs DstRC:$vdst),(outs));

  // VOP3b instructions are a special case with a second explicit
  // output. This is manually overridden for them.
  field dag Outs32 = Outs;
  field dag Outs64 = Outs;
  field dag OutsDPP = getOutsExt<HasDst, DstVT, DstRCDPP>.ret;
  field dag OutsSDWA = getOutsExt<HasDst, DstVT, DstRCDPP>.ret;

  field dag Ins32 = getIns32<Src0RC32, Src1RC32, NumSrcArgs>.ret;
  field dag Ins64 = getIns64<Src0RC64, Src1RC64, Src2RC64, NumSrcArgs,
                             HasModifiers>.ret;
  field dag InsDPP = getInsDPP<Src0DPP, Src1DPP, NumSrcArgs, HasModifiers>.ret;
  field dag InsSDWA = getInsSDWA<Src0SDWA, Src1SDWA, NumSrcArgs, HasModifiers, DstVT>.ret;

  field string Asm32 = getAsm32<HasDst, NumSrcArgs, DstVT>.ret;
  field string Asm64 = getAsm64<HasDst, NumSrcArgs, HasModifiers, DstVT>.ret;
  field string AsmDPP = getAsmDPP<HasDst, NumSrcArgs, HasModifiers, DstVT>.ret;
  field string AsmSDWA = getAsmSDWA<HasDst, NumSrcArgs, HasModifiers, DstVT>.ret;
}

class VOP_NO_EXT <VOPProfile p> : VOPProfile <p.ArgVT> {
  let HasExt = 0;
}

// FIXME: I think these F16/I16 profiles will need to use f16/i16 types in order
//        for the instruction patterns to work.
def VOP_F16_F16 : VOPProfile <[f16, f16, untyped, untyped]>;
def VOP_F16_I16 : VOPProfile <[f16, i32, untyped, untyped]>;
def VOP_I16_F16 : VOPProfile <[i32, f16, untyped, untyped]>;

def VOP_F16_F16_F16 : VOPProfile <[f16, f16, f16, untyped]>;
def VOP_F16_F16_I16 : VOPProfile <[f16, f16, i32, untyped]>;
def VOP_I16_I16_I16 : VOPProfile <[i32, i32, i32, untyped]>;

def VOP_I16_I16_I16_I16 : VOPProfile <[i32, i32, i32, i32, untyped]>;
def VOP_F16_F16_F16_F16 : VOPProfile <[f16, f16, f16, f16, untyped]>;

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

def VOP_F32_F32_F32 : VOPProfile <[f32, f32, f32, untyped]>;
def VOP_F32_F32_I32 : VOPProfile <[f32, f32, i32, untyped]>;
def VOP_F64_F64_F64 : VOPProfile <[f64, f64, f64, untyped]>;
def VOP_F64_F64_I32 : VOPProfile <[f64, f64, i32, untyped]>;
def VOP_I32_F32_F32 : VOPProfile <[i32, f32, f32, untyped]>;
def VOP_I32_F32_I32 : VOPProfile <[i32, f32, i32, untyped]>;
def VOP_I32_I32_I32 : VOPProfile <[i32, i32, i32, untyped]>;

// Restrict src0 to be VGPR
def VOP_I32_VI32_NO_EXT : VOPProfile<[i32, i32, untyped, untyped]> {
  let Src0RC32 = VRegSrc_32;
  let Src0RC64 = VRegSrc_32;

  let HasExt = 0;
}

// Special case because there are no true output operands.  Hack vdst
// to be a src operand. The custom inserter must add a tied implicit
// def and use of the super register since there seems to be no way to
// add an implicit def of a virtual register in tablegen.
def VOP_MOVRELD : VOPProfile<[untyped, i32, untyped, untyped]> {
  let Src0RC32 = VOPDstOperand<VGPR_32>;
  let Src0RC64 = VOPDstOperand<VGPR_32>;

  let Outs = (outs);
  let Ins32 = (ins Src0RC32:$vdst, VSrc_32:$src0);
  let Ins64 = (ins Src0RC64:$vdst, VSrc_32:$src0);

  let InsDPP = (ins Src0RC32:$vdst, Src0RC32:$src0, dpp_ctrl:$dpp_ctrl, row_mask:$row_mask,
                    bank_mask:$bank_mask, bound_ctrl:$bound_ctrl);
  let InsSDWA = (ins Src0RC32:$vdst, IntInputMods:$src0_imodifiers, VCSrc_32:$src0,
                     clampmod:$clamp, dst_sel:$dst_sel, dst_unused:$dst_unused,
                     src0_sel:$src0_sel);

  let Asm32 = getAsm32<1, 1>.ret;
  let Asm64 = getAsm64<1, 1, 0>.ret;
  let AsmDPP = getAsmDPP<1, 1, 0>.ret;
  let AsmSDWA = getAsmSDWA<1, 1, 0>.ret;

  let HasExt = 0;
  let HasDst = 0;
}

// Write out to vcc or arbitrary SGPR.
def VOP2b_I32_I1_I32_I32 : VOPProfile<[i32, i32, i32, untyped]> {
  let Asm32 = "$vdst, vcc, $src0, $src1";
  let Asm64 = "$vdst, $sdst, $src0, $src1";
  let Outs32 = (outs DstRC:$vdst);
  let Outs64 = (outs DstRC:$vdst, SReg_64:$sdst);
}

// Write out to vcc or arbitrary SGPR and read in from vcc or
// arbitrary SGPR.
def VOP2b_I32_I1_I32_I32_I1 : VOPProfile<[i32, i32, i32, i1]> {
  // We use VCSrc_32 to exclude literal constants, even though the
  // encoding normally allows them since the implicit VCC use means
  // using one would always violate the constant bus
  // restriction. SGPRs are still allowed because it should
  // technically be possible to use VCC again as src0.
  let Src0RC32 = VCSrc_32;
  let Asm32 = "$vdst, vcc, $src0, $src1, vcc";
  let Asm64 = "$vdst, $sdst, $src0, $src1, $src2";
  let Outs32 = (outs DstRC:$vdst);
  let Outs64 = (outs DstRC:$vdst, SReg_64:$sdst);

  // Suppress src2 implied by type since the 32-bit encoding uses an
  // implicit VCC use.
  let Ins32 = (ins Src0RC32:$src0, Src1RC32:$src1);
}

// Read in from vcc or arbitrary SGPR
def VOP2e_I32_I32_I32_I1 : VOPProfile<[i32, i32, i32, i1]> {
  let Src0RC32 = VCSrc_32; // See comment in def VOP2b_I32_I1_I32_I32_I1 above.
  let Asm32 = "$vdst, $src0, $src1, vcc";
  let Asm64 = "$vdst, $src0, $src1, $src2";
  let Outs32 = (outs DstRC:$vdst);
  let Outs64 = (outs DstRC:$vdst);

  // Suppress src2 implied by type since the 32-bit encoding uses an
  // implicit VCC use.
  let Ins32 = (ins Src0RC32:$src0, Src1RC32:$src1);
}

class VOP3b_Profile<ValueType vt> : VOPProfile<[vt, vt, vt, vt]> {
  let Outs64 = (outs DstRC:$vdst, SReg_64:$sdst);
  let Asm64 = "$vdst, $sdst, $src0_modifiers, $src1_modifiers, $src2_modifiers"#"$clamp"#"$omod";
}

def VOP3b_F32_I1_F32_F32_F32 : VOP3b_Profile<f32> {
  // FIXME: Hack to stop printing _e64
  let DstRC = RegisterOperand<VGPR_32>;
}

def VOP3b_F64_I1_F64_F64_F64 : VOP3b_Profile<f64> {
  // FIXME: Hack to stop printing _e64
  let DstRC = RegisterOperand<VReg_64>;
}

// VOPC instructions are a special case because for the 32-bit
// encoding, we want to display the implicit vcc write as if it were
// an explicit $dst.
class VOPC_Profile<ValueType vt0, ValueType vt1 = vt0> : VOPProfile <[i1, vt0, vt1, untyped]> {
  let Asm32 = "vcc, $src0, $src1";
  // The destination for 32-bit encoding is implicit.
  let HasDst32 = 0;
  let Outs64 = (outs DstRC:$sdst);
}

class VOPC_Class_Profile<ValueType vt> : VOPC_Profile<vt, i32> {
  let Ins64 = (ins FPInputMods:$src0_modifiers, Src0RC64:$src0, Src1RC64:$src1);
  let Asm64 = "$sdst, $src0_modifiers, $src1";
  let InsSDWA = (ins FPInputMods:$src0_fmodifiers, Src0RC64:$src0,
                     IntInputMods:$src1_imodifiers, Src1RC64:$src1,
                     clampmod:$clamp, src0_sel:$src0_sel, src1_sel:$src1_sel);
  let AsmSDWA = " vcc, $src0_fmodifiers, $src1_imodifiers$clamp $src0_sel $src1_sel";

}

def VOPC_I1_F32_F32 : VOPC_Profile<f32>;
def VOPC_I1_F64_F64 : VOPC_Profile<f64>;
def VOPC_I1_I32_I32 : VOPC_Profile<i32>;
def VOPC_I1_I64_I64 : VOPC_Profile<i64>;

def VOPC_I1_F32_I32 : VOPC_Class_Profile<f32>;
def VOPC_I1_F64_I32 : VOPC_Class_Profile<f64>;

def VOP_I64_I64_I32 : VOPProfile <[i64, i64, i32, untyped]>;
def VOP_I64_I32_I64 : VOPProfile <[i64, i32, i64, untyped]>;
def VOP_I64_I64_I64 : VOPProfile <[i64, i64, i64, untyped]>;

def VOP_F32_F32_F32_F32 : VOPProfile <[f32, f32, f32, f32]>;
def VOP_MADAK : VOPProfile <[f32, f32, f32, f32]> {
  field dag Ins32 = (ins VCSrc_32:$src0, VGPR_32:$src1, u32kimm:$imm);
  field string Asm32 = "$vdst, $src0, $src1, $imm";
  field bit HasExt = 0;
}
def VOP_MADMK : VOPProfile <[f32, f32, f32, f32]> {
  field dag Ins32 = (ins VCSrc_32:$src0, u32kimm:$imm, VGPR_32:$src1);
  field string Asm32 = "$vdst, $src0, $imm, $src1";
  field bit HasExt = 0;
}
def VOP_MAC : VOPProfile <[f32, f32, f32, f32]> {
  let Ins32 = (ins Src0RC32:$src0, Src1RC32:$src1, VGPR_32:$src2);
  let Ins64 = getIns64<Src0RC64, Src1RC64, RegisterOperand<VGPR_32>, 3,
                             HasModifiers>.ret;
  let InsDPP = (ins FPInputMods:$src0_modifiers, Src0RC32:$src0,
                    FPInputMods:$src1_modifiers, Src1RC32:$src1,
                    VGPR_32:$src2, // stub argument
                    dpp_ctrl:$dpp_ctrl, row_mask:$row_mask,
                    bank_mask:$bank_mask, bound_ctrl:$bound_ctrl);
  let InsSDWA = (ins FPInputMods:$src0_fmodifiers, Src0RC32:$src0,
                     FPInputMods:$src1_fmodifiers, Src1RC32:$src1,
                     VGPR_32:$src2, // stub argument
                     clampmod:$clamp, dst_sel:$dst_sel, dst_unused:$dst_unused,
                     src0_sel:$src0_sel, src1_sel:$src1_sel);
  let Asm32 = getAsm32<1, 2, f32>.ret;
  let Asm64 = getAsm64<1, 2, HasModifiers, f32>.ret;
  let AsmDPP = getAsmDPP<1, 2, HasModifiers, f32>.ret;
  let AsmSDWA = getAsmSDWA<1, 2, HasModifiers, f32>.ret;
}
def VOP_F64_F64_F64_F64 : VOPProfile <[f64, f64, f64, f64]>;
def VOP_I32_I32_I32_I32 : VOPProfile <[i32, i32, i32, i32]>;
def VOP_I64_I32_I32_I64 : VOPProfile <[i64, i32, i32, i64]>;
def VOP_I32_F32_I32_I32 : VOPProfile <[i32, f32, i32, i32]>;

// This class is used only with VOPC instructions. Use $sdst for out operand
class SIInstAlias <string asm, Instruction inst, VOPProfile p> :
    InstAlias <asm, (inst)>, PredicateControl {

  field bit isCompare;
  field bit isCommutable;

  let ResultInst =
    !if (p.HasDst32,
      !if (!eq(p.NumSrcArgs, 0),
        // 1 dst, 0 src
        (inst p.DstRC:$sdst),
      !if (!eq(p.NumSrcArgs, 1),
        // 1 dst, 1 src
        (inst p.DstRC:$sdst, p.Src0RC32:$src0),
      !if (!eq(p.NumSrcArgs, 2),
        // 1 dst, 2 src
        (inst p.DstRC:$sdst, p.Src0RC32:$src0, p.Src1RC32:$src1),
      // else - unreachable
        (inst)))),
    // else
      !if (!eq(p.NumSrcArgs, 2),
        // 0 dst, 2 src
        (inst p.Src0RC32:$src0, p.Src1RC32:$src1),
      !if (!eq(p.NumSrcArgs, 1),
        // 0 dst, 1 src
        (inst p.Src0RC32:$src1),
      // else
        // 0 dst, 0 src
        (inst))));
}

class SIInstAliasSI <string asm, string op_name, VOPProfile p> :
  SIInstAlias <asm, !cast<Instruction>(op_name#"_e32_si"), p> {
  let AssemblerPredicate = SIAssemblerPredicate;
}

class SIInstAliasVI <string asm, string op_name, VOPProfile p> :
  SIInstAlias <asm, !cast<Instruction>(op_name#"_e32_vi"), p> {
  let AssemblerPredicates = [isVI];
}

multiclass SIInstAliasBuilder <string asm, VOPProfile p> {

  def : SIInstAliasSI <asm, NAME, p>;

  def : SIInstAliasVI <asm, NAME, p>;
}

class VOP <string opName> {
  string OpName = opName;
}

class VOP2_REV <string revOp, bit isOrig> {
  string RevOp = revOp;
  bit IsOrig = isOrig;
}

class AtomicNoRet <string noRetOp, bit isRet> {
  string NoRetOp = noRetOp;
  bit IsRet = isRet;
}

class VOP1_Pseudo <dag outs, dag ins, list<dag> pattern, string opName> :
  VOP1Common <outs, ins, "", pattern>,
  VOP <opName>,
  SIMCInstr <opName#"_e32", SIEncodingFamily.NONE>,
  MnemonicAlias<opName#"_e32", opName> {
  let isPseudo = 1;
  let isCodeGenOnly = 1;

  field bits<8> vdst;
  field bits<9> src0;
}

class VOP1_Real_si <string opName, vop1 op, dag outs, dag ins, string asm> :
  VOP1<op.SI, outs, ins, asm, []>,
  SIMCInstr <opName#"_e32", SIEncodingFamily.SI> {
  let AssemblerPredicate = SIAssemblerPredicate;
  let DecoderNamespace = "SICI";
  let DisableDecoder = DisableSIDecoder;
}

class VOP1_Real_vi <string opName, vop1 op, dag outs, dag ins, string asm> :
  VOP1<op.VI, outs, ins, asm, []>,
  SIMCInstr <opName#"_e32", SIEncodingFamily.VI> {
  let AssemblerPredicates = [isVI];
  let DecoderNamespace = "VI";
  let DisableDecoder = DisableVIDecoder;
}

multiclass VOP1_m <vop1 op, string opName, VOPProfile p, list<dag> pattern,
                   string asm = opName#p.Asm32> {
  def "" : VOP1_Pseudo <p.Outs, p.Ins32, pattern, opName>;

  def _si : VOP1_Real_si <opName, op, p.Outs, p.Ins32, asm>;

  def _vi : VOP1_Real_vi <opName, op, p.Outs, p.Ins32, asm>;

}

class VOP1_DPP <vop1 op, string opName, VOPProfile p> :
  VOP1_DPPe <op.VI>,
  VOP_DPP <p.OutsDPP, p.InsDPP, opName#p.AsmDPP, [], p.HasModifiers> {
  let AssemblerPredicates = !if(p.HasExt, [isVI], [DisableInst]);
  let DecoderNamespace = "DPP";
  let DisableDecoder = DisableVIDecoder;
  let src0_modifiers = !if(p.HasModifiers, ?, 0);
  let src1_modifiers = 0;
}

class SDWADisableFields <VOPProfile p> {
  bits<8> src0 = !if(!eq(p.NumSrcArgs, 0), 0, ?);
  bits<3> src0_sel = !if(!eq(p.NumSrcArgs, 0), 6, ?);
  bits<2> src0_fmodifiers = !if(!eq(p.NumSrcArgs, 0),
                                0,
                                !if(p.HasModifiers, ?, 0));
  bits<1> src0_imodifiers = !if(!eq(p.NumSrcArgs, 0),
                                0,
                                !if(p.HasModifiers, 0, ?));
  bits<3> src1_sel = !if(!eq(p.NumSrcArgs, 0), 6,
                         !if(!eq(p.NumSrcArgs, 1), 6,
                             ?));
  bits<2> src1_fmodifiers = !if(!eq(p.NumSrcArgs, 0), 0,
                                !if(!eq(p.NumSrcArgs, 1), 0,
                                    !if(p.HasModifiers, ?, 0)));
  bits<1> src1_imodifiers = !if(!eq(p.NumSrcArgs, 0), 0,
                                !if(!eq(p.NumSrcArgs, 1), 0,
                                    !if(p.HasModifiers, 0, ?)));
  bits<3> dst_sel = !if(p.HasDst, ?, 6);
  bits<2> dst_unused = !if(p.HasDst, ?, 2);
  bits<1> clamp = !if(!eq(p.NumSrcArgs, 0), 0, ?);
}

class VOP1_SDWA <vop1 op, string opName, VOPProfile p> :
  VOP1_SDWAe <op.VI>,
  VOP_SDWA <p.OutsSDWA, p.InsSDWA, opName#p.AsmSDWA, [], p.HasModifiers>,
  SDWADisableFields <p> {
  let AsmMatchConverter = "cvtSdwaVOP1";
  let AssemblerPredicates = !if(p.HasExt, [isVI], [DisableInst]);
  let DecoderNamespace = "SDWA";
  let DisableDecoder = DisableVIDecoder;
}

multiclass VOP1SI_m <vop1 op, string opName, VOPProfile p, list<dag> pattern,
                     string asm = opName#p.Asm32> {

  def "" : VOP1_Pseudo <p.Outs, p.Ins32, pattern, opName>;

  def _si : VOP1_Real_si <opName, op, p.Outs, p.Ins32, asm>;
}

class VOP2_Pseudo <dag outs, dag ins, list<dag> pattern, string opName> :
  VOP2Common <outs, ins, "", pattern>,
  VOP <opName>,
  SIMCInstr<opName#"_e32", SIEncodingFamily.NONE>,
  MnemonicAlias<opName#"_e32", opName> {
  let isPseudo = 1;
  let isCodeGenOnly = 1;
}

class VOP2_Real_si <string opName, vop2 op, dag outs, dag ins, string asm> :
  VOP2 <op.SI, outs, ins, opName#asm, []>,
  SIMCInstr <opName#"_e32", SIEncodingFamily.SI> {
  let AssemblerPredicates = [isSICI];
  let DecoderNamespace = "SICI";
  let DisableDecoder = DisableSIDecoder;
}

class VOP2_Real_vi <string opName, vop2 op, dag outs, dag ins, string asm> :
  VOP2 <op.VI, outs, ins, opName#asm, []>,
  SIMCInstr <opName#"_e32", SIEncodingFamily.VI> {
  let AssemblerPredicates = [isVI];
  let DecoderNamespace = "VI";
  let DisableDecoder = DisableVIDecoder;
}

multiclass VOP2SI_m <vop2 op, string opName, VOPProfile p, list<dag> pattern,
                     string revOp> {

  def "" : VOP2_Pseudo <p.Outs32, p.Ins32, pattern, opName>,
           VOP2_REV<revOp#"_e32", !eq(revOp, opName)>;

  def _si : VOP2_Real_si <opName, op, p.Outs32, p.Ins32, p.Asm32>;
}

multiclass VOP2_m <vop2 op, string opName, VOPProfile p, list <dag> pattern,
                   string revOp> {

  def "" : VOP2_Pseudo <p.Outs32, p.Ins32, pattern, opName>,
           VOP2_REV<revOp#"_e32", !eq(revOp, opName)>;

  def _si : VOP2_Real_si <opName, op, p.Outs32, p.Ins32, p.Asm32>;

  def _vi : VOP2_Real_vi <opName, op, p.Outs32, p.Ins32, p.Asm32>;

}

class VOP2_DPP <vop2 op, string opName, VOPProfile p> :
  VOP2_DPPe <op.VI>,
  VOP_DPP <p.OutsDPP, p.InsDPP, opName#p.AsmDPP, [], p.HasModifiers> {
  let AssemblerPredicates = !if(p.HasExt, [isVI], [DisableInst]);
  let DecoderNamespace = "DPP";
  let DisableDecoder = DisableVIDecoder;
  let src0_modifiers = !if(p.HasModifiers, ?, 0);
  let src1_modifiers = !if(p.HasModifiers, ?, 0);
}

class VOP2_SDWA <vop2 op, string opName, VOPProfile p> :
  VOP2_SDWAe <op.VI>,
  VOP_SDWA <p.OutsSDWA, p.InsSDWA, opName#p.AsmSDWA, [], p.HasModifiers>,
  SDWADisableFields <p> {
  let AsmMatchConverter = "cvtSdwaVOP2";
  let AssemblerPredicates = !if(p.HasExt, [isVI], [DisableInst]);
  let DecoderNamespace = "SDWA";
  let DisableDecoder = DisableVIDecoder;
}

class VOP3DisableFields <bit HasSrc1, bit HasSrc2, bit HasModifiers> {

  bits<2> src0_modifiers = !if(HasModifiers, ?, 0);
  bits<2> src1_modifiers = !if(HasModifiers, !if(HasSrc1, ?, 0), 0);
  bits<2> src2_modifiers = !if(HasModifiers, !if(HasSrc2, ?, 0), 0);
  bits<2> omod = !if(HasModifiers, ?, 0);
  bits<1> clamp = !if(HasModifiers, ?, 0);
  bits<9> src1 = !if(HasSrc1, ?, 0);
  bits<9> src2 = !if(HasSrc2, ?, 0);
}

class VOP3DisableModFields <bit HasSrc0Mods,
                            bit HasSrc1Mods = 0,
                            bit HasSrc2Mods = 0,
                            bit HasOutputMods = 0> {
  bits<2> src0_modifiers = !if(HasSrc0Mods, ?, 0);
  bits<2> src1_modifiers = !if(HasSrc1Mods, ?, 0);
  bits<2> src2_modifiers = !if(HasSrc2Mods, ?, 0);
  bits<2> omod = !if(HasOutputMods, ?, 0);
  bits<1> clamp = !if(HasOutputMods, ?, 0);
}

class VOP3_Pseudo <dag outs, dag ins, list<dag> pattern, string opName,
                   bit HasMods = 0, bit VOP3Only = 0> :
  VOP3Common <outs, ins, "", pattern, HasMods, VOP3Only>,
  VOP <opName>,
  SIMCInstr<opName#"_e64", SIEncodingFamily.NONE>,
  MnemonicAlias<opName#"_e64", opName> {
  let isPseudo = 1;
  let isCodeGenOnly = 1;

  field bit vdst;
  field bit src0;
}

class VOP3_Real_si <bits<9> op, dag outs, dag ins, string asm, string opName,
                    bit HasMods = 0, bit VOP3Only = 0> :
  VOP3Common <outs, ins, asm, [], HasMods, VOP3Only>,
  VOP3e <op>,
  SIMCInstr<opName#"_e64", SIEncodingFamily.SI> {
  let AssemblerPredicates = [isSICI];
  let DecoderNamespace = "SICI";
  let DisableDecoder = DisableSIDecoder;
}

class VOP3_Real_vi <bits<10> op, dag outs, dag ins, string asm, string opName,
                    bit HasMods = 0, bit VOP3Only = 0> :
  VOP3Common <outs, ins, asm, [], HasMods, VOP3Only>,
  VOP3e_vi <op>,
  SIMCInstr <opName#"_e64", SIEncodingFamily.VI> {
  let AssemblerPredicates = [isVI];
  let DecoderNamespace = "VI";
  let DisableDecoder = DisableVIDecoder;
}

class VOP3_C_Real_si <bits<9> op, dag outs, dag ins, string asm, string opName,
                     bit HasMods = 0, bit VOP3Only = 0> :
  VOP3Common <outs, ins, asm, [], HasMods, VOP3Only>,
  VOP3ce <op>,
  SIMCInstr<opName#"_e64", SIEncodingFamily.SI> {
  let AssemblerPredicates = [isSICI];
  let DecoderNamespace = "SICI";
  let DisableDecoder = DisableSIDecoder;
}

class VOP3_C_Real_vi <bits<10> op, dag outs, dag ins, string asm, string opName,
                      bit HasMods = 0, bit VOP3Only = 0> :
  VOP3Common <outs, ins, asm, [], HasMods, VOP3Only>,
  VOP3ce_vi <op>,
  SIMCInstr <opName#"_e64", SIEncodingFamily.VI> {
  let AssemblerPredicates = [isVI];
  let DecoderNamespace = "VI";
  let DisableDecoder = DisableVIDecoder;
}

class VOP3b_Real_si <bits<9> op, dag outs, dag ins, string asm, string opName,
                     bit HasMods = 0, bit VOP3Only = 0> :
  VOP3Common <outs, ins, asm, [], HasMods, VOP3Only>,
  VOP3be <op>,
  SIMCInstr<opName#"_e64", SIEncodingFamily.SI> {
  let AssemblerPredicates = [isSICI];
  let DecoderNamespace = "SICI";
  let DisableDecoder = DisableSIDecoder;
}

class VOP3b_Real_vi <bits<10> op, dag outs, dag ins, string asm, string opName,
                     bit HasMods = 0, bit VOP3Only = 0> :
  VOP3Common <outs, ins, asm, [], HasMods, VOP3Only>,
  VOP3be_vi <op>,
  SIMCInstr <opName#"_e64", SIEncodingFamily.VI> {
  let AssemblerPredicates = [isVI];
  let DecoderNamespace = "VI";
  let DisableDecoder = DisableVIDecoder;
}

class VOP3e_Real_si <bits<9> op, dag outs, dag ins, string asm, string opName,
                     bit HasMods = 0, bit VOP3Only = 0> :
  VOP3Common <outs, ins, asm, [], HasMods, VOP3Only>,
  VOP3e <op>,
  SIMCInstr<opName#"_e64", SIEncodingFamily.SI> {
  let AssemblerPredicates = [isSICI];
  let DecoderNamespace = "SICI";
  let DisableDecoder = DisableSIDecoder;
}

class VOP3e_Real_vi <bits<10> op, dag outs, dag ins, string asm, string opName,
                     bit HasMods = 0, bit VOP3Only = 0> :
  VOP3Common <outs, ins, asm, [], HasMods, VOP3Only>,
  VOP3e_vi <op>,
  SIMCInstr <opName#"_e64", SIEncodingFamily.VI> {
  let AssemblerPredicates = [isVI];
  let DecoderNamespace = "VI";
  let DisableDecoder = DisableVIDecoder;
}

multiclass VOP3_m <vop op, dag outs, dag ins, string asm, list<dag> pattern,
                   string opName, int NumSrcArgs, bit HasMods = 1, bit VOP3Only = 0> {

  def "" : VOP3_Pseudo <outs, ins, pattern, opName>;

  def _si : VOP3_Real_si <op.SI3, outs, ins, asm, opName, HasMods, VOP3Only>,
            VOP3DisableFields<!if(!eq(NumSrcArgs, 1), 0, 1),
                              !if(!eq(NumSrcArgs, 2), 0, 1),
                              HasMods>;
  def _vi : VOP3_Real_vi <op.VI3, outs, ins, asm, opName, HasMods, VOP3Only>,
            VOP3DisableFields<!if(!eq(NumSrcArgs, 1), 0, 1),
                              !if(!eq(NumSrcArgs, 2), 0, 1),
                              HasMods>;
}

multiclass VOP3_1_m <vop op, dag outs, dag ins, string asm,
                     list<dag> pattern, string opName, bit HasMods = 1> {

  def "" : VOP3_Pseudo <outs, ins, pattern, opName, HasMods>;

  def _si : VOP3_Real_si <op.SI3, outs, ins, asm, opName, HasMods>,
            VOP3DisableFields<0, 0, HasMods>;

  def _vi : VOP3_Real_vi <op.VI3, outs, ins, asm, opName, HasMods>,
            VOP3DisableFields<0, 0, HasMods>;
}

multiclass VOP3SI_1_m <vop op, dag outs, dag ins, string asm,
                     list<dag> pattern, string opName, bit HasMods = 1> {

  def "" : VOP3_Pseudo <outs, ins, pattern, opName, HasMods>;

  def _si : VOP3_Real_si <op.SI3, outs, ins, asm, opName, HasMods>,
            VOP3DisableFields<0, 0, HasMods>;
  // No VI instruction. This class is for SI only.
}

multiclass VOP3_2_m <vop op, dag outs, dag ins, string asm,
                     list<dag> pattern, string opName, string revOp,
                     bit HasMods = 1> {

  def "" : VOP3_Pseudo <outs, ins, pattern, opName, HasMods>,
           VOP2_REV<revOp#"_e64", !eq(revOp, opName)>;

  def _si : VOP3_Real_si <op.SI3, outs, ins, asm, opName, HasMods>,
            VOP3DisableFields<1, 0, HasMods>;

  def _vi : VOP3_Real_vi <op.VI3, outs, ins, asm, opName, HasMods>,
            VOP3DisableFields<1, 0, HasMods>;
}

multiclass VOP3SI_2_m <vop op, dag outs, dag ins, string asm,
                     list<dag> pattern, string opName, string revOp,
                     bit HasMods = 1> {

  def "" : VOP3_Pseudo <outs, ins, pattern, opName, HasMods>,
           VOP2_REV<revOp#"_e64", !eq(revOp, opName)>;

  def _si : VOP3_Real_si <op.SI3, outs, ins, asm, opName, HasMods>,
            VOP3DisableFields<1, 0, HasMods>;

  // No VI instruction. This class is for SI only.
}

// Two operand VOP3b instruction that may have a 3rd SGPR bool operand
// instead of an implicit VCC as in the VOP2b format.
multiclass VOP3b_2_3_m <vop op, dag outs, dag ins, string asm,
                        list<dag> pattern, string opName, string revOp,
                        bit HasMods = 1, bit useSrc2Input = 0, bit VOP3Only = 0> {
  def "" : VOP3_Pseudo <outs, ins, pattern, opName, HasMods, VOP3Only>;

  def _si : VOP3b_Real_si <op.SI3, outs, ins, asm, opName, HasMods, VOP3Only>,
            VOP3DisableFields<1, useSrc2Input, HasMods>;

  def _vi : VOP3b_Real_vi <op.VI3, outs, ins, asm, opName, HasMods, VOP3Only>,
            VOP3DisableFields<1, useSrc2Input, HasMods>;
}

// Same as VOP3b_2_3_m but no 2nd destination (sdst), e.g. v_cndmask_b32.
multiclass VOP3e_2_3_m <vop op, dag outs, dag ins, string asm,
                        list<dag> pattern, string opName, string revOp,
                        bit HasMods = 1, bit useSrc2Input = 0, bit VOP3Only = 0> {
  def "" : VOP3_Pseudo <outs, ins, pattern, opName, HasMods, VOP3Only>;

  def _si : VOP3e_Real_si <op.SI3, outs, ins, asm, opName, HasMods, VOP3Only>,
            VOP3DisableFields<1, useSrc2Input, HasMods>;

  def _vi : VOP3e_Real_vi <op.VI3, outs, ins, asm, opName, HasMods, VOP3Only>,
            VOP3DisableFields<1, useSrc2Input, HasMods>;
}

multiclass VOP3_C_m <vop op, dag outs, dag ins, string asm,
                     list<dag> pattern, string opName,
                     bit HasMods, bit defExec,
                     string revOp, list<SchedReadWrite> sched> {

  def "" : VOP3_Pseudo <outs, ins, pattern, opName, HasMods>,
           VOP2_REV<revOp#"_e64", !eq(revOp, opName)> {
    let Defs = !if(defExec, [EXEC], []);
    let SchedRW = sched;
  }

  def _si : VOP3_C_Real_si <op.SI3, outs, ins, asm, opName, HasMods>,
            VOP3DisableFields<1, 0, HasMods> {
    let Defs = !if(defExec, [EXEC], []);
    let SchedRW = sched;
  }

  def _vi : VOP3_C_Real_vi <op.VI3, outs, ins, asm, opName, HasMods>,
            VOP3DisableFields<1, 0, HasMods> {
    let Defs = !if(defExec, [EXEC], []);
    let SchedRW = sched;
  }
}

// An instruction that is VOP2 on SI and VOP3 on VI, no modifiers.
multiclass VOP2SI_3VI_m <vop3 op, string opName, dag outs, dag ins,
                         string asm, list<dag> pattern = []> {
  let isPseudo = 1, isCodeGenOnly = 1 in {
    def "" : VOPAnyCommon <outs, ins, "", pattern>,
             SIMCInstr<opName, SIEncodingFamily.NONE>;
  }

  def _si : VOP2 <op.SI3{5-0}, outs, ins, asm, []>,
            SIMCInstr <opName, SIEncodingFamily.SI> {
            let AssemblerPredicates = [isSICI];
            let DecoderNamespace = "SICI";
            let DisableDecoder = DisableSIDecoder;
  }

  def _vi : VOP3Common <outs, ins, asm, []>,
            VOP3e_vi <op.VI3>,
            VOP3DisableFields <1, 0, 0>,
            SIMCInstr <opName, SIEncodingFamily.VI> {
            let AssemblerPredicates = [isVI];
            let DecoderNamespace = "VI";
            let DisableDecoder = DisableVIDecoder;
  }
}

multiclass VOP1_Helper <vop1 op, string opName, VOPProfile p, list<dag> pat32,
                        list<dag> pat64> {

  defm _e32 : VOP1_m <op, opName, p, pat32>;

  defm _e64 : VOP3_1_m <op, p.Outs, p.Ins64, opName#p.Asm64, pat64, opName,
                        p.HasModifiers>;

  def _dpp : VOP1_DPP <op, opName, p>;

  def _sdwa : VOP1_SDWA <op, opName, p>;
}

multiclass VOP1Inst <vop1 op, string opName, VOPProfile P,
                     SDPatternOperator node = null_frag> : VOP1_Helper <
  op, opName, P, [],
  !if(P.HasModifiers,
      [(set P.DstVT:$vdst, (node (P.Src0VT (VOP3Mods0 P.Src0VT:$src0,
                                i32:$src0_modifiers, i1:$clamp, i32:$omod))))],
      [(set P.DstVT:$vdst, (node P.Src0VT:$src0))])
>;

multiclass VOP1InstSI <vop1 op, string opName, VOPProfile P,
                       SDPatternOperator node = null_frag> {

  defm _e32 : VOP1SI_m <op, opName, P, []>;

  defm _e64 : VOP3SI_1_m <op, P.Outs, P.Ins64, opName#P.Asm64,
    !if(P.HasModifiers,
      [(set P.DstVT:$vdst, (node (P.Src0VT (VOP3Mods0 P.Src0VT:$src0,
                                i32:$src0_modifiers, i1:$clamp, i32:$omod))))],
      [(set P.DstVT:$vdst, (node P.Src0VT:$src0))]),
    opName, P.HasModifiers>;
}

multiclass VOP2_Helper <vop2 op, string opName, VOPProfile p, list<dag> pat32,
                        list<dag> pat64, string revOp> {

  defm _e32 : VOP2_m <op, opName, p, pat32, revOp>;

  defm _e64 : VOP3_2_m <op, p.Outs, p.Ins64, opName#p.Asm64, pat64, opName,
                        revOp, p.HasModifiers>;

  def _dpp : VOP2_DPP <op, opName, p>;

  def _sdwa : VOP2_SDWA <op, opName, p>;
}

multiclass VOP2Inst <vop2 op, string opName, VOPProfile P,
                     SDPatternOperator node = null_frag,
                     string revOp = opName> : VOP2_Helper <
  op, opName, P, [],
  !if(P.HasModifiers,
      [(set P.DstVT:$vdst,
           (node (P.Src0VT (VOP3Mods0 P.Src0VT:$src0, i32:$src0_modifiers,
                                      i1:$clamp, i32:$omod)),
                 (P.Src1VT (VOP3Mods P.Src1VT:$src1, i32:$src1_modifiers))))],
      [(set P.DstVT:$vdst, (node P.Src0VT:$src0, P.Src1VT:$src1))]),
  revOp
>;

multiclass VOP2InstSI <vop2 op, string opName, VOPProfile P,
                       SDPatternOperator node = null_frag,
                       string revOp = opName> {

  defm _e32 : VOP2SI_m <op, opName, P, [], revOp>;

  defm _e64 : VOP3SI_2_m <op, P.Outs, P.Ins64, opName#P.Asm64,
    !if(P.HasModifiers,
        [(set P.DstVT:$vdst,
             (node (P.Src0VT (VOP3Mods0 P.Src0VT:$src0, i32:$src0_modifiers,
                                        i1:$clamp, i32:$omod)),
                   (P.Src1VT (VOP3Mods P.Src1VT:$src1, i32:$src1_modifiers))))],
        [(set P.DstVT:$vdst, (node P.Src0VT:$src0, P.Src1VT:$src1))]),
    opName, revOp, P.HasModifiers>;
}

multiclass VOP2e_Helper <vop2 op, string opName, VOPProfile p,
                         list<dag> pat32, list<dag> pat64,
                         string revOp, bit useSGPRInput> {

  let SchedRW = [Write32Bit] in {
    let Uses = !if(useSGPRInput, [VCC, EXEC], [EXEC]) in {
      defm _e32 : VOP2_m <op, opName, p, pat32, revOp>;
    }

    defm _e64 : VOP3e_2_3_m <op, p.Outs64, p.Ins64, opName#p.Asm64, pat64,
                             opName, revOp, p.HasModifiers, useSGPRInput>;
  }
}

multiclass VOP2eInst <vop2 op, string opName, VOPProfile P,
                      SDPatternOperator node = null_frag,
                      string revOp = opName> : VOP2e_Helper <
  op, opName, P, [],
  !if(P.HasModifiers,
      [(set P.DstVT:$vdst,
           (node (P.Src0VT (VOP3Mods0 P.Src0VT:$src0, i32:$src0_modifiers,
                                      i1:$clamp, i32:$omod)),
                 (P.Src1VT (VOP3Mods P.Src1VT:$src1, i32:$src1_modifiers))))],
      [(set P.DstVT:$vdst, (node P.Src0VT:$src0, P.Src1VT:$src1))]),
  revOp, !eq(P.NumSrcArgs, 3)
>;

multiclass VOP2b_Helper <vop2 op, string opName, VOPProfile p,
                         list<dag> pat32, list<dag> pat64,
                         string revOp, bit useSGPRInput> {

  let SchedRW = [Write32Bit, WriteSALU] in {
    let Uses = !if(useSGPRInput, [VCC, EXEC], [EXEC]), Defs = [VCC] in {
      defm _e32 : VOP2_m <op, opName, p, pat32, revOp>;
    }

    defm _e64 : VOP3b_2_3_m <op, p.Outs64, p.Ins64, opName#p.Asm64, pat64,
                             opName, revOp, p.HasModifiers, useSGPRInput>;
  }
}

multiclass VOP2bInst <vop2 op, string opName, VOPProfile P,
                      SDPatternOperator node = null_frag,
                      string revOp = opName> : VOP2b_Helper <
  op, opName, P, [],
  !if(P.HasModifiers,
      [(set P.DstVT:$vdst,
           (node (P.Src0VT (VOP3Mods0 P.Src0VT:$src0, i32:$src0_modifiers,
                                      i1:$clamp, i32:$omod)),
                 (P.Src1VT (VOP3Mods P.Src1VT:$src1, i32:$src1_modifiers))))],
      [(set P.DstVT:$vdst, (node P.Src0VT:$src0, P.Src1VT:$src1))]),
  revOp, !eq(P.NumSrcArgs, 3)
>;

// A VOP2 instruction that is VOP3-only on VI.
multiclass VOP2_VI3_Helper <vop23 op, string opName, VOPProfile p,
                            list<dag> pat32, list<dag> pat64, string revOp> {

  defm _e32 : VOP2SI_m <op, opName, p, pat32, revOp>;

  defm _e64 : VOP3_2_m <op, p.Outs, p.Ins64, opName#p.Asm64, pat64, opName,
                        revOp, p.HasModifiers>;
}

multiclass VOP2_VI3_Inst <vop23 op, string opName, VOPProfile P,
                          SDPatternOperator node = null_frag,
                          string revOp = opName>
                          : VOP2_VI3_Helper <
  op, opName, P, [],
  !if(P.HasModifiers,
      [(set P.DstVT:$vdst,
           (node (P.Src0VT (VOP3Mods0 P.Src0VT:$src0, i32:$src0_modifiers,
                                      i1:$clamp, i32:$omod)),
                 (P.Src1VT (VOP3Mods P.Src1VT:$src1, i32:$src1_modifiers))))],
      [(set P.DstVT:$vdst, (node P.Src0VT:$src0, P.Src1VT:$src1))]),
  revOp
>;

multiclass VOP2MADK <vop2 op, string opName, VOPProfile P, list<dag> pattern = []> {

  def "" : VOP2_Pseudo <P.Outs, P.Ins32, pattern, opName>;

let isCodeGenOnly = 0 in {
  def _si : VOP2Common <P.Outs, P.Ins32,
                        !strconcat(opName, P.Asm32), []>,
            SIMCInstr <opName#"_e32", SIEncodingFamily.SI>,
            VOP2_MADKe <op.SI> {
            let AssemblerPredicates = [isSICI];
            let DecoderNamespace = "SICI";
            let DisableDecoder = DisableSIDecoder;
            }

  def _vi : VOP2Common <P.Outs, P.Ins32,
                        !strconcat(opName, P.Asm32), []>,
            SIMCInstr <opName#"_e32", SIEncodingFamily.VI>,
            VOP2_MADKe <op.VI> {
            let AssemblerPredicates = [isVI];
            let DecoderNamespace = "VI";
            let DisableDecoder = DisableVIDecoder;
            }
} // End isCodeGenOnly = 0
}

class VOPC_Pseudo <dag ins, list<dag> pattern, string opName> :
  VOPCCommon <ins, "", pattern>,
  VOP <opName>,
  SIMCInstr<opName#"_e32", SIEncodingFamily.NONE> {
  let isPseudo = 1;
  let isCodeGenOnly = 1;
}

class VOPC_SDWA <vopc op, string opName, bit DefExec, VOPProfile p> :
    VOPC_SDWAe <op.VI>,
    VOP_SDWA <p.OutsSDWA, p.InsSDWA, opName#p.AsmSDWA, [], p.HasModifiers>,
    SDWADisableFields <p> {
  let Defs = !if(DefExec, [VCC, EXEC], [VCC]);
  let hasSideEffects = DefExec;
  let AsmMatchConverter = "cvtSdwaVOPC";
  let AssemblerPredicates = !if(p.HasExt, [isVI], [DisableInst]);
  let DecoderNamespace = "SDWA";
  let DisableDecoder = DisableVIDecoder;
}

multiclass VOPC_m <vopc op, dag ins, string op_asm, list<dag> pattern,
                   string opName, bit DefExec, VOPProfile p,
                   list<SchedReadWrite> sched,
                   string revOpName = "", string asm = opName#"_e32 "#op_asm,
                   string alias_asm = opName#" "#op_asm> {
  def "" : VOPC_Pseudo <ins, pattern, opName>,
           VOP2_REV<revOpName#"_e32", !eq(revOpName, opName)> {
    let Defs = !if(DefExec, [VCC, EXEC], [VCC]);
    let SchedRW = sched;
    let isConvergent = DefExec;
  }

  let AssemblerPredicates = [isSICI] in {
    def _si : VOPC<op.SI, ins, asm, []>,
              SIMCInstr <opName#"_e32", SIEncodingFamily.SI> {
      let Defs = !if(DefExec, [VCC, EXEC], [VCC]);
      let isConvergent = DefExec;
      let SchedRW = sched;
      let DecoderNamespace = "SICI";
      let DisableDecoder = DisableSIDecoder;
    }

  } // End AssemblerPredicates = [isSICI]

  let AssemblerPredicates = [isVI] in {
    def _vi : VOPC<op.VI, ins, asm, []>,
              SIMCInstr <opName#"_e32", SIEncodingFamily.VI> {
      let Defs = !if(DefExec, [VCC, EXEC], [VCC]);
      let isConvergent = DefExec;
      let SchedRW = sched;
      let DecoderNamespace = "VI";
      let DisableDecoder = DisableVIDecoder;
    }

  } // End AssemblerPredicates = [isVI]

  defm : SIInstAliasBuilder<alias_asm, p>;
}

multiclass VOPC_Helper <vopc op, string opName, list<dag> pat32,
                        list<dag> pat64, bit DefExec, string revOp,
                        VOPProfile p, list<SchedReadWrite> sched> {
  defm _e32 : VOPC_m <op, p.Ins32, p.Asm32, pat32, opName, DefExec, p, sched,
                      revOp>;

  defm _e64 : VOP3_C_m <op, (outs VOPDstS64:$sdst), p.Ins64, opName#p.Asm64, pat64,
                        opName, p.HasModifiers, DefExec, revOp, sched>;

  def _sdwa : VOPC_SDWA <op, opName, DefExec, p>;
}

// Special case for class instructions which only have modifiers on
// the 1st source operand.
multiclass VOPC_Class_Helper <vopc op, string opName,
                              list<dag> pat64, bit DefExec, string revOp,
                              VOPProfile p, list<SchedReadWrite> sched> {
  defm _e32 : VOPC_m <op, p.Ins32, p.Asm32, [], opName, DefExec, p, sched>;

  defm _e64 : VOP3_C_m <op, (outs VOPDstS64:$sdst), p.Ins64, opName#p.Asm64, pat64,
                        opName, p.HasModifiers, DefExec, revOp, sched>,
                        VOP3DisableModFields<1, 0, 0>;

  def _sdwa : VOPC_SDWA <op, opName, DefExec, p> {
    let src1_fmodifiers = 0;
    let src1_imodifiers = ?;
  }
}

multiclass VOPCInst <vopc op, string opName,
                     VOPProfile P, PatLeaf cond = COND_NULL,
                     string revOp = opName,
                     bit DefExec = 0,
                     list<SchedReadWrite> sched = [Write32Bit]> :
                     VOPC_Helper <
  op, opName, [],
  !if(P.HasModifiers,
      [(set i1:$sdst,
          (setcc (P.Src0VT (VOP3Mods0 P.Src0VT:$src0, i32:$src0_modifiers,
                                      i1:$clamp, i32:$omod)),
                 (P.Src1VT (VOP3Mods P.Src1VT:$src1, i32:$src1_modifiers)),
                 cond))],
      [(set i1:$sdst, (setcc P.Src0VT:$src0, P.Src1VT:$src1, cond))]),
  DefExec, revOp, P, sched
>;

multiclass VOPCClassInst <vopc op, string opName, VOPProfile P,
                     bit DefExec = 0,
                     list<SchedReadWrite> sched> : VOPC_Class_Helper <
  op, opName,
  [(set i1:$sdst,
     (AMDGPUfp_class
       (P.Src0VT (VOP3Mods0Clamp0OMod P.Src0VT:$src0, i32:$src0_modifiers)),
       P.Src1VT:$src1))],
  DefExec, opName, P, sched
>;

multiclass VOPC_F32 <vopc op, string opName, PatLeaf cond = COND_NULL, string revOp = opName> :
  VOPCInst <op, opName, VOPC_I1_F32_F32, cond, revOp>;

multiclass VOPC_F64 <vopc op, string opName, PatLeaf cond = COND_NULL, string revOp = opName> :
  VOPCInst <op, opName, VOPC_I1_F64_F64, cond, revOp, 0, [WriteDoubleAdd]>;

multiclass VOPC_I32 <vopc op, string opName, PatLeaf cond = COND_NULL, string revOp = opName> :
  VOPCInst <op, opName, VOPC_I1_I32_I32, cond, revOp>;

multiclass VOPC_I64 <vopc op, string opName, PatLeaf cond = COND_NULL, string revOp = opName> :
  VOPCInst <op, opName, VOPC_I1_I64_I64, cond, revOp, 0, [Write64Bit]>;


multiclass VOPCX <vopc op, string opName, VOPProfile P,
                  PatLeaf cond = COND_NULL,
                  list<SchedReadWrite> sched,
                  string revOp = "">
  : VOPCInst <op, opName, P, cond, revOp, 1, sched>;

multiclass VOPCX_F32 <vopc op, string opName, string revOp = opName> :
  VOPCX <op, opName, VOPC_I1_F32_F32, COND_NULL, [Write32Bit], revOp>;

multiclass VOPCX_F64 <vopc op, string opName, string revOp = opName> :
  VOPCX <op, opName, VOPC_I1_F64_F64, COND_NULL, [WriteDoubleAdd], revOp>;

multiclass VOPCX_I32 <vopc op, string opName, string revOp = opName> :
  VOPCX <op, opName, VOPC_I1_I32_I32, COND_NULL, [Write32Bit], revOp>;

multiclass VOPCX_I64 <vopc op, string opName, string revOp = opName> :
  VOPCX <op, opName, VOPC_I1_I64_I64, COND_NULL, [Write64Bit], revOp>;


multiclass VOPC_CLASS_F32 <vopc op, string opName> :
  VOPCClassInst <op, opName, VOPC_I1_F32_I32, 0, [Write32Bit]>;

multiclass VOPCX_CLASS_F32 <vopc op, string opName> :
  VOPCClassInst <op, opName, VOPC_I1_F32_I32, 1, [Write32Bit]>;

multiclass VOPC_CLASS_F64 <vopc op, string opName> :
  VOPCClassInst <op, opName, VOPC_I1_F64_I32, 0, [WriteDoubleAdd]>;

multiclass VOPCX_CLASS_F64 <vopc op, string opName> :
  VOPCClassInst <op, opName, VOPC_I1_F64_I32, 1, [WriteDoubleAdd]>;


multiclass VOP3_Helper <vop3 op, string opName, dag outs, dag ins, string asm,
                        list<dag> pat, int NumSrcArgs, bit HasMods,
                        bit VOP3Only = 0> : VOP3_m <
    op, outs, ins, opName#" "#asm, pat, opName, NumSrcArgs, HasMods, VOP3Only
>;

multiclass VOP3Inst <vop3 op, string opName, VOPProfile P,
                     SDPatternOperator node = null_frag, bit VOP3Only = 0> :
  VOP3_Helper <
  op, opName, (outs P.DstRC.RegClass:$vdst), P.Ins64, P.Asm64,
  !if(!eq(P.NumSrcArgs, 3),
    !if(P.HasModifiers,
        [(set P.DstVT:$vdst,
            (node (P.Src0VT (VOP3Mods0 P.Src0VT:$src0, i32:$src0_modifiers,
                                       i1:$clamp, i32:$omod)),
                  (P.Src1VT (VOP3Mods P.Src1VT:$src1, i32:$src1_modifiers)),
                  (P.Src2VT (VOP3Mods P.Src2VT:$src2, i32:$src2_modifiers))))],
        [(set P.DstVT:$vdst, (node P.Src0VT:$src0, P.Src1VT:$src1,
                                  P.Src2VT:$src2))]),
  !if(!eq(P.NumSrcArgs, 2),
    !if(P.HasModifiers,
        [(set P.DstVT:$vdst,
            (node (P.Src0VT (VOP3Mods0 P.Src0VT:$src0, i32:$src0_modifiers,
                                       i1:$clamp, i32:$omod)),
                  (P.Src1VT (VOP3Mods P.Src1VT:$src1, i32:$src1_modifiers))))],
        [(set P.DstVT:$vdst, (node P.Src0VT:$src0, P.Src1VT:$src1))])
  /* P.NumSrcArgs == 1 */,
    !if(P.HasModifiers,
        [(set P.DstVT:$vdst,
            (node (P.Src0VT (VOP3Mods0 P.Src0VT:$src0, i32:$src0_modifiers,
                                       i1:$clamp, i32:$omod))))],
        [(set P.DstVT:$vdst, (node P.Src0VT:$src0))]))),
  P.NumSrcArgs, P.HasModifiers, VOP3Only
>;

// Special case for v_div_fmas_{f32|f64}, since it seems to be the
// only VOP instruction that implicitly reads VCC.
multiclass VOP3_VCC_Inst <vop3 op, string opName,
                          VOPProfile P,
                          SDPatternOperator node = null_frag> : VOP3_Helper <
  op, opName,
  (outs P.DstRC.RegClass:$vdst),
  (ins FPInputMods:$src0_modifiers, P.Src0RC64:$src0,
       FPInputMods:$src1_modifiers, P.Src1RC64:$src1,
       FPInputMods:$src2_modifiers, P.Src2RC64:$src2,
       clampmod:$clamp,
       omod:$omod),
  "$vdst, $src0_modifiers, $src1_modifiers, $src2_modifiers"#"$clamp"#"$omod",
  [(set P.DstVT:$vdst,
            (node (P.Src0VT (VOP3Mods0 P.Src0VT:$src0, i32:$src0_modifiers,
                                       i1:$clamp, i32:$omod)),
                  (P.Src1VT (VOP3Mods P.Src1VT:$src1, i32:$src1_modifiers)),
                  (P.Src2VT (VOP3Mods P.Src2VT:$src2, i32:$src2_modifiers)),
                  (i1 VCC)))],
  3, 1
>;

multiclass VOP3bInst <vop op, string opName, VOPProfile P, list<dag> pattern = [], bit VOP3Only = 0> :
  VOP3b_2_3_m <
  op, P.Outs64, P.Ins64,
  opName#" "#P.Asm64, pattern,
  opName, "", 1, 1, VOP3Only
>;

class Vop3ModPat<Instruction Inst, VOPProfile P, SDPatternOperator node> : Pat<
  (node (P.Src0VT (VOP3Mods0 P.Src0VT:$src0, i32:$src0_modifiers, i1:$clamp, i32:$omod)),
        (P.Src1VT (VOP3Mods P.Src1VT:$src1, i32:$src1_modifiers)),
        (P.Src2VT (VOP3Mods P.Src2VT:$src2, i32:$src2_modifiers))),
  (Inst i32:$src0_modifiers, P.Src0VT:$src0,
        i32:$src1_modifiers, P.Src1VT:$src1,
        i32:$src2_modifiers, P.Src2VT:$src2,
        i1:$clamp,
        i32:$omod)>;

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
// MTBUF classes
//===----------------------------------------------------------------------===//

class MTBUF_Pseudo <string opName, dag outs, dag ins, list<dag> pattern> :
  MTBUF <outs, ins, "", pattern>,
  SIMCInstr<opName, SIEncodingFamily.NONE> {
  let isPseudo = 1;
  let isCodeGenOnly = 1;
}

class MTBUF_Real_si <bits<3> op, string opName, dag outs, dag ins,
                    string asm> :
  MTBUF <outs, ins, asm, []>,
  MTBUFe <op>,
  SIMCInstr<opName, SIEncodingFamily.SI> {
  let DecoderNamespace="SICI";
  let DisableDecoder = DisableSIDecoder;
}

class MTBUF_Real_vi <bits<4> op, string opName, dag outs, dag ins, string asm> :
  MTBUF <outs, ins, asm, []>,
  MTBUFe_vi <op>,
  SIMCInstr <opName, SIEncodingFamily.VI> {
  let DecoderNamespace="VI";
  let DisableDecoder = DisableVIDecoder;
}

multiclass MTBUF_m <bits<3> op, string opName, dag outs, dag ins, string asm,
                    list<dag> pattern> {

  def "" : MTBUF_Pseudo <opName, outs, ins, pattern>;

  def _si : MTBUF_Real_si <op, opName, outs, ins, asm>;

  def _vi : MTBUF_Real_vi <{0, op{2}, op{1}, op{0}}, opName, outs, ins, asm>;

}

let mayStore = 1, mayLoad = 0 in {

multiclass MTBUF_Store_Helper <bits<3> op, string opName,
                               RegisterClass regClass> : MTBUF_m <
  op, opName, (outs),
  (ins regClass:$vdata, u16imm:$offset, i1imm:$offen, i1imm:$idxen, i1imm:$glc,
   i1imm:$addr64, i8imm:$dfmt, i8imm:$nfmt, VGPR_32:$vaddr,
   SReg_128:$srsrc, i1imm:$slc, i1imm:$tfe, SCSrc_32:$soffset),
  opName#" $vdata, $offset, $offen, $idxen, $glc, $addr64, $dfmt,"
        #" $nfmt, $vaddr, $srsrc, $slc, $tfe, $soffset", []
>;

} // mayStore = 1, mayLoad = 0

let mayLoad = 1, mayStore = 0 in {

multiclass MTBUF_Load_Helper <bits<3> op, string opName,
                              RegisterClass regClass> : MTBUF_m <
  op, opName, (outs regClass:$dst),
  (ins u16imm:$offset, i1imm:$offen, i1imm:$idxen, i1imm:$glc, i1imm:$addr64,
       i8imm:$dfmt, i8imm:$nfmt, VGPR_32:$vaddr, SReg_128:$srsrc,
       i1imm:$slc, i1imm:$tfe, SCSrc_32:$soffset),
  opName#" $dst, $offset, $offen, $idxen, $glc, $addr64, $dfmt,"
        #" $nfmt, $vaddr, $srsrc, $slc, $tfe, $soffset", []
>;

} // mayLoad = 1, mayStore = 0

//===----------------------------------------------------------------------===//
// MUBUF classes
//===----------------------------------------------------------------------===//

class mubuf <bits<7> si, bits<7> vi = si> {
  field bits<7> SI = si;
  field bits<7> VI = vi;
}

let isCodeGenOnly = 0 in {

class MUBUF_si <bits<7> op, dag outs, dag ins, string asm, list<dag> pattern> :
  MUBUF <outs, ins, asm, pattern>, MUBUFe <op> {
  let lds  = 0;
}

} // End let isCodeGenOnly = 0

class MUBUF_vi <bits<7> op, dag outs, dag ins, string asm, list<dag> pattern> :
  MUBUF <outs, ins, asm, pattern>, MUBUFe_vi <op> {
  let lds = 0;
}

class MUBUFAddr64Table <bit is_addr64, string suffix = ""> {
  bit IsAddr64 = is_addr64;
  string OpName = NAME # suffix;
}

class MUBUF_Pseudo <string opName, dag outs, dag ins, list<dag> pattern> :
  MUBUF <outs, ins, "", pattern>,
  SIMCInstr<opName, SIEncodingFamily.NONE> {
  let isPseudo = 1;
  let isCodeGenOnly = 1;

  // dummy fields, so that we can use let statements around multiclasses
  bits<1> offen;
  bits<1> idxen;
  bits<8> vaddr;
  bits<1> glc;
  bits<1> slc;
  bits<1> tfe;
  bits<8> soffset;
}

class MUBUF_Real_si <mubuf op, string opName, dag outs, dag ins,
                     string asm> :
  MUBUF <outs, ins, asm, []>,
  MUBUFe <op.SI>,
  SIMCInstr<opName, SIEncodingFamily.SI> {
  let lds = 0;
  let AssemblerPredicate = SIAssemblerPredicate;
  let DecoderNamespace="SICI";
  let DisableDecoder = DisableSIDecoder;
}

class MUBUF_Real_vi <mubuf op, string opName, dag outs, dag ins,
                     string asm> :
  MUBUF <outs, ins, asm, []>,
  MUBUFe_vi <op.VI>,
  SIMCInstr<opName, SIEncodingFamily.VI> {
  let lds = 0;
  let AssemblerPredicate = VIAssemblerPredicate;
  let DecoderNamespace="VI";
  let DisableDecoder = DisableVIDecoder;
}

multiclass MUBUF_m <mubuf op, string opName, dag outs, dag ins, string asm,
                    list<dag> pattern> {

  def "" : MUBUF_Pseudo <opName, outs, ins, pattern>,
           MUBUFAddr64Table <0>;

  let DisableWQM = 1 in {
    def "_exact" : MUBUF_Pseudo <opName, outs, ins, []>;
  }

  let addr64 = 0, isCodeGenOnly = 0 in {
    def _si : MUBUF_Real_si <op, opName, outs, ins, asm>;
  }

  def _vi : MUBUF_Real_vi <op, opName, outs, ins, asm>;
}

multiclass MUBUFAddr64_m <mubuf op, string opName, dag outs,
                          dag ins, string asm, list<dag> pattern> {

  def "" : MUBUF_Pseudo <opName, outs, ins, pattern>,
           MUBUFAddr64Table <1>;

  let addr64 = 1, isCodeGenOnly = 0 in {
    def _si : MUBUF_Real_si <op, opName, outs, ins, asm>;
  }

  // There is no VI version. If the pseudo is selected, it should be lowered
  // for VI appropriately.
}

multiclass MUBUFAtomicOffset_m <mubuf op, string opName, dag outs, dag ins,
                                string asm, list<dag> pattern, bit is_return> {

  def "" : MUBUF_Pseudo <opName, outs, ins, pattern>,
           MUBUFAddr64Table <0, !if(is_return, "_RTN", "")>,
           AtomicNoRet<NAME#"_OFFSET", is_return>;

  let offen = 0, idxen = 0, tfe = 0, vaddr = 0 in {
    let addr64 = 0 in {
      def _si : MUBUF_Real_si <op, opName, outs, ins, asm>;
    }

    def _vi : MUBUF_Real_vi <op, opName, outs, ins, asm>;
  }
}

multiclass MUBUFAtomicAddr64_m <mubuf op, string opName, dag outs, dag ins,
                                string asm, list<dag> pattern, bit is_return> {

  def "" : MUBUF_Pseudo <opName, outs, ins, pattern>,
           MUBUFAddr64Table <1, !if(is_return, "_RTN", "")>,
           AtomicNoRet<NAME#"_ADDR64", is_return>;

  let offen = 0, idxen = 0, addr64 = 1, tfe = 0 in {
    def _si : MUBUF_Real_si <op, opName, outs, ins, asm>;
  }

  // There is no VI version. If the pseudo is selected, it should be lowered
  // for VI appropriately.
}

multiclass MUBUFAtomicOther_m <mubuf op, string opName, dag outs, dag ins,
                               string asm, list<dag> pattern, bit is_return> {

  def "" : MUBUF_Pseudo <opName, outs, ins, pattern>,
           AtomicNoRet<opName, is_return>;

  let tfe = 0 in {
    let addr64 = 0 in {
      def _si : MUBUF_Real_si <op, opName, outs, ins, asm>;
    }

    def _vi : MUBUF_Real_vi <op, opName, outs, ins, asm>;
  }
}

multiclass MUBUF_Atomic <mubuf op, string name, RegisterClass rc,
                         ValueType vt, SDPatternOperator atomic> {

  let mayStore = 1, mayLoad = 1, hasPostISelHook = 1, hasSideEffects = 1,
      DisableWQM = 1 in {

    // No return variants
    let glc = 0, AsmMatchConverter = "cvtMubufAtomic" in {

      defm _ADDR64 : MUBUFAtomicAddr64_m <
        op, name#"_addr64", (outs),
        (ins rc:$vdata, VReg_64:$vaddr, SReg_128:$srsrc,
             SCSrc_32:$soffset, offset:$offset, slc:$slc),
        name#" $vdata, $vaddr, $srsrc, $soffset addr64$offset$slc", [], 0
      >;

      defm _OFFSET : MUBUFAtomicOffset_m <
        op, name#"_offset", (outs),
        (ins rc:$vdata, SReg_128:$srsrc, SCSrc_32:$soffset, offset:$offset,
             slc:$slc),
        name#" $vdata, off, $srsrc, $soffset$offset$slc", [], 0
      >;

      let offen = 1, idxen = 0 in {
        defm _OFFEN : MUBUFAtomicOther_m <
          op, name#"_offen", (outs),
          (ins rc:$vdata, VGPR_32:$vaddr, SReg_128:$srsrc, SCSrc_32:$soffset,
                offset:$offset, slc:$slc),
          name#" $vdata, $vaddr, $srsrc, $soffset offen$offset$slc", [], 0
        >;
      }

      let offen = 0, idxen = 1 in {
        defm _IDXEN : MUBUFAtomicOther_m <
          op, name#"_idxen", (outs),
          (ins rc:$vdata, VGPR_32:$vaddr, SReg_128:$srsrc, SCSrc_32:$soffset,
                offset:$offset, slc:$slc),
          name#" $vdata, $vaddr, $srsrc, $soffset idxen$offset$slc", [], 0
        >;
      }

      let offen = 1, idxen = 1 in {
        defm _BOTHEN : MUBUFAtomicOther_m <
          op, name#"_bothen", (outs),
          (ins rc:$vdata, VReg_64:$vaddr, SReg_128:$srsrc, SCSrc_32:$soffset,
                offset:$offset, slc:$slc),
          name#" $vdata, $vaddr, $srsrc, $soffset idxen offen$offset$slc",
          [], 0
        >;
      }
    } // glc = 0

    // Variant that return values
    let glc = 1, Constraints = "$vdata = $vdata_in",
        AsmMatchConverter = "cvtMubufAtomicReturn",
        DisableEncoding = "$vdata_in"  in {

      defm _RTN_ADDR64 : MUBUFAtomicAddr64_m <
        op, name#"_rtn_addr64", (outs rc:$vdata),
        (ins rc:$vdata_in, VReg_64:$vaddr, SReg_128:$srsrc,
             SCSrc_32:$soffset, offset:$offset, slc:$slc),
        name#" $vdata, $vaddr, $srsrc, $soffset addr64$offset glc$slc",
        [(set vt:$vdata,
         (atomic (MUBUFAddr64Atomic v4i32:$srsrc, i64:$vaddr, i32:$soffset,
	                            i16:$offset, i1:$slc), vt:$vdata_in))], 1
      >;

      defm _RTN_OFFSET : MUBUFAtomicOffset_m <
        op, name#"_rtn_offset", (outs rc:$vdata),
        (ins rc:$vdata_in, SReg_128:$srsrc, SCSrc_32:$soffset,
             offset:$offset, slc:$slc),
        name#" $vdata, off, $srsrc, $soffset$offset glc$slc",
        [(set vt:$vdata,
         (atomic (MUBUFOffsetAtomic v4i32:$srsrc, i32:$soffset, i16:$offset,
                                    i1:$slc), vt:$vdata_in))], 1
      >;

      let offen = 1, idxen = 0 in {
        defm _RTN_OFFEN : MUBUFAtomicOther_m <
          op, name#"_rtn_offen", (outs rc:$vdata),
          (ins rc:$vdata_in, VGPR_32:$vaddr, SReg_128:$srsrc, SCSrc_32:$soffset,
                offset:$offset, slc:$slc),
          name#" $vdata, $vaddr, $srsrc, $soffset offen$offset glc$slc",
          [], 1
        >;
      }

      let offen = 0, idxen = 1 in {
        defm _RTN_IDXEN : MUBUFAtomicOther_m <
          op, name#"_rtn_idxen", (outs rc:$vdata),
          (ins rc:$vdata_in, VGPR_32:$vaddr, SReg_128:$srsrc, SCSrc_32:$soffset,
                offset:$offset, slc:$slc),
          name#" $vdata, $vaddr, $srsrc, $soffset idxen$offset glc$slc",
          [], 1
        >;
      }

      let offen = 1, idxen = 1 in {
        defm _RTN_BOTHEN : MUBUFAtomicOther_m <
          op, name#"_rtn_bothen", (outs rc:$vdata),
          (ins rc:$vdata_in, VReg_64:$vaddr, SReg_128:$srsrc, SCSrc_32:$soffset,
                offset:$offset, slc:$slc),
          name#" $vdata, $vaddr, $srsrc, $soffset idxen offen$offset glc$slc",
          [], 1
        >;
      }
    } // glc = 1

  } // mayStore = 1, mayLoad = 1, hasPostISelHook = 1
}

// FIXME: tfe can't be an operand because it requires a separate
// opcode because it needs an N+1 register class dest register.
multiclass MUBUF_Load_Helper <mubuf op, string name, RegisterClass regClass,
                              ValueType load_vt = i32,
                              SDPatternOperator ld = null_frag> {

  let mayLoad = 1, mayStore = 0 in {
    let offen = 0, idxen = 0, vaddr = 0 in {
      defm _OFFSET : MUBUF_m <op, name#"_offset", (outs regClass:$vdata),
                           (ins SReg_128:$srsrc, SCSrc_32:$soffset,
                           offset:$offset, glc:$glc, slc:$slc, tfe:$tfe),
                           name#" $vdata, off, $srsrc, $soffset$offset$glc$slc$tfe",
                           [(set load_vt:$vdata, (ld (MUBUFOffset v4i32:$srsrc,
                                                     i32:$soffset, i16:$offset,
                                                     i1:$glc, i1:$slc, i1:$tfe)))]>;
    }

    let offen = 1, idxen = 0  in {
      defm _OFFEN  : MUBUF_m <op, name#"_offen", (outs regClass:$vdata),
                           (ins VGPR_32:$vaddr, SReg_128:$srsrc,
                           SCSrc_32:$soffset, offset:$offset, glc:$glc, slc:$slc,
                           tfe:$tfe),
                           name#" $vdata, $vaddr, $srsrc, $soffset offen$offset$glc$slc$tfe", []>;
    }

    let offen = 0, idxen = 1 in {
      defm _IDXEN  : MUBUF_m <op, name#"_idxen", (outs regClass:$vdata),
                           (ins VGPR_32:$vaddr, SReg_128:$srsrc,
                           SCSrc_32:$soffset, offset:$offset, glc:$glc,
                           slc:$slc, tfe:$tfe),
                           name#" $vdata, $vaddr, $srsrc, $soffset idxen$offset$glc$slc$tfe", []>;
    }

    let offen = 1, idxen = 1 in {
      defm _BOTHEN : MUBUF_m <op, name#"_bothen", (outs regClass:$vdata),
                           (ins VReg_64:$vaddr, SReg_128:$srsrc, SCSrc_32:$soffset,
                           offset:$offset, glc:$glc, slc:$slc, tfe:$tfe),
                           name#" $vdata, $vaddr, $srsrc, $soffset idxen offen$offset$glc$slc$tfe", []>;
    }

    let offen = 0, idxen = 0 in {
      defm _ADDR64 : MUBUFAddr64_m <op, name#"_addr64", (outs regClass:$vdata),
                           (ins VReg_64:$vaddr, SReg_128:$srsrc,
                                SCSrc_32:$soffset, offset:$offset,
				glc:$glc, slc:$slc, tfe:$tfe),
                           name#" $vdata, $vaddr, $srsrc, $soffset addr64$offset$glc$slc$tfe",
                           [(set load_vt:$vdata, (ld (MUBUFAddr64 v4i32:$srsrc,
                                                  i64:$vaddr, i32:$soffset,
                                                  i16:$offset, i1:$glc, i1:$slc,
						  i1:$tfe)))]>;
    }
  }
}

multiclass MUBUF_Store_Helper <mubuf op, string name, RegisterClass vdataClass,
                          ValueType store_vt = i32, SDPatternOperator st = null_frag> {
  let mayLoad = 0, mayStore = 1 in {
    let offen = 0, idxen = 0, vaddr = 0 in {
      defm _OFFSET : MUBUF_m <op, name#"_offset",(outs),
                              (ins vdataClass:$vdata, SReg_128:$srsrc, SCSrc_32:$soffset,
                              offset:$offset, glc:$glc, slc:$slc, tfe:$tfe),
                              name#" $vdata, off, $srsrc, $soffset$offset$glc$slc$tfe",
                              [(st store_vt:$vdata, (MUBUFOffset v4i32:$srsrc, i32:$soffset,
                                   i16:$offset, i1:$glc, i1:$slc, i1:$tfe))]>;
    } // offen = 0, idxen = 0, vaddr = 0

    let offen = 1, idxen = 0  in {
      defm _OFFEN : MUBUF_m <op, name#"_offen", (outs),
                             (ins vdataClass:$vdata, VGPR_32:$vaddr, SReg_128:$srsrc,
                              SCSrc_32:$soffset, offset:$offset, glc:$glc,
                              slc:$slc, tfe:$tfe),
                             name#" $vdata, $vaddr, $srsrc, $soffset offen"#
                             "$offset$glc$slc$tfe", []>;
    } // end offen = 1, idxen = 0

    let offen = 0, idxen = 1 in {
      defm _IDXEN  : MUBUF_m <op, name#"_idxen", (outs),
                           (ins vdataClass:$vdata, VGPR_32:$vaddr, SReg_128:$srsrc,
                           SCSrc_32:$soffset, offset:$offset, glc:$glc,
                           slc:$slc, tfe:$tfe),
                           name#" $vdata, $vaddr, $srsrc, $soffset idxen$offset$glc$slc$tfe", []>;
    }

    let offen = 1, idxen = 1 in {
      defm _BOTHEN : MUBUF_m <op, name#"_bothen", (outs),
                           (ins vdataClass:$vdata, VReg_64:$vaddr, SReg_128:$srsrc, SCSrc_32:$soffset,
                           offset:$offset, glc:$glc, slc:$slc, tfe:$tfe),
                           name#" $vdata, $vaddr, $srsrc, $soffset idxen offen$offset$glc$slc$tfe", []>;
    }

    let offen = 0, idxen = 0 in {
      defm _ADDR64 : MUBUFAddr64_m <op, name#"_addr64", (outs),
                                    (ins vdataClass:$vdata, VReg_64:$vaddr, SReg_128:$srsrc,
                                         SCSrc_32:$soffset,
                                         offset:$offset, glc:$glc, slc:$slc,
                                         tfe:$tfe),
                                    name#" $vdata, $vaddr, $srsrc, $soffset addr64"#
                                         "$offset$glc$slc$tfe",
                                    [(st store_vt:$vdata,
                                      (MUBUFAddr64 v4i32:$srsrc, i64:$vaddr,
                                                   i32:$soffset, i16:$offset,
                                                   i1:$glc, i1:$slc, i1:$tfe))]>;
    }
  } // End mayLoad = 0, mayStore = 1
}

// For cache invalidation instructions.
multiclass MUBUF_Invalidate <mubuf op, string opName, SDPatternOperator node> {
  let hasSideEffects = 1, mayStore = 1, AsmMatchConverter = "" in {
    def "" : MUBUF_Pseudo <opName, (outs), (ins), [(node)]>;

    // Set everything to 0.
    let offset = 0, offen = 0, idxen = 0, glc = 0, vaddr = 0,
        vdata = 0, srsrc = 0, slc = 0, tfe = 0, soffset = 0 in {
      let addr64 = 0 in {
        def _si : MUBUF_Real_si <op, opName, (outs), (ins), opName>;
      }

      def _vi : MUBUF_Real_vi <op, opName, (outs), (ins), opName>;
    }
  } // End hasSideEffects = 1, mayStore = 1, AsmMatchConverter = ""
}

//===----------------------------------------------------------------------===//
// FLAT classes
//===----------------------------------------------------------------------===//

class flat <bits<7> ci, bits<7> vi = ci> {
  field bits<7> CI = ci;
  field bits<7> VI = vi;
}

class FLAT_Pseudo <string opName, dag outs, dag ins, list<dag> pattern> :
     FLAT <0, outs, ins, "", pattern>,
      SIMCInstr<opName, SIEncodingFamily.NONE> {
  let isPseudo = 1;
  let isCodeGenOnly = 1;
}

class FLAT_Real_ci <bits<7> op, string opName, dag outs, dag ins, string asm> :
    FLAT <op, outs, ins, asm, []>,
    SIMCInstr<opName, SIEncodingFamily.SI> {
  let AssemblerPredicate = isCIOnly;
  let DecoderNamespace="CI";
}

class FLAT_Real_vi <bits<7> op, string opName, dag outs, dag ins, string asm> :
    FLAT <op, outs, ins, asm, []>,
    SIMCInstr<opName, SIEncodingFamily.VI> {
  let AssemblerPredicate = VIAssemblerPredicate;
  let DecoderNamespace="VI";
  let DisableDecoder = DisableVIDecoder;
}

multiclass FLAT_AtomicRet_m <flat op, dag outs, dag ins, string asm,
                   list<dag> pattern> {
  def "" : FLAT_Pseudo <NAME#"_RTN", outs, ins, pattern>,
               AtomicNoRet <NAME, 1>;

  def _ci : FLAT_Real_ci <op.CI, NAME#"_RTN", outs, ins, asm>;

  def _vi : FLAT_Real_vi <op.VI, NAME#"_RTN", outs, ins, asm>;
}

multiclass FLAT_Load_Helper <flat op, string asm_name,
    RegisterClass regClass,
    dag outs = (outs regClass:$vdst),
    dag ins = (ins VReg_64:$addr, glc:$glc, slc:$slc, tfe:$tfe),
    string asm = asm_name#" $vdst, $addr$glc$slc$tfe"> {

  let data = 0, mayLoad = 1 in {

    def "" : FLAT_Pseudo <NAME, outs, ins, []>;

    def _ci : FLAT_Real_ci <op.CI, NAME, outs, ins, asm>;

    def _vi : FLAT_Real_vi <op.VI, NAME, outs, ins, asm>;
  }
}

multiclass FLAT_Store_Helper <flat op, string asm_name,
    RegisterClass vdataClass,
    dag outs = (outs),
    dag ins = (ins VReg_64:$addr, vdataClass:$data, glc:$glc,
                   slc:$slc, tfe:$tfe),
    string asm = asm_name#" $addr, $data$glc$slc$tfe"> {

  let mayLoad = 0, mayStore = 1, vdst = 0 in {

    def "" : FLAT_Pseudo <NAME, outs, ins, []>;

    def _ci : FLAT_Real_ci <op.CI, NAME, outs, ins, asm>;

    def _vi : FLAT_Real_vi <op.VI, NAME, outs, ins, asm>;
  }
}

multiclass FLAT_ATOMIC <flat op, string asm_name, RegisterClass vdst_rc,
    ValueType vt, SDPatternOperator atomic = null_frag,
    ValueType data_vt = vt,
    RegisterClass data_rc = vdst_rc,
    string asm_noret = asm_name#" $addr, $data"#"$slc"#"$tfe"> {

  let mayLoad = 1, mayStore = 1, glc = 0, vdst = 0 in {
    def "" : FLAT_Pseudo <NAME, (outs),
                          (ins VReg_64:$addr, data_rc:$data,
                               slc:$slc, tfe:$tfe), []>,
             AtomicNoRet <NAME, 0>;

    def _ci : FLAT_Real_ci <op.CI, NAME, (outs),
                            (ins VReg_64:$addr, data_rc:$data,
                                 slc:$slc, tfe:$tfe),
                            asm_noret>;

    def _vi : FLAT_Real_vi <op.VI, NAME, (outs),
                            (ins VReg_64:$addr, data_rc:$data,
                                 slc:$slc, tfe:$tfe),
                            asm_noret>;
  }

  let glc = 1, hasPostISelHook = 1 in {
    defm _RTN : FLAT_AtomicRet_m <
      op, (outs vdst_rc:$vdst),
      (ins VReg_64:$addr, data_rc:$data, slc:$slc, tfe:$tfe),
      asm_name#" $vdst, $addr, $data glc$slc$tfe",
      [(set vt:$vdst,
         (atomic (FLATAtomic i64:$addr, i1:$slc, i1:$tfe), data_vt:$data))]
    >;
  }
}

class MIMG_Mask <string op, int channels> {
  string Op = op;
  int Channels = channels;
}

class mimg <bits<7> si, bits<7> vi = si> {
  field bits<7> SI = si;
  field bits<7> VI = vi;
}

class MIMG_Helper <dag outs, dag ins, string asm,
                   string dns=""> : MIMG<outs, ins, asm,[]> {
  let mayLoad = 1;
  let mayStore = 0;
  let hasPostISelHook = 1;
  let DecoderNamespace = dns;
  let isAsmParserOnly = !if(!eq(dns,""), 1, 0);
  let AsmMatchConverter = "cvtMIMG";
}

class MIMG_NoSampler_Helper <bits<7> op, string asm,
                             RegisterClass dst_rc,
                             RegisterClass addr_rc,
                             string dns=""> : MIMG_Helper <
  (outs dst_rc:$vdata),
  (ins addr_rc:$vaddr, SReg_256:$srsrc,
       dmask:$dmask, unorm:$unorm, glc:$glc, slc:$slc,
       r128:$r128, tfe:$tfe, lwe:$lwe, da:$da),
  asm#" $vdata, $vaddr, $srsrc$dmask$unorm$glc$slc$r128$tfe$lwe$da",
  dns>, MIMGe<op> {
  let ssamp = 0;
}

multiclass MIMG_NoSampler_Src_Helper <bits<7> op, string asm,
                                      RegisterClass dst_rc,
                                      int channels> {
  def _V1 : MIMG_NoSampler_Helper <op, asm, dst_rc, VGPR_32,
                                   !if(!eq(channels, 1), "AMDGPU", "")>,
            MIMG_Mask<asm#"_V1", channels>;
  def _V2 : MIMG_NoSampler_Helper <op, asm, dst_rc, VReg_64>,
            MIMG_Mask<asm#"_V2", channels>;
  def _V4 : MIMG_NoSampler_Helper <op, asm, dst_rc, VReg_128>,
            MIMG_Mask<asm#"_V4", channels>;
}

multiclass MIMG_NoSampler <bits<7> op, string asm> {
  defm _V1 : MIMG_NoSampler_Src_Helper <op, asm, VGPR_32, 1>;
  defm _V2 : MIMG_NoSampler_Src_Helper <op, asm, VReg_64, 2>;
  defm _V3 : MIMG_NoSampler_Src_Helper <op, asm, VReg_96, 3>;
  defm _V4 : MIMG_NoSampler_Src_Helper <op, asm, VReg_128, 4>;
}

class MIMG_Store_Helper <bits<7> op, string asm,
                         RegisterClass data_rc,
                         RegisterClass addr_rc> : MIMG_Helper <
  (outs),
  (ins data_rc:$vdata, addr_rc:$vaddr, SReg_256:$srsrc,
       dmask:$dmask, unorm:$unorm, glc:$glc, slc:$slc,
       r128:$r128, tfe:$tfe, lwe:$lwe, da:$da),
  asm#" $vdata, $vaddr, $srsrc$dmask$unorm$glc$slc$r128$tfe$lwe$da"
     >, MIMGe<op> {
  let ssamp = 0;
  let mayLoad = 1; // TableGen requires this for matching with the intrinsics
  let mayStore = 1;
  let hasSideEffects = 1;
  let hasPostISelHook = 0;
  let DisableWQM = 1;
}

multiclass MIMG_Store_Addr_Helper <bits<7> op, string asm,
                                  RegisterClass data_rc,
                                  int channels> {
  def _V1 : MIMG_Store_Helper <op, asm, data_rc, VGPR_32>,
            MIMG_Mask<asm#"_V1", channels>;
  def _V2 : MIMG_Store_Helper <op, asm, data_rc, VReg_64>,
            MIMG_Mask<asm#"_V2", channels>;
  def _V4 : MIMG_Store_Helper <op, asm, data_rc, VReg_128>,
            MIMG_Mask<asm#"_V4", channels>;
}

multiclass MIMG_Store <bits<7> op, string asm> {
  defm _V1 : MIMG_Store_Addr_Helper <op, asm, VGPR_32, 1>;
  defm _V2 : MIMG_Store_Addr_Helper <op, asm, VReg_64, 2>;
  defm _V3 : MIMG_Store_Addr_Helper <op, asm, VReg_96, 3>;
  defm _V4 : MIMG_Store_Addr_Helper <op, asm, VReg_128, 4>;
}

class MIMG_Atomic_Helper <string asm, RegisterClass data_rc,
                          RegisterClass addr_rc> : MIMG_Helper <
    (outs data_rc:$vdst),
    (ins data_rc:$vdata, addr_rc:$vaddr, SReg_256:$srsrc,
         dmask:$dmask, unorm:$unorm, glc:$glc, slc:$slc,
         r128:$r128, tfe:$tfe, lwe:$lwe, da:$da),
    asm#" $vdst, $vaddr, $srsrc$dmask$unorm$glc$slc$r128$tfe$lwe$da"
  > {
  let mayStore = 1;
  let hasSideEffects = 1;
  let hasPostISelHook = 0;
  let DisableWQM = 1;
  let Constraints = "$vdst = $vdata";
  let AsmMatchConverter = "cvtMIMGAtomic";
}

class MIMG_Atomic_Real_si<mimg op, string name, string asm,
  RegisterClass data_rc, RegisterClass addr_rc> :
  MIMG_Atomic_Helper<asm, data_rc, addr_rc>,
  SIMCInstr<name, SIEncodingFamily.SI>,
  MIMGe<op.SI> {
  let isCodeGenOnly = 0;
  let AssemblerPredicates = [isSICI];
  let DecoderNamespace = "SICI";
  let DisableDecoder = DisableSIDecoder;
}

class MIMG_Atomic_Real_vi<mimg op, string name, string asm,
  RegisterClass data_rc, RegisterClass addr_rc> :
  MIMG_Atomic_Helper<asm, data_rc, addr_rc>,
  SIMCInstr<name, SIEncodingFamily.VI>,
  MIMGe<op.VI> {
  let isCodeGenOnly = 0;
  let AssemblerPredicates = [isVI];
  let DecoderNamespace = "VI";
  let DisableDecoder = DisableVIDecoder;
}

multiclass MIMG_Atomic_Helper_m <mimg op, string name, string asm,
                                 RegisterClass data_rc, RegisterClass addr_rc> {
  let isPseudo = 1, isCodeGenOnly = 1 in {
    def "" : MIMG_Atomic_Helper<asm, data_rc, addr_rc>,
             SIMCInstr<name, SIEncodingFamily.NONE>;
  }

  let ssamp = 0 in {
    def _si : MIMG_Atomic_Real_si<op, name, asm, data_rc, addr_rc>;

    def _vi : MIMG_Atomic_Real_vi<op, name, asm, data_rc, addr_rc>;
  }
}

multiclass MIMG_Atomic <mimg op, string asm, RegisterClass data_rc = VGPR_32> {
  defm _V1 : MIMG_Atomic_Helper_m <op, asm # "_V1", asm, data_rc, VGPR_32>;
  defm _V2 : MIMG_Atomic_Helper_m <op, asm # "_V2", asm, data_rc, VReg_64>;
  defm _V4 : MIMG_Atomic_Helper_m <op, asm # "_V3", asm, data_rc, VReg_128>;
}

class MIMG_Sampler_Helper <bits<7> op, string asm,
                           RegisterClass dst_rc,
                           RegisterClass src_rc,
                           int wqm,
                           string dns=""> : MIMG_Helper <
  (outs dst_rc:$vdata),
  (ins src_rc:$vaddr, SReg_256:$srsrc, SReg_128:$ssamp,
       dmask:$dmask, unorm:$unorm, glc:$glc, slc:$slc,
       r128:$r128, tfe:$tfe, lwe:$lwe, da:$da),
  asm#" $vdata, $vaddr, $srsrc, $ssamp$dmask$unorm$glc$slc$r128$tfe$lwe$da",
  dns>, MIMGe<op> {
  let WQM = wqm;
}

multiclass MIMG_Sampler_Src_Helper <bits<7> op, string asm,
                                    RegisterClass dst_rc,
                                    int channels, int wqm> {
  def _V1 : MIMG_Sampler_Helper <op, asm, dst_rc, VGPR_32, wqm,
                                 !if(!eq(channels, 1), "AMDGPU", "")>,
            MIMG_Mask<asm#"_V1", channels>;
  def _V2 : MIMG_Sampler_Helper <op, asm, dst_rc, VReg_64, wqm>,
            MIMG_Mask<asm#"_V2", channels>;
  def _V4 : MIMG_Sampler_Helper <op, asm, dst_rc, VReg_128, wqm>,
            MIMG_Mask<asm#"_V4", channels>;
  def _V8 : MIMG_Sampler_Helper <op, asm, dst_rc, VReg_256, wqm>,
            MIMG_Mask<asm#"_V8", channels>;
  def _V16 : MIMG_Sampler_Helper <op, asm, dst_rc, VReg_512, wqm>,
            MIMG_Mask<asm#"_V16", channels>;
}

multiclass MIMG_Sampler <bits<7> op, string asm, int wqm=0> {
  defm _V1 : MIMG_Sampler_Src_Helper<op, asm, VGPR_32, 1, wqm>;
  defm _V2 : MIMG_Sampler_Src_Helper<op, asm, VReg_64, 2, wqm>;
  defm _V3 : MIMG_Sampler_Src_Helper<op, asm, VReg_96, 3, wqm>;
  defm _V4 : MIMG_Sampler_Src_Helper<op, asm, VReg_128, 4, wqm>;
}

multiclass MIMG_Sampler_WQM <bits<7> op, string asm> : MIMG_Sampler<op, asm, 1>;

class MIMG_Gather_Helper <bits<7> op, string asm,
                          RegisterClass dst_rc,
                          RegisterClass src_rc, int wqm> : MIMG <
  (outs dst_rc:$vdata),
  (ins src_rc:$vaddr, SReg_256:$srsrc, SReg_128:$ssamp,
       dmask:$dmask, unorm:$unorm, glc:$glc, slc:$slc,
       r128:$r128, tfe:$tfe, lwe:$lwe, da:$da),
  asm#" $vdata, $vaddr, $srsrc, $ssamp$dmask$unorm$glc$slc$r128$tfe$lwe$da",
  []>, MIMGe<op> {
  let mayLoad = 1;
  let mayStore = 0;

  // DMASK was repurposed for GATHER4. 4 components are always
  // returned and DMASK works like a swizzle - it selects
  // the component to fetch. The only useful DMASK values are
  // 1=red, 2=green, 4=blue, 8=alpha. (e.g. 1 returns
  // (red,red,red,red) etc.) The ISA document doesn't mention
  // this.
  // Therefore, disable all code which updates DMASK by setting this:
  let Gather4 = 1;
  let hasPostISelHook = 0;
  let WQM = wqm;

  let isAsmParserOnly = 1; // TBD: fix it later
}

multiclass MIMG_Gather_Src_Helper <bits<7> op, string asm,
                                    RegisterClass dst_rc,
                                    int channels, int wqm> {
  def _V1 : MIMG_Gather_Helper <op, asm, dst_rc, VGPR_32, wqm>,
            MIMG_Mask<asm#"_V1", channels>;
  def _V2 : MIMG_Gather_Helper <op, asm, dst_rc, VReg_64, wqm>,
            MIMG_Mask<asm#"_V2", channels>;
  def _V4 : MIMG_Gather_Helper <op, asm, dst_rc, VReg_128, wqm>,
            MIMG_Mask<asm#"_V4", channels>;
  def _V8 : MIMG_Gather_Helper <op, asm, dst_rc, VReg_256, wqm>,
            MIMG_Mask<asm#"_V8", channels>;
  def _V16 : MIMG_Gather_Helper <op, asm, dst_rc, VReg_512, wqm>,
            MIMG_Mask<asm#"_V16", channels>;
}

multiclass MIMG_Gather <bits<7> op, string asm, int wqm=0> {
  defm _V1 : MIMG_Gather_Src_Helper<op, asm, VGPR_32, 1, wqm>;
  defm _V2 : MIMG_Gather_Src_Helper<op, asm, VReg_64, 2, wqm>;
  defm _V3 : MIMG_Gather_Src_Helper<op, asm, VReg_96, 3, wqm>;
  defm _V4 : MIMG_Gather_Src_Helper<op, asm, VReg_128, 4, wqm>;
}

multiclass MIMG_Gather_WQM <bits<7> op, string asm> : MIMG_Gather<op, asm, 1>;

//===----------------------------------------------------------------------===//
// Vector instruction mappings
//===----------------------------------------------------------------------===//

// Maps an opcode in e32 form to its e64 equivalent
def getVOPe64 : InstrMapping {
  let FilterClass = "VOP";
  let RowFields = ["OpName"];
  let ColFields = ["Size"];
  let KeyCol = ["4"];
  let ValueCols = [["8"]];
}

// Maps an opcode in e64 form to its e32 equivalent
def getVOPe32 : InstrMapping {
  let FilterClass = "VOP";
  let RowFields = ["OpName"];
  let ColFields = ["Size"];
  let KeyCol = ["8"];
  let ValueCols = [["4"]];
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
  let FilterClass = "VOP2_REV";
  let RowFields = ["RevOp"];
  let ColFields = ["IsOrig"];
  let KeyCol = ["0"];
  let ValueCols = [["1"]];
}

// Maps an original opcode to its commuted version
def getCommuteRev : InstrMapping {
  let FilterClass = "VOP2_REV";
  let RowFields = ["RevOp"];
  let ColFields = ["IsOrig"];
  let KeyCol = ["1"];
  let ValueCols = [["0"]];
}

def getCommuteCmpOrig : InstrMapping {
  let FilterClass = "VOP2_REV";
  let RowFields = ["RevOp"];
  let ColFields = ["IsOrig"];
  let KeyCol = ["0"];
  let ValueCols = [["1"]];
}

// Maps an original opcode to its commuted version
def getCommuteCmpRev : InstrMapping {
  let FilterClass = "VOP2_REV";
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
include "VIInstructions.td"

include "DSInstructions.td"
