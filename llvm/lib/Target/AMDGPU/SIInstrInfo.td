//===-- SIInstrInfo.td - SI Instruction Infos -------------*- tablegen -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
def isCI : Predicate<"Subtarget->getGeneration() "
                      ">= AMDGPUSubtarget::SEA_ISLANDS">;
def isCIOnly : Predicate<"Subtarget->getGeneration() =="
                         "AMDGPUSubtarget::SEA_ISLANDS">,
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

// Specify an SMRD opcode for SI and SMEM opcode for VI

// FIXME: This should really be bits<5> si, Tablegen crashes if
// parameter default value is other parameter with different bit size
class smrd<bits<8> si, bits<8> vi = si> {
  field bits<5> SI = si{4-0};
  field bits<8> VI = vi;
}

// Execpt for the NONE field, this must be kept in sync with the SISubtarget enum
// in AMDGPUInstrInfo.cpp
def SISubtarget {
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

def SIconstdata_ptr : SDNode<
  "AMDGPUISD::CONST_DATA_PTR", SDTypeProfile <1, 1, [SDTCisVT<0, i64>,
                                                     SDTCisVT<0, i64>]>
>;

//===----------------------------------------------------------------------===//
// PatFrags for FLAT instructions
//===----------------------------------------------------------------------===//

class flat_ld <SDPatternOperator ld> : PatFrag<(ops node:$ptr),
                                               (ld node:$ptr), [{
  return isFlatLoad(dyn_cast<LoadSDNode>(N)) ||
         isGlobalLoad(dyn_cast<LoadSDNode>(N)) ||
         isConstantLoad(cast<LoadSDNode>(N), -1);
}]>;

def flat_load : flat_ld <load>;
def flat_az_extloadi8 : flat_ld <az_extloadi8>;
def flat_sextloadi8 : flat_ld <sextloadi8>;
def flat_az_extloadi16 : flat_ld <az_extloadi16>;
def flat_sextloadi16 : flat_ld <sextloadi16>;

class flat_st <SDPatternOperator st> : PatFrag<(ops node:$val, node:$ptr),
                                               (st node:$val, node:$ptr), [{
  return isFlatStore(dyn_cast<StoreSDNode>(N)) ||
         isGlobalStore(dyn_cast<StoreSDNode>(N));
}]>;

def flat_store: flat_st <store>;
def flat_truncstorei8 : flat_st <truncstorei8>;
def flat_truncstorei16 : flat_st <truncstorei16>;


def mubuf_load : PatFrag <(ops node:$ptr), (load node:$ptr), [{
	return isGlobalLoad(cast<LoadSDNode>(N)) ||
         isConstantLoad(cast<LoadSDNode>(N), -1);
}]>;

def smrd_load : PatFrag <(ops node:$ptr), (load node:$ptr), [{
  return isConstantLoad(cast<LoadSDNode>(N), -1) &&
  static_cast<const SITargetLowering *>(getTargetLowering())->isMemOpUniform(N);
}]>;

//===----------------------------------------------------------------------===//
// SDNodes and PatFrag for local loads and stores to enable s_mov_b32 m0, -1
// to be glued to the memory instructions.
//===----------------------------------------------------------------------===//

def SIld_local : SDNode <"ISD::LOAD", SDTLoad,
  [SDNPHasChain, SDNPMayLoad, SDNPMemOperand, SDNPInGlue]
>;

def si_ld_local : PatFrag <(ops node:$ptr), (SIld_local node:$ptr), [{
  return isLocalLoad(cast<LoadSDNode>(N));
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
  return isLocalStore(cast<StoreSDNode>(N));
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

multiclass SIAtomicM0Glue2 <string op_name> {

  def _glue : SDNode <"ISD::ATOMIC_"#op_name, SDTAtomic2,
    [SDNPHasChain, SDNPMayStore, SDNPMayLoad, SDNPMemOperand, SDNPInGlue]
  >;

  def _local : local_binary_atomic_op <!cast<SDNode>(NAME#"_glue")>;
}

defm si_atomic_load_add : SIAtomicM0Glue2 <"LOAD_ADD">;
defm si_atomic_load_and : SIAtomicM0Glue2 <"LOAD_AND">;
defm si_atomic_load_min : SIAtomicM0Glue2 <"LOAD_MIN">;
defm si_atomic_load_max : SIAtomicM0Glue2 <"LOAD_MAX">;
defm si_atomic_load_or : SIAtomicM0Glue2 <"LOAD_OR">;
defm si_atomic_load_sub : SIAtomicM0Glue2 <"LOAD_SUB">;
defm si_atomic_load_xor : SIAtomicM0Glue2 <"LOAD_XOR">;
defm si_atomic_load_umin : SIAtomicM0Glue2 <"LOAD_UMIN">;
defm si_atomic_load_umax : SIAtomicM0Glue2 <"LOAD_UMAX">;
defm si_atomic_swap : SIAtomicM0Glue2 <"SWAP">;

def si_atomic_cmp_swap_glue : SDNode <"ISD::ATOMIC_CMP_SWAP", SDTAtomic3,
  [SDNPHasChain, SDNPMayStore, SDNPMayLoad, SDNPMemOperand, SDNPInGlue]
>;

defm si_atomic_cmp_swap : AtomicCmpSwapLocal <si_atomic_cmp_swap_glue>;

// Transformation function, extract the lower 32bit of a 64bit immediate
def LO32 : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(N->getZExtValue() & 0xffffffff, SDLoc(N),
                                   MVT::i32);
}]>;

def LO32f : SDNodeXForm<fpimm, [{
  APInt V = N->getValueAPF().bitcastToAPInt().trunc(32);
  return CurDAG->getTargetConstantFP(APFloat(APFloat::IEEEsingle, V), MVT::f32);
}]>;

// Transformation function, extract the upper 32bit of a 64bit immediate
def HI32 : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(N->getZExtValue() >> 32, SDLoc(N), MVT::i32);
}]>;

def HI32f : SDNodeXForm<fpimm, [{
  APInt V = N->getValueAPF().bitcastToAPInt().lshr(32).trunc(32);
  return CurDAG->getTargetConstantFP(APFloat(APFloat::IEEEsingle, V), SDLoc(N),
                                     MVT::f32);
}]>;

def IMM8bitDWORD : PatLeaf <(imm),
  [{return (N->getZExtValue() & ~0x3FC) == 0;}]
>;

def as_dword_i32imm : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(N->getZExtValue() >> 2, SDLoc(N), MVT::i32);
}]>;

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

def IMM8bit : PatLeaf <(imm),
  [{return isUInt<8>(N->getZExtValue());}]
>;

def IMM12bit : PatLeaf <(imm),
  [{return isUInt<12>(N->getZExtValue());}]
>;

def IMM16bit : PatLeaf <(imm),
  [{return isUInt<16>(N->getZExtValue());}]
>;

def IMM20bit : PatLeaf <(imm),
  [{return isUInt<20>(N->getZExtValue());}]
>;

def IMM32bit : PatLeaf <(imm),
  [{return isUInt<32>(N->getZExtValue());}]
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
  if (Subtarget->getGeneration() < AMDGPUSubtarget::SOUTHERN_ISLANDS) {
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

def FRAMEri32 : Operand<iPTR> {
  let MIOperandInfo = (ops i32:$ptr, i32imm:$index);
}

def SoppBrTarget : AsmOperandClass {
  let Name = "SoppBrTarget";
  let ParserMethod = "parseSOppBrTarget";
}

def sopp_brtarget : Operand<OtherVT> {
  let EncoderMethod = "getSOPPBrEncoding";
  let OperandType = "OPERAND_PCREL";
  let ParserMatchClass = SoppBrTarget;
}

def const_ga : Operand<iPTR>;

include "SIInstrFormats.td"
include "VIInstrFormats.td"

def MubufOffsetMatchClass : AsmOperandClass {
  let Name = "MubufOffset";
  let ParserMethod = "parseMubufOptionalOps";
  let RenderMethod = "addImmOperands";
}

class DSOffsetBaseMatchClass <string parser> : AsmOperandClass {
  let Name = "DSOffset"#parser;
  let ParserMethod = parser;
  let RenderMethod = "addImmOperands";
  let PredicateMethod = "isDSOffset";
}

def DSOffsetMatchClass : DSOffsetBaseMatchClass <"parseDSOptionalOps">;
def DSOffsetGDSMatchClass : DSOffsetBaseMatchClass <"parseDSOffsetOptional">;

def DSOffset01MatchClass : AsmOperandClass {
  let Name = "DSOffset1";
  let ParserMethod = "parseDSOff01OptionalOps";
  let RenderMethod = "addImmOperands";
  let PredicateMethod = "isDSOffset01";
}

class GDSBaseMatchClass <string parser> : AsmOperandClass {
  let Name = "GDS"#parser;
  let PredicateMethod = "isImm";
  let ParserMethod = parser;
  let RenderMethod = "addImmOperands";
}

def GDSMatchClass : GDSBaseMatchClass <"parseDSOptionalOps">;
def GDS01MatchClass : GDSBaseMatchClass <"parseDSOff01OptionalOps">;

class GLCBaseMatchClass <string parser> : AsmOperandClass {
  let Name = "GLC"#parser;
  let PredicateMethod = "isImm";
  let ParserMethod = parser;
  let RenderMethod = "addImmOperands";
}

def GLCMubufMatchClass : GLCBaseMatchClass <"parseMubufOptionalOps">;
def GLCFlatMatchClass : GLCBaseMatchClass <"parseFlatOptionalOps">;

class SLCBaseMatchClass <string parser> : AsmOperandClass {
  let Name = "SLC"#parser;
  let PredicateMethod = "isImm";
  let ParserMethod = parser;
  let RenderMethod = "addImmOperands";
}

def SLCMubufMatchClass : SLCBaseMatchClass <"parseMubufOptionalOps">;
def SLCFlatMatchClass : SLCBaseMatchClass <"parseFlatOptionalOps">;
def SLCFlatAtomicMatchClass : SLCBaseMatchClass <"parseFlatAtomicOptionalOps">;

class TFEBaseMatchClass <string parser> : AsmOperandClass {
  let Name = "TFE"#parser;
  let PredicateMethod = "isImm";
  let ParserMethod = parser;
  let RenderMethod = "addImmOperands";
}

def TFEMubufMatchClass : TFEBaseMatchClass <"parseMubufOptionalOps">;
def TFEFlatMatchClass : TFEBaseMatchClass <"parseFlatOptionalOps">;
def TFEFlatAtomicMatchClass : TFEBaseMatchClass <"parseFlatAtomicOptionalOps">;

def OModMatchClass : AsmOperandClass {
  let Name = "OMod";
  let PredicateMethod = "isImm";
  let ParserMethod = "parseVOP3OptionalOps";
  let RenderMethod = "addImmOperands";
}

def ClampMatchClass : AsmOperandClass {
  let Name = "Clamp";
  let PredicateMethod = "isImm";
  let ParserMethod = "parseVOP3OptionalOps";
  let RenderMethod = "addImmOperands";
}

class SMRDOffsetBaseMatchClass <string predicate> : AsmOperandClass {
  let Name = "SMRDOffset"#predicate;
  let PredicateMethod = predicate;
  let RenderMethod = "addImmOperands";
}

def SMRDOffsetMatchClass : SMRDOffsetBaseMatchClass <"isSMRDOffset">;
def SMRDLiteralOffsetMatchClass : SMRDOffsetBaseMatchClass <
  "isSMRDLiteralOffset"
>;

let OperandType = "OPERAND_IMMEDIATE" in {

def offen : Operand<i1> {
  let PrintMethod = "printOffen";
}
def idxen : Operand<i1> {
  let PrintMethod = "printIdxen";
}
def addr64 : Operand<i1> {
  let PrintMethod = "printAddr64";
}
def mbuf_offset : Operand<i16> {
  let PrintMethod = "printMBUFOffset";
  let ParserMatchClass = MubufOffsetMatchClass;
}
class ds_offset_base <AsmOperandClass mc> : Operand<i16> {
  let PrintMethod = "printDSOffset";
  let ParserMatchClass = mc;
}
def ds_offset : ds_offset_base <DSOffsetMatchClass>;
def ds_offset_gds : ds_offset_base <DSOffsetGDSMatchClass>;

def ds_offset0 : Operand<i8> {
  let PrintMethod = "printDSOffset0";
  let ParserMatchClass = DSOffset01MatchClass;
}
def ds_offset1 : Operand<i8> {
  let PrintMethod = "printDSOffset1";
  let ParserMatchClass = DSOffset01MatchClass;
}
class gds_base <AsmOperandClass mc> : Operand <i1> {
  let PrintMethod = "printGDS";
  let ParserMatchClass = mc;
}
def gds : gds_base <GDSMatchClass>;

def gds01 : gds_base <GDS01MatchClass>;

class glc_base <AsmOperandClass mc> : Operand <i1> {
  let PrintMethod = "printGLC";
  let ParserMatchClass = mc;
}

def glc : glc_base <GLCMubufMatchClass>;
def glc_flat : glc_base <GLCFlatMatchClass>;

class slc_base <AsmOperandClass mc> : Operand <i1> {
  let PrintMethod = "printSLC";
  let ParserMatchClass = mc;
}

def slc : slc_base <SLCMubufMatchClass>;
def slc_flat : slc_base <SLCFlatMatchClass>;
def slc_flat_atomic : slc_base <SLCFlatAtomicMatchClass>;

class tfe_base <AsmOperandClass mc> : Operand <i1> {
  let PrintMethod = "printTFE";
  let ParserMatchClass = mc;
}

def tfe : tfe_base <TFEMubufMatchClass>;
def tfe_flat : tfe_base <TFEFlatMatchClass>;
def tfe_flat_atomic : tfe_base <TFEFlatAtomicMatchClass>;

def omod : Operand <i32> {
  let PrintMethod = "printOModSI";
  let ParserMatchClass = OModMatchClass;
}

def ClampMod : Operand <i1> {
  let PrintMethod = "printClampSI";
  let ParserMatchClass = ClampMatchClass;
}

def smrd_offset : Operand <i32> {
  let PrintMethod = "printU32ImmOperand";
  let ParserMatchClass = SMRDOffsetMatchClass;
}

def smrd_literal_offset : Operand <i32> {
  let PrintMethod = "printU32ImmOperand";
  let ParserMatchClass = SMRDLiteralOffsetMatchClass;
}

} // End OperandType = "OPERAND_IMMEDIATE"

def VOPDstS64 : VOPDstOperand <SReg_64>;

//===----------------------------------------------------------------------===//
// Complex patterns
//===----------------------------------------------------------------------===//

def DS1Addr1Offset : ComplexPattern<i32, 2, "SelectDS1Addr1Offset">;
def DS64Bit4ByteAligned : ComplexPattern<i32, 3, "SelectDS64Bit4ByteAligned">;

def MUBUFAddr32 : ComplexPattern<i64, 9, "SelectMUBUFAddr32">;
def MUBUFAddr64 : ComplexPattern<i64, 7, "SelectMUBUFAddr64">;
def MUBUFAddr64Atomic : ComplexPattern<i64, 5, "SelectMUBUFAddr64">;
def MUBUFScratch : ComplexPattern<i64, 4, "SelectMUBUFScratch">;
def MUBUFOffset : ComplexPattern<i64, 6, "SelectMUBUFOffset">;
def MUBUFOffsetAtomic : ComplexPattern<i64, 4, "SelectMUBUFOffset">;

def SMRDImm   : ComplexPattern<i64, 2, "SelectSMRDImm">;
def SMRDImm32 : ComplexPattern<i64, 2, "SelectSMRDImm32">;
def SMRDSgpr  : ComplexPattern<i64, 2, "SelectSMRDSgpr">;
def SMRDBufferImm   : ComplexPattern<i32, 1, "SelectSMRDBufferImm">;
def SMRDBufferImm32 : ComplexPattern<i32, 1, "SelectSMRDBufferImm32">;
def SMRDBufferSgpr  : ComplexPattern<i32, 1, "SelectSMRDBufferSgpr">;

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
}

multiclass EXP_m {

  let isPseudo = 1, isCodeGenOnly = 1 in {
    def "" : EXPCommon, SIMCInstr <"exp", SISubtarget.NONE> ;
  }

  def _si : EXPCommon, SIMCInstr <"exp", SISubtarget.SI>, EXPe;

  def _vi : EXPCommon, SIMCInstr <"exp", SISubtarget.VI>, EXPe_vi;
}

//===----------------------------------------------------------------------===//
// Scalar classes
//===----------------------------------------------------------------------===//

class SOP1_Pseudo <string opName, dag outs, dag ins, list<dag> pattern> :
  SOP1 <outs, ins, "", pattern>,
  SIMCInstr<opName, SISubtarget.NONE> {
  let isPseudo = 1;
  let isCodeGenOnly = 1;
}

class SOP1_Real_si <sop1 op, string opName, dag outs, dag ins, string asm> :
  SOP1 <outs, ins, asm, []>,
  SOP1e <op.SI>,
  SIMCInstr<opName, SISubtarget.SI> {
  let isCodeGenOnly = 0;
  let AssemblerPredicates = [isSICI];
}

class SOP1_Real_vi <sop1 op, string opName, dag outs, dag ins, string asm> :
  SOP1 <outs, ins, asm, []>,
  SOP1e <op.VI>,
  SIMCInstr<opName, SISubtarget.VI> {
  let isCodeGenOnly = 0;
  let AssemblerPredicates = [isVI];
}

multiclass SOP1_m <sop1 op, string opName, dag outs, dag ins, string asm,
                   list<dag> pattern> {

  def "" : SOP1_Pseudo <opName, outs, ins, pattern>;

  def _si : SOP1_Real_si <op, opName, outs, ins, asm>;

  def _vi : SOP1_Real_vi <op, opName, outs, ins, asm>;

}

multiclass SOP1_32 <sop1 op, string opName, list<dag> pattern> : SOP1_m <
    op, opName, (outs SReg_32:$dst), (ins SSrc_32:$src0),
    opName#" $dst, $src0", pattern
>;

multiclass SOP1_64 <sop1 op, string opName, list<dag> pattern> : SOP1_m <
    op, opName, (outs SReg_64:$dst), (ins SSrc_64:$src0),
    opName#" $dst, $src0", pattern
>;

// no input, 64-bit output.
multiclass SOP1_64_0 <sop1 op, string opName, list<dag> pattern> {
  def "" : SOP1_Pseudo <opName, (outs SReg_64:$dst), (ins), pattern>;

  def _si : SOP1_Real_si <op, opName, (outs SReg_64:$dst), (ins),
    opName#" $dst"> {
    let ssrc0 = 0;
  }

  def _vi : SOP1_Real_vi <op, opName, (outs SReg_64:$dst), (ins),
    opName#" $dst"> {
    let ssrc0 = 0;
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
    op, opName, (outs SReg_32:$dst), (ins SSrc_64:$src0),
    opName#" $dst, $src0", pattern
>;

class SOP2_Pseudo<string opName, dag outs, dag ins, list<dag> pattern> :
  SOP2<outs, ins, "", pattern>,
  SIMCInstr<opName, SISubtarget.NONE> {
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
  SIMCInstr<opName, SISubtarget.SI> {
  let AssemblerPredicates = [isSICI];
}

class SOP2_Real_vi<sop2 op, string opName, dag outs, dag ins, string asm> :
  SOP2<outs, ins, asm, []>,
  SOP2e<op.VI>,
  SIMCInstr<opName, SISubtarget.VI> {
  let AssemblerPredicates = [isVI];
}

multiclass SOP2_m <sop2 op, string opName, dag outs, dag ins, string asm,
                   list<dag> pattern> {

  def "" : SOP2_Pseudo <opName, outs, ins, pattern>;

  def _si : SOP2_Real_si <op, opName, outs, ins, asm>;

  def _vi : SOP2_Real_vi <op, opName, outs, ins, asm>;

}

multiclass SOP2_32 <sop2 op, string opName, list<dag> pattern> : SOP2_m <
    op, opName, (outs SReg_32:$dst), (ins SSrc_32:$src0, SSrc_32:$src1),
    opName#" $dst, $src0, $src1", pattern
>;

multiclass SOP2_64 <sop2 op, string opName, list<dag> pattern> : SOP2_m <
    op, opName, (outs SReg_64:$dst), (ins SSrc_64:$src0, SSrc_64:$src1),
    opName#" $dst, $src0, $src1", pattern
>;

multiclass SOP2_64_32 <sop2 op, string opName, list<dag> pattern> : SOP2_m <
    op, opName, (outs SReg_64:$dst), (ins SSrc_64:$src0, SSrc_32:$src1),
    opName#" $dst, $src0, $src1", pattern
>;

class SOPC_Helper <bits<7> op, RegisterOperand rc, ValueType vt,
                    string opName, PatLeaf cond> : SOPC <
  op, (outs), (ins rc:$src0, rc:$src1),
  opName#" $src0, $src1", []> {
  let Defs = [SCC];
}

class SOPC_32<bits<7> op, string opName, PatLeaf cond = COND_NULL>
  : SOPC_Helper<op, SSrc_32, i32, opName, cond>;

class SOPC_64<bits<7> op, string opName, PatLeaf cond = COND_NULL>
  : SOPC_Helper<op, SSrc_64, i64, opName, cond>;

class SOPK_Pseudo <string opName, dag outs, dag ins, list<dag> pattern> :
  SOPK <outs, ins, "", pattern>,
  SIMCInstr<opName, SISubtarget.NONE> {
  let isPseudo = 1;
  let isCodeGenOnly = 1;
}

class SOPK_Real_si <sopk op, string opName, dag outs, dag ins, string asm> :
  SOPK <outs, ins, asm, []>,
  SOPKe <op.SI>,
  SIMCInstr<opName, SISubtarget.SI> {
  let AssemblerPredicates = [isSICI];
  let isCodeGenOnly = 0;
}

class SOPK_Real_vi <sopk op, string opName, dag outs, dag ins, string asm> :
  SOPK <outs, ins, asm, []>,
  SOPKe <op.VI>,
  SIMCInstr<opName, SISubtarget.VI> {
  let AssemblerPredicates = [isVI];
  let isCodeGenOnly = 0;
}

multiclass SOPK_m <sopk op, string opName, dag outs, dag ins, string opAsm,
                   string asm = opName#opAsm> {
  def "" : SOPK_Pseudo <opName, outs, ins, []>;

  def _si : SOPK_Real_si <op, opName, outs, ins, asm>;

  def _vi : SOPK_Real_vi <op, opName, outs, ins, asm>;

}

multiclass SOPK_32 <sopk op, string opName, list<dag> pattern> {
  def "" : SOPK_Pseudo <opName, (outs SReg_32:$dst), (ins u16imm:$src0),
    pattern>;

  def _si : SOPK_Real_si <op, opName, (outs SReg_32:$dst), (ins u16imm:$src0),
    opName#" $dst, $src0">;

  def _vi : SOPK_Real_vi <op, opName, (outs SReg_32:$dst), (ins u16imm:$src0),
    opName#" $dst, $src0">;
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
            SIMCInstr<opName, SISubtarget.SI> {
              let AssemblerPredicates = [isSICI];
              let isCodeGenOnly = 0;
            }

  def _vi : SOPK <outs, ins, asm, []>,
            SOPK64e <op.VI>,
            SIMCInstr<opName, SISubtarget.VI> {
              let AssemblerPredicates = [isVI];
              let isCodeGenOnly = 0;
            }
}
//===----------------------------------------------------------------------===//
// SMRD classes
//===----------------------------------------------------------------------===//

class SMRD_Pseudo <string opName, dag outs, dag ins, list<dag> pattern> :
  SMRD <outs, ins, "", pattern>,
  SIMCInstr<opName, SISubtarget.NONE> {
  let isPseudo = 1;
  let isCodeGenOnly = 1;
}

class SMRD_Real_si <bits<5> op, string opName, bit imm, dag outs, dag ins,
                    string asm> :
  SMRD <outs, ins, asm, []>,
  SMRDe <op, imm>,
  SIMCInstr<opName, SISubtarget.SI> {
  let AssemblerPredicates = [isSICI];
}

class SMRD_Real_vi <bits<8> op, string opName, bit imm, dag outs, dag ins,
                    string asm, list<dag> pattern = []> :
  SMRD <outs, ins, asm, pattern>,
  SMEMe_vi <op, imm>,
  SIMCInstr<opName, SISubtarget.VI> {
  let AssemblerPredicates = [isVI];
}

multiclass SMRD_m <smrd op, string opName, bit imm, dag outs, dag ins,
                   string asm, list<dag> pattern> {

  def "" : SMRD_Pseudo <opName, outs, ins, pattern>;

  def _si : SMRD_Real_si <op.SI, opName, imm, outs, ins, asm>;

  // glc is only applicable to scalar stores, which are not yet
  // implemented.
  let glc = 0 in {
    def _vi : SMRD_Real_vi <op.VI, opName, imm, outs, ins, asm>;
  }
}

multiclass SMRD_Inval <smrd op, string opName,
                       SDPatternOperator node> {
  let hasSideEffects = 1, mayStore = 1 in {
    def "" : SMRD_Pseudo <opName, (outs), (ins), [(node)]>;

    let sbase = 0, offset = 0 in {
      let sdst = 0 in {
        def _si : SMRD_Real_si <op.SI, opName, 0, (outs), (ins), opName>;
      }

      let glc = 0, sdata = 0 in {
        def _vi : SMRD_Real_vi <op.VI, opName, 0, (outs), (ins), opName>;
      }
    }
  }
}

class SMEM_Inval <bits<8> op, string opName, SDPatternOperator node> :
  SMRD_Real_vi<op, opName, 0, (outs), (ins), opName, [(node)]> {
  let hasSideEffects = 1;
  let mayStore = 1;
  let sbase = 0;
  let sdata = 0;
  let glc = 0;
  let offset = 0;
}

multiclass SMRD_Helper <smrd op, string opName, RegisterClass baseClass,
                        RegisterClass dstClass> {
  defm _IMM : SMRD_m <
    op, opName#"_IMM", 1, (outs dstClass:$dst),
    (ins baseClass:$sbase, smrd_offset:$offset),
    opName#" $dst, $sbase, $offset", []
  >;

  def _IMM_ci : SMRD <
    (outs dstClass:$dst), (ins baseClass:$sbase, smrd_literal_offset:$offset),
    opName#" $dst, $sbase, $offset", []>, SMRD_IMMe_ci <op.SI> {
    let AssemblerPredicates = [isCIOnly];
  }

  defm _SGPR : SMRD_m <
    op, opName#"_SGPR", 0, (outs dstClass:$dst),
    (ins baseClass:$sbase, SReg_32:$soff),
    opName#" $dst, $sbase, $soff", []
  >;
}

//===----------------------------------------------------------------------===//
// Vector ALU classes
//===----------------------------------------------------------------------===//

// This must always be right before the operand being input modified.
def InputMods : OperandWithDefaultOps <i32, (ops (i32 0))> {
  let PrintMethod = "printOperandAndMods";
}

def InputModsMatchClass : AsmOperandClass {
  let Name = "RegWithInputMods";
}

def InputModsNoDefault : Operand <i32> {
  let PrintMethod = "printOperandAndMods";
  let ParserMatchClass = InputModsMatchClass;
}

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

// Returns the register class to use for source 1 of VOP[12C] for the
// given VT.
class getVOPSrc1ForVT<ValueType VT> {
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
  bit ret = !if(!eq(SrcVT.Value, f32.Value), 1,
            !if(!eq(SrcVT.Value, f64.Value), 1, 0));
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
    !if (!eq(NumSrcArgs, 1),
      !if (!eq(HasModifiers, 1),
        // VOP1 with modifiers
        (ins InputModsNoDefault:$src0_modifiers, Src0RC:$src0,
             ClampMod:$clamp, omod:$omod)
      /* else */,
        // VOP1 without modifiers
        (ins Src0RC:$src0)
      /* endif */ ),
    !if (!eq(NumSrcArgs, 2),
      !if (!eq(HasModifiers, 1),
        // VOP 2 with modifiers
        (ins InputModsNoDefault:$src0_modifiers, Src0RC:$src0,
             InputModsNoDefault:$src1_modifiers, Src1RC:$src1,
             ClampMod:$clamp, omod:$omod)
      /* else */,
        // VOP2 without modifiers
        (ins Src0RC:$src0, Src1RC:$src1)
      /* endif */ )
    /* NumSrcArgs == 3 */,
      !if (!eq(HasModifiers, 1),
        // VOP3 with modifiers
        (ins InputModsNoDefault:$src0_modifiers, Src0RC:$src0,
             InputModsNoDefault:$src1_modifiers, Src1RC:$src1,
             InputModsNoDefault:$src2_modifiers, Src2RC:$src2,
             ClampMod:$clamp, omod:$omod)
      /* else */,
        // VOP3 without modifiers
        (ins Src0RC:$src0, Src1RC:$src1, Src2RC:$src2)
      /* endif */ )));
}

// Returns the assembly string for the inputs and outputs of a VOP[12C]
// instruction.  This does not add the _e32 suffix, so it can be reused
// by getAsm64.
class getAsm32 <bit HasDst, int NumSrcArgs> {
  string dst = "$dst";
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
class getAsm64 <bit HasDst, int NumSrcArgs, bit HasModifiers> {
  string src0 = !if(!eq(NumSrcArgs, 1), "$src0_modifiers", "$src0_modifiers,");
  string src1 = !if(!eq(NumSrcArgs, 1), "",
                   !if(!eq(NumSrcArgs, 2), " $src1_modifiers",
                                           " $src1_modifiers,"));
  string src2 = !if(!eq(NumSrcArgs, 3), " $src2_modifiers", "");
  string ret =
  !if(!eq(HasModifiers, 0),
      getAsm32<HasDst, NumSrcArgs>.ret,
      "$dst, "#src0#src1#src2#"$clamp"#"$omod");
}

class VOPProfile <list<ValueType> _ArgVT> {

  field list<ValueType> ArgVT = _ArgVT;

  field ValueType DstVT = ArgVT[0];
  field ValueType Src0VT = ArgVT[1];
  field ValueType Src1VT = ArgVT[2];
  field ValueType Src2VT = ArgVT[3];
  field RegisterOperand DstRC = getVALUDstForVT<DstVT>.ret;
  field RegisterOperand Src0RC32 = getVOPSrc0ForVT<Src0VT>.ret;
  field RegisterClass Src1RC32 = getVOPSrc1ForVT<Src1VT>.ret;
  field RegisterOperand Src0RC64 = getVOP3SrcForVT<Src0VT>.ret;
  field RegisterOperand Src1RC64 = getVOP3SrcForVT<Src1VT>.ret;
  field RegisterOperand Src2RC64 = getVOP3SrcForVT<Src2VT>.ret;

  field bit HasDst = !if(!eq(DstVT.Value, untyped.Value), 0, 1);
  field bit HasDst32 = HasDst;
  field int NumSrcArgs = getNumSrcArgs<Src0VT, Src1VT, Src2VT>.ret;
  field bit HasModifiers = hasModifiers<Src0VT>.ret;

  field dag Outs = !if(HasDst,(outs DstRC:$dst),(outs));

  // VOP3b instructions are a special case with a second explicit
  // output. This is manually overridden for them.
  field dag Outs32 = Outs;
  field dag Outs64 = Outs;

  field dag Ins32 = getIns32<Src0RC32, Src1RC32, NumSrcArgs>.ret;
  field dag Ins64 = getIns64<Src0RC64, Src1RC64, Src2RC64, NumSrcArgs,
                             HasModifiers>.ret;

  field string Asm32 = getAsm32<HasDst, NumSrcArgs>.ret;
  field string Asm64 = getAsm64<HasDst, NumSrcArgs, HasModifiers>.ret;
}

// FIXME: I think these F16/I16 profiles will need to use f16/i16 types in order
//        for the instruction patterns to work.
def VOP_F16_F16 : VOPProfile <[f16, f16, untyped, untyped]>;
def VOP_F16_I16 : VOPProfile <[f16, i32, untyped, untyped]>;
def VOP_I16_F16 : VOPProfile <[i32, f16, untyped, untyped]>;

def VOP_F16_F16_F16 : VOPProfile <[f16, f16, f16, untyped]>;
def VOP_F16_F16_I16 : VOPProfile <[f16, f16, i32, untyped]>;
def VOP_I16_I16_I16 : VOPProfile <[i32, i32, i32, untyped]>;

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

// Write out to vcc or arbitrary SGPR.
def VOP2b_I32_I1_I32_I32 : VOPProfile<[i32, i32, i32, untyped]> {
  let Asm32 = "$dst, vcc, $src0, $src1";
  let Asm64 = "$dst, $sdst, $src0, $src1";
  let Outs32 = (outs DstRC:$dst);
  let Outs64 = (outs DstRC:$dst, SReg_64:$sdst);
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
  let Asm32 = "$dst, vcc, $src0, $src1, vcc";
  let Asm64 = "$dst, $sdst, $src0, $src1, $src2";
  let Outs32 = (outs DstRC:$dst);
  let Outs64 = (outs DstRC:$dst, SReg_64:$sdst);

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
}

class VOPC_Class_Profile<ValueType vt> : VOPC_Profile<vt, i32> {
  let Ins64 = (ins InputModsNoDefault:$src0_modifiers, Src0RC64:$src0, Src1RC64:$src1);
  let Asm64 = "$dst, $src0_modifiers, $src1";
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
def VOP_CNDMASK : VOPProfile <[i32, i32, i32, untyped]> {
  let Ins32 = (ins Src0RC32:$src0, Src1RC32:$src1);
  let Ins64 = (ins Src0RC64:$src0, Src1RC64:$src1, SSrc_64:$src2);
  let Asm64 = "$dst, $src0, $src1, $src2";
}

def VOP_F32_F32_F32_F32 : VOPProfile <[f32, f32, f32, f32]>;
def VOP_MADK : VOPProfile <[f32, f32, f32, f32]> {
  field dag Ins = (ins VCSrc_32:$src0, VGPR_32:$vsrc1, u32imm:$src2);
  field string Asm = "$dst, $src0, $vsrc1, $src2";
}
def VOP_MAC : VOPProfile <[f32, f32, f32, f32]> {
  let Ins32 = (ins Src0RC32:$src0, Src1RC32:$src1, VGPR_32:$src2);
  let Ins64 = getIns64<Src0RC64, Src1RC64, RegisterOperand<VGPR_32>, 3,
                             HasModifiers>.ret;
  let Asm32 = getAsm32<1, 2>.ret;
  let Asm64 = getAsm64<1, 2, HasModifiers>.ret;
}
def VOP_F64_F64_F64_F64 : VOPProfile <[f64, f64, f64, f64]>;
def VOP_I32_I32_I32_I32 : VOPProfile <[i32, i32, i32, i32]>;
def VOP_I64_I32_I32_I64 : VOPProfile <[i64, i32, i32, i64]>;

class SIInstAlias <string asm, Instruction inst, VOPProfile p> :
    InstAlias <asm, (inst)>, PredicateControl {

  field bit isCompare;
  field bit isCommutable;

  let ResultInst =
    !if (p.HasDst32,
      !if (!eq(p.NumSrcArgs, 0),
        // 1 dst, 0 src
        (inst p.DstRC:$dst),
      !if (!eq(p.NumSrcArgs, 1),
        // 1 dst, 1 src
        (inst p.DstRC:$dst, p.Src0RC32:$src0),
      !if (!eq(p.NumSrcArgs, 2),
        // 1 dst, 2 src
        (inst p.DstRC:$dst, p.Src0RC32:$src0, p.Src1RC32:$src1),
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
  SIMCInstr <opName#"_e32", SISubtarget.NONE>,
  MnemonicAlias<opName#"_e32", opName> {
  let isPseudo = 1;
  let isCodeGenOnly = 1;

  field bits<8> vdst;
  field bits<9> src0;
}

class VOP1_Real_si <string opName, vop1 op, dag outs, dag ins, string asm> :
  VOP1<op.SI, outs, ins, asm, []>,
  SIMCInstr <opName#"_e32", SISubtarget.SI> {
  let AssemblerPredicate = SIAssemblerPredicate;
}

class VOP1_Real_vi <string opName, vop1 op, dag outs, dag ins, string asm> :
  VOP1<op.VI, outs, ins, asm, []>,
  SIMCInstr <opName#"_e32", SISubtarget.VI> {
  let AssemblerPredicates = [isVI];
}

multiclass VOP1_m <vop1 op, string opName, VOPProfile p, list<dag> pattern,
                   string asm = opName#p.Asm32> {
  def "" : VOP1_Pseudo <p.Outs, p.Ins32, pattern, opName>;

  def _si : VOP1_Real_si <opName, op, p.Outs, p.Ins32, asm>;

  def _vi : VOP1_Real_vi <opName, op, p.Outs, p.Ins32, asm>;

}

multiclass VOP1SI_m <vop1 op, string opName, VOPProfile p, list<dag> pattern,
                     string asm = opName#p.Asm32> {

  def "" : VOP1_Pseudo <p.Outs, p.Ins32, pattern, opName>;

  def _si : VOP1_Real_si <opName, op, p.Outs, p.Ins32, asm>;
}

class VOP2_Pseudo <dag outs, dag ins, list<dag> pattern, string opName> :
  VOP2Common <outs, ins, "", pattern>,
  VOP <opName>,
  SIMCInstr<opName#"_e32", SISubtarget.NONE>,
  MnemonicAlias<opName#"_e32", opName> {
  let isPseudo = 1;
  let isCodeGenOnly = 1;
}

class VOP2_Real_si <string opName, vop2 op, dag outs, dag ins, string asm> :
  VOP2 <op.SI, outs, ins, opName#asm, []>,
  SIMCInstr <opName#"_e32", SISubtarget.SI> {
  let AssemblerPredicates = [isSICI];
}

class VOP2_Real_vi <string opName, vop2 op, dag outs, dag ins, string asm> :
  VOP2 <op.VI, outs, ins, opName#asm, []>,
  SIMCInstr <opName#"_e32", SISubtarget.VI> {
  let AssemblerPredicates = [isVI];
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

class VOP3_Pseudo <dag outs, dag ins, list<dag> pattern, string opName> :
  VOP3Common <outs, ins, "", pattern>,
  VOP <opName>,
  SIMCInstr<opName#"_e64", SISubtarget.NONE>,
  MnemonicAlias<opName#"_e64", opName> {
  let isPseudo = 1;
  let isCodeGenOnly = 1;

  field bit vdst;
  field bit src0;
}

class VOP3_Real_si <bits<9> op, dag outs, dag ins, string asm, string opName> :
  VOP3Common <outs, ins, asm, []>,
  VOP3e <op>,
  SIMCInstr<opName#"_e64", SISubtarget.SI> {
  let AssemblerPredicates = [isSICI];
}

class VOP3_Real_vi <bits<10> op, dag outs, dag ins, string asm, string opName> :
  VOP3Common <outs, ins, asm, []>,
  VOP3e_vi <op>,
  SIMCInstr <opName#"_e64", SISubtarget.VI> {
  let AssemblerPredicates = [isVI];
}

class VOP3b_Real_si <bits<9> op, dag outs, dag ins, string asm, string opName> :
  VOP3Common <outs, ins, asm, []>,
  VOP3be <op>,
  SIMCInstr<opName#"_e64", SISubtarget.SI> {
  let AssemblerPredicates = [isSICI];
}

class VOP3b_Real_vi <bits<10> op, dag outs, dag ins, string asm, string opName> :
  VOP3Common <outs, ins, asm, []>,
  VOP3be_vi <op>,
  SIMCInstr <opName#"_e64", SISubtarget.VI> {
  let AssemblerPredicates = [isVI];
}

multiclass VOP3_m <vop op, dag outs, dag ins, string asm, list<dag> pattern,
                   string opName, int NumSrcArgs, bit HasMods = 1> {

  def "" : VOP3_Pseudo <outs, ins, pattern, opName>;

  def _si : VOP3_Real_si <op.SI3, outs, ins, asm, opName>,
            VOP3DisableFields<!if(!eq(NumSrcArgs, 1), 0, 1),
                              !if(!eq(NumSrcArgs, 2), 0, 1),
                              HasMods>;
  def _vi : VOP3_Real_vi <op.VI3, outs, ins, asm, opName>,
            VOP3DisableFields<!if(!eq(NumSrcArgs, 1), 0, 1),
                              !if(!eq(NumSrcArgs, 2), 0, 1),
                              HasMods>;
}

multiclass VOP3_1_m <vop op, dag outs, dag ins, string asm,
                     list<dag> pattern, string opName, bit HasMods = 1> {

  def "" : VOP3_Pseudo <outs, ins, pattern, opName>;

  def _si : VOP3_Real_si <op.SI3, outs, ins, asm, opName>,
            VOP3DisableFields<0, 0, HasMods>;

  def _vi : VOP3_Real_vi <op.VI3, outs, ins, asm, opName>,
            VOP3DisableFields<0, 0, HasMods>;
}

multiclass VOP3SI_1_m <vop op, dag outs, dag ins, string asm,
                     list<dag> pattern, string opName, bit HasMods = 1> {

  def "" : VOP3_Pseudo <outs, ins, pattern, opName>;

  def _si : VOP3_Real_si <op.SI3, outs, ins, asm, opName>,
            VOP3DisableFields<0, 0, HasMods>;
  // No VI instruction. This class is for SI only.
}

multiclass VOP3_2_m <vop op, dag outs, dag ins, string asm,
                     list<dag> pattern, string opName, string revOp,
                     bit HasMods = 1> {

  def "" : VOP3_Pseudo <outs, ins, pattern, opName>,
           VOP2_REV<revOp#"_e64", !eq(revOp, opName)>;

  def _si : VOP3_Real_si <op.SI3, outs, ins, asm, opName>,
            VOP3DisableFields<1, 0, HasMods>;

  def _vi : VOP3_Real_vi <op.VI3, outs, ins, asm, opName>,
            VOP3DisableFields<1, 0, HasMods>;
}

multiclass VOP3SI_2_m <vop op, dag outs, dag ins, string asm,
                     list<dag> pattern, string opName, string revOp,
                     bit HasMods = 1> {

  def "" : VOP3_Pseudo <outs, ins, pattern, opName>,
           VOP2_REV<revOp#"_e64", !eq(revOp, opName)>;

  def _si : VOP3_Real_si <op.SI3, outs, ins, asm, opName>,
            VOP3DisableFields<1, 0, HasMods>;

  // No VI instruction. This class is for SI only.
}

// Two operand VOP3b instruction that may have a 3rd SGPR bool operand
// instead of an implicit VCC as in the VOP2b format.
multiclass VOP3b_2_3_m <vop op, dag outs, dag ins, string asm,
                        list<dag> pattern, string opName, string revOp,
                        bit HasMods = 1, bit useSrc2Input = 0> {
  def "" : VOP3_Pseudo <outs, ins, pattern, opName>;

  def _si : VOP3b_Real_si <op.SI3, outs, ins, asm, opName>,
            VOP3DisableFields<1, useSrc2Input, HasMods>;

  def _vi : VOP3b_Real_vi <op.VI3, outs, ins, asm, opName>,
            VOP3DisableFields<1, useSrc2Input, HasMods>;
}

multiclass VOP3_C_m <vop op, dag outs, dag ins, string asm,
                     list<dag> pattern, string opName,
                     bit HasMods, bit defExec,
                     string revOp, list<SchedReadWrite> sched> {

  def "" : VOP3_Pseudo <outs, ins, pattern, opName>,
           VOP2_REV<revOp#"_e64", !eq(revOp, opName)> {
    let Defs = !if(defExec, [EXEC], []);
    let SchedRW = sched;
  }

  def _si : VOP3_Real_si <op.SI3, outs, ins, asm, opName>,
            VOP3DisableFields<1, 0, HasMods> {
    let Defs = !if(defExec, [EXEC], []);
    let SchedRW = sched;
  }

  def _vi : VOP3_Real_vi <op.VI3, outs, ins, asm, opName>,
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
             SIMCInstr<opName, SISubtarget.NONE>;
  }

  def _si : VOP2 <op.SI3{5-0}, outs, ins, asm, []>,
            SIMCInstr <opName, SISubtarget.SI> {
            let AssemblerPredicates = [isSICI];
  }

  def _vi : VOP3Common <outs, ins, asm, []>,
            VOP3e_vi <op.VI3>,
            VOP3DisableFields <1, 0, 0>,
            SIMCInstr <opName, SISubtarget.VI> {
            let AssemblerPredicates = [isVI];
  }
}

multiclass VOP1_Helper <vop1 op, string opName, VOPProfile p, list<dag> pat32,
                        list<dag> pat64> {

  defm _e32 : VOP1_m <op, opName, p, pat32>;

  defm _e64 : VOP3_1_m <op, p.Outs, p.Ins64, opName#p.Asm64, pat64, opName,
                        p.HasModifiers>;
}

multiclass VOP1Inst <vop1 op, string opName, VOPProfile P,
                     SDPatternOperator node = null_frag> : VOP1_Helper <
  op, opName, P, [],
  !if(P.HasModifiers,
      [(set P.DstVT:$dst, (node (P.Src0VT (VOP3Mods0 P.Src0VT:$src0,
                                i32:$src0_modifiers, i1:$clamp, i32:$omod))))],
      [(set P.DstVT:$dst, (node P.Src0VT:$src0))])
>;

multiclass VOP1InstSI <vop1 op, string opName, VOPProfile P,
                       SDPatternOperator node = null_frag> {

  defm _e32 : VOP1SI_m <op, opName, P, []>;

  defm _e64 : VOP3SI_1_m <op, P.Outs, P.Ins64, opName#P.Asm64,
    !if(P.HasModifiers,
      [(set P.DstVT:$dst, (node (P.Src0VT (VOP3Mods0 P.Src0VT:$src0,
                                i32:$src0_modifiers, i1:$clamp, i32:$omod))))],
      [(set P.DstVT:$dst, (node P.Src0VT:$src0))]),
    opName, P.HasModifiers>;
}

multiclass VOP2_Helper <vop2 op, string opName, VOPProfile p, list<dag> pat32,
                        list<dag> pat64, string revOp> {

  defm _e32 : VOP2_m <op, opName, p, pat32, revOp>;

  defm _e64 : VOP3_2_m <op, p.Outs, p.Ins64, opName#p.Asm64, pat64, opName,
                        revOp, p.HasModifiers>;
}

multiclass VOP2Inst <vop2 op, string opName, VOPProfile P,
                     SDPatternOperator node = null_frag,
                     string revOp = opName> : VOP2_Helper <
  op, opName, P, [],
  !if(P.HasModifiers,
      [(set P.DstVT:$dst,
           (node (P.Src0VT (VOP3Mods0 P.Src0VT:$src0, i32:$src0_modifiers,
                                      i1:$clamp, i32:$omod)),
                 (P.Src1VT (VOP3Mods P.Src1VT:$src1, i32:$src1_modifiers))))],
      [(set P.DstVT:$dst, (node P.Src0VT:$src0, P.Src1VT:$src1))]),
  revOp
>;

multiclass VOP2InstSI <vop2 op, string opName, VOPProfile P,
                       SDPatternOperator node = null_frag,
                       string revOp = opName> {

  defm _e32 : VOP2SI_m <op, opName, P, [], revOp>;

  defm _e64 : VOP3SI_2_m <op, P.Outs, P.Ins64, opName#P.Asm64,
    !if(P.HasModifiers,
        [(set P.DstVT:$dst,
             (node (P.Src0VT (VOP3Mods0 P.Src0VT:$src0, i32:$src0_modifiers,
                                        i1:$clamp, i32:$omod)),
                   (P.Src1VT (VOP3Mods P.Src1VT:$src1, i32:$src1_modifiers))))],
        [(set P.DstVT:$dst, (node P.Src0VT:$src0, P.Src1VT:$src1))]),
    opName, revOp, P.HasModifiers>;
}

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
      [(set P.DstVT:$dst,
           (node (P.Src0VT (VOP3Mods0 P.Src0VT:$src0, i32:$src0_modifiers,
                                      i1:$clamp, i32:$omod)),
                 (P.Src1VT (VOP3Mods P.Src1VT:$src1, i32:$src1_modifiers))))],
      [(set P.DstVT:$dst, (node P.Src0VT:$src0, P.Src1VT:$src1))]),
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
      [(set P.DstVT:$dst,
           (node (P.Src0VT (VOP3Mods0 P.Src0VT:$src0, i32:$src0_modifiers,
                                      i1:$clamp, i32:$omod)),
                 (P.Src1VT (VOP3Mods P.Src1VT:$src1, i32:$src1_modifiers))))],
      [(set P.DstVT:$dst, (node P.Src0VT:$src0, P.Src1VT:$src1))]),
  revOp
>;

multiclass VOP2MADK <vop2 op, string opName, list<dag> pattern = []> {

  def "" : VOP2_Pseudo <VOP_MADK.Outs, VOP_MADK.Ins, pattern, opName>;

let isCodeGenOnly = 0 in {
  def _si : VOP2Common <VOP_MADK.Outs, VOP_MADK.Ins,
                        !strconcat(opName, VOP_MADK.Asm), []>,
            SIMCInstr <opName#"_e32", SISubtarget.SI>,
            VOP2_MADKe <op.SI> {
            let AssemblerPredicates = [isSICI];
            }

  def _vi : VOP2Common <VOP_MADK.Outs, VOP_MADK.Ins,
                        !strconcat(opName, VOP_MADK.Asm), []>,
            SIMCInstr <opName#"_e32", SISubtarget.VI>,
            VOP2_MADKe <op.VI> {
            let AssemblerPredicates = [isVI];
            }
} // End isCodeGenOnly = 0
}

class VOPC_Pseudo <dag ins, list<dag> pattern, string opName> :
  VOPCCommon <ins, "", pattern>,
  VOP <opName>,
  SIMCInstr<opName#"_e32", SISubtarget.NONE> {
  let isPseudo = 1;
  let isCodeGenOnly = 1;
}

multiclass VOPC_m <vopc op, dag ins, string op_asm, list<dag> pattern,
                   string opName, bit DefExec, VOPProfile p,
                   list<SchedReadWrite> sched,
                   string revOpName = "", string asm = opName#"_e32 "#op_asm,
                   string alias_asm = opName#" "#op_asm> {
  def "" : VOPC_Pseudo <ins, pattern, opName> {
    let Defs = !if(DefExec, [VCC, EXEC], [VCC]);
    let SchedRW = sched;
  }

  let AssemblerPredicates = [isSICI] in {
    def _si : VOPC<op.SI, ins, asm, []>,
              SIMCInstr <opName#"_e32", SISubtarget.SI> {
      let Defs = !if(DefExec, [VCC, EXEC], [VCC]);
      let hasSideEffects = DefExec;
      let SchedRW = sched;
    }

  } // End AssemblerPredicates = [isSICI]

  let AssemblerPredicates = [isVI] in {
    def _vi : VOPC<op.VI, ins, asm, []>,
              SIMCInstr <opName#"_e32", SISubtarget.VI> {
      let Defs = !if(DefExec, [VCC, EXEC], [VCC]);
      let hasSideEffects = DefExec;
      let SchedRW = sched;
    }

  } // End AssemblerPredicates = [isVI]

  defm : SIInstAliasBuilder<alias_asm, p>;
}

multiclass VOPC_Helper <vopc op, string opName, list<dag> pat32,
                        list<dag> pat64, bit DefExec, string revOp,
                        VOPProfile p, list<SchedReadWrite> sched> {
  defm _e32 : VOPC_m <op, p.Ins32, p.Asm32, pat32, opName, DefExec, p, sched>;

  defm _e64 : VOP3_C_m <op, (outs VOPDstS64:$dst), p.Ins64, opName#p.Asm64, pat64,
                        opName, p.HasModifiers, DefExec, revOp, sched>;
}

// Special case for class instructions which only have modifiers on
// the 1st source operand.
multiclass VOPC_Class_Helper <vopc op, string opName, list<dag> pat32,
                              list<dag> pat64, bit DefExec, string revOp,
                              VOPProfile p, list<SchedReadWrite> sched> {
  defm _e32 : VOPC_m <op, p.Ins32, p.Asm32, pat32, opName, DefExec, p, sched>;

  defm _e64 : VOP3_C_m <op, (outs VOPDstS64:$dst), p.Ins64, opName#p.Asm64, pat64,
                        opName, p.HasModifiers, DefExec, revOp, sched>,
                        VOP3DisableModFields<1, 0, 0>;
}

multiclass VOPCInst <vopc op, string opName,
                     VOPProfile P, PatLeaf cond = COND_NULL,
                     string revOp = opName,
                     bit DefExec = 0,
                     list<SchedReadWrite> sched = [Write32Bit]> :
                     VOPC_Helper <
  op, opName, [],
  !if(P.HasModifiers,
      [(set i1:$dst,
          (setcc (P.Src0VT (VOP3Mods0 P.Src0VT:$src0, i32:$src0_modifiers,
                                      i1:$clamp, i32:$omod)),
                 (P.Src1VT (VOP3Mods P.Src1VT:$src1, i32:$src1_modifiers)),
                 cond))],
      [(set i1:$dst, (setcc P.Src0VT:$src0, P.Src1VT:$src1, cond))]),
  DefExec, revOp, P, sched
>;

multiclass VOPCClassInst <vopc op, string opName, VOPProfile P,
                     bit DefExec = 0,
                     list<SchedReadWrite> sched> : VOPC_Class_Helper <
  op, opName, [],
  !if(P.HasModifiers,
      [(set i1:$dst,
          (AMDGPUfp_class (P.Src0VT (VOP3Mods0Clamp0OMod P.Src0VT:$src0, i32:$src0_modifiers)), P.Src1VT:$src1))],
      [(set i1:$dst, (AMDGPUfp_class P.Src0VT:$src0, P.Src1VT:$src1))]),
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

multiclass VOP3_Helper <vop3 op, string opName, dag outs, dag ins, string asm,
                        list<dag> pat, int NumSrcArgs, bit HasMods> : VOP3_m <
    op, outs, ins, opName#" "#asm, pat, opName, NumSrcArgs, HasMods
>;

multiclass VOPC_CLASS_F32 <vopc op, string opName> :
  VOPCClassInst <op, opName, VOPC_I1_F32_I32, 0, [Write32Bit]>;

multiclass VOPCX_CLASS_F32 <vopc op, string opName> :
  VOPCClassInst <op, opName, VOPC_I1_F32_I32, 1, [Write32Bit]>;

multiclass VOPC_CLASS_F64 <vopc op, string opName> :
  VOPCClassInst <op, opName, VOPC_I1_F64_I32, 0, [WriteDoubleAdd]>;

multiclass VOPCX_CLASS_F64 <vopc op, string opName> :
  VOPCClassInst <op, opName, VOPC_I1_F64_I32, 1, [WriteDoubleAdd]>;

multiclass VOP3Inst <vop3 op, string opName, VOPProfile P,
                     SDPatternOperator node = null_frag> : VOP3_Helper <
  op, opName, (outs P.DstRC.RegClass:$dst), P.Ins64, P.Asm64,
  !if(!eq(P.NumSrcArgs, 3),
    !if(P.HasModifiers,
        [(set P.DstVT:$dst,
            (node (P.Src0VT (VOP3Mods0 P.Src0VT:$src0, i32:$src0_modifiers,
                                       i1:$clamp, i32:$omod)),
                  (P.Src1VT (VOP3Mods P.Src1VT:$src1, i32:$src1_modifiers)),
                  (P.Src2VT (VOP3Mods P.Src2VT:$src2, i32:$src2_modifiers))))],
        [(set P.DstVT:$dst, (node P.Src0VT:$src0, P.Src1VT:$src1,
                                  P.Src2VT:$src2))]),
  !if(!eq(P.NumSrcArgs, 2),
    !if(P.HasModifiers,
        [(set P.DstVT:$dst,
            (node (P.Src0VT (VOP3Mods0 P.Src0VT:$src0, i32:$src0_modifiers,
                                       i1:$clamp, i32:$omod)),
                  (P.Src1VT (VOP3Mods P.Src1VT:$src1, i32:$src1_modifiers))))],
        [(set P.DstVT:$dst, (node P.Src0VT:$src0, P.Src1VT:$src1))])
  /* P.NumSrcArgs == 1 */,
    !if(P.HasModifiers,
        [(set P.DstVT:$dst,
            (node (P.Src0VT (VOP3Mods0 P.Src0VT:$src0, i32:$src0_modifiers,
                                       i1:$clamp, i32:$omod))))],
        [(set P.DstVT:$dst, (node P.Src0VT:$src0))]))),
  P.NumSrcArgs, P.HasModifiers
>;

// Special case for v_div_fmas_{f32|f64}, since it seems to be the
// only VOP instruction that implicitly reads VCC.
multiclass VOP3_VCC_Inst <vop3 op, string opName,
                          VOPProfile P,
                          SDPatternOperator node = null_frag> : VOP3_Helper <
  op, opName,
  (outs P.DstRC.RegClass:$dst),
  (ins InputModsNoDefault:$src0_modifiers, P.Src0RC64:$src0,
       InputModsNoDefault:$src1_modifiers, P.Src1RC64:$src1,
       InputModsNoDefault:$src2_modifiers, P.Src2RC64:$src2,
       ClampMod:$clamp,
       omod:$omod),
  "$dst, $src0_modifiers, $src1_modifiers, $src2_modifiers"#"$clamp"#"$omod",
  [(set P.DstVT:$dst,
            (node (P.Src0VT (VOP3Mods0 P.Src0VT:$src0, i32:$src0_modifiers,
                                       i1:$clamp, i32:$omod)),
                  (P.Src1VT (VOP3Mods P.Src1VT:$src1, i32:$src1_modifiers)),
                  (P.Src2VT (VOP3Mods P.Src2VT:$src2, i32:$src2_modifiers)),
                  (i1 VCC)))],
  3, 1
>;

multiclass VOP3bInst <vop op, string opName, VOPProfile P, list<dag> pattern = []> :
  VOP3b_2_3_m <
  op, P.Outs64, P.Ins64,
  opName#" "#P.Asm64, pattern,
  opName, "", 1, 1
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
  SIMCInstr<opName, SISubtarget.NONE> {
  let isPseudo = 1;
  let isCodeGenOnly = 1;
}

class VINTRP_Real_si <bits <2> op, string opName, dag outs, dag ins,
                      string asm> :
  VINTRPCommon <outs, ins, asm, []>,
  VINTRPe <op>,
  SIMCInstr<opName, SISubtarget.SI>;

class VINTRP_Real_vi <bits <2> op, string opName, dag outs, dag ins,
                      string asm> :
  VINTRPCommon <outs, ins, asm, []>,
  VINTRPe_vi <op>,
  SIMCInstr<opName, SISubtarget.VI>;

multiclass VINTRP_m <bits <2> op, dag outs, dag ins, string asm,
                     list<dag> pattern = []> {
  def "" : VINTRP_Pseudo <NAME, outs, ins, pattern>;

  def _si : VINTRP_Real_si <op, NAME, outs, ins, asm>;

  def _vi : VINTRP_Real_vi <op, NAME, outs, ins, asm>;
}

//===----------------------------------------------------------------------===//
// Vector I/O classes
//===----------------------------------------------------------------------===//

class DS_Pseudo <string opName, dag outs, dag ins, list<dag> pattern> :
  DS <outs, ins, "", pattern>,
  SIMCInstr <opName, SISubtarget.NONE> {
  let isPseudo = 1;
  let isCodeGenOnly = 1;
}

class DS_Real_si <bits<8> op, string opName, dag outs, dag ins, string asm> :
  DS <outs, ins, asm, []>,
  DSe <op>,
  SIMCInstr <opName, SISubtarget.SI> {
  let isCodeGenOnly = 0;
}

class DS_Real_vi <bits<8> op, string opName, dag outs, dag ins, string asm> :
  DS <outs, ins, asm, []>,
  DSe_vi <op>,
  SIMCInstr <opName, SISubtarget.VI>;

class DS_Off16_Real_si <bits<8> op, string opName, dag outs, dag ins, string asm> :
  DS_Real_si <op,opName, outs, ins, asm> {

  // Single load interpret the 2 i8imm operands as a single i16 offset.
  bits<16> offset;
  let offset0 = offset{7-0};
  let offset1 = offset{15-8};
  let isCodeGenOnly = 0;
}

class DS_Off16_Real_vi <bits<8> op, string opName, dag outs, dag ins, string asm> :
  DS_Real_vi <op, opName, outs, ins, asm> {

  // Single load interpret the 2 i8imm operands as a single i16 offset.
  bits<16> offset;
  let offset0 = offset{7-0};
  let offset1 = offset{15-8};
}

multiclass DS_1A_RET <bits<8> op, string opName, RegisterClass rc,
  dag outs = (outs rc:$vdst),
  dag ins = (ins VGPR_32:$addr, ds_offset:$offset, gds:$gds),
  string asm = opName#" $vdst, $addr"#"$offset$gds"> {

  def "" : DS_Pseudo <opName, outs, ins, []>;

  let data0 = 0, data1 = 0 in {
    def _si : DS_Off16_Real_si <op, opName, outs, ins, asm>;
    def _vi : DS_Off16_Real_vi <op, opName, outs, ins, asm>;
  }
}

multiclass DS_1A_Off8_RET <bits<8> op, string opName, RegisterClass rc,
  dag outs = (outs rc:$vdst),
  dag ins = (ins VGPR_32:$addr, ds_offset0:$offset0, ds_offset1:$offset1,
                 gds01:$gds),
  string asm = opName#" $vdst, $addr"#"$offset0"#"$offset1$gds"> {

  def "" : DS_Pseudo <opName, outs, ins, []>;

  let data0 = 0, data1 = 0, AsmMatchConverter = "cvtDSOffset01" in {
    def _si : DS_Real_si <op, opName, outs, ins, asm>;
    def _vi : DS_Real_vi <op, opName, outs, ins, asm>;
  }
}

multiclass DS_1A1D_NORET <bits<8> op, string opName, RegisterClass rc,
  dag outs = (outs),
  dag ins = (ins VGPR_32:$addr, rc:$data0, ds_offset:$offset, gds:$gds),
  string asm = opName#" $addr, $data0"#"$offset$gds"> {

  def "" : DS_Pseudo <opName, outs, ins, []>,
           AtomicNoRet<opName, 0>;

  let data1 = 0, vdst = 0 in {
    def _si : DS_Off16_Real_si <op, opName, outs, ins, asm>;
    def _vi : DS_Off16_Real_vi <op, opName, outs, ins, asm>;
  }
}

multiclass DS_1A1D_Off8_NORET <bits<8> op, string opName, RegisterClass rc,
  dag outs = (outs),
  dag ins = (ins VGPR_32:$addr, rc:$data0, rc:$data1,
              ds_offset0:$offset0, ds_offset1:$offset1, gds01:$gds),
  string asm = opName#" $addr, $data0, $data1"#"$offset0"#"$offset1"#"$gds"> {

  def "" : DS_Pseudo <opName, outs, ins, []>;

  let vdst = 0, AsmMatchConverter = "cvtDSOffset01" in {
    def _si : DS_Real_si <op, opName, outs, ins, asm>;
    def _vi : DS_Real_vi <op, opName, outs, ins, asm>;
  }
}

multiclass DS_1A1D_RET <bits<8> op, string opName, RegisterClass rc,
                        string noRetOp = "",
  dag outs = (outs rc:$vdst),
  dag ins = (ins VGPR_32:$addr, rc:$data0, ds_offset:$offset, gds:$gds),
  string asm = opName#" $vdst, $addr, $data0"#"$offset$gds"> {

  let hasPostISelHook = 1 in {
    def "" : DS_Pseudo <opName, outs, ins, []>,
             AtomicNoRet<noRetOp, 1>;

    let data1 = 0 in {
      def _si : DS_Off16_Real_si <op, opName, outs, ins, asm>;
      def _vi : DS_Off16_Real_vi <op, opName, outs, ins, asm>;
    }
  }
}

multiclass DS_1A2D_RET_m <bits<8> op, string opName, RegisterClass rc,
                          string noRetOp = "", dag ins,
  dag outs = (outs rc:$vdst),
  string asm = opName#" $vdst, $addr, $data0, $data1"#"$offset"#"$gds"> {

  let hasPostISelHook = 1 in {
    def "" : DS_Pseudo <opName, outs, ins, []>,
             AtomicNoRet<noRetOp, 1>;

    def _si : DS_Off16_Real_si <op, opName, outs, ins, asm>;
    def _vi : DS_Off16_Real_vi <op, opName, outs, ins, asm>;
  }
}

multiclass DS_1A2D_RET <bits<8> op, string asm, RegisterClass rc,
                        string noRetOp = "", RegisterClass src = rc> :
  DS_1A2D_RET_m <op, asm, rc, noRetOp,
                 (ins VGPR_32:$addr, src:$data0, src:$data1,
                      ds_offset:$offset, gds:$gds)
>;

multiclass DS_1A2D_NORET <bits<8> op, string opName, RegisterClass rc,
                          string noRetOp = opName,
  dag outs = (outs),
  dag ins = (ins VGPR_32:$addr, rc:$data0, rc:$data1,
                 ds_offset:$offset, gds:$gds),
  string asm = opName#" $addr, $data0, $data1"#"$offset"#"$gds"> {

  def "" : DS_Pseudo <opName, outs, ins, []>,
           AtomicNoRet<noRetOp, 0>;

  let vdst = 0 in {
    def _si : DS_Off16_Real_si <op, opName, outs, ins, asm>;
    def _vi : DS_Off16_Real_vi <op, opName, outs, ins, asm>;
  }
}

multiclass DS_0A_RET <bits<8> op, string opName,
  dag outs = (outs VGPR_32:$vdst),
  dag ins = (ins ds_offset:$offset, gds:$gds),
  string asm = opName#" $vdst"#"$offset"#"$gds"> {

  let mayLoad = 1, mayStore = 1 in {
    def "" : DS_Pseudo <opName, outs, ins, []>;

    let addr = 0, data0 = 0, data1 = 0 in {
      def _si : DS_Off16_Real_si <op, opName, outs, ins, asm>;
      def _vi : DS_Off16_Real_vi <op, opName, outs, ins, asm>;
    } // end addr = 0, data0 = 0, data1 = 0
  } // end mayLoad = 1, mayStore = 1
}

multiclass DS_1A_RET_GDS <bits<8> op, string opName,
  dag outs = (outs VGPR_32:$vdst),
  dag ins = (ins VGPR_32:$addr, ds_offset_gds:$offset),
  string asm = opName#" $vdst, $addr"#"$offset gds"> {

  def "" : DS_Pseudo <opName, outs, ins, []>;

  let data0 = 0, data1 = 0, gds = 1 in {
    def _si : DS_Off16_Real_si <op, opName, outs, ins, asm>;
    def _vi : DS_Off16_Real_vi <op, opName, outs, ins, asm>;
  } // end data0 = 0, data1 = 0, gds = 1
}

multiclass DS_1A_GDS <bits<8> op, string opName,
  dag outs = (outs),
  dag ins = (ins VGPR_32:$addr),
  string asm = opName#" $addr gds"> {

  def "" : DS_Pseudo <opName, outs, ins, []>;

  let vdst = 0, data0 = 0, data1 = 0, offset0 = 0, offset1 = 0, gds = 1 in {
    def _si : DS_Real_si <op, opName, outs, ins, asm>;
    def _vi : DS_Real_vi <op, opName, outs, ins, asm>;
  } // end vdst = 0, data = 0, data1 = 0, gds = 1
}

multiclass DS_1A <bits<8> op, string opName,
  dag outs = (outs),
  dag ins = (ins VGPR_32:$addr, ds_offset:$offset, gds:$gds),
  string asm = opName#" $addr"#"$offset"#"$gds"> {

  let mayLoad = 1, mayStore = 1 in {
    def "" : DS_Pseudo <opName, outs, ins, []>;

    let vdst = 0, data0 = 0, data1 = 0 in {
      def _si : DS_Off16_Real_si <op, opName, outs, ins, asm>;
      def _vi : DS_Off16_Real_vi <op, opName, outs, ins, asm>;
    } // let vdst = 0, data0 = 0, data1 = 0
  } // end mayLoad = 1, mayStore = 1
}

//===----------------------------------------------------------------------===//
// MTBUF classes
//===----------------------------------------------------------------------===//

class MTBUF_Pseudo <string opName, dag outs, dag ins, list<dag> pattern> :
  MTBUF <outs, ins, "", pattern>,
  SIMCInstr<opName, SISubtarget.NONE> {
  let isPseudo = 1;
  let isCodeGenOnly = 1;
}

class MTBUF_Real_si <bits<3> op, string opName, dag outs, dag ins,
                    string asm> :
  MTBUF <outs, ins, asm, []>,
  MTBUFe <op>,
  SIMCInstr<opName, SISubtarget.SI>;

class MTBUF_Real_vi <bits<4> op, string opName, dag outs, dag ins, string asm> :
  MTBUF <outs, ins, asm, []>,
  MTBUFe_vi <op>,
  SIMCInstr <opName, SISubtarget.VI>;

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
  SIMCInstr<opName, SISubtarget.NONE> {
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
  SIMCInstr<opName, SISubtarget.SI> {
  let lds = 0;
}

class MUBUF_Real_vi <mubuf op, string opName, dag outs, dag ins,
                     string asm> :
  MUBUF <outs, ins, asm, []>,
  MUBUFe_vi <op.VI>,
  SIMCInstr<opName, SISubtarget.VI> {
  let lds = 0;
}

multiclass MUBUF_m <mubuf op, string opName, dag outs, dag ins, string asm,
                    list<dag> pattern> {

  def "" : MUBUF_Pseudo <opName, outs, ins, pattern>,
           MUBUFAddr64Table <0>;

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

multiclass MUBUF_Atomic <mubuf op, string name, RegisterClass rc,
                         ValueType vt, SDPatternOperator atomic> {

  let mayStore = 1, mayLoad = 1, hasPostISelHook = 1 in {

    // No return variants
    let glc = 0 in {

      defm _ADDR64 : MUBUFAtomicAddr64_m <
        op, name#"_addr64", (outs),
        (ins rc:$vdata, VReg_64:$vaddr, SReg_128:$srsrc,
             SCSrc_32:$soffset, mbuf_offset:$offset, slc:$slc),
        name#" $vdata, $vaddr, $srsrc, $soffset addr64"#"$offset"#"$slc", [], 0
      >;

      defm _OFFSET : MUBUFAtomicOffset_m <
        op, name#"_offset", (outs),
        (ins rc:$vdata, SReg_128:$srsrc, SCSrc_32:$soffset, mbuf_offset:$offset,
             slc:$slc),
        name#" $vdata, $srsrc, $soffset"#"$offset"#"$slc", [], 0
      >;
    } // glc = 0

    // Variant that return values
    let glc = 1, Constraints = "$vdata = $vdata_in",
        DisableEncoding = "$vdata_in"  in {

      defm _RTN_ADDR64 : MUBUFAtomicAddr64_m <
        op, name#"_rtn_addr64", (outs rc:$vdata),
        (ins rc:$vdata_in, VReg_64:$vaddr, SReg_128:$srsrc,
             SCSrc_32:$soffset, mbuf_offset:$offset, slc:$slc),
        name#" $vdata, $vaddr, $srsrc, $soffset addr64"#"$offset"#" glc"#"$slc",
        [(set vt:$vdata,
         (atomic (MUBUFAddr64Atomic v4i32:$srsrc, i64:$vaddr, i32:$soffset,
	                            i16:$offset, i1:$slc), vt:$vdata_in))], 1
      >;

      defm _RTN_OFFSET : MUBUFAtomicOffset_m <
        op, name#"_rtn_offset", (outs rc:$vdata),
        (ins rc:$vdata_in, SReg_128:$srsrc, SCSrc_32:$soffset,
             mbuf_offset:$offset, slc:$slc),
        name#" $vdata, $srsrc, $soffset"#"$offset"#" glc$slc",
        [(set vt:$vdata,
         (atomic (MUBUFOffsetAtomic v4i32:$srsrc, i32:$soffset, i16:$offset,
                                    i1:$slc), vt:$vdata_in))], 1
      >;

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
                           mbuf_offset:$offset, glc:$glc, slc:$slc, tfe:$tfe),
                           name#" $vdata, $srsrc, $soffset"#"$offset"#"$glc"#"$slc"#"$tfe",
                           [(set load_vt:$vdata, (ld (MUBUFOffset v4i32:$srsrc,
                                                     i32:$soffset, i16:$offset,
                                                     i1:$glc, i1:$slc, i1:$tfe)))]>;
    }

    let offen = 1, idxen = 0  in {
      defm _OFFEN  : MUBUF_m <op, name#"_offen", (outs regClass:$vdata),
                           (ins VGPR_32:$vaddr, SReg_128:$srsrc,
                           SCSrc_32:$soffset, mbuf_offset:$offset, glc:$glc, slc:$slc,
                           tfe:$tfe),
                           name#" $vdata, $vaddr, $srsrc, $soffset offen"#"$offset"#"$glc"#"$slc"#"$tfe", []>;
    }

    let offen = 0, idxen = 1 in {
      defm _IDXEN  : MUBUF_m <op, name#"_idxen", (outs regClass:$vdata),
                           (ins VGPR_32:$vaddr, SReg_128:$srsrc,
                           SCSrc_32:$soffset, mbuf_offset:$offset, glc:$glc,
                           slc:$slc, tfe:$tfe),
                           name#" $vdata, $vaddr, $srsrc, $soffset idxen"#"$offset"#"$glc"#"$slc"#"$tfe", []>;
    }

    let offen = 1, idxen = 1 in {
      defm _BOTHEN : MUBUF_m <op, name#"_bothen", (outs regClass:$vdata),
                           (ins VReg_64:$vaddr, SReg_128:$srsrc, SCSrc_32:$soffset,
                           mbuf_offset:$offset, glc:$glc, slc:$slc, tfe:$tfe),
                           name#" $vdata, $vaddr, $srsrc, $soffset idxen offen"#"$offset"#"$glc"#"$slc"#"$tfe", []>;
    }

    let offen = 0, idxen = 0 in {
      defm _ADDR64 : MUBUFAddr64_m <op, name#"_addr64", (outs regClass:$vdata),
                           (ins VReg_64:$vaddr, SReg_128:$srsrc,
                                SCSrc_32:$soffset, mbuf_offset:$offset,
				glc:$glc, slc:$slc, tfe:$tfe),
                           name#" $vdata, $vaddr, $srsrc, $soffset addr64"#"$offset"#
                                "$glc"#"$slc"#"$tfe",
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
    defm : MUBUF_m <op, name, (outs),
                    (ins vdataClass:$vdata, VGPR_32:$vaddr, SReg_128:$srsrc, SCSrc_32:$soffset,
                    mbuf_offset:$offset, offen:$offen, idxen:$idxen, glc:$glc, slc:$slc,
                    tfe:$tfe),
                    name#" $vdata, $vaddr, $srsrc, $soffset"#"$offen"#"$idxen"#"$offset"#
                         "$glc"#"$slc"#"$tfe", []>;

    let offen = 0, idxen = 0, vaddr = 0 in {
      defm _OFFSET : MUBUF_m <op, name#"_offset",(outs),
                              (ins vdataClass:$vdata, SReg_128:$srsrc, SCSrc_32:$soffset,
                              mbuf_offset:$offset, glc:$glc, slc:$slc, tfe:$tfe),
                              name#" $vdata, $srsrc, $soffset"#"$offset"#"$glc"#"$slc"#"$tfe",
                              [(st store_vt:$vdata, (MUBUFOffset v4i32:$srsrc, i32:$soffset,
                                   i16:$offset, i1:$glc, i1:$slc, i1:$tfe))]>;
    } // offen = 0, idxen = 0, vaddr = 0

    let offen = 1, idxen = 0  in {
      defm _OFFEN : MUBUF_m <op, name#"_offen", (outs),
                             (ins vdataClass:$vdata, VGPR_32:$vaddr, SReg_128:$srsrc,
                              SCSrc_32:$soffset, mbuf_offset:$offset, glc:$glc,
                              slc:$slc, tfe:$tfe),
                             name#" $vdata, $vaddr, $srsrc, $soffset offen"#"$offset"#
                             "$glc"#"$slc"#"$tfe", []>;
    } // end offen = 1, idxen = 0

    let offen = 0, idxen = 1 in {
      defm _IDXEN  : MUBUF_m <op, name#"_idxen", (outs),
                           (ins vdataClass:$vdata, VGPR_32:$vaddr, SReg_128:$srsrc,
                           SCSrc_32:$soffset, mbuf_offset:$offset, glc:$glc,
                           slc:$slc, tfe:$tfe),
                           name#" $vdata, $vaddr, $srsrc, $soffset idxen"#"$offset"#"$glc"#"$slc"#"$tfe", []>;
    }

    let offen = 1, idxen = 1 in {
      defm _BOTHEN : MUBUF_m <op, name#"_bothen", (outs),
                           (ins vdataClass:$vdata, VReg_64:$vaddr, SReg_128:$srsrc, SCSrc_32:$soffset,
                           mbuf_offset:$offset, glc:$glc, slc:$slc, tfe:$tfe),
                           name#" $vdata, $vaddr, $srsrc, $soffset idxen offen"#"$offset"#"$glc"#"$slc"#"$tfe", []>;
    }

    let offen = 0, idxen = 0 in {
      defm _ADDR64 : MUBUFAddr64_m <op, name#"_addr64", (outs),
                                    (ins vdataClass:$vdata, VReg_64:$vaddr, SReg_128:$srsrc,
                                         SCSrc_32:$soffset,
                                         mbuf_offset:$offset, glc:$glc, slc:$slc,
                                         tfe:$tfe),
                                    name#" $vdata, $vaddr, $srsrc, $soffset addr64"#
                                         "$offset"#"$glc"#"$slc"#"$tfe",
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
      SIMCInstr<opName, SISubtarget.NONE> {
  let isPseudo = 1;
  let isCodeGenOnly = 1;
}

class FLAT_Real_ci <bits<7> op, string opName, dag outs, dag ins, string asm> :
    FLAT <op, outs, ins, asm, []>,
    SIMCInstr<opName, SISubtarget.SI> {
  let AssemblerPredicate = isCIOnly;
}

class FLAT_Real_vi <bits<7> op, string opName, dag outs, dag ins, string asm> :
    FLAT <op, outs, ins, asm, []>,
    SIMCInstr<opName, SISubtarget.VI> {
  let AssemblerPredicate = VIAssemblerPredicate;
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
    dag ins = (ins VReg_64:$addr, glc_flat:$glc, slc_flat:$slc, tfe_flat:$tfe),
    string asm = asm_name#" $vdst, $addr"#"$glc"#"$slc"#"$tfe"> {

  let data = 0, mayLoad = 1 in {

    def "" : FLAT_Pseudo <NAME, outs, ins, []>;

    def _ci : FLAT_Real_ci <op.CI, NAME, outs, ins, asm>;

    def _vi : FLAT_Real_vi <op.VI, NAME, outs, ins, asm>;
  }
}

multiclass FLAT_Store_Helper <flat op, string asm_name,
    RegisterClass vdataClass,
    dag outs = (outs),
    dag ins = (ins vdataClass:$data, VReg_64:$addr, glc_flat:$glc,
                   slc_flat:$slc, tfe_flat:$tfe),
    string asm = asm_name#" $data, $addr"#"$glc"#"$slc"#"$tfe"> {

  let mayLoad = 0, mayStore = 1, vdst = 0 in {

    def "" : FLAT_Pseudo <NAME, outs, ins, []>;

    def _ci : FLAT_Real_ci <op.CI, NAME, outs, ins, asm>;

    def _vi : FLAT_Real_vi <op.VI, NAME, outs, ins, asm>;
  }
}

multiclass FLAT_ATOMIC <flat op, string asm_name, RegisterClass vdst_rc,
    RegisterClass data_rc = vdst_rc,
    dag outs_noret = (outs),
    string asm_noret = asm_name#" $addr, $data"#"$slc"#"$tfe"> {

  let mayLoad = 1, mayStore = 1, glc = 0, vdst = 0 in {
    def "" : FLAT_Pseudo <NAME, outs_noret,
                          (ins VReg_64:$addr, data_rc:$data,
                               slc_flat_atomic:$slc, tfe_flat_atomic:$tfe), []>,
             AtomicNoRet <NAME, 0>;

    def _ci : FLAT_Real_ci <op.CI, NAME, outs_noret,
                            (ins VReg_64:$addr, data_rc:$data,
                                 slc_flat_atomic:$slc, tfe_flat_atomic:$tfe),
                            asm_noret>;

    def _vi : FLAT_Real_vi <op.VI, NAME, outs_noret,
                            (ins VReg_64:$addr, data_rc:$data,
                                 slc_flat_atomic:$slc, tfe_flat_atomic:$tfe),
                            asm_noret>;
  }

  let glc = 1, hasPostISelHook = 1 in {
    defm _RTN : FLAT_AtomicRet_m <op, (outs vdst_rc:$vdst),
                        (ins VReg_64:$addr, data_rc:$data, slc_flat_atomic:$slc,
                             tfe_flat_atomic:$tfe),
                        asm_name#" $vdst, $addr, $data glc"#"$slc"#"$tfe", []>;
  }
}

class MIMG_Mask <string op, int channels> {
  string Op = op;
  int Channels = channels;
}

class MIMG_NoSampler_Helper <bits<7> op, string asm,
                             RegisterClass dst_rc,
                             RegisterClass src_rc> : MIMG <
  op,
  (outs dst_rc:$vdata),
  (ins i32imm:$dmask, i1imm:$unorm, i1imm:$glc, i1imm:$da, i1imm:$r128,
       i1imm:$tfe, i1imm:$lwe, i1imm:$slc, src_rc:$vaddr,
       SReg_256:$srsrc),
  asm#" $vdata, $dmask, $unorm, $glc, $da, $r128,"
     #" $tfe, $lwe, $slc, $vaddr, $srsrc",
  []> {
  let ssamp = 0;
  let mayLoad = 1;
  let mayStore = 0;
  let hasPostISelHook = 1;
}

multiclass MIMG_NoSampler_Src_Helper <bits<7> op, string asm,
                                      RegisterClass dst_rc,
                                      int channels> {
  def _V1 : MIMG_NoSampler_Helper <op, asm, dst_rc, VGPR_32>,
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

class MIMG_Sampler_Helper <bits<7> op, string asm,
                           RegisterClass dst_rc,
                           RegisterClass src_rc, int wqm> : MIMG <
  op,
  (outs dst_rc:$vdata),
  (ins i32imm:$dmask, i1imm:$unorm, i1imm:$glc, i1imm:$da, i1imm:$r128,
       i1imm:$tfe, i1imm:$lwe, i1imm:$slc, src_rc:$vaddr,
       SReg_256:$srsrc, SReg_128:$ssamp),
  asm#" $vdata, $dmask, $unorm, $glc, $da, $r128,"
     #" $tfe, $lwe, $slc, $vaddr, $srsrc, $ssamp",
  []> {
  let mayLoad = 1;
  let mayStore = 0;
  let hasPostISelHook = 1;
  let WQM = wqm;
}

multiclass MIMG_Sampler_Src_Helper <bits<7> op, string asm,
                                    RegisterClass dst_rc,
                                    int channels, int wqm> {
  def _V1 : MIMG_Sampler_Helper <op, asm, dst_rc, VGPR_32, wqm>,
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

multiclass MIMG_Sampler <bits<7> op, string asm> {
  defm _V1 : MIMG_Sampler_Src_Helper<op, asm, VGPR_32, 1, 0>;
  defm _V2 : MIMG_Sampler_Src_Helper<op, asm, VReg_64, 2, 0>;
  defm _V3 : MIMG_Sampler_Src_Helper<op, asm, VReg_96, 3, 0>;
  defm _V4 : MIMG_Sampler_Src_Helper<op, asm, VReg_128, 4, 0>;
}

multiclass MIMG_Sampler_WQM <bits<7> op, string asm> {
  defm _V1 : MIMG_Sampler_Src_Helper<op, asm, VGPR_32, 1, 1>;
  defm _V2 : MIMG_Sampler_Src_Helper<op, asm, VReg_64, 2, 1>;
  defm _V3 : MIMG_Sampler_Src_Helper<op, asm, VReg_96, 3, 1>;
  defm _V4 : MIMG_Sampler_Src_Helper<op, asm, VReg_128, 4, 1>;
}

class MIMG_Gather_Helper <bits<7> op, string asm,
                          RegisterClass dst_rc,
                          RegisterClass src_rc, int wqm> : MIMG <
  op,
  (outs dst_rc:$vdata),
  (ins i32imm:$dmask, i1imm:$unorm, i1imm:$glc, i1imm:$da, i1imm:$r128,
       i1imm:$tfe, i1imm:$lwe, i1imm:$slc, src_rc:$vaddr,
       SReg_256:$srsrc, SReg_128:$ssamp),
  asm#" $vdata, $dmask, $unorm, $glc, $da, $r128,"
     #" $tfe, $lwe, $slc, $vaddr, $srsrc, $ssamp",
  []> {
  let mayLoad = 1;
  let mayStore = 0;

  // DMASK was repurposed for GATHER4. 4 components are always
  // returned and DMASK works like a swizzle - it selects
  // the component to fetch. The only useful DMASK values are
  // 1=red, 2=green, 4=blue, 8=alpha. (e.g. 1 returns
  // (red,red,red,red) etc.) The ISA document doesn't mention
  // this.
  // Therefore, disable all code which updates DMASK by setting these two:
  let MIMG = 0;
  let hasPostISelHook = 0;
  let WQM = wqm;
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

multiclass MIMG_Gather <bits<7> op, string asm> {
  defm _V1 : MIMG_Gather_Src_Helper<op, asm, VGPR_32, 1, 0>;
  defm _V2 : MIMG_Gather_Src_Helper<op, asm, VReg_64, 2, 0>;
  defm _V3 : MIMG_Gather_Src_Helper<op, asm, VReg_96, 3, 0>;
  defm _V4 : MIMG_Gather_Src_Helper<op, asm, VReg_128, 4, 0>;
}

multiclass MIMG_Gather_WQM <bits<7> op, string asm> {
  defm _V1 : MIMG_Gather_Src_Helper<op, asm, VGPR_32, 1, 1>;
  defm _V2 : MIMG_Gather_Src_Helper<op, asm, VReg_64, 2, 1>;
  defm _V3 : MIMG_Gather_Src_Helper<op, asm, VReg_96, 3, 1>;
  defm _V4 : MIMG_Gather_Src_Helper<op, asm, VReg_128, 4, 1>;
}

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
  let KeyCol = [!cast<string>(SISubtarget.NONE)];
  let ValueCols = [[!cast<string>(SISubtarget.SI)],[!cast<string>(SISubtarget.VI)]];
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
