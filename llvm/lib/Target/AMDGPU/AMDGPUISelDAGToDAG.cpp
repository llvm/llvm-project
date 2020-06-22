//===-- AMDGPUISelDAGToDAG.cpp - A dag to dag inst selector for AMDGPU ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//==-----------------------------------------------------------------------===//
//
/// \file
/// Defines an instruction selector for the AMDGPU target.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "AMDGPUArgumentUsageInfo.h"
#include "AMDGPUISelLowering.h" // For AMDGPUISD
#include "AMDGPUInstrInfo.h"
#include "AMDGPUPerfHintAnalysis.h"
#include "AMDGPUSubtarget.h"
#include "AMDGPUTargetMachine.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIDefines.h"
#include "SIISelLowering.h"
#include "SIInstrInfo.h"
#include "SIMachineFunctionInfo.h"
#include "SIRegisterInfo.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/LegacyDivergenceAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/CodeGen/FunctionLoweringInfo.h"
#include "llvm/CodeGen/ISDOpcodes.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/CodeGen/SelectionDAGNodes.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/InitializePasses.h"
#ifdef EXPENSIVE_CHECKS
#include "llvm/IR/Dominators.h"
#endif
#include "llvm/IR/Instruction.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MachineValueType.h"
#include "llvm/Support/MathExtras.h"
#include <cassert>
#include <cstdint>
#include <new>
#include <vector>

#define DEBUG_TYPE "isel"

using namespace llvm;

namespace llvm {

class R600InstrInfo;

} // end namespace llvm

//===----------------------------------------------------------------------===//
// Instruction Selector Implementation
//===----------------------------------------------------------------------===//

namespace {

static bool isNullConstantOrUndef(SDValue V) {
  if (V.isUndef())
    return true;

  ConstantSDNode *Const = dyn_cast<ConstantSDNode>(V);
  return Const != nullptr && Const->isNullValue();
}

static bool getConstantValue(SDValue N, uint32_t &Out) {
  // This is only used for packed vectors, where ussing 0 for undef should
  // always be good.
  if (N.isUndef()) {
    Out = 0;
    return true;
  }

  if (const ConstantSDNode *C = dyn_cast<ConstantSDNode>(N)) {
    Out = C->getAPIntValue().getSExtValue();
    return true;
  }

  if (const ConstantFPSDNode *C = dyn_cast<ConstantFPSDNode>(N)) {
    Out = C->getValueAPF().bitcastToAPInt().getSExtValue();
    return true;
  }

  return false;
}

// TODO: Handle undef as zero
static SDNode *packConstantV2I16(const SDNode *N, SelectionDAG &DAG,
                                 bool Negate = false) {
  assert(N->getOpcode() == ISD::BUILD_VECTOR && N->getNumOperands() == 2);
  uint32_t LHSVal, RHSVal;
  if (getConstantValue(N->getOperand(0), LHSVal) &&
      getConstantValue(N->getOperand(1), RHSVal)) {
    SDLoc SL(N);
    uint32_t K = Negate ?
      (-LHSVal & 0xffff) | (-RHSVal << 16) :
      (LHSVal & 0xffff) | (RHSVal << 16);
    return DAG.getMachineNode(AMDGPU::S_MOV_B32, SL, N->getValueType(0),
                              DAG.getTargetConstant(K, SL, MVT::i32));
  }

  return nullptr;
}

static SDNode *packNegConstantV2I16(const SDNode *N, SelectionDAG &DAG) {
  return packConstantV2I16(N, DAG, true);
}

/// AMDGPU specific code to select AMDGPU machine instructions for
/// SelectionDAG operations.
class AMDGPUDAGToDAGISel : public SelectionDAGISel {
  // Subtarget - Keep a pointer to the AMDGPU Subtarget around so that we can
  // make the right decision when generating code for different targets.
  const GCNSubtarget *Subtarget;

  // Default FP mode for the current function.
  AMDGPU::SIModeRegisterDefaults Mode;

  bool EnableLateStructurizeCFG;

public:
  explicit AMDGPUDAGToDAGISel(TargetMachine *TM = nullptr,
                              CodeGenOpt::Level OptLevel = CodeGenOpt::Default)
    : SelectionDAGISel(*TM, OptLevel) {
    EnableLateStructurizeCFG = AMDGPUTargetMachine::EnableLateStructurizeCFG;
  }
  ~AMDGPUDAGToDAGISel() override = default;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<AMDGPUArgumentUsageInfo>();
    AU.addRequired<LegacyDivergenceAnalysis>();
#ifdef EXPENSIVE_CHECKS
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<LoopInfoWrapperPass>();
#endif
    SelectionDAGISel::getAnalysisUsage(AU);
  }

  bool matchLoadD16FromBuildVector(SDNode *N) const;

  bool runOnMachineFunction(MachineFunction &MF) override;
  void PreprocessISelDAG() override;
  void Select(SDNode *N) override;
  StringRef getPassName() const override;
  void PostprocessISelDAG() override;

protected:
  void SelectBuildVector(SDNode *N, unsigned RegClassID);

private:
  std::pair<SDValue, SDValue> foldFrameIndex(SDValue N) const;
  bool isNoNanSrc(SDValue N) const;
  bool isInlineImmediate(const SDNode *N, bool Negated = false) const;
  bool isNegInlineImmediate(const SDNode *N) const {
    return isInlineImmediate(N, true);
  }

  bool isInlineImmediate16(int64_t Imm) const {
    return AMDGPU::isInlinableLiteral16(Imm, Subtarget->hasInv2PiInlineImm());
  }

  bool isInlineImmediate32(int64_t Imm) const {
    return AMDGPU::isInlinableLiteral32(Imm, Subtarget->hasInv2PiInlineImm());
  }

  bool isInlineImmediate64(int64_t Imm) const {
    return AMDGPU::isInlinableLiteral64(Imm, Subtarget->hasInv2PiInlineImm());
  }

  bool isInlineImmediate(const APFloat &Imm) const {
    return Subtarget->getInstrInfo()->isInlineConstant(Imm);
  }

  bool isVGPRImm(const SDNode *N) const;
  bool isUniformLoad(const SDNode *N) const;
  bool isUniformBr(const SDNode *N) const;

  MachineSDNode *buildSMovImm64(SDLoc &DL, uint64_t Val, EVT VT) const;

  SDNode *glueCopyToOp(SDNode *N, SDValue NewChain, SDValue Glue) const;
  SDNode *glueCopyToM0(SDNode *N, SDValue Val) const;
  SDNode *glueCopyToM0LDSInit(SDNode *N) const;

  const TargetRegisterClass *getOperandRegClass(SDNode *N, unsigned OpNo) const;
  virtual bool SelectADDRVTX_READ(SDValue Addr, SDValue &Base, SDValue &Offset);
  virtual bool SelectADDRIndirect(SDValue Addr, SDValue &Base, SDValue &Offset);
  bool isDSOffsetLegal(SDValue Base, unsigned Offset,
                       unsigned OffsetBits) const;
  bool SelectDS1Addr1Offset(SDValue Ptr, SDValue &Base, SDValue &Offset) const;
  bool SelectDS64Bit4ByteAligned(SDValue Ptr, SDValue &Base, SDValue &Offset0,
                                 SDValue &Offset1) const;
  bool SelectMUBUF(SDValue Addr, SDValue &SRsrc, SDValue &VAddr,
                   SDValue &SOffset, SDValue &Offset, SDValue &Offen,
                   SDValue &Idxen, SDValue &Addr64, SDValue &GLC, SDValue &SLC,
                   SDValue &TFE, SDValue &DLC, SDValue &SWZ) const;
  bool SelectMUBUFAddr64(SDValue Addr, SDValue &SRsrc, SDValue &VAddr,
                         SDValue &SOffset, SDValue &Offset, SDValue &GLC,
                         SDValue &SLC, SDValue &TFE, SDValue &DLC,
                         SDValue &SWZ) const;
  bool SelectMUBUFAddr64(SDValue Addr, SDValue &SRsrc,
                         SDValue &VAddr, SDValue &SOffset, SDValue &Offset,
                         SDValue &SLC) const;
  bool SelectMUBUFScratchOffen(SDNode *Parent,
                               SDValue Addr, SDValue &RSrc, SDValue &VAddr,
                               SDValue &SOffset, SDValue &ImmOffset) const;
  bool SelectMUBUFScratchOffset(SDNode *Parent,
                                SDValue Addr, SDValue &SRsrc, SDValue &Soffset,
                                SDValue &Offset) const;

  bool SelectMUBUFOffset(SDValue Addr, SDValue &SRsrc, SDValue &SOffset,
                         SDValue &Offset, SDValue &GLC, SDValue &SLC,
                         SDValue &TFE, SDValue &DLC, SDValue &SWZ) const;
  bool SelectMUBUFOffset(SDValue Addr, SDValue &SRsrc, SDValue &Soffset,
                         SDValue &Offset, SDValue &SLC) const;
  bool SelectMUBUFOffset(SDValue Addr, SDValue &SRsrc, SDValue &Soffset,
                         SDValue &Offset) const;

  template <bool IsSigned>
  bool SelectFlatOffset(SDNode *N, SDValue Addr, SDValue &VAddr,
                        SDValue &Offset, SDValue &SLC) const;
  bool SelectFlatAtomic(SDNode *N, SDValue Addr, SDValue &VAddr,
                        SDValue &Offset, SDValue &SLC) const;
  bool SelectFlatAtomicSigned(SDNode *N, SDValue Addr, SDValue &VAddr,
                              SDValue &Offset, SDValue &SLC) const;

  bool SelectSMRDOffset(SDValue ByteOffsetNode, SDValue &Offset,
                        bool &Imm) const;
  SDValue Expand32BitAddress(SDValue Addr) const;
  bool SelectSMRD(SDValue Addr, SDValue &SBase, SDValue &Offset,
                  bool &Imm) const;
  bool SelectSMRDImm(SDValue Addr, SDValue &SBase, SDValue &Offset) const;
  bool SelectSMRDImm32(SDValue Addr, SDValue &SBase, SDValue &Offset) const;
  bool SelectSMRDSgpr(SDValue Addr, SDValue &SBase, SDValue &Offset) const;
  bool SelectSMRDBufferImm(SDValue Addr, SDValue &Offset) const;
  bool SelectSMRDBufferImm32(SDValue Addr, SDValue &Offset) const;
  bool SelectMOVRELOffset(SDValue Index, SDValue &Base, SDValue &Offset) const;

  bool SelectVOP3Mods_NNaN(SDValue In, SDValue &Src, SDValue &SrcMods) const;
  bool SelectVOP3ModsImpl(SDValue In, SDValue &Src, unsigned &SrcMods) const;
  bool SelectVOP3Mods(SDValue In, SDValue &Src, SDValue &SrcMods) const;
  bool SelectVOP3NoMods(SDValue In, SDValue &Src) const;
  bool SelectVOP3Mods0(SDValue In, SDValue &Src, SDValue &SrcMods,
                       SDValue &Clamp, SDValue &Omod) const;
  bool SelectVOP3NoMods0(SDValue In, SDValue &Src, SDValue &SrcMods,
                         SDValue &Clamp, SDValue &Omod) const;

  bool SelectVOP3OMods(SDValue In, SDValue &Src,
                       SDValue &Clamp, SDValue &Omod) const;

  bool SelectVOP3PMods(SDValue In, SDValue &Src, SDValue &SrcMods) const;

  bool SelectVOP3OpSel(SDValue In, SDValue &Src, SDValue &SrcMods) const;

  bool SelectVOP3OpSelMods(SDValue In, SDValue &Src, SDValue &SrcMods) const;
  bool SelectVOP3PMadMixModsImpl(SDValue In, SDValue &Src, unsigned &Mods) const;
  bool SelectVOP3PMadMixMods(SDValue In, SDValue &Src, SDValue &SrcMods) const;

  SDValue getHi16Elt(SDValue In) const;

  SDValue getMaterializedScalarImm32(int64_t Val, const SDLoc &DL) const;

  void SelectADD_SUB_I64(SDNode *N);
  void SelectAddcSubb(SDNode *N);
  void SelectUADDO_USUBO(SDNode *N);
  void SelectDIV_SCALE(SDNode *N);
  void SelectMAD_64_32(SDNode *N);
  void SelectFMA_W_CHAIN(SDNode *N);
  void SelectFMUL_W_CHAIN(SDNode *N);

  SDNode *getS_BFE(unsigned Opcode, const SDLoc &DL, SDValue Val,
                   uint32_t Offset, uint32_t Width);
  void SelectS_BFEFromShifts(SDNode *N);
  void SelectS_BFE(SDNode *N);
  bool isCBranchSCC(const SDNode *N) const;
  void SelectBRCOND(SDNode *N);
  void SelectFMAD_FMA(SDNode *N);
  void SelectATOMIC_CMP_SWAP(SDNode *N);
  void SelectDSAppendConsume(SDNode *N, unsigned IntrID);
  void SelectDS_GWS(SDNode *N, unsigned IntrID);
  void SelectInterpP1F16(SDNode *N);
  void SelectINTRINSIC_W_CHAIN(SDNode *N);
  void SelectINTRINSIC_WO_CHAIN(SDNode *N);
  void SelectINTRINSIC_VOID(SDNode *N);

protected:
  // Include the pieces autogenerated from the target description.
#include "AMDGPUGenDAGISel.inc"
};

class R600DAGToDAGISel : public AMDGPUDAGToDAGISel {
  const R600Subtarget *Subtarget;

  bool isConstantLoad(const MemSDNode *N, int cbID) const;
  bool SelectGlobalValueConstantOffset(SDValue Addr, SDValue& IntPtr);
  bool SelectGlobalValueVariableOffset(SDValue Addr, SDValue &BaseReg,
                                       SDValue& Offset);
public:
  explicit R600DAGToDAGISel(TargetMachine *TM, CodeGenOpt::Level OptLevel) :
      AMDGPUDAGToDAGISel(TM, OptLevel) {}

  void Select(SDNode *N) override;

  bool SelectADDRIndirect(SDValue Addr, SDValue &Base,
                          SDValue &Offset) override;
  bool SelectADDRVTX_READ(SDValue Addr, SDValue &Base,
                          SDValue &Offset) override;

  bool runOnMachineFunction(MachineFunction &MF) override;

  void PreprocessISelDAG() override {}

protected:
  // Include the pieces autogenerated from the target description.
#include "R600GenDAGISel.inc"
};

static SDValue stripBitcast(SDValue Val) {
  return Val.getOpcode() == ISD::BITCAST ? Val.getOperand(0) : Val;
}

// Figure out if this is really an extract of the high 16-bits of a dword.
static bool isExtractHiElt(SDValue In, SDValue &Out) {
  In = stripBitcast(In);
  if (In.getOpcode() != ISD::TRUNCATE)
    return false;

  SDValue Srl = In.getOperand(0);
  if (Srl.getOpcode() == ISD::SRL) {
    if (ConstantSDNode *ShiftAmt = dyn_cast<ConstantSDNode>(Srl.getOperand(1))) {
      if (ShiftAmt->getZExtValue() == 16) {
        Out = stripBitcast(Srl.getOperand(0));
        return true;
      }
    }
  }

  return false;
}

// Look through operations that obscure just looking at the low 16-bits of the
// same register.
static SDValue stripExtractLoElt(SDValue In) {
  if (In.getOpcode() == ISD::TRUNCATE) {
    SDValue Src = In.getOperand(0);
    if (Src.getValueType().getSizeInBits() == 32)
      return stripBitcast(Src);
  }

  return In;
}

}  // end anonymous namespace

INITIALIZE_PASS_BEGIN(AMDGPUDAGToDAGISel, "amdgpu-isel",
                      "AMDGPU DAG->DAG Pattern Instruction Selection", false, false)
INITIALIZE_PASS_DEPENDENCY(AMDGPUArgumentUsageInfo)
INITIALIZE_PASS_DEPENDENCY(AMDGPUPerfHintAnalysis)
INITIALIZE_PASS_DEPENDENCY(LegacyDivergenceAnalysis)
#ifdef EXPENSIVE_CHECKS
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
#endif
INITIALIZE_PASS_END(AMDGPUDAGToDAGISel, "amdgpu-isel",
                    "AMDGPU DAG->DAG Pattern Instruction Selection", false, false)

/// This pass converts a legalized DAG into a AMDGPU-specific
// DAG, ready for instruction scheduling.
FunctionPass *llvm::createAMDGPUISelDag(TargetMachine *TM,
                                        CodeGenOpt::Level OptLevel) {
  return new AMDGPUDAGToDAGISel(TM, OptLevel);
}

/// This pass converts a legalized DAG into a R600-specific
// DAG, ready for instruction scheduling.
FunctionPass *llvm::createR600ISelDag(TargetMachine *TM,
                                      CodeGenOpt::Level OptLevel) {
  return new R600DAGToDAGISel(TM, OptLevel);
}

bool AMDGPUDAGToDAGISel::runOnMachineFunction(MachineFunction &MF) {
#ifdef EXPENSIVE_CHECKS
  DominatorTree & DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  LoopInfo * LI = &getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  for (auto &L : LI->getLoopsInPreorder()) {
    assert(L->isLCSSAForm(DT));
  }
#endif
  Subtarget = &MF.getSubtarget<GCNSubtarget>();
  Mode = AMDGPU::SIModeRegisterDefaults(MF.getFunction());
  return SelectionDAGISel::runOnMachineFunction(MF);
}

bool AMDGPUDAGToDAGISel::matchLoadD16FromBuildVector(SDNode *N) const {
  assert(Subtarget->d16PreservesUnusedBits());
  MVT VT = N->getValueType(0).getSimpleVT();
  if (VT != MVT::v2i16 && VT != MVT::v2f16)
    return false;

  SDValue Lo = N->getOperand(0);
  SDValue Hi = N->getOperand(1);

  LoadSDNode *LdHi = dyn_cast<LoadSDNode>(stripBitcast(Hi));

  // build_vector lo, (load ptr) -> load_d16_hi ptr, lo
  // build_vector lo, (zextload ptr from i8) -> load_d16_hi_u8 ptr, lo
  // build_vector lo, (sextload ptr from i8) -> load_d16_hi_i8 ptr, lo

  // Need to check for possible indirect dependencies on the other half of the
  // vector to avoid introducing a cycle.
  if (LdHi && Hi.hasOneUse() && !LdHi->isPredecessorOf(Lo.getNode())) {
    SDVTList VTList = CurDAG->getVTList(VT, MVT::Other);

    SDValue TiedIn = CurDAG->getNode(ISD::SCALAR_TO_VECTOR, SDLoc(N), VT, Lo);
    SDValue Ops[] = {
      LdHi->getChain(), LdHi->getBasePtr(), TiedIn
    };

    unsigned LoadOp = AMDGPUISD::LOAD_D16_HI;
    if (LdHi->getMemoryVT() == MVT::i8) {
      LoadOp = LdHi->getExtensionType() == ISD::SEXTLOAD ?
        AMDGPUISD::LOAD_D16_HI_I8 : AMDGPUISD::LOAD_D16_HI_U8;
    } else {
      assert(LdHi->getMemoryVT() == MVT::i16);
    }

    SDValue NewLoadHi =
      CurDAG->getMemIntrinsicNode(LoadOp, SDLoc(LdHi), VTList,
                                  Ops, LdHi->getMemoryVT(),
                                  LdHi->getMemOperand());

    CurDAG->ReplaceAllUsesOfValueWith(SDValue(N, 0), NewLoadHi);
    CurDAG->ReplaceAllUsesOfValueWith(SDValue(LdHi, 1), NewLoadHi.getValue(1));
    return true;
  }

  // build_vector (load ptr), hi -> load_d16_lo ptr, hi
  // build_vector (zextload ptr from i8), hi -> load_d16_lo_u8 ptr, hi
  // build_vector (sextload ptr from i8), hi -> load_d16_lo_i8 ptr, hi
  LoadSDNode *LdLo = dyn_cast<LoadSDNode>(stripBitcast(Lo));
  if (LdLo && Lo.hasOneUse()) {
    SDValue TiedIn = getHi16Elt(Hi);
    if (!TiedIn || LdLo->isPredecessorOf(TiedIn.getNode()))
      return false;

    SDVTList VTList = CurDAG->getVTList(VT, MVT::Other);
    unsigned LoadOp = AMDGPUISD::LOAD_D16_LO;
    if (LdLo->getMemoryVT() == MVT::i8) {
      LoadOp = LdLo->getExtensionType() == ISD::SEXTLOAD ?
        AMDGPUISD::LOAD_D16_LO_I8 : AMDGPUISD::LOAD_D16_LO_U8;
    } else {
      assert(LdLo->getMemoryVT() == MVT::i16);
    }

    TiedIn = CurDAG->getNode(ISD::BITCAST, SDLoc(N), VT, TiedIn);

    SDValue Ops[] = {
      LdLo->getChain(), LdLo->getBasePtr(), TiedIn
    };

    SDValue NewLoadLo =
      CurDAG->getMemIntrinsicNode(LoadOp, SDLoc(LdLo), VTList,
                                  Ops, LdLo->getMemoryVT(),
                                  LdLo->getMemOperand());

    CurDAG->ReplaceAllUsesOfValueWith(SDValue(N, 0), NewLoadLo);
    CurDAG->ReplaceAllUsesOfValueWith(SDValue(LdLo, 1), NewLoadLo.getValue(1));
    return true;
  }

  return false;
}

void AMDGPUDAGToDAGISel::PreprocessISelDAG() {
  if (!Subtarget->d16PreservesUnusedBits())
    return;

  SelectionDAG::allnodes_iterator Position = CurDAG->allnodes_end();

  bool MadeChange = false;
  while (Position != CurDAG->allnodes_begin()) {
    SDNode *N = &*--Position;
    if (N->use_empty())
      continue;

    switch (N->getOpcode()) {
    case ISD::BUILD_VECTOR:
      MadeChange |= matchLoadD16FromBuildVector(N);
      break;
    default:
      break;
    }
  }

  if (MadeChange) {
    CurDAG->RemoveDeadNodes();
    LLVM_DEBUG(dbgs() << "After PreProcess:\n";
               CurDAG->dump(););
  }
}

bool AMDGPUDAGToDAGISel::isNoNanSrc(SDValue N) const {
  if (TM.Options.NoNaNsFPMath)
    return true;

  // TODO: Move into isKnownNeverNaN
  if (N->getFlags().isDefined())
    return N->getFlags().hasNoNaNs();

  return CurDAG->isKnownNeverNaN(N);
}

bool AMDGPUDAGToDAGISel::isInlineImmediate(const SDNode *N,
                                           bool Negated) const {
  if (N->isUndef())
    return true;

  const SIInstrInfo *TII = Subtarget->getInstrInfo();
  if (Negated) {
    if (const ConstantSDNode *C = dyn_cast<ConstantSDNode>(N))
      return TII->isInlineConstant(-C->getAPIntValue());

    if (const ConstantFPSDNode *C = dyn_cast<ConstantFPSDNode>(N))
      return TII->isInlineConstant(-C->getValueAPF().bitcastToAPInt());

  } else {
    if (const ConstantSDNode *C = dyn_cast<ConstantSDNode>(N))
      return TII->isInlineConstant(C->getAPIntValue());

    if (const ConstantFPSDNode *C = dyn_cast<ConstantFPSDNode>(N))
      return TII->isInlineConstant(C->getValueAPF().bitcastToAPInt());
  }

  return false;
}

/// Determine the register class for \p OpNo
/// \returns The register class of the virtual register that will be used for
/// the given operand number \OpNo or NULL if the register class cannot be
/// determined.
const TargetRegisterClass *AMDGPUDAGToDAGISel::getOperandRegClass(SDNode *N,
                                                          unsigned OpNo) const {
  if (!N->isMachineOpcode()) {
    if (N->getOpcode() == ISD::CopyToReg) {
      unsigned Reg = cast<RegisterSDNode>(N->getOperand(1))->getReg();
      if (Register::isVirtualRegister(Reg)) {
        MachineRegisterInfo &MRI = CurDAG->getMachineFunction().getRegInfo();
        return MRI.getRegClass(Reg);
      }

      const SIRegisterInfo *TRI
        = static_cast<const GCNSubtarget *>(Subtarget)->getRegisterInfo();
      return TRI->getPhysRegClass(Reg);
    }

    return nullptr;
  }

  switch (N->getMachineOpcode()) {
  default: {
    const MCInstrDesc &Desc =
        Subtarget->getInstrInfo()->get(N->getMachineOpcode());
    unsigned OpIdx = Desc.getNumDefs() + OpNo;
    if (OpIdx >= Desc.getNumOperands())
      return nullptr;
    int RegClass = Desc.OpInfo[OpIdx].RegClass;
    if (RegClass == -1)
      return nullptr;

    return Subtarget->getRegisterInfo()->getRegClass(RegClass);
  }
  case AMDGPU::REG_SEQUENCE: {
    unsigned RCID = cast<ConstantSDNode>(N->getOperand(0))->getZExtValue();
    const TargetRegisterClass *SuperRC =
        Subtarget->getRegisterInfo()->getRegClass(RCID);

    SDValue SubRegOp = N->getOperand(OpNo + 1);
    unsigned SubRegIdx = cast<ConstantSDNode>(SubRegOp)->getZExtValue();
    return Subtarget->getRegisterInfo()->getSubClassWithSubReg(SuperRC,
                                                              SubRegIdx);
  }
  }
}

SDNode *AMDGPUDAGToDAGISel::glueCopyToOp(SDNode *N, SDValue NewChain,
                                         SDValue Glue) const {
  SmallVector <SDValue, 8> Ops;
  Ops.push_back(NewChain); // Replace the chain.
  for (unsigned i = 1, e = N->getNumOperands(); i != e; ++i)
    Ops.push_back(N->getOperand(i));

  Ops.push_back(Glue);
  return CurDAG->MorphNodeTo(N, N->getOpcode(), N->getVTList(), Ops);
}

SDNode *AMDGPUDAGToDAGISel::glueCopyToM0(SDNode *N, SDValue Val) const {
  const SITargetLowering& Lowering =
    *static_cast<const SITargetLowering*>(getTargetLowering());

  assert(N->getOperand(0).getValueType() == MVT::Other && "Expected chain");

  SDValue M0 = Lowering.copyToM0(*CurDAG, N->getOperand(0), SDLoc(N), Val);
  return glueCopyToOp(N, M0, M0.getValue(1));
}

SDNode *AMDGPUDAGToDAGISel::glueCopyToM0LDSInit(SDNode *N) const {
  unsigned AS = cast<MemSDNode>(N)->getAddressSpace();
  if (AS == AMDGPUAS::LOCAL_ADDRESS) {
    if (Subtarget->ldsRequiresM0Init())
      return glueCopyToM0(N, CurDAG->getTargetConstant(-1, SDLoc(N), MVT::i32));
  } else if (AS == AMDGPUAS::REGION_ADDRESS) {
    MachineFunction &MF = CurDAG->getMachineFunction();
    unsigned Value = MF.getInfo<SIMachineFunctionInfo>()->getGDSSize();
    return
        glueCopyToM0(N, CurDAG->getTargetConstant(Value, SDLoc(N), MVT::i32));
  }
  return N;
}

MachineSDNode *AMDGPUDAGToDAGISel::buildSMovImm64(SDLoc &DL, uint64_t Imm,
                                                  EVT VT) const {
  SDNode *Lo = CurDAG->getMachineNode(
      AMDGPU::S_MOV_B32, DL, MVT::i32,
      CurDAG->getTargetConstant(Imm & 0xFFFFFFFF, DL, MVT::i32));
  SDNode *Hi =
      CurDAG->getMachineNode(AMDGPU::S_MOV_B32, DL, MVT::i32,
                             CurDAG->getTargetConstant(Imm >> 32, DL, MVT::i32));
  const SDValue Ops[] = {
      CurDAG->getTargetConstant(AMDGPU::SReg_64RegClassID, DL, MVT::i32),
      SDValue(Lo, 0), CurDAG->getTargetConstant(AMDGPU::sub0, DL, MVT::i32),
      SDValue(Hi, 0), CurDAG->getTargetConstant(AMDGPU::sub1, DL, MVT::i32)};

  return CurDAG->getMachineNode(TargetOpcode::REG_SEQUENCE, DL, VT, Ops);
}

void AMDGPUDAGToDAGISel::SelectBuildVector(SDNode *N, unsigned RegClassID) {
  EVT VT = N->getValueType(0);
  unsigned NumVectorElts = VT.getVectorNumElements();
  EVT EltVT = VT.getVectorElementType();
  SDLoc DL(N);
  SDValue RegClass = CurDAG->getTargetConstant(RegClassID, DL, MVT::i32);

  if (NumVectorElts == 1) {
    CurDAG->SelectNodeTo(N, AMDGPU::COPY_TO_REGCLASS, EltVT, N->getOperand(0),
                         RegClass);
    return;
  }

  assert(NumVectorElts <= 32 && "Vectors with more than 32 elements not "
                                  "supported yet");
  // 32 = Max Num Vector Elements
  // 2 = 2 REG_SEQUENCE operands per element (value, subreg index)
  // 1 = Vector Register Class
  SmallVector<SDValue, 32 * 2 + 1> RegSeqArgs(NumVectorElts * 2 + 1);

  bool IsGCN = CurDAG->getSubtarget().getTargetTriple().getArch() ==
               Triple::amdgcn;
  RegSeqArgs[0] = CurDAG->getTargetConstant(RegClassID, DL, MVT::i32);
  bool IsRegSeq = true;
  unsigned NOps = N->getNumOperands();
  for (unsigned i = 0; i < NOps; i++) {
    // XXX: Why is this here?
    if (isa<RegisterSDNode>(N->getOperand(i))) {
      IsRegSeq = false;
      break;
    }
    unsigned Sub = IsGCN ? SIRegisterInfo::getSubRegFromChannel(i)
                         : R600RegisterInfo::getSubRegFromChannel(i);
    RegSeqArgs[1 + (2 * i)] = N->getOperand(i);
    RegSeqArgs[1 + (2 * i) + 1] = CurDAG->getTargetConstant(Sub, DL, MVT::i32);
  }
  if (NOps != NumVectorElts) {
    // Fill in the missing undef elements if this was a scalar_to_vector.
    assert(N->getOpcode() == ISD::SCALAR_TO_VECTOR && NOps < NumVectorElts);
    MachineSDNode *ImpDef = CurDAG->getMachineNode(TargetOpcode::IMPLICIT_DEF,
                                                   DL, EltVT);
    for (unsigned i = NOps; i < NumVectorElts; ++i) {
      unsigned Sub = IsGCN ? SIRegisterInfo::getSubRegFromChannel(i)
                           : R600RegisterInfo::getSubRegFromChannel(i);
      RegSeqArgs[1 + (2 * i)] = SDValue(ImpDef, 0);
      RegSeqArgs[1 + (2 * i) + 1] =
          CurDAG->getTargetConstant(Sub, DL, MVT::i32);
    }
  }

  if (!IsRegSeq)
    SelectCode(N);
  CurDAG->SelectNodeTo(N, AMDGPU::REG_SEQUENCE, N->getVTList(), RegSeqArgs);
}

void AMDGPUDAGToDAGISel::Select(SDNode *N) {
  unsigned int Opc = N->getOpcode();
  if (N->isMachineOpcode()) {
    N->setNodeId(-1);
    return;   // Already selected.
  }

  // isa<MemSDNode> almost works but is slightly too permissive for some DS
  // intrinsics.
  if (Opc == ISD::LOAD || Opc == ISD::STORE || isa<AtomicSDNode>(N) ||
      (Opc == AMDGPUISD::ATOMIC_INC || Opc == AMDGPUISD::ATOMIC_DEC ||
       Opc == ISD::ATOMIC_LOAD_FADD ||
       Opc == AMDGPUISD::ATOMIC_LOAD_FMIN ||
       Opc == AMDGPUISD::ATOMIC_LOAD_FMAX ||
       Opc == AMDGPUISD::ATOMIC_LOAD_CSUB)) {
    N = glueCopyToM0LDSInit(N);
    SelectCode(N);
    return;
  }

  switch (Opc) {
  default:
    break;
  // We are selecting i64 ADD here instead of custom lower it during
  // DAG legalization, so we can fold some i64 ADDs used for address
  // calculation into the LOAD and STORE instructions.
  case ISD::ADDC:
  case ISD::ADDE:
  case ISD::SUBC:
  case ISD::SUBE: {
    if (N->getValueType(0) != MVT::i64)
      break;

    SelectADD_SUB_I64(N);
    return;
  }
  case ISD::ADDCARRY:
  case ISD::SUBCARRY:
    if (N->getValueType(0) != MVT::i32)
      break;

    SelectAddcSubb(N);
    return;
  case ISD::UADDO:
  case ISD::USUBO: {
    SelectUADDO_USUBO(N);
    return;
  }
  case AMDGPUISD::FMUL_W_CHAIN: {
    SelectFMUL_W_CHAIN(N);
    return;
  }
  case AMDGPUISD::FMA_W_CHAIN: {
    SelectFMA_W_CHAIN(N);
    return;
  }

  case ISD::SCALAR_TO_VECTOR:
  case ISD::BUILD_VECTOR: {
    EVT VT = N->getValueType(0);
    unsigned NumVectorElts = VT.getVectorNumElements();
    if (VT.getScalarSizeInBits() == 16) {
      if (Opc == ISD::BUILD_VECTOR && NumVectorElts == 2) {
        if (SDNode *Packed = packConstantV2I16(N, *CurDAG)) {
          ReplaceNode(N, Packed);
          return;
        }
      }

      break;
    }

    assert(VT.getVectorElementType().bitsEq(MVT::i32));
    unsigned RegClassID =
        SIRegisterInfo::getSGPRClassForBitWidth(NumVectorElts * 32)->getID();
    SelectBuildVector(N, RegClassID);
    return;
  }
  case ISD::BUILD_PAIR: {
    SDValue RC, SubReg0, SubReg1;
    SDLoc DL(N);
    if (N->getValueType(0) == MVT::i128) {
      RC = CurDAG->getTargetConstant(AMDGPU::SGPR_128RegClassID, DL, MVT::i32);
      SubReg0 = CurDAG->getTargetConstant(AMDGPU::sub0_sub1, DL, MVT::i32);
      SubReg1 = CurDAG->getTargetConstant(AMDGPU::sub2_sub3, DL, MVT::i32);
    } else if (N->getValueType(0) == MVT::i64) {
      RC = CurDAG->getTargetConstant(AMDGPU::SReg_64RegClassID, DL, MVT::i32);
      SubReg0 = CurDAG->getTargetConstant(AMDGPU::sub0, DL, MVT::i32);
      SubReg1 = CurDAG->getTargetConstant(AMDGPU::sub1, DL, MVT::i32);
    } else {
      llvm_unreachable("Unhandled value type for BUILD_PAIR");
    }
    const SDValue Ops[] = { RC, N->getOperand(0), SubReg0,
                            N->getOperand(1), SubReg1 };
    ReplaceNode(N, CurDAG->getMachineNode(TargetOpcode::REG_SEQUENCE, DL,
                                          N->getValueType(0), Ops));
    return;
  }

  case ISD::Constant:
  case ISD::ConstantFP: {
    if (N->getValueType(0).getSizeInBits() != 64 || isInlineImmediate(N))
      break;

    uint64_t Imm;
    if (ConstantFPSDNode *FP = dyn_cast<ConstantFPSDNode>(N))
      Imm = FP->getValueAPF().bitcastToAPInt().getZExtValue();
    else {
      ConstantSDNode *C = cast<ConstantSDNode>(N);
      Imm = C->getZExtValue();
    }

    SDLoc DL(N);
    ReplaceNode(N, buildSMovImm64(DL, Imm, N->getValueType(0)));
    return;
  }
  case AMDGPUISD::BFE_I32:
  case AMDGPUISD::BFE_U32: {
    // There is a scalar version available, but unlike the vector version which
    // has a separate operand for the offset and width, the scalar version packs
    // the width and offset into a single operand. Try to move to the scalar
    // version if the offsets are constant, so that we can try to keep extended
    // loads of kernel arguments in SGPRs.

    // TODO: Technically we could try to pattern match scalar bitshifts of
    // dynamic values, but it's probably not useful.
    ConstantSDNode *Offset = dyn_cast<ConstantSDNode>(N->getOperand(1));
    if (!Offset)
      break;

    ConstantSDNode *Width = dyn_cast<ConstantSDNode>(N->getOperand(2));
    if (!Width)
      break;

    bool Signed = Opc == AMDGPUISD::BFE_I32;

    uint32_t OffsetVal = Offset->getZExtValue();
    uint32_t WidthVal = Width->getZExtValue();

    ReplaceNode(N, getS_BFE(Signed ? AMDGPU::S_BFE_I32 : AMDGPU::S_BFE_U32,
                            SDLoc(N), N->getOperand(0), OffsetVal, WidthVal));
    return;
  }
  case AMDGPUISD::DIV_SCALE: {
    SelectDIV_SCALE(N);
    return;
  }
  case AMDGPUISD::MAD_I64_I32:
  case AMDGPUISD::MAD_U64_U32: {
    SelectMAD_64_32(N);
    return;
  }
  case ISD::CopyToReg: {
    const SITargetLowering& Lowering =
      *static_cast<const SITargetLowering*>(getTargetLowering());
    N = Lowering.legalizeTargetIndependentNode(N, *CurDAG);
    break;
  }
  case ISD::AND:
  case ISD::SRL:
  case ISD::SRA:
  case ISD::SIGN_EXTEND_INREG:
    if (N->getValueType(0) != MVT::i32)
      break;

    SelectS_BFE(N);
    return;
  case ISD::BRCOND:
    SelectBRCOND(N);
    return;
  case ISD::FMAD:
  case ISD::FMA:
    SelectFMAD_FMA(N);
    return;
  case AMDGPUISD::ATOMIC_CMP_SWAP:
    SelectATOMIC_CMP_SWAP(N);
    return;
  case AMDGPUISD::CVT_PKRTZ_F16_F32:
  case AMDGPUISD::CVT_PKNORM_I16_F32:
  case AMDGPUISD::CVT_PKNORM_U16_F32:
  case AMDGPUISD::CVT_PK_U16_U32:
  case AMDGPUISD::CVT_PK_I16_I32: {
    // Hack around using a legal type if f16 is illegal.
    if (N->getValueType(0) == MVT::i32) {
      MVT NewVT = Opc == AMDGPUISD::CVT_PKRTZ_F16_F32 ? MVT::v2f16 : MVT::v2i16;
      N = CurDAG->MorphNodeTo(N, N->getOpcode(), CurDAG->getVTList(NewVT),
                              { N->getOperand(0), N->getOperand(1) });
      SelectCode(N);
      return;
    }

    break;
  }
  case ISD::INTRINSIC_W_CHAIN: {
    SelectINTRINSIC_W_CHAIN(N);
    return;
  }
  case ISD::INTRINSIC_WO_CHAIN: {
    SelectINTRINSIC_WO_CHAIN(N);
    return;
  }
  case ISD::INTRINSIC_VOID: {
    SelectINTRINSIC_VOID(N);
    return;
  }
  }

  SelectCode(N);
}

bool AMDGPUDAGToDAGISel::isUniformBr(const SDNode *N) const {
  const BasicBlock *BB = FuncInfo->MBB->getBasicBlock();
  const Instruction *Term = BB->getTerminator();
  return Term->getMetadata("amdgpu.uniform") ||
         Term->getMetadata("structurizecfg.uniform");
}

StringRef AMDGPUDAGToDAGISel::getPassName() const {
  return "AMDGPU DAG->DAG Pattern Instruction Selection";
}

//===----------------------------------------------------------------------===//
// Complex Patterns
//===----------------------------------------------------------------------===//

bool AMDGPUDAGToDAGISel::SelectADDRVTX_READ(SDValue Addr, SDValue &Base,
                                            SDValue &Offset) {
  return false;
}

bool AMDGPUDAGToDAGISel::SelectADDRIndirect(SDValue Addr, SDValue &Base,
                                            SDValue &Offset) {
  ConstantSDNode *C;
  SDLoc DL(Addr);

  if ((C = dyn_cast<ConstantSDNode>(Addr))) {
    Base = CurDAG->getRegister(R600::INDIRECT_BASE_ADDR, MVT::i32);
    Offset = CurDAG->getTargetConstant(C->getZExtValue(), DL, MVT::i32);
  } else if ((Addr.getOpcode() == AMDGPUISD::DWORDADDR) &&
             (C = dyn_cast<ConstantSDNode>(Addr.getOperand(0)))) {
    Base = CurDAG->getRegister(R600::INDIRECT_BASE_ADDR, MVT::i32);
    Offset = CurDAG->getTargetConstant(C->getZExtValue(), DL, MVT::i32);
  } else if ((Addr.getOpcode() == ISD::ADD || Addr.getOpcode() == ISD::OR) &&
            (C = dyn_cast<ConstantSDNode>(Addr.getOperand(1)))) {
    Base = Addr.getOperand(0);
    Offset = CurDAG->getTargetConstant(C->getZExtValue(), DL, MVT::i32);
  } else {
    Base = Addr;
    Offset = CurDAG->getTargetConstant(0, DL, MVT::i32);
  }

  return true;
}

SDValue AMDGPUDAGToDAGISel::getMaterializedScalarImm32(int64_t Val,
                                                       const SDLoc &DL) const {
  SDNode *Mov = CurDAG->getMachineNode(
    AMDGPU::S_MOV_B32, DL, MVT::i32,
    CurDAG->getTargetConstant(Val, DL, MVT::i32));
  return SDValue(Mov, 0);
}

// FIXME: Should only handle addcarry/subcarry
void AMDGPUDAGToDAGISel::SelectADD_SUB_I64(SDNode *N) {
  SDLoc DL(N);
  SDValue LHS = N->getOperand(0);
  SDValue RHS = N->getOperand(1);

  unsigned Opcode = N->getOpcode();
  bool ConsumeCarry = (Opcode == ISD::ADDE || Opcode == ISD::SUBE);
  bool ProduceCarry =
      ConsumeCarry || Opcode == ISD::ADDC || Opcode == ISD::SUBC;
  bool IsAdd = Opcode == ISD::ADD || Opcode == ISD::ADDC || Opcode == ISD::ADDE;

  SDValue Sub0 = CurDAG->getTargetConstant(AMDGPU::sub0, DL, MVT::i32);
  SDValue Sub1 = CurDAG->getTargetConstant(AMDGPU::sub1, DL, MVT::i32);

  SDNode *Lo0 = CurDAG->getMachineNode(TargetOpcode::EXTRACT_SUBREG,
                                       DL, MVT::i32, LHS, Sub0);
  SDNode *Hi0 = CurDAG->getMachineNode(TargetOpcode::EXTRACT_SUBREG,
                                       DL, MVT::i32, LHS, Sub1);

  SDNode *Lo1 = CurDAG->getMachineNode(TargetOpcode::EXTRACT_SUBREG,
                                       DL, MVT::i32, RHS, Sub0);
  SDNode *Hi1 = CurDAG->getMachineNode(TargetOpcode::EXTRACT_SUBREG,
                                       DL, MVT::i32, RHS, Sub1);

  SDVTList VTList = CurDAG->getVTList(MVT::i32, MVT::Glue);

  static const unsigned OpcMap[2][2][2] = {
      {{AMDGPU::S_SUB_U32, AMDGPU::S_ADD_U32},
       {AMDGPU::V_SUB_I32_e32, AMDGPU::V_ADD_I32_e32}},
      {{AMDGPU::S_SUBB_U32, AMDGPU::S_ADDC_U32},
       {AMDGPU::V_SUBB_U32_e32, AMDGPU::V_ADDC_U32_e32}}};

  unsigned Opc = OpcMap[0][N->isDivergent()][IsAdd];
  unsigned CarryOpc = OpcMap[1][N->isDivergent()][IsAdd];

  SDNode *AddLo;
  if (!ConsumeCarry) {
    SDValue Args[] = { SDValue(Lo0, 0), SDValue(Lo1, 0) };
    AddLo = CurDAG->getMachineNode(Opc, DL, VTList, Args);
  } else {
    SDValue Args[] = { SDValue(Lo0, 0), SDValue(Lo1, 0), N->getOperand(2) };
    AddLo = CurDAG->getMachineNode(CarryOpc, DL, VTList, Args);
  }
  SDValue AddHiArgs[] = {
    SDValue(Hi0, 0),
    SDValue(Hi1, 0),
    SDValue(AddLo, 1)
  };
  SDNode *AddHi = CurDAG->getMachineNode(CarryOpc, DL, VTList, AddHiArgs);

  SDValue RegSequenceArgs[] = {
    CurDAG->getTargetConstant(AMDGPU::SReg_64RegClassID, DL, MVT::i32),
    SDValue(AddLo,0),
    Sub0,
    SDValue(AddHi,0),
    Sub1,
  };
  SDNode *RegSequence = CurDAG->getMachineNode(AMDGPU::REG_SEQUENCE, DL,
                                               MVT::i64, RegSequenceArgs);

  if (ProduceCarry) {
    // Replace the carry-use
    ReplaceUses(SDValue(N, 1), SDValue(AddHi, 1));
  }

  // Replace the remaining uses.
  ReplaceNode(N, RegSequence);
}

void AMDGPUDAGToDAGISel::SelectAddcSubb(SDNode *N) {
  SDLoc DL(N);
  SDValue LHS = N->getOperand(0);
  SDValue RHS = N->getOperand(1);
  SDValue CI = N->getOperand(2);

  if (N->isDivergent()) {
    unsigned Opc = N->getOpcode() == ISD::ADDCARRY ? AMDGPU::V_ADDC_U32_e64
                                                   : AMDGPU::V_SUBB_U32_e64;
    CurDAG->SelectNodeTo(
        N, Opc, N->getVTList(),
        {LHS, RHS, CI,
         CurDAG->getTargetConstant(0, {}, MVT::i1) /*clamp bit*/});
  } else {
    unsigned Opc = N->getOpcode() == ISD::ADDCARRY ? AMDGPU::S_ADD_CO_PSEUDO
                                                   : AMDGPU::S_SUB_CO_PSEUDO;
    CurDAG->SelectNodeTo(N, Opc, N->getVTList(), {LHS, RHS, CI});
  }
}

void AMDGPUDAGToDAGISel::SelectUADDO_USUBO(SDNode *N) {
  // The name of the opcodes are misleading. v_add_i32/v_sub_i32 have unsigned
  // carry out despite the _i32 name. These were renamed in VI to _U32.
  // FIXME: We should probably rename the opcodes here.
  bool IsAdd = N->getOpcode() == ISD::UADDO;
  bool IsVALU = N->isDivergent();

  for (SDNode::use_iterator UI = N->use_begin(), E = N->use_end(); UI != E;
       ++UI)
    if (UI.getUse().getResNo() == 1) {
      if ((IsAdd && (UI->getOpcode() != ISD::ADDCARRY)) ||
          (!IsAdd && (UI->getOpcode() != ISD::SUBCARRY))) {
        IsVALU = true;
        break;
      }
    }

  if (IsVALU) {
    unsigned Opc = IsAdd ? AMDGPU::V_ADD_I32_e64 : AMDGPU::V_SUB_I32_e64;

    CurDAG->SelectNodeTo(
        N, Opc, N->getVTList(),
        {N->getOperand(0), N->getOperand(1),
         CurDAG->getTargetConstant(0, {}, MVT::i1) /*clamp bit*/});
  } else {
    unsigned Opc = N->getOpcode() == ISD::UADDO ? AMDGPU::S_UADDO_PSEUDO
                                                : AMDGPU::S_USUBO_PSEUDO;

    CurDAG->SelectNodeTo(N, Opc, N->getVTList(),
                         {N->getOperand(0), N->getOperand(1)});
  }
}

void AMDGPUDAGToDAGISel::SelectFMA_W_CHAIN(SDNode *N) {
  SDLoc SL(N);
  //  src0_modifiers, src0,  src1_modifiers, src1, src2_modifiers, src2, clamp, omod
  SDValue Ops[10];

  SelectVOP3Mods0(N->getOperand(1), Ops[1], Ops[0], Ops[6], Ops[7]);
  SelectVOP3Mods(N->getOperand(2), Ops[3], Ops[2]);
  SelectVOP3Mods(N->getOperand(3), Ops[5], Ops[4]);
  Ops[8] = N->getOperand(0);
  Ops[9] = N->getOperand(4);

  CurDAG->SelectNodeTo(N, AMDGPU::V_FMA_F32, N->getVTList(), Ops);
}

void AMDGPUDAGToDAGISel::SelectFMUL_W_CHAIN(SDNode *N) {
  SDLoc SL(N);
  //    src0_modifiers, src0,  src1_modifiers, src1, clamp, omod
  SDValue Ops[8];

  SelectVOP3Mods0(N->getOperand(1), Ops[1], Ops[0], Ops[4], Ops[5]);
  SelectVOP3Mods(N->getOperand(2), Ops[3], Ops[2]);
  Ops[6] = N->getOperand(0);
  Ops[7] = N->getOperand(3);

  CurDAG->SelectNodeTo(N, AMDGPU::V_MUL_F32_e64, N->getVTList(), Ops);
}

// We need to handle this here because tablegen doesn't support matching
// instructions with multiple outputs.
void AMDGPUDAGToDAGISel::SelectDIV_SCALE(SDNode *N) {
  SDLoc SL(N);
  EVT VT = N->getValueType(0);

  assert(VT == MVT::f32 || VT == MVT::f64);

  unsigned Opc
    = (VT == MVT::f64) ? AMDGPU::V_DIV_SCALE_F64 : AMDGPU::V_DIV_SCALE_F32;

  SDValue Ops[] = { N->getOperand(0), N->getOperand(1), N->getOperand(2) };
  CurDAG->SelectNodeTo(N, Opc, N->getVTList(), Ops);
}

// We need to handle this here because tablegen doesn't support matching
// instructions with multiple outputs.
void AMDGPUDAGToDAGISel::SelectMAD_64_32(SDNode *N) {
  SDLoc SL(N);
  bool Signed = N->getOpcode() == AMDGPUISD::MAD_I64_I32;
  unsigned Opc = Signed ? AMDGPU::V_MAD_I64_I32 : AMDGPU::V_MAD_U64_U32;

  SDValue Clamp = CurDAG->getTargetConstant(0, SL, MVT::i1);
  SDValue Ops[] = { N->getOperand(0), N->getOperand(1), N->getOperand(2),
                    Clamp };
  CurDAG->SelectNodeTo(N, Opc, N->getVTList(), Ops);
}

bool AMDGPUDAGToDAGISel::isDSOffsetLegal(SDValue Base, unsigned Offset,
                                         unsigned OffsetBits) const {
  if ((OffsetBits == 16 && !isUInt<16>(Offset)) ||
      (OffsetBits == 8 && !isUInt<8>(Offset)))
    return false;

  if (Subtarget->hasUsableDSOffset() ||
      Subtarget->unsafeDSOffsetFoldingEnabled())
    return true;

  // On Southern Islands instruction with a negative base value and an offset
  // don't seem to work.
  return CurDAG->SignBitIsZero(Base);
}

bool AMDGPUDAGToDAGISel::SelectDS1Addr1Offset(SDValue Addr, SDValue &Base,
                                              SDValue &Offset) const {
  SDLoc DL(Addr);
  if (CurDAG->isBaseWithConstantOffset(Addr)) {
    SDValue N0 = Addr.getOperand(0);
    SDValue N1 = Addr.getOperand(1);
    ConstantSDNode *C1 = cast<ConstantSDNode>(N1);
    if (isDSOffsetLegal(N0, C1->getSExtValue(), 16)) {
      // (add n0, c0)
      Base = N0;
      Offset = CurDAG->getTargetConstant(C1->getZExtValue(), DL, MVT::i16);
      return true;
    }
  } else if (Addr.getOpcode() == ISD::SUB) {
    // sub C, x -> add (sub 0, x), C
    if (const ConstantSDNode *C = dyn_cast<ConstantSDNode>(Addr.getOperand(0))) {
      int64_t ByteOffset = C->getSExtValue();
      if (isUInt<16>(ByteOffset)) {
        SDValue Zero = CurDAG->getTargetConstant(0, DL, MVT::i32);

        // XXX - This is kind of hacky. Create a dummy sub node so we can check
        // the known bits in isDSOffsetLegal. We need to emit the selected node
        // here, so this is thrown away.
        SDValue Sub = CurDAG->getNode(ISD::SUB, DL, MVT::i32,
                                      Zero, Addr.getOperand(1));

        if (isDSOffsetLegal(Sub, ByteOffset, 16)) {
          SmallVector<SDValue, 3> Opnds;
          Opnds.push_back(Zero);
          Opnds.push_back(Addr.getOperand(1));

          // FIXME: Select to VOP3 version for with-carry.
          unsigned SubOp = AMDGPU::V_SUB_I32_e32;
          if (Subtarget->hasAddNoCarry()) {
            SubOp = AMDGPU::V_SUB_U32_e64;
            Opnds.push_back(
                CurDAG->getTargetConstant(0, {}, MVT::i1)); // clamp bit
          }

          MachineSDNode *MachineSub =
              CurDAG->getMachineNode(SubOp, DL, MVT::i32, Opnds);

          Base = SDValue(MachineSub, 0);
          Offset = CurDAG->getTargetConstant(ByteOffset, DL, MVT::i16);
          return true;
        }
      }
    }
  } else if (const ConstantSDNode *CAddr = dyn_cast<ConstantSDNode>(Addr)) {
    // If we have a constant address, prefer to put the constant into the
    // offset. This can save moves to load the constant address since multiple
    // operations can share the zero base address register, and enables merging
    // into read2 / write2 instructions.

    SDLoc DL(Addr);

    if (isUInt<16>(CAddr->getZExtValue())) {
      SDValue Zero = CurDAG->getTargetConstant(0, DL, MVT::i32);
      MachineSDNode *MovZero = CurDAG->getMachineNode(AMDGPU::V_MOV_B32_e32,
                                 DL, MVT::i32, Zero);
      Base = SDValue(MovZero, 0);
      Offset = CurDAG->getTargetConstant(CAddr->getZExtValue(), DL, MVT::i16);
      return true;
    }
  }

  // default case
  Base = Addr;
  Offset = CurDAG->getTargetConstant(0, SDLoc(Addr), MVT::i16);
  return true;
}

// TODO: If offset is too big, put low 16-bit into offset.
bool AMDGPUDAGToDAGISel::SelectDS64Bit4ByteAligned(SDValue Addr, SDValue &Base,
                                                   SDValue &Offset0,
                                                   SDValue &Offset1) const {
  SDLoc DL(Addr);

  if (CurDAG->isBaseWithConstantOffset(Addr)) {
    SDValue N0 = Addr.getOperand(0);
    SDValue N1 = Addr.getOperand(1);
    ConstantSDNode *C1 = cast<ConstantSDNode>(N1);
    unsigned DWordOffset0 = C1->getZExtValue() / 4;
    unsigned DWordOffset1 = DWordOffset0 + 1;
    // (add n0, c0)
    if (isDSOffsetLegal(N0, DWordOffset1, 8)) {
      Base = N0;
      Offset0 = CurDAG->getTargetConstant(DWordOffset0, DL, MVT::i8);
      Offset1 = CurDAG->getTargetConstant(DWordOffset1, DL, MVT::i8);
      return true;
    }
  } else if (Addr.getOpcode() == ISD::SUB) {
    // sub C, x -> add (sub 0, x), C
    if (const ConstantSDNode *C = dyn_cast<ConstantSDNode>(Addr.getOperand(0))) {
      unsigned DWordOffset0 = C->getZExtValue() / 4;
      unsigned DWordOffset1 = DWordOffset0 + 1;

      if (isUInt<8>(DWordOffset0)) {
        SDLoc DL(Addr);
        SDValue Zero = CurDAG->getTargetConstant(0, DL, MVT::i32);

        // XXX - This is kind of hacky. Create a dummy sub node so we can check
        // the known bits in isDSOffsetLegal. We need to emit the selected node
        // here, so this is thrown away.
        SDValue Sub = CurDAG->getNode(ISD::SUB, DL, MVT::i32,
                                      Zero, Addr.getOperand(1));

        if (isDSOffsetLegal(Sub, DWordOffset1, 8)) {
          SmallVector<SDValue, 3> Opnds;
          Opnds.push_back(Zero);
          Opnds.push_back(Addr.getOperand(1));
          unsigned SubOp = AMDGPU::V_SUB_I32_e32;
          if (Subtarget->hasAddNoCarry()) {
            SubOp = AMDGPU::V_SUB_U32_e64;
            Opnds.push_back(
                CurDAG->getTargetConstant(0, {}, MVT::i1)); // clamp bit
          }

          MachineSDNode *MachineSub
            = CurDAG->getMachineNode(SubOp, DL, MVT::i32, Opnds);

          Base = SDValue(MachineSub, 0);
          Offset0 = CurDAG->getTargetConstant(DWordOffset0, DL, MVT::i8);
          Offset1 = CurDAG->getTargetConstant(DWordOffset1, DL, MVT::i8);
          return true;
        }
      }
    }
  } else if (const ConstantSDNode *CAddr = dyn_cast<ConstantSDNode>(Addr)) {
    unsigned DWordOffset0 = CAddr->getZExtValue() / 4;
    unsigned DWordOffset1 = DWordOffset0 + 1;
    assert(4 * DWordOffset0 == CAddr->getZExtValue());

    if (isUInt<8>(DWordOffset0) && isUInt<8>(DWordOffset1)) {
      SDValue Zero = CurDAG->getTargetConstant(0, DL, MVT::i32);
      MachineSDNode *MovZero
        = CurDAG->getMachineNode(AMDGPU::V_MOV_B32_e32,
                                 DL, MVT::i32, Zero);
      Base = SDValue(MovZero, 0);
      Offset0 = CurDAG->getTargetConstant(DWordOffset0, DL, MVT::i8);
      Offset1 = CurDAG->getTargetConstant(DWordOffset1, DL, MVT::i8);
      return true;
    }
  }

  // default case

  Base = Addr;
  Offset0 = CurDAG->getTargetConstant(0, DL, MVT::i8);
  Offset1 = CurDAG->getTargetConstant(1, DL, MVT::i8);
  return true;
}

bool AMDGPUDAGToDAGISel::SelectMUBUF(SDValue Addr, SDValue &Ptr,
                                     SDValue &VAddr, SDValue &SOffset,
                                     SDValue &Offset, SDValue &Offen,
                                     SDValue &Idxen, SDValue &Addr64,
                                     SDValue &GLC, SDValue &SLC,
                                     SDValue &TFE, SDValue &DLC,
                                     SDValue &SWZ) const {
  // Subtarget prefers to use flat instruction
  // FIXME: This should be a pattern predicate and not reach here
  if (Subtarget->useFlatForGlobal())
    return false;

  SDLoc DL(Addr);

  if (!GLC.getNode())
    GLC = CurDAG->getTargetConstant(0, DL, MVT::i1);
  if (!SLC.getNode())
    SLC = CurDAG->getTargetConstant(0, DL, MVT::i1);
  TFE = CurDAG->getTargetConstant(0, DL, MVT::i1);
  DLC = CurDAG->getTargetConstant(0, DL, MVT::i1);
  SWZ = CurDAG->getTargetConstant(0, DL, MVT::i1);

  Idxen = CurDAG->getTargetConstant(0, DL, MVT::i1);
  Offen = CurDAG->getTargetConstant(0, DL, MVT::i1);
  Addr64 = CurDAG->getTargetConstant(0, DL, MVT::i1);
  SOffset = CurDAG->getTargetConstant(0, DL, MVT::i32);

  ConstantSDNode *C1 = nullptr;
  SDValue N0 = Addr;
  if (CurDAG->isBaseWithConstantOffset(Addr)) {
    C1 = cast<ConstantSDNode>(Addr.getOperand(1));
    if (isUInt<32>(C1->getZExtValue()))
      N0 = Addr.getOperand(0);
    else
      C1 = nullptr;
  }

  if (N0.getOpcode() == ISD::ADD) {
    // (add N2, N3) -> addr64, or
    // (add (add N2, N3), C1) -> addr64
    SDValue N2 = N0.getOperand(0);
    SDValue N3 = N0.getOperand(1);
    Addr64 = CurDAG->getTargetConstant(1, DL, MVT::i1);

    if (N2->isDivergent()) {
      if (N3->isDivergent()) {
        // Both N2 and N3 are divergent. Use N0 (the result of the add) as the
        // addr64, and construct the resource from a 0 address.
        Ptr = SDValue(buildSMovImm64(DL, 0, MVT::v2i32), 0);
        VAddr = N0;
      } else {
        // N2 is divergent, N3 is not.
        Ptr = N3;
        VAddr = N2;
      }
    } else {
      // N2 is not divergent.
      Ptr = N2;
      VAddr = N3;
    }
    Offset = CurDAG->getTargetConstant(0, DL, MVT::i16);
  } else if (N0->isDivergent()) {
    // N0 is divergent. Use it as the addr64, and construct the resource from a
    // 0 address.
    Ptr = SDValue(buildSMovImm64(DL, 0, MVT::v2i32), 0);
    VAddr = N0;
    Addr64 = CurDAG->getTargetConstant(1, DL, MVT::i1);
  } else {
    // N0 -> offset, or
    // (N0 + C1) -> offset
    VAddr = CurDAG->getTargetConstant(0, DL, MVT::i32);
    Ptr = N0;
  }

  if (!C1) {
    // No offset.
    Offset = CurDAG->getTargetConstant(0, DL, MVT::i16);
    return true;
  }

  if (SIInstrInfo::isLegalMUBUFImmOffset(C1->getZExtValue())) {
    // Legal offset for instruction.
    Offset = CurDAG->getTargetConstant(C1->getZExtValue(), DL, MVT::i16);
    return true;
  }

  // Illegal offset, store it in soffset.
  Offset = CurDAG->getTargetConstant(0, DL, MVT::i16);
  SOffset =
      SDValue(CurDAG->getMachineNode(
                  AMDGPU::S_MOV_B32, DL, MVT::i32,
                  CurDAG->getTargetConstant(C1->getZExtValue(), DL, MVT::i32)),
              0);
  return true;
}

bool AMDGPUDAGToDAGISel::SelectMUBUFAddr64(SDValue Addr, SDValue &SRsrc,
                                           SDValue &VAddr, SDValue &SOffset,
                                           SDValue &Offset, SDValue &GLC,
                                           SDValue &SLC, SDValue &TFE,
                                           SDValue &DLC, SDValue &SWZ) const {
  SDValue Ptr, Offen, Idxen, Addr64;

  // addr64 bit was removed for volcanic islands.
  // FIXME: This should be a pattern predicate and not reach here
  if (!Subtarget->hasAddr64())
    return false;

  if (!SelectMUBUF(Addr, Ptr, VAddr, SOffset, Offset, Offen, Idxen, Addr64,
              GLC, SLC, TFE, DLC, SWZ))
    return false;

  ConstantSDNode *C = cast<ConstantSDNode>(Addr64);
  if (C->getSExtValue()) {
    SDLoc DL(Addr);

    const SITargetLowering& Lowering =
      *static_cast<const SITargetLowering*>(getTargetLowering());

    SRsrc = SDValue(Lowering.wrapAddr64Rsrc(*CurDAG, DL, Ptr), 0);
    return true;
  }

  return false;
}

bool AMDGPUDAGToDAGISel::SelectMUBUFAddr64(SDValue Addr, SDValue &SRsrc,
                                           SDValue &VAddr, SDValue &SOffset,
                                           SDValue &Offset,
                                           SDValue &SLC) const {
  SLC = CurDAG->getTargetConstant(0, SDLoc(Addr), MVT::i1);
  SDValue GLC, TFE, DLC, SWZ;

  return SelectMUBUFAddr64(Addr, SRsrc, VAddr, SOffset, Offset, GLC, SLC, TFE, DLC, SWZ);
}

static bool isStackPtrRelative(const MachinePointerInfo &PtrInfo) {
  auto PSV = PtrInfo.V.dyn_cast<const PseudoSourceValue *>();
  return PSV && PSV->isStack();
}

std::pair<SDValue, SDValue> AMDGPUDAGToDAGISel::foldFrameIndex(SDValue N) const {
  SDLoc DL(N);
  const MachineFunction &MF = CurDAG->getMachineFunction();
  const SIMachineFunctionInfo *Info = MF.getInfo<SIMachineFunctionInfo>();

  if (auto FI = dyn_cast<FrameIndexSDNode>(N)) {
    SDValue TFI = CurDAG->getTargetFrameIndex(FI->getIndex(),
                                              FI->getValueType(0));

    // If we can resolve this to a frame index access, this will be relative to
    // either the stack or frame pointer SGPR.
    return std::make_pair(
        TFI, CurDAG->getRegister(Info->getStackPtrOffsetReg(), MVT::i32));
  }

  // If we don't know this private access is a local stack object, it needs to
  // be relative to the entry point's scratch wave offset.
  return std::make_pair(N, CurDAG->getTargetConstant(0, DL, MVT::i32));
}

bool AMDGPUDAGToDAGISel::SelectMUBUFScratchOffen(SDNode *Parent,
                                                 SDValue Addr, SDValue &Rsrc,
                                                 SDValue &VAddr, SDValue &SOffset,
                                                 SDValue &ImmOffset) const {

  SDLoc DL(Addr);
  MachineFunction &MF = CurDAG->getMachineFunction();
  const SIMachineFunctionInfo *Info = MF.getInfo<SIMachineFunctionInfo>();

  Rsrc = CurDAG->getRegister(Info->getScratchRSrcReg(), MVT::v4i32);

  if (ConstantSDNode *CAddr = dyn_cast<ConstantSDNode>(Addr)) {
    int64_t Imm = CAddr->getSExtValue();
    const int64_t NullPtr =
        AMDGPUTargetMachine::getNullPointerValue(AMDGPUAS::PRIVATE_ADDRESS);
    // Don't fold null pointer.
    if (Imm != NullPtr) {
      SDValue HighBits = CurDAG->getTargetConstant(Imm & ~4095, DL, MVT::i32);
      MachineSDNode *MovHighBits = CurDAG->getMachineNode(
        AMDGPU::V_MOV_B32_e32, DL, MVT::i32, HighBits);
      VAddr = SDValue(MovHighBits, 0);

      // In a call sequence, stores to the argument stack area are relative to the
      // stack pointer.
      const MachinePointerInfo &PtrInfo
        = cast<MemSDNode>(Parent)->getPointerInfo();
      SOffset = isStackPtrRelative(PtrInfo)
        ? CurDAG->getRegister(Info->getStackPtrOffsetReg(), MVT::i32)
        : CurDAG->getTargetConstant(0, DL, MVT::i32);
      ImmOffset = CurDAG->getTargetConstant(Imm & 4095, DL, MVT::i16);
      return true;
    }
  }

  if (CurDAG->isBaseWithConstantOffset(Addr)) {
    // (add n0, c1)

    SDValue N0 = Addr.getOperand(0);
    SDValue N1 = Addr.getOperand(1);

    // Offsets in vaddr must be positive if range checking is enabled.
    //
    // The total computation of vaddr + soffset + offset must not overflow.  If
    // vaddr is negative, even if offset is 0 the sgpr offset add will end up
    // overflowing.
    //
    // Prior to gfx9, MUBUF instructions with the vaddr offset enabled would
    // always perform a range check. If a negative vaddr base index was used,
    // this would fail the range check. The overall address computation would
    // compute a valid address, but this doesn't happen due to the range
    // check. For out-of-bounds MUBUF loads, a 0 is returned.
    //
    // Therefore it should be safe to fold any VGPR offset on gfx9 into the
    // MUBUF vaddr, but not on older subtargets which can only do this if the
    // sign bit is known 0.
    ConstantSDNode *C1 = cast<ConstantSDNode>(N1);
    if (SIInstrInfo::isLegalMUBUFImmOffset(C1->getZExtValue()) &&
        (!Subtarget->privateMemoryResourceIsRangeChecked() ||
         CurDAG->SignBitIsZero(N0))) {
      std::tie(VAddr, SOffset) = foldFrameIndex(N0);
      ImmOffset = CurDAG->getTargetConstant(C1->getZExtValue(), DL, MVT::i16);
      return true;
    }
  }

  // (node)
  std::tie(VAddr, SOffset) = foldFrameIndex(Addr);
  ImmOffset = CurDAG->getTargetConstant(0, DL, MVT::i16);
  return true;
}

bool AMDGPUDAGToDAGISel::SelectMUBUFScratchOffset(SDNode *Parent,
                                                  SDValue Addr,
                                                  SDValue &SRsrc,
                                                  SDValue &SOffset,
                                                  SDValue &Offset) const {
  ConstantSDNode *CAddr = dyn_cast<ConstantSDNode>(Addr);
  if (!CAddr || !SIInstrInfo::isLegalMUBUFImmOffset(CAddr->getZExtValue()))
    return false;

  SDLoc DL(Addr);
  MachineFunction &MF = CurDAG->getMachineFunction();
  const SIMachineFunctionInfo *Info = MF.getInfo<SIMachineFunctionInfo>();

  SRsrc = CurDAG->getRegister(Info->getScratchRSrcReg(), MVT::v4i32);

  const MachinePointerInfo &PtrInfo = cast<MemSDNode>(Parent)->getPointerInfo();

  // FIXME: Get from MachinePointerInfo? We should only be using the frame
  // offset if we know this is in a call sequence.
  SOffset = isStackPtrRelative(PtrInfo)
                ? CurDAG->getRegister(Info->getStackPtrOffsetReg(), MVT::i32)
                : CurDAG->getTargetConstant(0, DL, MVT::i32);

  Offset = CurDAG->getTargetConstant(CAddr->getZExtValue(), DL, MVT::i16);
  return true;
}

bool AMDGPUDAGToDAGISel::SelectMUBUFOffset(SDValue Addr, SDValue &SRsrc,
                                           SDValue &SOffset, SDValue &Offset,
                                           SDValue &GLC, SDValue &SLC,
                                           SDValue &TFE, SDValue &DLC,
                                           SDValue &SWZ) const {
  SDValue Ptr, VAddr, Offen, Idxen, Addr64;
  const SIInstrInfo *TII =
    static_cast<const SIInstrInfo *>(Subtarget->getInstrInfo());

  if (!SelectMUBUF(Addr, Ptr, VAddr, SOffset, Offset, Offen, Idxen, Addr64,
              GLC, SLC, TFE, DLC, SWZ))
    return false;

  if (!cast<ConstantSDNode>(Offen)->getSExtValue() &&
      !cast<ConstantSDNode>(Idxen)->getSExtValue() &&
      !cast<ConstantSDNode>(Addr64)->getSExtValue()) {
    uint64_t Rsrc = TII->getDefaultRsrcDataFormat() |
                    APInt::getAllOnesValue(32).getZExtValue(); // Size
    SDLoc DL(Addr);

    const SITargetLowering& Lowering =
      *static_cast<const SITargetLowering*>(getTargetLowering());

    SRsrc = SDValue(Lowering.buildRSRC(*CurDAG, DL, Ptr, 0, Rsrc), 0);
    return true;
  }
  return false;
}

bool AMDGPUDAGToDAGISel::SelectMUBUFOffset(SDValue Addr, SDValue &SRsrc,
                                           SDValue &Soffset, SDValue &Offset
                                           ) const {
  SDValue GLC, SLC, TFE, DLC, SWZ;

  return SelectMUBUFOffset(Addr, SRsrc, Soffset, Offset, GLC, SLC, TFE, DLC, SWZ);
}
bool AMDGPUDAGToDAGISel::SelectMUBUFOffset(SDValue Addr, SDValue &SRsrc,
                                           SDValue &Soffset, SDValue &Offset,
                                           SDValue &SLC) const {
  SDValue GLC, TFE, DLC, SWZ;

  return SelectMUBUFOffset(Addr, SRsrc, Soffset, Offset, GLC, SLC, TFE, DLC, SWZ);
}

// Find a load or store from corresponding pattern root.
// Roots may be build_vector, bitconvert or their combinations.
static MemSDNode* findMemSDNode(SDNode *N) {
  N = AMDGPUTargetLowering::stripBitcast(SDValue(N,0)).getNode();
  if (MemSDNode *MN = dyn_cast<MemSDNode>(N))
    return MN;
  assert(isa<BuildVectorSDNode>(N));
  for (SDValue V : N->op_values())
    if (MemSDNode *MN =
          dyn_cast<MemSDNode>(AMDGPUTargetLowering::stripBitcast(V)))
      return MN;
  llvm_unreachable("cannot find MemSDNode in the pattern!");
}

template <bool IsSigned>
bool AMDGPUDAGToDAGISel::SelectFlatOffset(SDNode *N,
                                          SDValue Addr,
                                          SDValue &VAddr,
                                          SDValue &Offset,
                                          SDValue &SLC) const {
  int64_t OffsetVal = 0;

  if (Subtarget->hasFlatInstOffsets() &&
      (!Subtarget->hasFlatSegmentOffsetBug() ||
       findMemSDNode(N)->getAddressSpace() != AMDGPUAS::FLAT_ADDRESS) &&
      CurDAG->isBaseWithConstantOffset(Addr)) {
    SDValue N0 = Addr.getOperand(0);
    SDValue N1 = Addr.getOperand(1);
    uint64_t COffsetVal = cast<ConstantSDNode>(N1)->getSExtValue();

    const SIInstrInfo *TII = Subtarget->getInstrInfo();
    unsigned AS = findMemSDNode(N)->getAddressSpace();
    if (TII->isLegalFLATOffset(COffsetVal, AS, IsSigned)) {
      Addr = N0;
      OffsetVal = COffsetVal;
    } else {
      // If the offset doesn't fit, put the low bits into the offset field and
      // add the rest.

      SDLoc DL(N);
      uint64_t ImmField;
      const unsigned NumBits = TII->getNumFlatOffsetBits(AS, IsSigned);
      if (IsSigned) {
        ImmField = SignExtend64(COffsetVal, NumBits);

        // Don't use a negative offset field if the base offset is positive.
        // Since the scheduler currently relies on the offset field, doing so
        // could result in strange scheduling decisions.

        // TODO: Should we not do this in the opposite direction as well?
        if (static_cast<int64_t>(COffsetVal) > 0) {
          if (static_cast<int64_t>(ImmField) < 0) {
            const uint64_t OffsetMask = maskTrailingOnes<uint64_t>(NumBits - 1);
            ImmField = COffsetVal & OffsetMask;
          }
        }
      } else {
        // TODO: Should we do this for a negative offset?
        const uint64_t OffsetMask = maskTrailingOnes<uint64_t>(NumBits);
        ImmField = COffsetVal & OffsetMask;
      }

      uint64_t RemainderOffset = COffsetVal - ImmField;

      assert(TII->isLegalFLATOffset(ImmField, AS, IsSigned));
      assert(RemainderOffset + ImmField == COffsetVal);

      OffsetVal = ImmField;

      // TODO: Should this try to use a scalar add pseudo if the base address is
      // uniform and saddr is usable?
      SDValue Sub0 = CurDAG->getTargetConstant(AMDGPU::sub0, DL, MVT::i32);
      SDValue Sub1 = CurDAG->getTargetConstant(AMDGPU::sub1, DL, MVT::i32);

      SDNode *N0Lo = CurDAG->getMachineNode(TargetOpcode::EXTRACT_SUBREG,
                                            DL, MVT::i32, N0, Sub0);
      SDNode *N0Hi = CurDAG->getMachineNode(TargetOpcode::EXTRACT_SUBREG,
                                            DL, MVT::i32, N0, Sub1);

      SDValue AddOffsetLo
        = getMaterializedScalarImm32(Lo_32(RemainderOffset), DL);
      SDValue AddOffsetHi
        = getMaterializedScalarImm32(Hi_32(RemainderOffset), DL);

      SDVTList VTs = CurDAG->getVTList(MVT::i32, MVT::i1);
      SDValue Clamp = CurDAG->getTargetConstant(0, DL, MVT::i1);

      SDNode *Add = CurDAG->getMachineNode(
        AMDGPU::V_ADD_I32_e64, DL, VTs,
        {AddOffsetLo, SDValue(N0Lo, 0), Clamp});

      SDNode *Addc = CurDAG->getMachineNode(
        AMDGPU::V_ADDC_U32_e64, DL, VTs,
        {AddOffsetHi, SDValue(N0Hi, 0), SDValue(Add, 1), Clamp});

      SDValue RegSequenceArgs[] = {
        CurDAG->getTargetConstant(AMDGPU::VReg_64RegClassID, DL, MVT::i32),
        SDValue(Add, 0), Sub0, SDValue(Addc, 0), Sub1
      };

      Addr = SDValue(CurDAG->getMachineNode(AMDGPU::REG_SEQUENCE, DL,
                                            MVT::i64, RegSequenceArgs), 0);
    }
  }

  VAddr = Addr;
  Offset = CurDAG->getTargetConstant(OffsetVal, SDLoc(), MVT::i16);
  SLC = CurDAG->getTargetConstant(0, SDLoc(), MVT::i1);
  return true;
}

bool AMDGPUDAGToDAGISel::SelectFlatAtomic(SDNode *N,
                                          SDValue Addr,
                                          SDValue &VAddr,
                                          SDValue &Offset,
                                          SDValue &SLC) const {
  return SelectFlatOffset<false>(N, Addr, VAddr, Offset, SLC);
}

bool AMDGPUDAGToDAGISel::SelectFlatAtomicSigned(SDNode *N,
                                                SDValue Addr,
                                                SDValue &VAddr,
                                                SDValue &Offset,
                                                SDValue &SLC) const {
  return SelectFlatOffset<true>(N, Addr, VAddr, Offset, SLC);
}

bool AMDGPUDAGToDAGISel::SelectSMRDOffset(SDValue ByteOffsetNode,
                                          SDValue &Offset, bool &Imm) const {
  ConstantSDNode *C = dyn_cast<ConstantSDNode>(ByteOffsetNode);
  if (!C) {
    if (ByteOffsetNode.getValueType().isScalarInteger() &&
        ByteOffsetNode.getValueType().getSizeInBits() == 32) {
      Offset = ByteOffsetNode;
      Imm = false;
      return true;
    }
    if (ByteOffsetNode.getOpcode() == ISD::ZERO_EXTEND) {
      if (ByteOffsetNode.getOperand(0).getValueType().getSizeInBits() == 32) {
        Offset = ByteOffsetNode.getOperand(0);
        Imm = false;
        return true;
      }
    }
    return false;
  }

  SDLoc SL(ByteOffsetNode);
  // GFX9 and GFX10 have signed byte immediate offsets.
  int64_t ByteOffset = C->getSExtValue();
  Optional<int64_t> EncodedOffset =
      AMDGPU::getSMRDEncodedOffset(*Subtarget, ByteOffset, false);
  if (EncodedOffset) {
    Offset = CurDAG->getTargetConstant(*EncodedOffset, SL, MVT::i32);
    Imm = true;
    return true;
  }

  // SGPR and literal offsets are unsigned.
  if (ByteOffset < 0)
    return false;

  EncodedOffset = AMDGPU::getSMRDEncodedLiteralOffset32(*Subtarget, ByteOffset);
  if (EncodedOffset) {
    Offset = CurDAG->getTargetConstant(*EncodedOffset, SL, MVT::i32);
    return true;
  }

  if (!isUInt<32>(ByteOffset) && !isInt<32>(ByteOffset))
    return false;

  SDValue C32Bit = CurDAG->getTargetConstant(ByteOffset, SL, MVT::i32);
  Offset = SDValue(
      CurDAG->getMachineNode(AMDGPU::S_MOV_B32, SL, MVT::i32, C32Bit), 0);

  return true;
}

SDValue AMDGPUDAGToDAGISel::Expand32BitAddress(SDValue Addr) const {
  if (Addr.getValueType() != MVT::i32)
    return Addr;

  // Zero-extend a 32-bit address.
  SDLoc SL(Addr);

  const MachineFunction &MF = CurDAG->getMachineFunction();
  const SIMachineFunctionInfo *Info = MF.getInfo<SIMachineFunctionInfo>();
  unsigned AddrHiVal = Info->get32BitAddressHighBits();
  SDValue AddrHi = CurDAG->getTargetConstant(AddrHiVal, SL, MVT::i32);

  const SDValue Ops[] = {
    CurDAG->getTargetConstant(AMDGPU::SReg_64_XEXECRegClassID, SL, MVT::i32),
    Addr,
    CurDAG->getTargetConstant(AMDGPU::sub0, SL, MVT::i32),
    SDValue(CurDAG->getMachineNode(AMDGPU::S_MOV_B32, SL, MVT::i32, AddrHi),
            0),
    CurDAG->getTargetConstant(AMDGPU::sub1, SL, MVT::i32),
  };

  return SDValue(CurDAG->getMachineNode(AMDGPU::REG_SEQUENCE, SL, MVT::i64,
                                        Ops), 0);
}

bool AMDGPUDAGToDAGISel::SelectSMRD(SDValue Addr, SDValue &SBase,
                                     SDValue &Offset, bool &Imm) const {
  SDLoc SL(Addr);

  // A 32-bit (address + offset) should not cause unsigned 32-bit integer
  // wraparound, because s_load instructions perform the addition in 64 bits.
  if ((Addr.getValueType() != MVT::i32 ||
       Addr->getFlags().hasNoUnsignedWrap()) &&
      (CurDAG->isBaseWithConstantOffset(Addr) ||
       Addr.getOpcode() == ISD::ADD)) {
    SDValue N0 = Addr.getOperand(0);
    SDValue N1 = Addr.getOperand(1);

    if (SelectSMRDOffset(N1, Offset, Imm)) {
      SBase = Expand32BitAddress(N0);
      return true;
    }
  }
  SBase = Expand32BitAddress(Addr);
  Offset = CurDAG->getTargetConstant(0, SL, MVT::i32);
  Imm = true;
  return true;
}

bool AMDGPUDAGToDAGISel::SelectSMRDImm(SDValue Addr, SDValue &SBase,
                                       SDValue &Offset) const {
  bool Imm = false;
  return SelectSMRD(Addr, SBase, Offset, Imm) && Imm;
}

bool AMDGPUDAGToDAGISel::SelectSMRDImm32(SDValue Addr, SDValue &SBase,
                                         SDValue &Offset) const {

  assert(Subtarget->getGeneration() == AMDGPUSubtarget::SEA_ISLANDS);

  bool Imm = false;
  if (!SelectSMRD(Addr, SBase, Offset, Imm))
    return false;

  return !Imm && isa<ConstantSDNode>(Offset);
}

bool AMDGPUDAGToDAGISel::SelectSMRDSgpr(SDValue Addr, SDValue &SBase,
                                        SDValue &Offset) const {
  bool Imm = false;
  return SelectSMRD(Addr, SBase, Offset, Imm) && !Imm &&
         !isa<ConstantSDNode>(Offset);
}

bool AMDGPUDAGToDAGISel::SelectSMRDBufferImm(SDValue Addr,
                                             SDValue &Offset) const {
  if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(Addr)) {
    // The immediate offset for S_BUFFER instructions is unsigned.
    if (auto Imm =
            AMDGPU::getSMRDEncodedOffset(*Subtarget, C->getZExtValue(), true)) {
      Offset = CurDAG->getTargetConstant(*Imm, SDLoc(Addr), MVT::i32);
      return true;
    }
  }

  return false;
}

bool AMDGPUDAGToDAGISel::SelectSMRDBufferImm32(SDValue Addr,
                                               SDValue &Offset) const {
  assert(Subtarget->getGeneration() == AMDGPUSubtarget::SEA_ISLANDS);

  if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(Addr)) {
    if (auto Imm = AMDGPU::getSMRDEncodedLiteralOffset32(*Subtarget,
                                                         C->getZExtValue())) {
      Offset = CurDAG->getTargetConstant(*Imm, SDLoc(Addr), MVT::i32);
      return true;
    }
  }

  return false;
}

bool AMDGPUDAGToDAGISel::SelectMOVRELOffset(SDValue Index,
                                            SDValue &Base,
                                            SDValue &Offset) const {
  SDLoc DL(Index);

  if (CurDAG->isBaseWithConstantOffset(Index)) {
    SDValue N0 = Index.getOperand(0);
    SDValue N1 = Index.getOperand(1);
    ConstantSDNode *C1 = cast<ConstantSDNode>(N1);

    // (add n0, c0)
    // Don't peel off the offset (c0) if doing so could possibly lead
    // the base (n0) to be negative.
    // (or n0, |c0|) can never change a sign given isBaseWithConstantOffset.
    if (C1->getSExtValue() <= 0 || CurDAG->SignBitIsZero(N0) ||
        (Index->getOpcode() == ISD::OR && C1->getSExtValue() >= 0)) {
      Base = N0;
      Offset = CurDAG->getTargetConstant(C1->getZExtValue(), DL, MVT::i32);
      return true;
    }
  }

  if (isa<ConstantSDNode>(Index))
    return false;

  Base = Index;
  Offset = CurDAG->getTargetConstant(0, DL, MVT::i32);
  return true;
}

SDNode *AMDGPUDAGToDAGISel::getS_BFE(unsigned Opcode, const SDLoc &DL,
                                     SDValue Val, uint32_t Offset,
                                     uint32_t Width) {
  // Transformation function, pack the offset and width of a BFE into
  // the format expected by the S_BFE_I32 / S_BFE_U32. In the second
  // source, bits [5:0] contain the offset and bits [22:16] the width.
  uint32_t PackedVal = Offset | (Width << 16);
  SDValue PackedConst = CurDAG->getTargetConstant(PackedVal, DL, MVT::i32);

  return CurDAG->getMachineNode(Opcode, DL, MVT::i32, Val, PackedConst);
}

void AMDGPUDAGToDAGISel::SelectS_BFEFromShifts(SDNode *N) {
  // "(a << b) srl c)" ---> "BFE_U32 a, (c-b), (32-c)
  // "(a << b) sra c)" ---> "BFE_I32 a, (c-b), (32-c)
  // Predicate: 0 < b <= c < 32

  const SDValue &Shl = N->getOperand(0);
  ConstantSDNode *B = dyn_cast<ConstantSDNode>(Shl->getOperand(1));
  ConstantSDNode *C = dyn_cast<ConstantSDNode>(N->getOperand(1));

  if (B && C) {
    uint32_t BVal = B->getZExtValue();
    uint32_t CVal = C->getZExtValue();

    if (0 < BVal && BVal <= CVal && CVal < 32) {
      bool Signed = N->getOpcode() == ISD::SRA;
      unsigned Opcode = Signed ? AMDGPU::S_BFE_I32 : AMDGPU::S_BFE_U32;

      ReplaceNode(N, getS_BFE(Opcode, SDLoc(N), Shl.getOperand(0), CVal - BVal,
                              32 - CVal));
      return;
    }
  }
  SelectCode(N);
}

void AMDGPUDAGToDAGISel::SelectS_BFE(SDNode *N) {
  switch (N->getOpcode()) {
  case ISD::AND:
    if (N->getOperand(0).getOpcode() == ISD::SRL) {
      // "(a srl b) & mask" ---> "BFE_U32 a, b, popcount(mask)"
      // Predicate: isMask(mask)
      const SDValue &Srl = N->getOperand(0);
      ConstantSDNode *Shift = dyn_cast<ConstantSDNode>(Srl.getOperand(1));
      ConstantSDNode *Mask = dyn_cast<ConstantSDNode>(N->getOperand(1));

      if (Shift && Mask) {
        uint32_t ShiftVal = Shift->getZExtValue();
        uint32_t MaskVal = Mask->getZExtValue();

        if (isMask_32(MaskVal)) {
          uint32_t WidthVal = countPopulation(MaskVal);

          ReplaceNode(N, getS_BFE(AMDGPU::S_BFE_U32, SDLoc(N),
                                  Srl.getOperand(0), ShiftVal, WidthVal));
          return;
        }
      }
    }
    break;
  case ISD::SRL:
    if (N->getOperand(0).getOpcode() == ISD::AND) {
      // "(a & mask) srl b)" ---> "BFE_U32 a, b, popcount(mask >> b)"
      // Predicate: isMask(mask >> b)
      const SDValue &And = N->getOperand(0);
      ConstantSDNode *Shift = dyn_cast<ConstantSDNode>(N->getOperand(1));
      ConstantSDNode *Mask = dyn_cast<ConstantSDNode>(And->getOperand(1));

      if (Shift && Mask) {
        uint32_t ShiftVal = Shift->getZExtValue();
        uint32_t MaskVal = Mask->getZExtValue() >> ShiftVal;

        if (isMask_32(MaskVal)) {
          uint32_t WidthVal = countPopulation(MaskVal);

          ReplaceNode(N, getS_BFE(AMDGPU::S_BFE_U32, SDLoc(N),
                                  And.getOperand(0), ShiftVal, WidthVal));
          return;
        }
      }
    } else if (N->getOperand(0).getOpcode() == ISD::SHL) {
      SelectS_BFEFromShifts(N);
      return;
    }
    break;
  case ISD::SRA:
    if (N->getOperand(0).getOpcode() == ISD::SHL) {
      SelectS_BFEFromShifts(N);
      return;
    }
    break;

  case ISD::SIGN_EXTEND_INREG: {
    // sext_inreg (srl x, 16), i8 -> bfe_i32 x, 16, 8
    SDValue Src = N->getOperand(0);
    if (Src.getOpcode() != ISD::SRL)
      break;

    const ConstantSDNode *Amt = dyn_cast<ConstantSDNode>(Src.getOperand(1));
    if (!Amt)
      break;

    unsigned Width = cast<VTSDNode>(N->getOperand(1))->getVT().getSizeInBits();
    ReplaceNode(N, getS_BFE(AMDGPU::S_BFE_I32, SDLoc(N), Src.getOperand(0),
                            Amt->getZExtValue(), Width));
    return;
  }
  }

  SelectCode(N);
}

bool AMDGPUDAGToDAGISel::isCBranchSCC(const SDNode *N) const {
  assert(N->getOpcode() == ISD::BRCOND);
  if (!N->hasOneUse())
    return false;

  SDValue Cond = N->getOperand(1);
  if (Cond.getOpcode() == ISD::CopyToReg)
    Cond = Cond.getOperand(2);

  if (Cond.getOpcode() != ISD::SETCC || !Cond.hasOneUse())
    return false;

  MVT VT = Cond.getOperand(0).getSimpleValueType();
  if (VT == MVT::i32)
    return true;

  if (VT == MVT::i64) {
    auto ST = static_cast<const GCNSubtarget *>(Subtarget);

    ISD::CondCode CC = cast<CondCodeSDNode>(Cond.getOperand(2))->get();
    return (CC == ISD::SETEQ || CC == ISD::SETNE) && ST->hasScalarCompareEq64();
  }

  return false;
}

void AMDGPUDAGToDAGISel::SelectBRCOND(SDNode *N) {
  SDValue Cond = N->getOperand(1);

  if (Cond.isUndef()) {
    CurDAG->SelectNodeTo(N, AMDGPU::SI_BR_UNDEF, MVT::Other,
                         N->getOperand(2), N->getOperand(0));
    return;
  }

  const GCNSubtarget *ST = static_cast<const GCNSubtarget *>(Subtarget);
  const SIRegisterInfo *TRI = ST->getRegisterInfo();

  bool UseSCCBr = isCBranchSCC(N) && isUniformBr(N);
  unsigned BrOp = UseSCCBr ? AMDGPU::S_CBRANCH_SCC1 : AMDGPU::S_CBRANCH_VCCNZ;
  Register CondReg = UseSCCBr ? AMDGPU::SCC : TRI->getVCC();
  SDLoc SL(N);

  if (!UseSCCBr) {
    // This is the case that we are selecting to S_CBRANCH_VCCNZ.  We have not
    // analyzed what generates the vcc value, so we do not know whether vcc
    // bits for disabled lanes are 0.  Thus we need to mask out bits for
    // disabled lanes.
    //
    // For the case that we select S_CBRANCH_SCC1 and it gets
    // changed to S_CBRANCH_VCCNZ in SIFixSGPRCopies, SIFixSGPRCopies calls
    // SIInstrInfo::moveToVALU which inserts the S_AND).
    //
    // We could add an analysis of what generates the vcc value here and omit
    // the S_AND when is unnecessary. But it would be better to add a separate
    // pass after SIFixSGPRCopies to do the unnecessary S_AND removal, so it
    // catches both cases.
    Cond = SDValue(CurDAG->getMachineNode(ST->isWave32() ? AMDGPU::S_AND_B32
                                                         : AMDGPU::S_AND_B64,
                     SL, MVT::i1,
                     CurDAG->getRegister(ST->isWave32() ? AMDGPU::EXEC_LO
                                                        : AMDGPU::EXEC,
                                         MVT::i1),
                    Cond),
                   0);
  }

  SDValue VCC = CurDAG->getCopyToReg(N->getOperand(0), SL, CondReg, Cond);
  CurDAG->SelectNodeTo(N, BrOp, MVT::Other,
                       N->getOperand(2), // Basic Block
                       VCC.getValue(0));
}

void AMDGPUDAGToDAGISel::SelectFMAD_FMA(SDNode *N) {
  MVT VT = N->getSimpleValueType(0);
  bool IsFMA = N->getOpcode() == ISD::FMA;
  if (VT != MVT::f32 || (!Subtarget->hasMadMixInsts() &&
                         !Subtarget->hasFmaMixInsts()) ||
      ((IsFMA && Subtarget->hasMadMixInsts()) ||
       (!IsFMA && Subtarget->hasFmaMixInsts()))) {
    SelectCode(N);
    return;
  }

  SDValue Src0 = N->getOperand(0);
  SDValue Src1 = N->getOperand(1);
  SDValue Src2 = N->getOperand(2);
  unsigned Src0Mods, Src1Mods, Src2Mods;

  // Avoid using v_mad_mix_f32/v_fma_mix_f32 unless there is actually an operand
  // using the conversion from f16.
  bool Sel0 = SelectVOP3PMadMixModsImpl(Src0, Src0, Src0Mods);
  bool Sel1 = SelectVOP3PMadMixModsImpl(Src1, Src1, Src1Mods);
  bool Sel2 = SelectVOP3PMadMixModsImpl(Src2, Src2, Src2Mods);

  assert((IsFMA || !Mode.allFP32Denormals()) &&
         "fmad selected with denormals enabled");
  // TODO: We can select this with f32 denormals enabled if all the sources are
  // converted from f16 (in which case fmad isn't legal).

  if (Sel0 || Sel1 || Sel2) {
    // For dummy operands.
    SDValue Zero = CurDAG->getTargetConstant(0, SDLoc(), MVT::i32);
    SDValue Ops[] = {
      CurDAG->getTargetConstant(Src0Mods, SDLoc(), MVT::i32), Src0,
      CurDAG->getTargetConstant(Src1Mods, SDLoc(), MVT::i32), Src1,
      CurDAG->getTargetConstant(Src2Mods, SDLoc(), MVT::i32), Src2,
      CurDAG->getTargetConstant(0, SDLoc(), MVT::i1),
      Zero, Zero
    };

    CurDAG->SelectNodeTo(N,
                         IsFMA ? AMDGPU::V_FMA_MIX_F32 : AMDGPU::V_MAD_MIX_F32,
                         MVT::f32, Ops);
  } else {
    SelectCode(N);
  }
}

// This is here because there isn't a way to use the generated sub0_sub1 as the
// subreg index to EXTRACT_SUBREG in tablegen.
void AMDGPUDAGToDAGISel::SelectATOMIC_CMP_SWAP(SDNode *N) {
  MemSDNode *Mem = cast<MemSDNode>(N);
  unsigned AS = Mem->getAddressSpace();
  if (AS == AMDGPUAS::FLAT_ADDRESS) {
    SelectCode(N);
    return;
  }

  MVT VT = N->getSimpleValueType(0);
  bool Is32 = (VT == MVT::i32);
  SDLoc SL(N);

  MachineSDNode *CmpSwap = nullptr;
  if (Subtarget->hasAddr64()) {
    SDValue SRsrc, VAddr, SOffset, Offset, SLC;

    if (SelectMUBUFAddr64(Mem->getBasePtr(), SRsrc, VAddr, SOffset, Offset, SLC)) {
      unsigned Opcode = Is32 ? AMDGPU::BUFFER_ATOMIC_CMPSWAP_ADDR64_RTN :
        AMDGPU::BUFFER_ATOMIC_CMPSWAP_X2_ADDR64_RTN;
      SDValue CmpVal = Mem->getOperand(2);

      // XXX - Do we care about glue operands?

      SDValue Ops[] = {
        CmpVal, VAddr, SRsrc, SOffset, Offset, SLC, Mem->getChain()
      };

      CmpSwap = CurDAG->getMachineNode(Opcode, SL, Mem->getVTList(), Ops);
    }
  }

  if (!CmpSwap) {
    SDValue SRsrc, SOffset, Offset, SLC;
    if (SelectMUBUFOffset(Mem->getBasePtr(), SRsrc, SOffset, Offset, SLC)) {
      unsigned Opcode = Is32 ? AMDGPU::BUFFER_ATOMIC_CMPSWAP_OFFSET_RTN :
        AMDGPU::BUFFER_ATOMIC_CMPSWAP_X2_OFFSET_RTN;

      SDValue CmpVal = Mem->getOperand(2);
      SDValue Ops[] = {
        CmpVal, SRsrc, SOffset, Offset, SLC, Mem->getChain()
      };

      CmpSwap = CurDAG->getMachineNode(Opcode, SL, Mem->getVTList(), Ops);
    }
  }

  if (!CmpSwap) {
    SelectCode(N);
    return;
  }

  MachineMemOperand *MMO = Mem->getMemOperand();
  CurDAG->setNodeMemRefs(CmpSwap, {MMO});

  unsigned SubReg = Is32 ? AMDGPU::sub0 : AMDGPU::sub0_sub1;
  SDValue Extract
    = CurDAG->getTargetExtractSubreg(SubReg, SL, VT, SDValue(CmpSwap, 0));

  ReplaceUses(SDValue(N, 0), Extract);
  ReplaceUses(SDValue(N, 1), SDValue(CmpSwap, 1));
  CurDAG->RemoveDeadNode(N);
}

void AMDGPUDAGToDAGISel::SelectDSAppendConsume(SDNode *N, unsigned IntrID) {
  // The address is assumed to be uniform, so if it ends up in a VGPR, it will
  // be copied to an SGPR with readfirstlane.
  unsigned Opc = IntrID == Intrinsic::amdgcn_ds_append ?
    AMDGPU::DS_APPEND : AMDGPU::DS_CONSUME;

  SDValue Chain = N->getOperand(0);
  SDValue Ptr = N->getOperand(2);
  MemIntrinsicSDNode *M = cast<MemIntrinsicSDNode>(N);
  MachineMemOperand *MMO = M->getMemOperand();
  bool IsGDS = M->getAddressSpace() == AMDGPUAS::REGION_ADDRESS;

  SDValue Offset;
  if (CurDAG->isBaseWithConstantOffset(Ptr)) {
    SDValue PtrBase = Ptr.getOperand(0);
    SDValue PtrOffset = Ptr.getOperand(1);

    const APInt &OffsetVal = cast<ConstantSDNode>(PtrOffset)->getAPIntValue();
    if (isDSOffsetLegal(PtrBase, OffsetVal.getZExtValue(), 16)) {
      N = glueCopyToM0(N, PtrBase);
      Offset = CurDAG->getTargetConstant(OffsetVal, SDLoc(), MVT::i32);
    }
  }

  if (!Offset) {
    N = glueCopyToM0(N, Ptr);
    Offset = CurDAG->getTargetConstant(0, SDLoc(), MVT::i32);
  }

  SDValue Ops[] = {
    Offset,
    CurDAG->getTargetConstant(IsGDS, SDLoc(), MVT::i32),
    Chain,
    N->getOperand(N->getNumOperands() - 1) // New glue
  };

  SDNode *Selected = CurDAG->SelectNodeTo(N, Opc, N->getVTList(), Ops);
  CurDAG->setNodeMemRefs(cast<MachineSDNode>(Selected), {MMO});
}

static unsigned gwsIntrinToOpcode(unsigned IntrID) {
  switch (IntrID) {
  case Intrinsic::amdgcn_ds_gws_init:
    return AMDGPU::DS_GWS_INIT;
  case Intrinsic::amdgcn_ds_gws_barrier:
    return AMDGPU::DS_GWS_BARRIER;
  case Intrinsic::amdgcn_ds_gws_sema_v:
    return AMDGPU::DS_GWS_SEMA_V;
  case Intrinsic::amdgcn_ds_gws_sema_br:
    return AMDGPU::DS_GWS_SEMA_BR;
  case Intrinsic::amdgcn_ds_gws_sema_p:
    return AMDGPU::DS_GWS_SEMA_P;
  case Intrinsic::amdgcn_ds_gws_sema_release_all:
    return AMDGPU::DS_GWS_SEMA_RELEASE_ALL;
  default:
    llvm_unreachable("not a gws intrinsic");
  }
}

void AMDGPUDAGToDAGISel::SelectDS_GWS(SDNode *N, unsigned IntrID) {
  if (IntrID == Intrinsic::amdgcn_ds_gws_sema_release_all &&
      !Subtarget->hasGWSSemaReleaseAll()) {
    // Let this error.
    SelectCode(N);
    return;
  }

  // Chain, intrinsic ID, vsrc, offset
  const bool HasVSrc = N->getNumOperands() == 4;
  assert(HasVSrc || N->getNumOperands() == 3);

  SDLoc SL(N);
  SDValue BaseOffset = N->getOperand(HasVSrc ? 3 : 2);
  int ImmOffset = 0;
  MemIntrinsicSDNode *M = cast<MemIntrinsicSDNode>(N);
  MachineMemOperand *MMO = M->getMemOperand();

  // Don't worry if the offset ends up in a VGPR. Only one lane will have
  // effect, so SIFixSGPRCopies will validly insert readfirstlane.

  // The resource id offset is computed as (<isa opaque base> + M0[21:16] +
  // offset field) % 64. Some versions of the programming guide omit the m0
  // part, or claim it's from offset 0.
  if (ConstantSDNode *ConstOffset = dyn_cast<ConstantSDNode>(BaseOffset)) {
    // If we have a constant offset, try to use the 0 in m0 as the base.
    // TODO: Look into changing the default m0 initialization value. If the
    // default -1 only set the low 16-bits, we could leave it as-is and add 1 to
    // the immediate offset.
    glueCopyToM0(N, CurDAG->getTargetConstant(0, SL, MVT::i32));
    ImmOffset = ConstOffset->getZExtValue();
  } else {
    if (CurDAG->isBaseWithConstantOffset(BaseOffset)) {
      ImmOffset = BaseOffset.getConstantOperandVal(1);
      BaseOffset = BaseOffset.getOperand(0);
    }

    // Prefer to do the shift in an SGPR since it should be possible to use m0
    // as the result directly. If it's already an SGPR, it will be eliminated
    // later.
    SDNode *SGPROffset
      = CurDAG->getMachineNode(AMDGPU::V_READFIRSTLANE_B32, SL, MVT::i32,
                               BaseOffset);
    // Shift to offset in m0
    SDNode *M0Base
      = CurDAG->getMachineNode(AMDGPU::S_LSHL_B32, SL, MVT::i32,
                               SDValue(SGPROffset, 0),
                               CurDAG->getTargetConstant(16, SL, MVT::i32));
    glueCopyToM0(N, SDValue(M0Base, 0));
  }

  SDValue Chain = N->getOperand(0);
  SDValue OffsetField = CurDAG->getTargetConstant(ImmOffset, SL, MVT::i32);

  // TODO: Can this just be removed from the instruction?
  SDValue GDS = CurDAG->getTargetConstant(1, SL, MVT::i1);

  const unsigned Opc = gwsIntrinToOpcode(IntrID);
  SmallVector<SDValue, 5> Ops;
  if (HasVSrc)
    Ops.push_back(N->getOperand(2));
  Ops.push_back(OffsetField);
  Ops.push_back(GDS);
  Ops.push_back(Chain);

  SDNode *Selected = CurDAG->SelectNodeTo(N, Opc, N->getVTList(), Ops);
  CurDAG->setNodeMemRefs(cast<MachineSDNode>(Selected), {MMO});
}

void AMDGPUDAGToDAGISel::SelectInterpP1F16(SDNode *N) {
  if (Subtarget->getLDSBankCount() != 16) {
    // This is a single instruction with a pattern.
    SelectCode(N);
    return;
  }

  SDLoc DL(N);

  // This requires 2 instructions. It is possible to write a pattern to support
  // this, but the generated isel emitter doesn't correctly deal with multiple
  // output instructions using the same physical register input. The copy to m0
  // is incorrectly placed before the second instruction.
  //
  // TODO: Match source modifiers.
  //
  // def : Pat <
  //   (int_amdgcn_interp_p1_f16
  //    (VOP3Mods f32:$src0, i32:$src0_modifiers),
  //                             (i32 timm:$attrchan), (i32 timm:$attr),
  //                             (i1 timm:$high), M0),
  //   (V_INTERP_P1LV_F16 $src0_modifiers, VGPR_32:$src0, timm:$attr,
  //       timm:$attrchan, 0,
  //       (V_INTERP_MOV_F32 2, timm:$attr, timm:$attrchan), timm:$high)> {
  //   let Predicates = [has16BankLDS];
  // }

  // 16 bank LDS
  SDValue ToM0 = CurDAG->getCopyToReg(CurDAG->getEntryNode(), DL, AMDGPU::M0,
                                      N->getOperand(5), SDValue());

  SDVTList VTs = CurDAG->getVTList(MVT::f32, MVT::Other);

  SDNode *InterpMov =
    CurDAG->getMachineNode(AMDGPU::V_INTERP_MOV_F32, DL, VTs, {
        CurDAG->getTargetConstant(2, DL, MVT::i32), // P0
        N->getOperand(3),  // Attr
        N->getOperand(2),  // Attrchan
        ToM0.getValue(1) // In glue
  });

  SDNode *InterpP1LV =
    CurDAG->getMachineNode(AMDGPU::V_INTERP_P1LV_F16, DL, MVT::f32, {
        CurDAG->getTargetConstant(0, DL, MVT::i32), // $src0_modifiers
        N->getOperand(1), // Src0
        N->getOperand(3), // Attr
        N->getOperand(2), // Attrchan
        CurDAG->getTargetConstant(0, DL, MVT::i32), // $src2_modifiers
        SDValue(InterpMov, 0), // Src2 - holds two f16 values selected by high
        N->getOperand(4), // high
        CurDAG->getTargetConstant(0, DL, MVT::i1), // $clamp
        CurDAG->getTargetConstant(0, DL, MVT::i32), // $omod
        SDValue(InterpMov, 1)
  });

  CurDAG->ReplaceAllUsesOfValueWith(SDValue(N, 0), SDValue(InterpP1LV, 0));
}

void AMDGPUDAGToDAGISel::SelectINTRINSIC_W_CHAIN(SDNode *N) {
  unsigned IntrID = cast<ConstantSDNode>(N->getOperand(1))->getZExtValue();
  switch (IntrID) {
  case Intrinsic::amdgcn_ds_append:
  case Intrinsic::amdgcn_ds_consume: {
    if (N->getValueType(0) != MVT::i32)
      break;
    SelectDSAppendConsume(N, IntrID);
    return;
  }
  }

  SelectCode(N);
}

void AMDGPUDAGToDAGISel::SelectINTRINSIC_WO_CHAIN(SDNode *N) {
  unsigned IntrID = cast<ConstantSDNode>(N->getOperand(0))->getZExtValue();
  unsigned Opcode;
  switch (IntrID) {
  case Intrinsic::amdgcn_wqm:
    Opcode = AMDGPU::WQM;
    break;
  case Intrinsic::amdgcn_softwqm:
    Opcode = AMDGPU::SOFT_WQM;
    break;
  case Intrinsic::amdgcn_wwm:
    Opcode = AMDGPU::WWM;
    break;
  case Intrinsic::amdgcn_interp_p1_f16:
    SelectInterpP1F16(N);
    return;
  default:
    SelectCode(N);
    return;
  }

  SDValue Src = N->getOperand(1);
  CurDAG->SelectNodeTo(N, Opcode, N->getVTList(), {Src});
}

void AMDGPUDAGToDAGISel::SelectINTRINSIC_VOID(SDNode *N) {
  unsigned IntrID = cast<ConstantSDNode>(N->getOperand(1))->getZExtValue();
  switch (IntrID) {
  case Intrinsic::amdgcn_ds_gws_init:
  case Intrinsic::amdgcn_ds_gws_barrier:
  case Intrinsic::amdgcn_ds_gws_sema_v:
  case Intrinsic::amdgcn_ds_gws_sema_br:
  case Intrinsic::amdgcn_ds_gws_sema_p:
  case Intrinsic::amdgcn_ds_gws_sema_release_all:
    SelectDS_GWS(N, IntrID);
    return;
  default:
    break;
  }

  SelectCode(N);
}

bool AMDGPUDAGToDAGISel::SelectVOP3ModsImpl(SDValue In, SDValue &Src,
                                            unsigned &Mods) const {
  Mods = 0;
  Src = In;

  if (Src.getOpcode() == ISD::FNEG) {
    Mods |= SISrcMods::NEG;
    Src = Src.getOperand(0);
  }

  if (Src.getOpcode() == ISD::FABS) {
    Mods |= SISrcMods::ABS;
    Src = Src.getOperand(0);
  }

  return true;
}

bool AMDGPUDAGToDAGISel::SelectVOP3Mods(SDValue In, SDValue &Src,
                                        SDValue &SrcMods) const {
  unsigned Mods;
  if (SelectVOP3ModsImpl(In, Src, Mods)) {
    SrcMods = CurDAG->getTargetConstant(Mods, SDLoc(In), MVT::i32);
    return true;
  }

  return false;
}

bool AMDGPUDAGToDAGISel::SelectVOP3Mods_NNaN(SDValue In, SDValue &Src,
                                             SDValue &SrcMods) const {
  SelectVOP3Mods(In, Src, SrcMods);
  return isNoNanSrc(Src);
}

bool AMDGPUDAGToDAGISel::SelectVOP3NoMods(SDValue In, SDValue &Src) const {
  if (In.getOpcode() == ISD::FABS || In.getOpcode() == ISD::FNEG)
    return false;

  Src = In;
  return true;
}

bool AMDGPUDAGToDAGISel::SelectVOP3Mods0(SDValue In, SDValue &Src,
                                         SDValue &SrcMods, SDValue &Clamp,
                                         SDValue &Omod) const {
  SDLoc DL(In);
  Clamp = CurDAG->getTargetConstant(0, DL, MVT::i1);
  Omod = CurDAG->getTargetConstant(0, DL, MVT::i1);

  return SelectVOP3Mods(In, Src, SrcMods);
}

bool AMDGPUDAGToDAGISel::SelectVOP3OMods(SDValue In, SDValue &Src,
                                         SDValue &Clamp, SDValue &Omod) const {
  Src = In;

  SDLoc DL(In);
  Clamp = CurDAG->getTargetConstant(0, DL, MVT::i1);
  Omod = CurDAG->getTargetConstant(0, DL, MVT::i1);

  return true;
}

bool AMDGPUDAGToDAGISel::SelectVOP3PMods(SDValue In, SDValue &Src,
                                         SDValue &SrcMods) const {
  unsigned Mods = 0;
  Src = In;

  if (Src.getOpcode() == ISD::FNEG) {
    Mods ^= (SISrcMods::NEG | SISrcMods::NEG_HI);
    Src = Src.getOperand(0);
  }

  if (Src.getOpcode() == ISD::BUILD_VECTOR) {
    unsigned VecMods = Mods;

    SDValue Lo = stripBitcast(Src.getOperand(0));
    SDValue Hi = stripBitcast(Src.getOperand(1));

    if (Lo.getOpcode() == ISD::FNEG) {
      Lo = stripBitcast(Lo.getOperand(0));
      Mods ^= SISrcMods::NEG;
    }

    if (Hi.getOpcode() == ISD::FNEG) {
      Hi = stripBitcast(Hi.getOperand(0));
      Mods ^= SISrcMods::NEG_HI;
    }

    if (isExtractHiElt(Lo, Lo))
      Mods |= SISrcMods::OP_SEL_0;

    if (isExtractHiElt(Hi, Hi))
      Mods |= SISrcMods::OP_SEL_1;

    Lo = stripExtractLoElt(Lo);
    Hi = stripExtractLoElt(Hi);

    if (Lo == Hi && !isInlineImmediate(Lo.getNode())) {
      // Really a scalar input. Just select from the low half of the register to
      // avoid packing.

      Src = Lo;
      SrcMods = CurDAG->getTargetConstant(Mods, SDLoc(In), MVT::i32);
      return true;
    }

    Mods = VecMods;
  }

  // Packed instructions do not have abs modifiers.
  Mods |= SISrcMods::OP_SEL_1;

  SrcMods = CurDAG->getTargetConstant(Mods, SDLoc(In), MVT::i32);
  return true;
}

bool AMDGPUDAGToDAGISel::SelectVOP3OpSel(SDValue In, SDValue &Src,
                                         SDValue &SrcMods) const {
  Src = In;
  // FIXME: Handle op_sel
  SrcMods = CurDAG->getTargetConstant(0, SDLoc(In), MVT::i32);
  return true;
}

bool AMDGPUDAGToDAGISel::SelectVOP3OpSelMods(SDValue In, SDValue &Src,
                                             SDValue &SrcMods) const {
  // FIXME: Handle op_sel
  return SelectVOP3Mods(In, Src, SrcMods);
}

// The return value is not whether the match is possible (which it always is),
// but whether or not it a conversion is really used.
bool AMDGPUDAGToDAGISel::SelectVOP3PMadMixModsImpl(SDValue In, SDValue &Src,
                                                   unsigned &Mods) const {
  Mods = 0;
  SelectVOP3ModsImpl(In, Src, Mods);

  if (Src.getOpcode() == ISD::FP_EXTEND) {
    Src = Src.getOperand(0);
    assert(Src.getValueType() == MVT::f16);
    Src = stripBitcast(Src);

    // Be careful about folding modifiers if we already have an abs. fneg is
    // applied last, so we don't want to apply an earlier fneg.
    if ((Mods & SISrcMods::ABS) == 0) {
      unsigned ModsTmp;
      SelectVOP3ModsImpl(Src, Src, ModsTmp);

      if ((ModsTmp & SISrcMods::NEG) != 0)
        Mods ^= SISrcMods::NEG;

      if ((ModsTmp & SISrcMods::ABS) != 0)
        Mods |= SISrcMods::ABS;
    }

    // op_sel/op_sel_hi decide the source type and source.
    // If the source's op_sel_hi is set, it indicates to do a conversion from fp16.
    // If the sources's op_sel is set, it picks the high half of the source
    // register.

    Mods |= SISrcMods::OP_SEL_1;
    if (isExtractHiElt(Src, Src)) {
      Mods |= SISrcMods::OP_SEL_0;

      // TODO: Should we try to look for neg/abs here?
    }

    return true;
  }

  return false;
}

bool AMDGPUDAGToDAGISel::SelectVOP3PMadMixMods(SDValue In, SDValue &Src,
                                               SDValue &SrcMods) const {
  unsigned Mods = 0;
  SelectVOP3PMadMixModsImpl(In, Src, Mods);
  SrcMods = CurDAG->getTargetConstant(Mods, SDLoc(In), MVT::i32);
  return true;
}

SDValue AMDGPUDAGToDAGISel::getHi16Elt(SDValue In) const {
  if (In.isUndef())
    return CurDAG->getUNDEF(MVT::i32);

  if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(In)) {
    SDLoc SL(In);
    return CurDAG->getConstant(C->getZExtValue() << 16, SL, MVT::i32);
  }

  if (ConstantFPSDNode *C = dyn_cast<ConstantFPSDNode>(In)) {
    SDLoc SL(In);
    return CurDAG->getConstant(
      C->getValueAPF().bitcastToAPInt().getZExtValue() << 16, SL, MVT::i32);
  }

  SDValue Src;
  if (isExtractHiElt(In, Src))
    return Src;

  return SDValue();
}

bool AMDGPUDAGToDAGISel::isVGPRImm(const SDNode * N) const {
  assert(CurDAG->getTarget().getTargetTriple().getArch() == Triple::amdgcn);

  const SIRegisterInfo *SIRI =
    static_cast<const SIRegisterInfo *>(Subtarget->getRegisterInfo());
  const SIInstrInfo * SII =
    static_cast<const SIInstrInfo *>(Subtarget->getInstrInfo());

  unsigned Limit = 0;
  bool AllUsesAcceptSReg = true;
  for (SDNode::use_iterator U = N->use_begin(), E = SDNode::use_end();
    Limit < 10 && U != E; ++U, ++Limit) {
    const TargetRegisterClass *RC = getOperandRegClass(*U, U.getOperandNo());

    // If the register class is unknown, it could be an unknown
    // register class that needs to be an SGPR, e.g. an inline asm
    // constraint
    if (!RC || SIRI->isSGPRClass(RC))
      return false;

    if (RC != &AMDGPU::VS_32RegClass) {
      AllUsesAcceptSReg = false;
      SDNode * User = *U;
      if (User->isMachineOpcode()) {
        unsigned Opc = User->getMachineOpcode();
        MCInstrDesc Desc = SII->get(Opc);
        if (Desc.isCommutable()) {
          unsigned OpIdx = Desc.getNumDefs() + U.getOperandNo();
          unsigned CommuteIdx1 = TargetInstrInfo::CommuteAnyOperandIndex;
          if (SII->findCommutedOpIndices(Desc, OpIdx, CommuteIdx1)) {
            unsigned CommutedOpNo = CommuteIdx1 - Desc.getNumDefs();
            const TargetRegisterClass *CommutedRC = getOperandRegClass(*U, CommutedOpNo);
            if (CommutedRC == &AMDGPU::VS_32RegClass)
              AllUsesAcceptSReg = true;
          }
        }
      }
      // If "AllUsesAcceptSReg == false" so far we haven't suceeded
      // commuting current user. This means have at least one use
      // that strictly require VGPR. Thus, we will not attempt to commute
      // other user instructions.
      if (!AllUsesAcceptSReg)
        break;
    }
  }
  return !AllUsesAcceptSReg && (Limit < 10);
}

bool AMDGPUDAGToDAGISel::isUniformLoad(const SDNode * N) const {
  auto Ld = cast<LoadSDNode>(N);

  return Ld->getAlignment() >= 4 &&
        (
          (
            (
              Ld->getAddressSpace() == AMDGPUAS::CONSTANT_ADDRESS       ||
              Ld->getAddressSpace() == AMDGPUAS::CONSTANT_ADDRESS_32BIT
            )
            &&
            !N->isDivergent()
          )
          ||
          (
            Subtarget->getScalarizeGlobalBehavior() &&
            Ld->getAddressSpace() == AMDGPUAS::GLOBAL_ADDRESS &&
            Ld->isSimple() &&
            !N->isDivergent() &&
            static_cast<const SITargetLowering *>(
              getTargetLowering())->isMemOpHasNoClobberedMemOperand(N)
          )
        );
}

void AMDGPUDAGToDAGISel::PostprocessISelDAG() {
  const AMDGPUTargetLowering& Lowering =
    *static_cast<const AMDGPUTargetLowering*>(getTargetLowering());
  bool IsModified = false;
  do {
    IsModified = false;

    // Go over all selected nodes and try to fold them a bit more
    SelectionDAG::allnodes_iterator Position = CurDAG->allnodes_begin();
    while (Position != CurDAG->allnodes_end()) {
      SDNode *Node = &*Position++;
      MachineSDNode *MachineNode = dyn_cast<MachineSDNode>(Node);
      if (!MachineNode)
        continue;

      SDNode *ResNode = Lowering.PostISelFolding(MachineNode, *CurDAG);
      if (ResNode != Node) {
        if (ResNode)
          ReplaceUses(Node, ResNode);
        IsModified = true;
      }
    }
    CurDAG->RemoveDeadNodes();
  } while (IsModified);
}

bool R600DAGToDAGISel::runOnMachineFunction(MachineFunction &MF) {
  Subtarget = &MF.getSubtarget<R600Subtarget>();
  return SelectionDAGISel::runOnMachineFunction(MF);
}

bool R600DAGToDAGISel::isConstantLoad(const MemSDNode *N, int CbId) const {
  if (!N->readMem())
    return false;
  if (CbId == -1)
    return N->getAddressSpace() == AMDGPUAS::CONSTANT_ADDRESS ||
           N->getAddressSpace() == AMDGPUAS::CONSTANT_ADDRESS_32BIT;

  return N->getAddressSpace() == AMDGPUAS::CONSTANT_BUFFER_0 + CbId;
}

bool R600DAGToDAGISel::SelectGlobalValueConstantOffset(SDValue Addr,
                                                         SDValue& IntPtr) {
  if (ConstantSDNode *Cst = dyn_cast<ConstantSDNode>(Addr)) {
    IntPtr = CurDAG->getIntPtrConstant(Cst->getZExtValue() / 4, SDLoc(Addr),
                                       true);
    return true;
  }
  return false;
}

bool R600DAGToDAGISel::SelectGlobalValueVariableOffset(SDValue Addr,
    SDValue& BaseReg, SDValue &Offset) {
  if (!isa<ConstantSDNode>(Addr)) {
    BaseReg = Addr;
    Offset = CurDAG->getIntPtrConstant(0, SDLoc(Addr), true);
    return true;
  }
  return false;
}

void R600DAGToDAGISel::Select(SDNode *N) {
  unsigned int Opc = N->getOpcode();
  if (N->isMachineOpcode()) {
    N->setNodeId(-1);
    return;   // Already selected.
  }

  switch (Opc) {
  default: break;
  case AMDGPUISD::BUILD_VERTICAL_VECTOR:
  case ISD::SCALAR_TO_VECTOR:
  case ISD::BUILD_VECTOR: {
    EVT VT = N->getValueType(0);
    unsigned NumVectorElts = VT.getVectorNumElements();
    unsigned RegClassID;
    // BUILD_VECTOR was lowered into an IMPLICIT_DEF + 4 INSERT_SUBREG
    // that adds a 128 bits reg copy when going through TwoAddressInstructions
    // pass. We want to avoid 128 bits copies as much as possible because they
    // can't be bundled by our scheduler.
    switch(NumVectorElts) {
    case 2: RegClassID = R600::R600_Reg64RegClassID; break;
    case 4:
      if (Opc == AMDGPUISD::BUILD_VERTICAL_VECTOR)
        RegClassID = R600::R600_Reg128VerticalRegClassID;
      else
        RegClassID = R600::R600_Reg128RegClassID;
      break;
    default: llvm_unreachable("Do not know how to lower this BUILD_VECTOR");
    }
    SelectBuildVector(N, RegClassID);
    return;
  }
  }

  SelectCode(N);
}

bool R600DAGToDAGISel::SelectADDRIndirect(SDValue Addr, SDValue &Base,
                                          SDValue &Offset) {
  ConstantSDNode *C;
  SDLoc DL(Addr);

  if ((C = dyn_cast<ConstantSDNode>(Addr))) {
    Base = CurDAG->getRegister(R600::INDIRECT_BASE_ADDR, MVT::i32);
    Offset = CurDAG->getTargetConstant(C->getZExtValue(), DL, MVT::i32);
  } else if ((Addr.getOpcode() == AMDGPUISD::DWORDADDR) &&
             (C = dyn_cast<ConstantSDNode>(Addr.getOperand(0)))) {
    Base = CurDAG->getRegister(R600::INDIRECT_BASE_ADDR, MVT::i32);
    Offset = CurDAG->getTargetConstant(C->getZExtValue(), DL, MVT::i32);
  } else if ((Addr.getOpcode() == ISD::ADD || Addr.getOpcode() == ISD::OR) &&
            (C = dyn_cast<ConstantSDNode>(Addr.getOperand(1)))) {
    Base = Addr.getOperand(0);
    Offset = CurDAG->getTargetConstant(C->getZExtValue(), DL, MVT::i32);
  } else {
    Base = Addr;
    Offset = CurDAG->getTargetConstant(0, DL, MVT::i32);
  }

  return true;
}

bool R600DAGToDAGISel::SelectADDRVTX_READ(SDValue Addr, SDValue &Base,
                                          SDValue &Offset) {
  ConstantSDNode *IMMOffset;

  if (Addr.getOpcode() == ISD::ADD
      && (IMMOffset = dyn_cast<ConstantSDNode>(Addr.getOperand(1)))
      && isInt<16>(IMMOffset->getZExtValue())) {

      Base = Addr.getOperand(0);
      Offset = CurDAG->getTargetConstant(IMMOffset->getZExtValue(), SDLoc(Addr),
                                         MVT::i32);
      return true;
  // If the pointer address is constant, we can move it to the offset field.
  } else if ((IMMOffset = dyn_cast<ConstantSDNode>(Addr))
             && isInt<16>(IMMOffset->getZExtValue())) {
    Base = CurDAG->getCopyFromReg(CurDAG->getEntryNode(),
                                  SDLoc(CurDAG->getEntryNode()),
                                  R600::ZERO, MVT::i32);
    Offset = CurDAG->getTargetConstant(IMMOffset->getZExtValue(), SDLoc(Addr),
                                       MVT::i32);
    return true;
  }

  // Default case, no offset
  Base = Addr;
  Offset = CurDAG->getTargetConstant(0, SDLoc(Addr), MVT::i32);
  return true;
}
