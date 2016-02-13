//===-- AMDGPUISelDAGToDAG.cpp - A dag to dag inst selector for AMDGPU ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//==-----------------------------------------------------------------------===//
//
/// \file
/// \brief Defines an instruction selector for the AMDGPU target.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUInstrInfo.h"
#include "AMDGPUIntrinsicInfo.h"
#include "AMDGPUISelLowering.h" // For AMDGPUISD
#include "AMDGPUSubtarget.h"
#include "SIISelLowering.h"
#include "SIMachineFunctionInfo.h"
#include "llvm/CodeGen/FunctionLoweringInfo.h"
#include "llvm/CodeGen/PseudoSourceValue.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/IR/DiagnosticInfo.h"

using namespace llvm;

namespace llvm {
class R600InstrInfo;
}

//===----------------------------------------------------------------------===//
// Instruction Selector Implementation
//===----------------------------------------------------------------------===//

namespace {

static bool isCBranchSCC(const SDNode *N) {
  assert(N->getOpcode() == ISD::BRCOND);
  if (!N->hasOneUse())
    return false;

  SDValue Cond = N->getOperand(1);
  if (Cond.getOpcode() == ISD::CopyToReg)
    Cond = Cond.getOperand(2);
  return Cond.getOpcode() == ISD::SETCC &&
         Cond.getOperand(0).getValueType() == MVT::i32 &&
	 Cond.hasOneUse();
}

/// AMDGPU specific code to select AMDGPU machine instructions for
/// SelectionDAG operations.
class AMDGPUDAGToDAGISel : public SelectionDAGISel {
  // Subtarget - Keep a pointer to the AMDGPU Subtarget around so that we can
  // make the right decision when generating code for different targets.
  const AMDGPUSubtarget *Subtarget;

public:
  AMDGPUDAGToDAGISel(TargetMachine &TM);
  virtual ~AMDGPUDAGToDAGISel();
  bool runOnMachineFunction(MachineFunction &MF) override;
  SDNode *Select(SDNode *N) override;
  const char *getPassName() const override;
  void PreprocessISelDAG() override;
  void PostprocessISelDAG() override;

private:
  bool isInlineImmediate(SDNode *N) const;
  bool FoldOperand(SDValue &Src, SDValue &Sel, SDValue &Neg, SDValue &Abs,
                   const R600InstrInfo *TII);
  bool FoldOperands(unsigned, const R600InstrInfo *, std::vector<SDValue> &);
  bool FoldDotOperands(unsigned, const R600InstrInfo *, std::vector<SDValue> &);

  // Complex pattern selectors
  bool SelectADDRParam(SDValue Addr, SDValue& R1, SDValue& R2);
  bool SelectADDR(SDValue N, SDValue &R1, SDValue &R2);
  bool SelectADDR64(SDValue N, SDValue &R1, SDValue &R2);

  static bool checkType(const Value *ptr, unsigned int addrspace);
  static bool checkPrivateAddress(const MachineMemOperand *Op);

  static bool isGlobalStore(const StoreSDNode *N);
  static bool isFlatStore(const StoreSDNode *N);
  static bool isPrivateStore(const StoreSDNode *N);
  static bool isLocalStore(const StoreSDNode *N);
  static bool isRegionStore(const StoreSDNode *N);

  bool isCPLoad(const LoadSDNode *N) const;
  bool isConstantLoad(const LoadSDNode *N, int cbID) const;
  bool isGlobalLoad(const LoadSDNode *N) const;
  bool isFlatLoad(const LoadSDNode *N) const;
  bool isParamLoad(const LoadSDNode *N) const;
  bool isPrivateLoad(const LoadSDNode *N) const;
  bool isLocalLoad(const LoadSDNode *N) const;
  bool isRegionLoad(const LoadSDNode *N) const;

  bool isUniformBr(const SDNode *N) const;

  SDNode *glueCopyToM0(SDNode *N) const;

  const TargetRegisterClass *getOperandRegClass(SDNode *N, unsigned OpNo) const;
  bool SelectGlobalValueConstantOffset(SDValue Addr, SDValue& IntPtr);
  bool SelectGlobalValueVariableOffset(SDValue Addr, SDValue &BaseReg,
                                       SDValue& Offset);
  bool SelectADDRVTX_READ(SDValue Addr, SDValue &Base, SDValue &Offset);
  bool SelectADDRIndirect(SDValue Addr, SDValue &Base, SDValue &Offset);
  bool isDSOffsetLegal(const SDValue &Base, unsigned Offset,
                       unsigned OffsetBits) const;
  bool SelectDS1Addr1Offset(SDValue Ptr, SDValue &Base, SDValue &Offset) const;
  bool SelectDS64Bit4ByteAligned(SDValue Ptr, SDValue &Base, SDValue &Offset0,
                                 SDValue &Offset1) const;
  bool SelectMUBUF(SDValue Addr, SDValue &SRsrc, SDValue &VAddr,
                   SDValue &SOffset, SDValue &Offset, SDValue &Offen,
                   SDValue &Idxen, SDValue &Addr64, SDValue &GLC, SDValue &SLC,
                   SDValue &TFE) const;
  bool SelectMUBUFAddr64(SDValue Addr, SDValue &SRsrc, SDValue &VAddr,
                         SDValue &SOffset, SDValue &Offset, SDValue &GLC,
                         SDValue &SLC, SDValue &TFE) const;
  bool SelectMUBUFAddr64(SDValue Addr, SDValue &SRsrc,
                         SDValue &VAddr, SDValue &SOffset, SDValue &Offset,
                         SDValue &SLC) const;
  bool SelectMUBUFScratch(SDValue Addr, SDValue &RSrc, SDValue &VAddr,
                          SDValue &SOffset, SDValue &ImmOffset) const;
  bool SelectMUBUFOffset(SDValue Addr, SDValue &SRsrc, SDValue &SOffset,
                         SDValue &Offset, SDValue &GLC, SDValue &SLC,
                         SDValue &TFE) const;
  bool SelectMUBUFOffset(SDValue Addr, SDValue &SRsrc, SDValue &Soffset,
                         SDValue &Offset, SDValue &GLC) const;
  bool SelectSMRDOffset(SDValue ByteOffsetNode, SDValue &Offset,
                        bool &Imm) const;
  bool SelectSMRD(SDValue Addr, SDValue &SBase, SDValue &Offset,
                  bool &Imm) const;
  bool SelectSMRDImm(SDValue Addr, SDValue &SBase, SDValue &Offset) const;
  bool SelectSMRDImm32(SDValue Addr, SDValue &SBase, SDValue &Offset) const;
  bool SelectSMRDSgpr(SDValue Addr, SDValue &SBase, SDValue &Offset) const;
  bool SelectSMRDBufferImm(SDValue Addr, SDValue &Offset) const;
  bool SelectSMRDBufferImm32(SDValue Addr, SDValue &Offset) const;
  bool SelectSMRDBufferSgpr(SDValue Addr, SDValue &Offset) const;
  SDNode *SelectAddrSpaceCast(SDNode *N);
  bool SelectVOP3Mods(SDValue In, SDValue &Src, SDValue &SrcMods) const;
  bool SelectVOP3NoMods(SDValue In, SDValue &Src, SDValue &SrcMods) const;
  bool SelectVOP3Mods0(SDValue In, SDValue &Src, SDValue &SrcMods,
                       SDValue &Clamp, SDValue &Omod) const;
  bool SelectVOP3NoMods0(SDValue In, SDValue &Src, SDValue &SrcMods,
                         SDValue &Clamp, SDValue &Omod) const;

  bool SelectVOP3Mods0Clamp(SDValue In, SDValue &Src, SDValue &SrcMods,
                            SDValue &Omod) const;
  bool SelectVOP3Mods0Clamp0OMod(SDValue In, SDValue &Src, SDValue &SrcMods,
                                 SDValue &Clamp,
                                 SDValue &Omod) const;

  SDNode *SelectADD_SUB_I64(SDNode *N);
  SDNode *SelectDIV_SCALE(SDNode *N);

  SDNode *getS_BFE(unsigned Opcode, SDLoc DL, SDValue Val,
                   uint32_t Offset, uint32_t Width);
  SDNode *SelectS_BFEFromShifts(SDNode *N);
  SDNode *SelectS_BFE(SDNode *N);
  SDNode *SelectBRCOND(SDNode *N);

  // Include the pieces autogenerated from the target description.
#include "AMDGPUGenDAGISel.inc"
};
}  // end anonymous namespace

/// \brief This pass converts a legalized DAG into a AMDGPU-specific
// DAG, ready for instruction scheduling.
FunctionPass *llvm::createAMDGPUISelDag(TargetMachine &TM) {
  return new AMDGPUDAGToDAGISel(TM);
}

AMDGPUDAGToDAGISel::AMDGPUDAGToDAGISel(TargetMachine &TM)
    : SelectionDAGISel(TM) {}

bool AMDGPUDAGToDAGISel::runOnMachineFunction(MachineFunction &MF) {
  Subtarget = &static_cast<const AMDGPUSubtarget &>(MF.getSubtarget());
  return SelectionDAGISel::runOnMachineFunction(MF);
}

AMDGPUDAGToDAGISel::~AMDGPUDAGToDAGISel() {
}

bool AMDGPUDAGToDAGISel::isInlineImmediate(SDNode *N) const {
  const SITargetLowering *TL
      = static_cast<const SITargetLowering *>(getTargetLowering());
  return TL->analyzeImmediate(N) == 0;
}

/// \brief Determine the register class for \p OpNo
/// \returns The register class of the virtual register that will be used for
/// the given operand number \OpNo or NULL if the register class cannot be
/// determined.
const TargetRegisterClass *AMDGPUDAGToDAGISel::getOperandRegClass(SDNode *N,
                                                          unsigned OpNo) const {
  if (!N->isMachineOpcode())
    return nullptr;

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

bool AMDGPUDAGToDAGISel::SelectADDRParam(
  SDValue Addr, SDValue& R1, SDValue& R2) {

  if (Addr.getOpcode() == ISD::FrameIndex) {
    if (FrameIndexSDNode *FIN = dyn_cast<FrameIndexSDNode>(Addr)) {
      R1 = CurDAG->getTargetFrameIndex(FIN->getIndex(), MVT::i32);
      R2 = CurDAG->getTargetConstant(0, SDLoc(Addr), MVT::i32);
    } else {
      R1 = Addr;
      R2 = CurDAG->getTargetConstant(0, SDLoc(Addr), MVT::i32);
    }
  } else if (Addr.getOpcode() == ISD::ADD) {
    R1 = Addr.getOperand(0);
    R2 = Addr.getOperand(1);
  } else {
    R1 = Addr;
    R2 = CurDAG->getTargetConstant(0, SDLoc(Addr), MVT::i32);
  }
  return true;
}

bool AMDGPUDAGToDAGISel::SelectADDR(SDValue Addr, SDValue& R1, SDValue& R2) {
  if (Addr.getOpcode() == ISD::TargetExternalSymbol ||
      Addr.getOpcode() == ISD::TargetGlobalAddress) {
    return false;
  }
  return SelectADDRParam(Addr, R1, R2);
}


bool AMDGPUDAGToDAGISel::SelectADDR64(SDValue Addr, SDValue& R1, SDValue& R2) {
  if (Addr.getOpcode() == ISD::TargetExternalSymbol ||
      Addr.getOpcode() == ISD::TargetGlobalAddress) {
    return false;
  }

  if (Addr.getOpcode() == ISD::FrameIndex) {
    if (FrameIndexSDNode *FIN = dyn_cast<FrameIndexSDNode>(Addr)) {
      R1 = CurDAG->getTargetFrameIndex(FIN->getIndex(), MVT::i64);
      R2 = CurDAG->getTargetConstant(0, SDLoc(Addr), MVT::i64);
    } else {
      R1 = Addr;
      R2 = CurDAG->getTargetConstant(0, SDLoc(Addr), MVT::i64);
    }
  } else if (Addr.getOpcode() == ISD::ADD) {
    R1 = Addr.getOperand(0);
    R2 = Addr.getOperand(1);
  } else {
    R1 = Addr;
    R2 = CurDAG->getTargetConstant(0, SDLoc(Addr), MVT::i64);
  }
  return true;
}

SDNode *AMDGPUDAGToDAGISel::glueCopyToM0(SDNode *N) const {
  if (Subtarget->getGeneration() < AMDGPUSubtarget::SOUTHERN_ISLANDS ||
      !checkType(cast<MemSDNode>(N)->getMemOperand()->getValue(),
                 AMDGPUAS::LOCAL_ADDRESS))
    return N;

  const SITargetLowering& Lowering =
      *static_cast<const SITargetLowering*>(getTargetLowering());

  // Write max value to m0 before each load operation

  SDValue M0 = Lowering.copyToM0(*CurDAG, CurDAG->getEntryNode(), SDLoc(N),
                                 CurDAG->getTargetConstant(-1, SDLoc(N), MVT::i32));

  SDValue Glue = M0.getValue(1);

  SmallVector <SDValue, 8> Ops;
  for (unsigned i = 0, e = N->getNumOperands(); i != e; ++i) {
     Ops.push_back(N->getOperand(i));
  }
  Ops.push_back(Glue);
  CurDAG->MorphNodeTo(N, N->getOpcode(), N->getVTList(), Ops);

  return N;
}

static unsigned selectSGPRVectorRegClassID(unsigned NumVectorElts) {
  switch (NumVectorElts) {
  case 1:
    return AMDGPU::SReg_32RegClassID;
  case 2:
    return AMDGPU::SReg_64RegClassID;
  case 4:
    return AMDGPU::SReg_128RegClassID;
  case 8:
    return AMDGPU::SReg_256RegClassID;
  case 16:
    return AMDGPU::SReg_512RegClassID;
  }

  llvm_unreachable("invalid vector size");
}

SDNode *AMDGPUDAGToDAGISel::Select(SDNode *N) {
  unsigned int Opc = N->getOpcode();
  if (N->isMachineOpcode()) {
    N->setNodeId(-1);
    return nullptr;   // Already selected.
  }

  if (isa<AtomicSDNode>(N))
    N = glueCopyToM0(N);

  switch (Opc) {
  default: break;
  // We are selecting i64 ADD here instead of custom lower it during
  // DAG legalization, so we can fold some i64 ADDs used for address
  // calculation into the LOAD and STORE instructions.
  case ISD::ADD:
  case ISD::SUB: {
    if (N->getValueType(0) != MVT::i64 ||
        Subtarget->getGeneration() < AMDGPUSubtarget::SOUTHERN_ISLANDS)
      break;

    return SelectADD_SUB_I64(N);
  }
  case ISD::SCALAR_TO_VECTOR:
  case AMDGPUISD::BUILD_VERTICAL_VECTOR:
  case ISD::BUILD_VECTOR: {
    unsigned RegClassID;
    const AMDGPURegisterInfo *TRI = Subtarget->getRegisterInfo();
    EVT VT = N->getValueType(0);
    unsigned NumVectorElts = VT.getVectorNumElements();
    EVT EltVT = VT.getVectorElementType();
    assert(EltVT.bitsEq(MVT::i32));
    if (Subtarget->getGeneration() >= AMDGPUSubtarget::SOUTHERN_ISLANDS) {
      RegClassID = selectSGPRVectorRegClassID(NumVectorElts);
    } else {
      // BUILD_VECTOR was lowered into an IMPLICIT_DEF + 4 INSERT_SUBREG
      // that adds a 128 bits reg copy when going through TwoAddressInstructions
      // pass. We want to avoid 128 bits copies as much as possible because they
      // can't be bundled by our scheduler.
      switch(NumVectorElts) {
      case 2: RegClassID = AMDGPU::R600_Reg64RegClassID; break;
      case 4:
        if (Opc == AMDGPUISD::BUILD_VERTICAL_VECTOR)
          RegClassID = AMDGPU::R600_Reg128VerticalRegClassID;
        else
          RegClassID = AMDGPU::R600_Reg128RegClassID;
        break;
      default: llvm_unreachable("Do not know how to lower this BUILD_VECTOR");
      }
    }

    SDLoc DL(N);
    SDValue RegClass = CurDAG->getTargetConstant(RegClassID, DL, MVT::i32);

    if (NumVectorElts == 1) {
      return CurDAG->SelectNodeTo(N, AMDGPU::COPY_TO_REGCLASS, EltVT,
                                  N->getOperand(0), RegClass);
    }

    assert(NumVectorElts <= 16 && "Vectors with more than 16 elements not "
                                  "supported yet");
    // 16 = Max Num Vector Elements
    // 2 = 2 REG_SEQUENCE operands per element (value, subreg index)
    // 1 = Vector Register Class
    SmallVector<SDValue, 16 * 2 + 1> RegSeqArgs(NumVectorElts * 2 + 1);

    RegSeqArgs[0] = CurDAG->getTargetConstant(RegClassID, DL, MVT::i32);
    bool IsRegSeq = true;
    unsigned NOps = N->getNumOperands();
    for (unsigned i = 0; i < NOps; i++) {
      // XXX: Why is this here?
      if (isa<RegisterSDNode>(N->getOperand(i))) {
        IsRegSeq = false;
        break;
      }
      RegSeqArgs[1 + (2 * i)] = N->getOperand(i);
      RegSeqArgs[1 + (2 * i) + 1] =
              CurDAG->getTargetConstant(TRI->getSubRegFromChannel(i), DL,
                                        MVT::i32);
    }

    if (NOps != NumVectorElts) {
      // Fill in the missing undef elements if this was a scalar_to_vector.
      assert(Opc == ISD::SCALAR_TO_VECTOR && NOps < NumVectorElts);

      MachineSDNode *ImpDef = CurDAG->getMachineNode(TargetOpcode::IMPLICIT_DEF,
                                                     DL, EltVT);
      for (unsigned i = NOps; i < NumVectorElts; ++i) {
        RegSeqArgs[1 + (2 * i)] = SDValue(ImpDef, 0);
        RegSeqArgs[1 + (2 * i) + 1] =
          CurDAG->getTargetConstant(TRI->getSubRegFromChannel(i), DL, MVT::i32);
      }
    }

    if (!IsRegSeq)
      break;
    return CurDAG->SelectNodeTo(N, AMDGPU::REG_SEQUENCE, N->getVTList(),
                                RegSeqArgs);
  }
  case ISD::BUILD_PAIR: {
    SDValue RC, SubReg0, SubReg1;
    if (Subtarget->getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS) {
      break;
    }
    SDLoc DL(N);
    if (N->getValueType(0) == MVT::i128) {
      RC = CurDAG->getTargetConstant(AMDGPU::SReg_128RegClassID, DL, MVT::i32);
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
    return CurDAG->getMachineNode(TargetOpcode::REG_SEQUENCE,
                                  DL, N->getValueType(0), Ops);
  }

  case ISD::Constant:
  case ISD::ConstantFP: {
    if (Subtarget->getGeneration() < AMDGPUSubtarget::SOUTHERN_ISLANDS ||
        N->getValueType(0).getSizeInBits() != 64 || isInlineImmediate(N))
      break;

    uint64_t Imm;
    if (ConstantFPSDNode *FP = dyn_cast<ConstantFPSDNode>(N))
      Imm = FP->getValueAPF().bitcastToAPInt().getZExtValue();
    else {
      ConstantSDNode *C = cast<ConstantSDNode>(N);
      Imm = C->getZExtValue();
    }

    SDLoc DL(N);
    SDNode *Lo = CurDAG->getMachineNode(AMDGPU::S_MOV_B32, DL, MVT::i32,
                                CurDAG->getConstant(Imm & 0xFFFFFFFF, DL,
                                                    MVT::i32));
    SDNode *Hi = CurDAG->getMachineNode(AMDGPU::S_MOV_B32, DL, MVT::i32,
                                CurDAG->getConstant(Imm >> 32, DL, MVT::i32));
    const SDValue Ops[] = {
      CurDAG->getTargetConstant(AMDGPU::SReg_64RegClassID, DL, MVT::i32),
      SDValue(Lo, 0), CurDAG->getTargetConstant(AMDGPU::sub0, DL, MVT::i32),
      SDValue(Hi, 0), CurDAG->getTargetConstant(AMDGPU::sub1, DL, MVT::i32)
    };

    return CurDAG->getMachineNode(TargetOpcode::REG_SEQUENCE, DL,
                                  N->getValueType(0), Ops);
  }
  case ISD::LOAD:
  case ISD::STORE: {
    N = glueCopyToM0(N);
    break;
  }

  case AMDGPUISD::BFE_I32:
  case AMDGPUISD::BFE_U32: {
    if (Subtarget->getGeneration() < AMDGPUSubtarget::SOUTHERN_ISLANDS)
      break;

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

    return getS_BFE(Signed ? AMDGPU::S_BFE_I32 : AMDGPU::S_BFE_U32, SDLoc(N),
                    N->getOperand(0), OffsetVal, WidthVal);
  }
  case AMDGPUISD::DIV_SCALE: {
    return SelectDIV_SCALE(N);
  }
  case ISD::CopyToReg: {
    const SITargetLowering& Lowering =
      *static_cast<const SITargetLowering*>(getTargetLowering());
    Lowering.legalizeTargetIndependentNode(N, *CurDAG);
    break;
  }
  case ISD::ADDRSPACECAST:
    return SelectAddrSpaceCast(N);
  case ISD::AND:
  case ISD::SRL:
  case ISD::SRA:
    if (N->getValueType(0) != MVT::i32 ||
        Subtarget->getGeneration() < AMDGPUSubtarget::SOUTHERN_ISLANDS)
      break;

    return SelectS_BFE(N);
  case ISD::BRCOND:
    return SelectBRCOND(N);
  }

  return SelectCode(N);
}

bool AMDGPUDAGToDAGISel::checkType(const Value *Ptr, unsigned AS) {
  assert(AS != 0 && "Use checkPrivateAddress instead.");
  if (!Ptr)
    return false;

  return Ptr->getType()->getPointerAddressSpace() == AS;
}

bool AMDGPUDAGToDAGISel::checkPrivateAddress(const MachineMemOperand *Op) {
  if (Op->getPseudoValue())
    return true;

  if (PointerType *PT = dyn_cast<PointerType>(Op->getValue()->getType()))
    return PT->getAddressSpace() == AMDGPUAS::PRIVATE_ADDRESS;

  return false;
}

bool AMDGPUDAGToDAGISel::isGlobalStore(const StoreSDNode *N) {
  return checkType(N->getMemOperand()->getValue(), AMDGPUAS::GLOBAL_ADDRESS);
}

bool AMDGPUDAGToDAGISel::isPrivateStore(const StoreSDNode *N) {
  const Value *MemVal = N->getMemOperand()->getValue();
  return (!checkType(MemVal, AMDGPUAS::LOCAL_ADDRESS) &&
          !checkType(MemVal, AMDGPUAS::GLOBAL_ADDRESS) &&
          !checkType(MemVal, AMDGPUAS::REGION_ADDRESS));
}

bool AMDGPUDAGToDAGISel::isLocalStore(const StoreSDNode *N) {
  return checkType(N->getMemOperand()->getValue(), AMDGPUAS::LOCAL_ADDRESS);
}

bool AMDGPUDAGToDAGISel::isFlatStore(const StoreSDNode *N) {
  return checkType(N->getMemOperand()->getValue(), AMDGPUAS::FLAT_ADDRESS);
}

bool AMDGPUDAGToDAGISel::isRegionStore(const StoreSDNode *N) {
  return checkType(N->getMemOperand()->getValue(), AMDGPUAS::REGION_ADDRESS);
}

bool AMDGPUDAGToDAGISel::isConstantLoad(const LoadSDNode *N, int CbId) const {
  const Value *MemVal = N->getMemOperand()->getValue();
  if (CbId == -1)
    return checkType(MemVal, AMDGPUAS::CONSTANT_ADDRESS);

  return checkType(MemVal, AMDGPUAS::CONSTANT_BUFFER_0 + CbId);
}

bool AMDGPUDAGToDAGISel::isGlobalLoad(const LoadSDNode *N) const {
  if (N->getAddressSpace() == AMDGPUAS::CONSTANT_ADDRESS)
    if (Subtarget->getGeneration() < AMDGPUSubtarget::SOUTHERN_ISLANDS ||
        N->getMemoryVT().bitsLT(MVT::i32))
      return true;

  return checkType(N->getMemOperand()->getValue(), AMDGPUAS::GLOBAL_ADDRESS);
}

bool AMDGPUDAGToDAGISel::isParamLoad(const LoadSDNode *N) const {
  return checkType(N->getMemOperand()->getValue(), AMDGPUAS::PARAM_I_ADDRESS);
}

bool AMDGPUDAGToDAGISel::isLocalLoad(const  LoadSDNode *N) const {
  return checkType(N->getMemOperand()->getValue(), AMDGPUAS::LOCAL_ADDRESS);
}

bool AMDGPUDAGToDAGISel::isFlatLoad(const  LoadSDNode *N) const {
  return checkType(N->getMemOperand()->getValue(), AMDGPUAS::FLAT_ADDRESS);
}

bool AMDGPUDAGToDAGISel::isRegionLoad(const  LoadSDNode *N) const {
  return checkType(N->getMemOperand()->getValue(), AMDGPUAS::REGION_ADDRESS);
}

bool AMDGPUDAGToDAGISel::isCPLoad(const LoadSDNode *N) const {
  MachineMemOperand *MMO = N->getMemOperand();
  if (checkPrivateAddress(N->getMemOperand())) {
    if (MMO) {
      const PseudoSourceValue *PSV = MMO->getPseudoValue();
      if (PSV && PSV->isConstantPool()) {
        return true;
      }
    }
  }
  return false;
}

bool AMDGPUDAGToDAGISel::isPrivateLoad(const LoadSDNode *N) const {
  if (checkPrivateAddress(N->getMemOperand())) {
    // Check to make sure we are not a constant pool load or a constant load
    // that is marked as a private load
    if (isCPLoad(N) || isConstantLoad(N, -1)) {
      return false;
    }
  }

  const Value *MemVal = N->getMemOperand()->getValue();
  if (!checkType(MemVal, AMDGPUAS::LOCAL_ADDRESS) &&
      !checkType(MemVal, AMDGPUAS::GLOBAL_ADDRESS) &&
      !checkType(MemVal, AMDGPUAS::FLAT_ADDRESS) &&
      !checkType(MemVal, AMDGPUAS::REGION_ADDRESS) &&
      !checkType(MemVal, AMDGPUAS::CONSTANT_ADDRESS) &&
      !checkType(MemVal, AMDGPUAS::PARAM_D_ADDRESS) &&
      !checkType(MemVal, AMDGPUAS::PARAM_I_ADDRESS)) {
    return true;
  }
  return false;
}

bool AMDGPUDAGToDAGISel::isUniformBr(const SDNode *N) const {
  const BasicBlock *BB = FuncInfo->MBB->getBasicBlock();
  return BB->getTerminator()->getMetadata("amdgpu.uniform");
}

const char *AMDGPUDAGToDAGISel::getPassName() const {
  return "AMDGPU DAG->DAG Pattern Instruction Selection";
}

//===----------------------------------------------------------------------===//
// Complex Patterns
//===----------------------------------------------------------------------===//

bool AMDGPUDAGToDAGISel::SelectGlobalValueConstantOffset(SDValue Addr,
                                                         SDValue& IntPtr) {
  if (ConstantSDNode *Cst = dyn_cast<ConstantSDNode>(Addr)) {
    IntPtr = CurDAG->getIntPtrConstant(Cst->getZExtValue() / 4, SDLoc(Addr),
                                       true);
    return true;
  }
  return false;
}

bool AMDGPUDAGToDAGISel::SelectGlobalValueVariableOffset(SDValue Addr,
    SDValue& BaseReg, SDValue &Offset) {
  if (!isa<ConstantSDNode>(Addr)) {
    BaseReg = Addr;
    Offset = CurDAG->getIntPtrConstant(0, SDLoc(Addr), true);
    return true;
  }
  return false;
}

bool AMDGPUDAGToDAGISel::SelectADDRVTX_READ(SDValue Addr, SDValue &Base,
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
                                  AMDGPU::ZERO, MVT::i32);
    Offset = CurDAG->getTargetConstant(IMMOffset->getZExtValue(), SDLoc(Addr),
                                       MVT::i32);
    return true;
  }

  // Default case, no offset
  Base = Addr;
  Offset = CurDAG->getTargetConstant(0, SDLoc(Addr), MVT::i32);
  return true;
}

bool AMDGPUDAGToDAGISel::SelectADDRIndirect(SDValue Addr, SDValue &Base,
                                            SDValue &Offset) {
  ConstantSDNode *C;
  SDLoc DL(Addr);

  if ((C = dyn_cast<ConstantSDNode>(Addr))) {
    Base = CurDAG->getRegister(AMDGPU::INDIRECT_BASE_ADDR, MVT::i32);
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

SDNode *AMDGPUDAGToDAGISel::SelectADD_SUB_I64(SDNode *N) {
  SDLoc DL(N);
  SDValue LHS = N->getOperand(0);
  SDValue RHS = N->getOperand(1);

  bool IsAdd = (N->getOpcode() == ISD::ADD);

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
  SDValue AddLoArgs[] = { SDValue(Lo0, 0), SDValue(Lo1, 0) };


  unsigned Opc = IsAdd ? AMDGPU::S_ADD_U32 : AMDGPU::S_SUB_U32;
  unsigned CarryOpc = IsAdd ? AMDGPU::S_ADDC_U32 : AMDGPU::S_SUBB_U32;

  SDNode *AddLo = CurDAG->getMachineNode( Opc, DL, VTList, AddLoArgs);
  SDValue Carry(AddLo, 1);
  SDNode *AddHi
    = CurDAG->getMachineNode(CarryOpc, DL, MVT::i32,
                             SDValue(Hi0, 0), SDValue(Hi1, 0), Carry);

  SDValue Args[5] = {
    CurDAG->getTargetConstant(AMDGPU::SReg_64RegClassID, DL, MVT::i32),
    SDValue(AddLo,0),
    Sub0,
    SDValue(AddHi,0),
    Sub1,
  };
  return CurDAG->SelectNodeTo(N, AMDGPU::REG_SEQUENCE, MVT::i64, Args);
}

// We need to handle this here because tablegen doesn't support matching
// instructions with multiple outputs.
SDNode *AMDGPUDAGToDAGISel::SelectDIV_SCALE(SDNode *N) {
  SDLoc SL(N);
  EVT VT = N->getValueType(0);

  assert(VT == MVT::f32 || VT == MVT::f64);

  unsigned Opc
    = (VT == MVT::f64) ? AMDGPU::V_DIV_SCALE_F64 : AMDGPU::V_DIV_SCALE_F32;

  // src0_modifiers, src0, src1_modifiers, src1, src2_modifiers, src2, clamp,
  // omod
  SDValue Ops[8];

  SelectVOP3Mods0(N->getOperand(0), Ops[1], Ops[0], Ops[6], Ops[7]);
  SelectVOP3Mods(N->getOperand(1), Ops[3], Ops[2]);
  SelectVOP3Mods(N->getOperand(2), Ops[5], Ops[4]);
  return CurDAG->SelectNodeTo(N, Opc, VT, MVT::i1, Ops);
}

bool AMDGPUDAGToDAGISel::isDSOffsetLegal(const SDValue &Base, unsigned Offset,
                                         unsigned OffsetBits) const {
  if ((OffsetBits == 16 && !isUInt<16>(Offset)) ||
      (OffsetBits == 8 && !isUInt<8>(Offset)))
    return false;

  if (Subtarget->getGeneration() >= AMDGPUSubtarget::SEA_ISLANDS ||
      Subtarget->unsafeDSOffsetFoldingEnabled())
    return true;

  // On Southern Islands instruction with a negative base value and an offset
  // don't seem to work.
  return CurDAG->SignBitIsZero(Base);
}

bool AMDGPUDAGToDAGISel::SelectDS1Addr1Offset(SDValue Addr, SDValue &Base,
                                              SDValue &Offset) const {
  if (CurDAG->isBaseWithConstantOffset(Addr)) {
    SDValue N0 = Addr.getOperand(0);
    SDValue N1 = Addr.getOperand(1);
    ConstantSDNode *C1 = cast<ConstantSDNode>(N1);
    if (isDSOffsetLegal(N0, C1->getSExtValue(), 16)) {
      // (add n0, c0)
      Base = N0;
      Offset = N1;
      return true;
    }
  } else if (Addr.getOpcode() == ISD::SUB) {
    // sub C, x -> add (sub 0, x), C
    if (const ConstantSDNode *C = dyn_cast<ConstantSDNode>(Addr.getOperand(0))) {
      int64_t ByteOffset = C->getSExtValue();
      if (isUInt<16>(ByteOffset)) {
        SDLoc DL(Addr);
        SDValue Zero = CurDAG->getTargetConstant(0, DL, MVT::i32);

        // XXX - This is kind of hacky. Create a dummy sub node so we can check
        // the known bits in isDSOffsetLegal. We need to emit the selected node
        // here, so this is thrown away.
        SDValue Sub = CurDAG->getNode(ISD::SUB, DL, MVT::i32,
                                      Zero, Addr.getOperand(1));

        if (isDSOffsetLegal(Sub, ByteOffset, 16)) {
          MachineSDNode *MachineSub
            = CurDAG->getMachineNode(AMDGPU::V_SUB_I32_e32, DL, MVT::i32,
                                     Zero, Addr.getOperand(1));

          Base = SDValue(MachineSub, 0);
          Offset = Addr.getOperand(0);
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
      Offset = Addr;
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
          MachineSDNode *MachineSub
            = CurDAG->getMachineNode(AMDGPU::V_SUB_I32_e32, DL, MVT::i32,
                                     Zero, Addr.getOperand(1));

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

static bool isLegalMUBUFImmOffset(const ConstantSDNode *Imm) {
  return isUInt<12>(Imm->getZExtValue());
}

bool AMDGPUDAGToDAGISel::SelectMUBUF(SDValue Addr, SDValue &Ptr,
                                     SDValue &VAddr, SDValue &SOffset,
                                     SDValue &Offset, SDValue &Offen,
                                     SDValue &Idxen, SDValue &Addr64,
                                     SDValue &GLC, SDValue &SLC,
                                     SDValue &TFE) const {
  // Subtarget prefers to use flat instruction
  if (Subtarget->useFlatForGlobal())
    return false;

  SDLoc DL(Addr);

  GLC = CurDAG->getTargetConstant(0, DL, MVT::i1);
  SLC = CurDAG->getTargetConstant(0, DL, MVT::i1);
  TFE = CurDAG->getTargetConstant(0, DL, MVT::i1);

  Idxen = CurDAG->getTargetConstant(0, DL, MVT::i1);
  Offen = CurDAG->getTargetConstant(0, DL, MVT::i1);
  Addr64 = CurDAG->getTargetConstant(0, DL, MVT::i1);
  SOffset = CurDAG->getTargetConstant(0, DL, MVT::i32);

  if (CurDAG->isBaseWithConstantOffset(Addr)) {
    SDValue N0 = Addr.getOperand(0);
    SDValue N1 = Addr.getOperand(1);
    ConstantSDNode *C1 = cast<ConstantSDNode>(N1);

    if (N0.getOpcode() == ISD::ADD) {
      // (add (add N2, N3), C1) -> addr64
      SDValue N2 = N0.getOperand(0);
      SDValue N3 = N0.getOperand(1);
      Addr64 = CurDAG->getTargetConstant(1, DL, MVT::i1);
      Ptr = N2;
      VAddr = N3;
    } else {

      // (add N0, C1) -> offset
      VAddr = CurDAG->getTargetConstant(0, DL, MVT::i32);
      Ptr = N0;
    }

    if (isLegalMUBUFImmOffset(C1)) {
        Offset = CurDAG->getTargetConstant(C1->getZExtValue(), DL, MVT::i16);
        return true;
    } else if (isUInt<32>(C1->getZExtValue())) {
      // Illegal offset, store it in soffset.
      Offset = CurDAG->getTargetConstant(0, DL, MVT::i16);
      SOffset = SDValue(CurDAG->getMachineNode(AMDGPU::S_MOV_B32, DL, MVT::i32,
                   CurDAG->getTargetConstant(C1->getZExtValue(), DL, MVT::i32)),
                        0);
      return true;
    }
  }

  if (Addr.getOpcode() == ISD::ADD) {
    // (add N0, N1) -> addr64
    SDValue N0 = Addr.getOperand(0);
    SDValue N1 = Addr.getOperand(1);
    Addr64 = CurDAG->getTargetConstant(1, DL, MVT::i1);
    Ptr = N0;
    VAddr = N1;
    Offset = CurDAG->getTargetConstant(0, DL, MVT::i16);
    return true;
  }

  // default case -> offset
  VAddr = CurDAG->getTargetConstant(0, DL, MVT::i32);
  Ptr = Addr;
  Offset = CurDAG->getTargetConstant(0, DL, MVT::i16);

  return true;
}

bool AMDGPUDAGToDAGISel::SelectMUBUFAddr64(SDValue Addr, SDValue &SRsrc,
                                           SDValue &VAddr, SDValue &SOffset,
                                           SDValue &Offset, SDValue &GLC,
                                           SDValue &SLC, SDValue &TFE) const {
  SDValue Ptr, Offen, Idxen, Addr64;

  // addr64 bit was removed for volcanic islands.
  if (Subtarget->getGeneration() >= AMDGPUSubtarget::VOLCANIC_ISLANDS)
    return false;

  if (!SelectMUBUF(Addr, Ptr, VAddr, SOffset, Offset, Offen, Idxen, Addr64,
              GLC, SLC, TFE))
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
  SDValue GLC, TFE;

  return SelectMUBUFAddr64(Addr, SRsrc, VAddr, SOffset, Offset, GLC, SLC, TFE);
}

bool AMDGPUDAGToDAGISel::SelectMUBUFScratch(SDValue Addr, SDValue &Rsrc,
                                            SDValue &VAddr, SDValue &SOffset,
                                            SDValue &ImmOffset) const {

  SDLoc DL(Addr);
  MachineFunction &MF = CurDAG->getMachineFunction();
  const SIMachineFunctionInfo *Info = MF.getInfo<SIMachineFunctionInfo>();

  Rsrc = CurDAG->getRegister(Info->getScratchRSrcReg(), MVT::v4i32);
  SOffset = CurDAG->getRegister(Info->getScratchWaveOffsetReg(), MVT::i32);

  // (add n0, c1)
  if (CurDAG->isBaseWithConstantOffset(Addr)) {
    SDValue N0 = Addr.getOperand(0);
    SDValue N1 = Addr.getOperand(1);
    // Offsets in vaddr must be positive.
    if (CurDAG->SignBitIsZero(N0)) {
      ConstantSDNode *C1 = cast<ConstantSDNode>(N1);
      if (isLegalMUBUFImmOffset(C1)) {
        VAddr = N0;
        ImmOffset = CurDAG->getTargetConstant(C1->getZExtValue(), DL, MVT::i16);
        return true;
      }
    }
  }

  // (node)
  VAddr = Addr;
  ImmOffset = CurDAG->getTargetConstant(0, DL, MVT::i16);
  return true;
}

bool AMDGPUDAGToDAGISel::SelectMUBUFOffset(SDValue Addr, SDValue &SRsrc,
                                           SDValue &SOffset, SDValue &Offset,
                                           SDValue &GLC, SDValue &SLC,
                                           SDValue &TFE) const {
  SDValue Ptr, VAddr, Offen, Idxen, Addr64;
  const SIInstrInfo *TII =
    static_cast<const SIInstrInfo *>(Subtarget->getInstrInfo());

  if (!SelectMUBUF(Addr, Ptr, VAddr, SOffset, Offset, Offen, Idxen, Addr64,
              GLC, SLC, TFE))
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
                                           SDValue &Soffset, SDValue &Offset,
                                           SDValue &GLC) const {
  SDValue SLC, TFE;

  return SelectMUBUFOffset(Addr, SRsrc, Soffset, Offset, GLC, SLC, TFE);
}

///
/// \param EncodedOffset This is the immediate value that will be encoded
///        directly into the instruction.  On SI/CI the \p EncodedOffset
///        will be in units of dwords and on VI+ it will be units of bytes.
static bool isLegalSMRDImmOffset(const AMDGPUSubtarget *ST,
                                 int64_t EncodedOffset) {
  return ST->getGeneration() < AMDGPUSubtarget::VOLCANIC_ISLANDS ?
     isUInt<8>(EncodedOffset) : isUInt<20>(EncodedOffset);
}

bool AMDGPUDAGToDAGISel::SelectSMRDOffset(SDValue ByteOffsetNode,
                                          SDValue &Offset, bool &Imm) const {

  // FIXME: Handle non-constant offsets.
  ConstantSDNode *C = dyn_cast<ConstantSDNode>(ByteOffsetNode);
  if (!C)
    return false;

  SDLoc SL(ByteOffsetNode);
  AMDGPUSubtarget::Generation Gen = Subtarget->getGeneration();
  int64_t ByteOffset = C->getSExtValue();
  int64_t EncodedOffset = Gen < AMDGPUSubtarget::VOLCANIC_ISLANDS ?
      ByteOffset >> 2 : ByteOffset;

  if (isLegalSMRDImmOffset(Subtarget, EncodedOffset)) {
    Offset = CurDAG->getTargetConstant(EncodedOffset, SL, MVT::i32);
    Imm = true;
    return true;
  }

  if (!isUInt<32>(EncodedOffset) || !isUInt<32>(ByteOffset))
    return false;

  if (Gen == AMDGPUSubtarget::SEA_ISLANDS && isUInt<32>(EncodedOffset)) {
    // 32-bit Immediates are supported on Sea Islands.
    Offset = CurDAG->getTargetConstant(EncodedOffset, SL, MVT::i32);
  } else {
    SDValue C32Bit = CurDAG->getTargetConstant(ByteOffset, SL, MVT::i32);
    Offset = SDValue(CurDAG->getMachineNode(AMDGPU::S_MOV_B32, SL, MVT::i32,
                                            C32Bit), 0);
  }
  Imm = false;
  return true;
}

bool AMDGPUDAGToDAGISel::SelectSMRD(SDValue Addr, SDValue &SBase,
                                     SDValue &Offset, bool &Imm) const {

  SDLoc SL(Addr);
  if (CurDAG->isBaseWithConstantOffset(Addr)) {
    SDValue N0 = Addr.getOperand(0);
    SDValue N1 = Addr.getOperand(1);

    if (SelectSMRDOffset(N1, Offset, Imm)) {
      SBase = N0;
      return true;
    }
  }
  SBase = Addr;
  Offset = CurDAG->getTargetConstant(0, SL, MVT::i32);
  Imm = true;
  return true;
}

bool AMDGPUDAGToDAGISel::SelectSMRDImm(SDValue Addr, SDValue &SBase,
                                       SDValue &Offset) const {
  bool Imm;
  return SelectSMRD(Addr, SBase, Offset, Imm) && Imm;
}

bool AMDGPUDAGToDAGISel::SelectSMRDImm32(SDValue Addr, SDValue &SBase,
                                         SDValue &Offset) const {

  if (Subtarget->getGeneration() != AMDGPUSubtarget::SEA_ISLANDS)
    return false;

  bool Imm;
  if (!SelectSMRD(Addr, SBase, Offset, Imm))
    return false;

  return !Imm && isa<ConstantSDNode>(Offset);
}

bool AMDGPUDAGToDAGISel::SelectSMRDSgpr(SDValue Addr, SDValue &SBase,
                                        SDValue &Offset) const {
  bool Imm;
  return SelectSMRD(Addr, SBase, Offset, Imm) && !Imm &&
         !isa<ConstantSDNode>(Offset);
}

bool AMDGPUDAGToDAGISel::SelectSMRDBufferImm(SDValue Addr,
                                             SDValue &Offset) const {
  bool Imm;
  return SelectSMRDOffset(Addr, Offset, Imm) && Imm;
}

bool AMDGPUDAGToDAGISel::SelectSMRDBufferImm32(SDValue Addr,
                                               SDValue &Offset) const {
  if (Subtarget->getGeneration() != AMDGPUSubtarget::SEA_ISLANDS)
    return false;

  bool Imm;
  if (!SelectSMRDOffset(Addr, Offset, Imm))
    return false;

  return !Imm && isa<ConstantSDNode>(Offset);
}

bool AMDGPUDAGToDAGISel::SelectSMRDBufferSgpr(SDValue Addr,
                                              SDValue &Offset) const {
  bool Imm;
  return SelectSMRDOffset(Addr, Offset, Imm) && !Imm &&
         !isa<ConstantSDNode>(Offset);
}

// FIXME: This is incorrect and only enough to be able to compile.
SDNode *AMDGPUDAGToDAGISel::SelectAddrSpaceCast(SDNode *N) {
  AddrSpaceCastSDNode *ASC = cast<AddrSpaceCastSDNode>(N);
  SDLoc DL(N);

  const MachineFunction &MF = CurDAG->getMachineFunction();
  DiagnosticInfoUnsupported NotImplemented(
      *MF.getFunction(), "addrspacecast not implemented", DL.getDebugLoc());
  CurDAG->getContext()->diagnose(NotImplemented);

  assert(Subtarget->hasFlatAddressSpace() &&
         "addrspacecast only supported with flat address space!");

  assert((ASC->getSrcAddressSpace() == AMDGPUAS::FLAT_ADDRESS ||
          ASC->getDestAddressSpace() == AMDGPUAS::FLAT_ADDRESS) &&
         "Can only cast to / from flat address space!");

  // The flat instructions read the address as the index of the VGPR holding the
  // address, so casting should just be reinterpreting the base VGPR, so just
  // insert trunc / bitcast / zext.

  SDValue Src = ASC->getOperand(0);
  EVT DestVT = ASC->getValueType(0);
  EVT SrcVT = Src.getValueType();

  unsigned SrcSize = SrcVT.getSizeInBits();
  unsigned DestSize = DestVT.getSizeInBits();

  if (SrcSize > DestSize) {
    assert(SrcSize == 64 && DestSize == 32);
    return CurDAG->getMachineNode(
      TargetOpcode::EXTRACT_SUBREG,
      DL,
      DestVT,
      Src,
      CurDAG->getTargetConstant(AMDGPU::sub0, DL, MVT::i32));
  }

  if (DestSize > SrcSize) {
    assert(SrcSize == 32 && DestSize == 64);

    // FIXME: This is probably wrong, we should never be defining
    // a register class with both VGPRs and SGPRs
    SDValue RC = CurDAG->getTargetConstant(AMDGPU::VS_64RegClassID, DL,
                                           MVT::i32);

    const SDValue Ops[] = {
      RC,
      Src,
      CurDAG->getTargetConstant(AMDGPU::sub0, DL, MVT::i32),
      SDValue(CurDAG->getMachineNode(AMDGPU::S_MOV_B32, DL, MVT::i32,
                                     CurDAG->getConstant(0, DL, MVT::i32)), 0),
      CurDAG->getTargetConstant(AMDGPU::sub1, DL, MVT::i32)
    };

    return CurDAG->getMachineNode(TargetOpcode::REG_SEQUENCE,
                                  DL, N->getValueType(0), Ops);
  }

  assert(SrcSize == 64 && DestSize == 64);
  return CurDAG->getNode(ISD::BITCAST, DL, DestVT, Src).getNode();
}

SDNode *AMDGPUDAGToDAGISel::getS_BFE(unsigned Opcode, SDLoc DL, SDValue Val,
                                     uint32_t Offset, uint32_t Width) {
  // Transformation function, pack the offset and width of a BFE into
  // the format expected by the S_BFE_I32 / S_BFE_U32. In the second
  // source, bits [5:0] contain the offset and bits [22:16] the width.
  uint32_t PackedVal = Offset | (Width << 16);
  SDValue PackedConst = CurDAG->getTargetConstant(PackedVal, DL, MVT::i32);

  return CurDAG->getMachineNode(Opcode, DL, MVT::i32, Val, PackedConst);
}

SDNode *AMDGPUDAGToDAGISel::SelectS_BFEFromShifts(SDNode *N) {
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

      return getS_BFE(Opcode, SDLoc(N), Shl.getOperand(0),
                      CVal - BVal, 32 - CVal);
    }
  }
  return SelectCode(N);
}

SDNode *AMDGPUDAGToDAGISel::SelectS_BFE(SDNode *N) {
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

          return getS_BFE(AMDGPU::S_BFE_U32, SDLoc(N), Srl.getOperand(0),
                          ShiftVal, WidthVal);
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

          return getS_BFE(AMDGPU::S_BFE_U32, SDLoc(N), And.getOperand(0),
                          ShiftVal, WidthVal);
        }
      }
    } else if (N->getOperand(0).getOpcode() == ISD::SHL)
      return SelectS_BFEFromShifts(N);
    break;
  case ISD::SRA:
    if (N->getOperand(0).getOpcode() == ISD::SHL)
      return SelectS_BFEFromShifts(N);
    break;
  }

  return SelectCode(N);
}

SDNode *AMDGPUDAGToDAGISel::SelectBRCOND(SDNode *N) {
  SDValue Cond = N->getOperand(1);

  if (isCBranchSCC(N)) {
    // This brcond will use S_CBRANCH_SCC*, so let tablegen handle it.
    return SelectCode(N);
  }

  // The result of VOPC instructions is or'd against ~EXEC before it is
  // written to vcc or another SGPR.  This means that the value '1' is always
  // written to the corresponding bit for results that are masked.  In order
  // to correctly check against vccz, we need to and VCC with the EXEC
  // register in order to clear the value from the masked bits.

  SDLoc SL(N);

  SDNode *MaskedCond =
        CurDAG->getMachineNode(AMDGPU::S_AND_B64, SL, MVT::i1,
                               CurDAG->getRegister(AMDGPU::EXEC, MVT::i1),
                               Cond);
  SDValue VCC = CurDAG->getCopyToReg(N->getOperand(0), SL, AMDGPU::VCC,
                                     SDValue(MaskedCond, 0),
                                     SDValue()); // Passing SDValue() adds a
                                                 // glue output.
  return CurDAG->SelectNodeTo(N, AMDGPU::S_CBRANCH_VCCNZ, MVT::Other,
                              N->getOperand(2), // Basic Block
                              VCC.getValue(0),  // Chain
                              VCC.getValue(1)); // Glue
}

bool AMDGPUDAGToDAGISel::SelectVOP3Mods(SDValue In, SDValue &Src,
                                        SDValue &SrcMods) const {

  unsigned Mods = 0;

  Src = In;

  if (Src.getOpcode() == ISD::FNEG) {
    Mods |= SISrcMods::NEG;
    Src = Src.getOperand(0);
  }

  if (Src.getOpcode() == ISD::FABS) {
    Mods |= SISrcMods::ABS;
    Src = Src.getOperand(0);
  }

  SrcMods = CurDAG->getTargetConstant(Mods, SDLoc(In), MVT::i32);

  return true;
}

bool AMDGPUDAGToDAGISel::SelectVOP3NoMods(SDValue In, SDValue &Src,
                                         SDValue &SrcMods) const {
  bool Res = SelectVOP3Mods(In, Src, SrcMods);
  return Res && cast<ConstantSDNode>(SrcMods)->isNullValue();
}

bool AMDGPUDAGToDAGISel::SelectVOP3Mods0(SDValue In, SDValue &Src,
                                         SDValue &SrcMods, SDValue &Clamp,
                                         SDValue &Omod) const {
  SDLoc DL(In);
  // FIXME: Handle Clamp and Omod
  Clamp = CurDAG->getTargetConstant(0, DL, MVT::i32);
  Omod = CurDAG->getTargetConstant(0, DL, MVT::i32);

  return SelectVOP3Mods(In, Src, SrcMods);
}

bool AMDGPUDAGToDAGISel::SelectVOP3NoMods0(SDValue In, SDValue &Src,
                                           SDValue &SrcMods, SDValue &Clamp,
                                           SDValue &Omod) const {
  bool Res = SelectVOP3Mods0(In, Src, SrcMods, Clamp, Omod);

  return Res && cast<ConstantSDNode>(SrcMods)->isNullValue() &&
                cast<ConstantSDNode>(Clamp)->isNullValue() &&
                cast<ConstantSDNode>(Omod)->isNullValue();
}

bool AMDGPUDAGToDAGISel::SelectVOP3Mods0Clamp(SDValue In, SDValue &Src,
                                              SDValue &SrcMods,
                                              SDValue &Omod) const {
  // FIXME: Handle Omod
  Omod = CurDAG->getTargetConstant(0, SDLoc(In), MVT::i32);

  return SelectVOP3Mods(In, Src, SrcMods);
}

bool AMDGPUDAGToDAGISel::SelectVOP3Mods0Clamp0OMod(SDValue In, SDValue &Src,
                                                   SDValue &SrcMods,
                                                   SDValue &Clamp,
                                                   SDValue &Omod) const {
  Clamp = Omod = CurDAG->getTargetConstant(0, SDLoc(In), MVT::i32);
  return SelectVOP3Mods(In, Src, SrcMods);
}

void AMDGPUDAGToDAGISel::PreprocessISelDAG() {
  bool Modified = false;

  // XXX - Other targets seem to be able to do this without a worklist.
  SmallVector<LoadSDNode *, 8> LoadsToReplace;
  SmallVector<StoreSDNode *, 8> StoresToReplace;

  for (SDNode &Node : CurDAG->allnodes()) {
    if (LoadSDNode *LD = dyn_cast<LoadSDNode>(&Node)) {
      EVT VT = LD->getValueType(0);
      if (VT != MVT::i64 || LD->getExtensionType() != ISD::NON_EXTLOAD)
        continue;

      // To simplify the TableGen patters, we replace all i64 loads with v2i32
      // loads.  Alternatively, we could promote i64 loads to v2i32 during DAG
      // legalization, however, so places (ExpandUnalignedLoad) in the DAG
      // legalizer assume that if i64 is legal, so doing this promotion early
      // can cause problems.
      LoadsToReplace.push_back(LD);
    } else if (StoreSDNode *ST = dyn_cast<StoreSDNode>(&Node)) {
      // Handle i64 stores here for the same reason mentioned above for loads.
      SDValue Value = ST->getValue();
      if (Value.getValueType() != MVT::i64 || ST->isTruncatingStore())
        continue;
      StoresToReplace.push_back(ST);
    }
  }

  for (LoadSDNode *LD : LoadsToReplace) {
    SDLoc SL(LD);

    SDValue NewLoad = CurDAG->getLoad(MVT::v2i32, SL, LD->getChain(),
                                      LD->getBasePtr(), LD->getMemOperand());
    SDValue BitCast = CurDAG->getNode(ISD::BITCAST, SL,
                                      MVT::i64, NewLoad);
    CurDAG->ReplaceAllUsesOfValueWith(SDValue(LD, 1), NewLoad.getValue(1));
    CurDAG->ReplaceAllUsesOfValueWith(SDValue(LD, 0), BitCast);
    Modified = true;
  }

  for (StoreSDNode *ST : StoresToReplace) {
    SDValue NewValue = CurDAG->getNode(ISD::BITCAST, SDLoc(ST),
                                       MVT::v2i32, ST->getValue());
    const SDValue StoreOps[] = {
      ST->getChain(),
      NewValue,
      ST->getBasePtr(),
      ST->getOffset()
    };

    CurDAG->UpdateNodeOperands(ST, StoreOps);
    Modified = true;
  }

  // XXX - Is this necessary?
  if (Modified)
    CurDAG->RemoveDeadNodes();
}

void AMDGPUDAGToDAGISel::PostprocessISelDAG() {
  const AMDGPUTargetLowering& Lowering =
    *static_cast<const AMDGPUTargetLowering*>(getTargetLowering());
  bool IsModified = false;
  do {
    IsModified = false;
    // Go over all selected nodes and try to fold them a bit more
    for (SDNode &Node : CurDAG->allnodes()) {
      MachineSDNode *MachineNode = dyn_cast<MachineSDNode>(&Node);
      if (!MachineNode)
        continue;

      SDNode *ResNode = Lowering.PostISelFolding(MachineNode, *CurDAG);
      if (ResNode != &Node) {
        ReplaceUses(&Node, ResNode);
        IsModified = true;
      }
    }
    CurDAG->RemoveDeadNodes();
  } while (IsModified);
}
