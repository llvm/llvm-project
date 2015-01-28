//===-- AMDILISelDAGToDAG.cpp - A dag to dag inst selector for AMDIL ------===//
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
#include "AMDGPUISelLowering.h" // For AMDGPUISD
#include "AMDGPURegisterInfo.h"
#include "AMDGPUSubtarget.h"
#include "R600InstrInfo.h"
#include "SIDefines.h"
#include "SIISelLowering.h"
#include "SIMachineFunctionInfo.h"
#include "llvm/CodeGen/FunctionLoweringInfo.h"
#include "llvm/CodeGen/PseudoSourceValue.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/IR/Function.h"

using namespace llvm;

//===----------------------------------------------------------------------===//
// Instruction Selector Implementation
//===----------------------------------------------------------------------===//

namespace {
/// AMDGPU specific code to select AMDGPU machine instructions for
/// SelectionDAG operations.
class AMDGPUDAGToDAGISel : public SelectionDAGISel {
  // Subtarget - Keep a pointer to the AMDGPU Subtarget around so that we can
  // make the right decision when generating code for different targets.
  const AMDGPUSubtarget &Subtarget;
public:
  AMDGPUDAGToDAGISel(TargetMachine &TM);
  virtual ~AMDGPUDAGToDAGISel();

  SDNode *Select(SDNode *N) override;
  const char *getPassName() const override;
  void PostprocessISelDAG() override;

private:
  bool isInlineImmediate(SDNode *N) const;
  inline SDValue getSmallIPtrImm(unsigned Imm);
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
  void SelectMUBUF(SDValue Addr, SDValue &SRsrc, SDValue &VAddr,
                   SDValue &SOffset, SDValue &Offset, SDValue &Offen,
                   SDValue &Idxen, SDValue &Addr64, SDValue &GLC, SDValue &SLC,
                   SDValue &TFE) const;
  bool SelectMUBUFAddr64(SDValue Addr, SDValue &SRsrc, SDValue &VAddr,
                         SDValue &Offset) const;
  bool SelectMUBUFAddr64(SDValue Addr, SDValue &SRsrc,
                         SDValue &VAddr, SDValue &Offset,
                         SDValue &SLC) const;
  bool SelectMUBUFScratch(SDValue Addr, SDValue &RSrc, SDValue &VAddr,
                          SDValue &SOffset, SDValue &ImmOffset) const;
  bool SelectMUBUFOffset(SDValue Addr, SDValue &SRsrc, SDValue &SOffset,
                         SDValue &Offset, SDValue &GLC, SDValue &SLC,
                         SDValue &TFE) const;
  bool SelectMUBUFOffset(SDValue Addr, SDValue &SRsrc, SDValue &Soffset,
                         SDValue &Offset, SDValue &GLC) const;
  SDNode *SelectAddrSpaceCast(SDNode *N);
  bool SelectVOP3Mods(SDValue In, SDValue &Src, SDValue &SrcMods) const;
  bool SelectVOP3Mods0(SDValue In, SDValue &Src, SDValue &SrcMods,
                       SDValue &Clamp, SDValue &Omod) const;

  bool SelectVOP3Mods0Clamp(SDValue In, SDValue &Src, SDValue &SrcMods,
                            SDValue &Omod) const;
  bool SelectVOP3Mods0Clamp0OMod(SDValue In, SDValue &Src, SDValue &SrcMods,
                                 SDValue &Clamp,
                                 SDValue &Omod) const;

  SDNode *SelectADD_SUB_I64(SDNode *N);
  SDNode *SelectDIV_SCALE(SDNode *N);

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
  : SelectionDAGISel(TM), Subtarget(TM.getSubtarget<AMDGPUSubtarget>()) {
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
        TM.getSubtargetImpl()->getInstrInfo()->get(N->getMachineOpcode());
    unsigned OpIdx = Desc.getNumDefs() + OpNo;
    if (OpIdx >= Desc.getNumOperands())
      return nullptr;
    int RegClass = Desc.OpInfo[OpIdx].RegClass;
    if (RegClass == -1)
      return nullptr;

    return TM.getSubtargetImpl()->getRegisterInfo()->getRegClass(RegClass);
  }
  case AMDGPU::REG_SEQUENCE: {
    unsigned RCID = cast<ConstantSDNode>(N->getOperand(0))->getZExtValue();
    const TargetRegisterClass *SuperRC =
        TM.getSubtargetImpl()->getRegisterInfo()->getRegClass(RCID);

    SDValue SubRegOp = N->getOperand(OpNo + 1);
    unsigned SubRegIdx = cast<ConstantSDNode>(SubRegOp)->getZExtValue();
    return TM.getSubtargetImpl()->getRegisterInfo()->getSubClassWithSubReg(
        SuperRC, SubRegIdx);
  }
  }
}

SDValue AMDGPUDAGToDAGISel::getSmallIPtrImm(unsigned int Imm) {
  return CurDAG->getTargetConstant(Imm, MVT::i32);
}

bool AMDGPUDAGToDAGISel::SelectADDRParam(
  SDValue Addr, SDValue& R1, SDValue& R2) {

  if (Addr.getOpcode() == ISD::FrameIndex) {
    if (FrameIndexSDNode *FIN = dyn_cast<FrameIndexSDNode>(Addr)) {
      R1 = CurDAG->getTargetFrameIndex(FIN->getIndex(), MVT::i32);
      R2 = CurDAG->getTargetConstant(0, MVT::i32);
    } else {
      R1 = Addr;
      R2 = CurDAG->getTargetConstant(0, MVT::i32);
    }
  } else if (Addr.getOpcode() == ISD::ADD) {
    R1 = Addr.getOperand(0);
    R2 = Addr.getOperand(1);
  } else {
    R1 = Addr;
    R2 = CurDAG->getTargetConstant(0, MVT::i32);
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
      R2 = CurDAG->getTargetConstant(0, MVT::i64);
    } else {
      R1 = Addr;
      R2 = CurDAG->getTargetConstant(0, MVT::i64);
    }
  } else if (Addr.getOpcode() == ISD::ADD) {
    R1 = Addr.getOperand(0);
    R2 = Addr.getOperand(1);
  } else {
    R1 = Addr;
    R2 = CurDAG->getTargetConstant(0, MVT::i64);
  }
  return true;
}

SDNode *AMDGPUDAGToDAGISel::Select(SDNode *N) {
  unsigned int Opc = N->getOpcode();
  if (N->isMachineOpcode()) {
    N->setNodeId(-1);
    return nullptr;   // Already selected.
  }

  const AMDGPUSubtarget &ST = TM.getSubtarget<AMDGPUSubtarget>();
  switch (Opc) {
  default: break;
  // We are selecting i64 ADD here instead of custom lower it during
  // DAG legalization, so we can fold some i64 ADDs used for address
  // calculation into the LOAD and STORE instructions.
  case ISD::ADD:
  case ISD::SUB: {
    if (N->getValueType(0) != MVT::i64 ||
        ST.getGeneration() < AMDGPUSubtarget::SOUTHERN_ISLANDS)
      break;

    return SelectADD_SUB_I64(N);
  }
  case ISD::SCALAR_TO_VECTOR:
  case AMDGPUISD::BUILD_VERTICAL_VECTOR:
  case ISD::BUILD_VECTOR: {
    unsigned RegClassID;
    const AMDGPURegisterInfo *TRI = static_cast<const AMDGPURegisterInfo *>(
        TM.getSubtargetImpl()->getRegisterInfo());
    const SIRegisterInfo *SIRI = static_cast<const SIRegisterInfo *>(
        TM.getSubtargetImpl()->getRegisterInfo());
    EVT VT = N->getValueType(0);
    unsigned NumVectorElts = VT.getVectorNumElements();
    EVT EltVT = VT.getVectorElementType();
    assert(EltVT.bitsEq(MVT::i32));
    if (ST.getGeneration() >= AMDGPUSubtarget::SOUTHERN_ISLANDS) {
      bool UseVReg = true;
      for (SDNode::use_iterator U = N->use_begin(), E = SDNode::use_end();
                                                    U != E; ++U) {
        if (!U->isMachineOpcode()) {
          continue;
        }
        const TargetRegisterClass *RC = getOperandRegClass(*U, U.getOperandNo());
        if (!RC) {
          continue;
        }
        if (SIRI->isSGPRClass(RC)) {
          UseVReg = false;
        }
      }
      switch(NumVectorElts) {
      case 1: RegClassID = UseVReg ? AMDGPU::VGPR_32RegClassID :
                                     AMDGPU::SReg_32RegClassID;
        break;
      case 2: RegClassID = UseVReg ? AMDGPU::VReg_64RegClassID :
                                     AMDGPU::SReg_64RegClassID;
        break;
      case 4: RegClassID = UseVReg ? AMDGPU::VReg_128RegClassID :
                                     AMDGPU::SReg_128RegClassID;
        break;
      case 8: RegClassID = UseVReg ? AMDGPU::VReg_256RegClassID :
                                     AMDGPU::SReg_256RegClassID;
        break;
      case 16: RegClassID = UseVReg ? AMDGPU::VReg_512RegClassID :
                                      AMDGPU::SReg_512RegClassID;
        break;
      default: llvm_unreachable("Do not know how to lower this BUILD_VECTOR");
      }
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

    SDValue RegClass = CurDAG->getTargetConstant(RegClassID, MVT::i32);

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

    RegSeqArgs[0] = CurDAG->getTargetConstant(RegClassID, MVT::i32);
    bool IsRegSeq = true;
    unsigned NOps = N->getNumOperands();
    for (unsigned i = 0; i < NOps; i++) {
      // XXX: Why is this here?
      if (dyn_cast<RegisterSDNode>(N->getOperand(i))) {
        IsRegSeq = false;
        break;
      }
      RegSeqArgs[1 + (2 * i)] = N->getOperand(i);
      RegSeqArgs[1 + (2 * i) + 1] =
              CurDAG->getTargetConstant(TRI->getSubRegFromChannel(i), MVT::i32);
    }

    if (NOps != NumVectorElts) {
      // Fill in the missing undef elements if this was a scalar_to_vector.
      assert(Opc == ISD::SCALAR_TO_VECTOR && NOps < NumVectorElts);

      MachineSDNode *ImpDef = CurDAG->getMachineNode(TargetOpcode::IMPLICIT_DEF,
                                                     SDLoc(N), EltVT);
      for (unsigned i = NOps; i < NumVectorElts; ++i) {
        RegSeqArgs[1 + (2 * i)] = SDValue(ImpDef, 0);
        RegSeqArgs[1 + (2 * i) + 1] =
          CurDAG->getTargetConstant(TRI->getSubRegFromChannel(i), MVT::i32);
      }
    }

    if (!IsRegSeq)
      break;
    return CurDAG->SelectNodeTo(N, AMDGPU::REG_SEQUENCE, N->getVTList(),
                                RegSeqArgs);
  }
  case ISD::BUILD_PAIR: {
    SDValue RC, SubReg0, SubReg1;
    if (ST.getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS) {
      break;
    }
    if (N->getValueType(0) == MVT::i128) {
      RC = CurDAG->getTargetConstant(AMDGPU::SReg_128RegClassID, MVT::i32);
      SubReg0 = CurDAG->getTargetConstant(AMDGPU::sub0_sub1, MVT::i32);
      SubReg1 = CurDAG->getTargetConstant(AMDGPU::sub2_sub3, MVT::i32);
    } else if (N->getValueType(0) == MVT::i64) {
      RC = CurDAG->getTargetConstant(AMDGPU::SReg_64RegClassID, MVT::i32);
      SubReg0 = CurDAG->getTargetConstant(AMDGPU::sub0, MVT::i32);
      SubReg1 = CurDAG->getTargetConstant(AMDGPU::sub1, MVT::i32);
    } else {
      llvm_unreachable("Unhandled value type for BUILD_PAIR");
    }
    const SDValue Ops[] = { RC, N->getOperand(0), SubReg0,
                            N->getOperand(1), SubReg1 };
    return CurDAG->getMachineNode(TargetOpcode::REG_SEQUENCE,
                                  SDLoc(N), N->getValueType(0), Ops);
  }

  case ISD::Constant:
  case ISD::ConstantFP: {
    const AMDGPUSubtarget &ST = TM.getSubtarget<AMDGPUSubtarget>();
    if (ST.getGeneration() < AMDGPUSubtarget::SOUTHERN_ISLANDS ||
        N->getValueType(0).getSizeInBits() != 64 || isInlineImmediate(N))
      break;

    uint64_t Imm;
    if (ConstantFPSDNode *FP = dyn_cast<ConstantFPSDNode>(N))
      Imm = FP->getValueAPF().bitcastToAPInt().getZExtValue();
    else {
      ConstantSDNode *C = cast<ConstantSDNode>(N);
      Imm = C->getZExtValue();
    }

    SDNode *Lo = CurDAG->getMachineNode(AMDGPU::S_MOV_B32, SDLoc(N), MVT::i32,
                                CurDAG->getConstant(Imm & 0xFFFFFFFF, MVT::i32));
    SDNode *Hi = CurDAG->getMachineNode(AMDGPU::S_MOV_B32, SDLoc(N), MVT::i32,
                                CurDAG->getConstant(Imm >> 32, MVT::i32));
    const SDValue Ops[] = {
      CurDAG->getTargetConstant(AMDGPU::SReg_64RegClassID, MVT::i32),
      SDValue(Lo, 0), CurDAG->getTargetConstant(AMDGPU::sub0, MVT::i32),
      SDValue(Hi, 0), CurDAG->getTargetConstant(AMDGPU::sub1, MVT::i32)
    };

    return CurDAG->getMachineNode(TargetOpcode::REG_SEQUENCE, SDLoc(N),
                                  N->getValueType(0), Ops);
  }

  case ISD::LOAD: {
    // To simplify the TableGen patters, we replace all i64 loads with
    // v2i32 loads.  Alternatively, we could promote i64 loads to v2i32
    // during DAG legalization, however, so places (ExpandUnalignedLoad)
    // in the DAG legalizer assume that if i64 is legal, so doing this
    // promotion early can cause problems.
    EVT VT = N->getValueType(0);
    LoadSDNode *LD = cast<LoadSDNode>(N);
    if (VT != MVT::i64 || LD->getExtensionType() != ISD::NON_EXTLOAD)
      break;

    SDValue NewLoad = CurDAG->getLoad(MVT::v2i32, SDLoc(N), LD->getChain(),
                                     LD->getBasePtr(), LD->getMemOperand());
    SDValue BitCast = CurDAG->getNode(ISD::BITCAST, SDLoc(N),
                                      MVT::i64, NewLoad);
    CurDAG->ReplaceAllUsesOfValueWith(SDValue(N, 1), NewLoad.getValue(1));
    CurDAG->ReplaceAllUsesOfValueWith(SDValue(N, 0), BitCast);
    SelectCode(NewLoad.getNode());
    N = BitCast.getNode();
    break;
  }

  case AMDGPUISD::REGISTER_LOAD: {
    if (ST.getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
      break;
    SDValue Addr, Offset;

    SelectADDRIndirect(N->getOperand(1), Addr, Offset);
    const SDValue Ops[] = {
      Addr,
      Offset,
      CurDAG->getTargetConstant(0, MVT::i32),
      N->getOperand(0),
    };
    return CurDAG->getMachineNode(AMDGPU::SI_RegisterLoad, SDLoc(N),
                                  CurDAG->getVTList(MVT::i32, MVT::i64, MVT::Other),
                                  Ops);
  }
  case AMDGPUISD::REGISTER_STORE: {
    if (ST.getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS)
      break;
    SDValue Addr, Offset;
    SelectADDRIndirect(N->getOperand(2), Addr, Offset);
    const SDValue Ops[] = {
      N->getOperand(1),
      Addr,
      Offset,
      CurDAG->getTargetConstant(0, MVT::i32),
      N->getOperand(0),
    };
    return CurDAG->getMachineNode(AMDGPU::SI_RegisterStorePseudo, SDLoc(N),
                                        CurDAG->getVTList(MVT::Other),
                                        Ops);
  }

  case AMDGPUISD::BFE_I32:
  case AMDGPUISD::BFE_U32: {
    if (ST.getGeneration() < AMDGPUSubtarget::SOUTHERN_ISLANDS)
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

    // Transformation function, pack the offset and width of a BFE into
    // the format expected by the S_BFE_I32 / S_BFE_U32. In the second
    // source, bits [5:0] contain the offset and bits [22:16] the width.

    uint32_t OffsetVal = Offset->getZExtValue();
    uint32_t WidthVal = Width->getZExtValue();

    uint32_t PackedVal = OffsetVal | WidthVal << 16;

    SDValue PackedOffsetWidth = CurDAG->getTargetConstant(PackedVal, MVT::i32);
    return CurDAG->getMachineNode(Signed ? AMDGPU::S_BFE_I32 : AMDGPU::S_BFE_U32,
                                  SDLoc(N),
                                  MVT::i32,
                                  N->getOperand(0),
                                  PackedOffsetWidth);

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
  if (N->getAddressSpace() == AMDGPUAS::CONSTANT_ADDRESS) {
    const AMDGPUSubtarget &ST = TM.getSubtarget<AMDGPUSubtarget>();
    if (ST.getGeneration() < AMDGPUSubtarget::SOUTHERN_ISLANDS ||
        N->getMemoryVT().bitsLT(MVT::i32)) {
      return true;
    }
  }
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
      if (PSV && PSV == PseudoSourceValue::getConstantPool()) {
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

const char *AMDGPUDAGToDAGISel::getPassName() const {
  return "AMDGPU DAG->DAG Pattern Instruction Selection";
}

#ifdef DEBUGTMP
#undef INT64_C
#endif
#undef DEBUGTMP

//===----------------------------------------------------------------------===//
// Complex Patterns
//===----------------------------------------------------------------------===//

bool AMDGPUDAGToDAGISel::SelectGlobalValueConstantOffset(SDValue Addr,
                                                         SDValue& IntPtr) {
  if (ConstantSDNode *Cst = dyn_cast<ConstantSDNode>(Addr)) {
    IntPtr = CurDAG->getIntPtrConstant(Cst->getZExtValue() / 4, true);
    return true;
  }
  return false;
}

bool AMDGPUDAGToDAGISel::SelectGlobalValueVariableOffset(SDValue Addr,
    SDValue& BaseReg, SDValue &Offset) {
  if (!isa<ConstantSDNode>(Addr)) {
    BaseReg = Addr;
    Offset = CurDAG->getIntPtrConstant(0, true);
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
      Offset = CurDAG->getTargetConstant(IMMOffset->getZExtValue(), MVT::i32);
      return true;
  // If the pointer address is constant, we can move it to the offset field.
  } else if ((IMMOffset = dyn_cast<ConstantSDNode>(Addr))
             && isInt<16>(IMMOffset->getZExtValue())) {
    Base = CurDAG->getCopyFromReg(CurDAG->getEntryNode(),
                                  SDLoc(CurDAG->getEntryNode()),
                                  AMDGPU::ZERO, MVT::i32);
    Offset = CurDAG->getTargetConstant(IMMOffset->getZExtValue(), MVT::i32);
    return true;
  }

  // Default case, no offset
  Base = Addr;
  Offset = CurDAG->getTargetConstant(0, MVT::i32);
  return true;
}

bool AMDGPUDAGToDAGISel::SelectADDRIndirect(SDValue Addr, SDValue &Base,
                                            SDValue &Offset) {
  ConstantSDNode *C;

  if ((C = dyn_cast<ConstantSDNode>(Addr))) {
    Base = CurDAG->getRegister(AMDGPU::INDIRECT_BASE_ADDR, MVT::i32);
    Offset = CurDAG->getTargetConstant(C->getZExtValue(), MVT::i32);
  } else if ((Addr.getOpcode() == ISD::ADD || Addr.getOpcode() == ISD::OR) &&
            (C = dyn_cast<ConstantSDNode>(Addr.getOperand(1)))) {
    Base = Addr.getOperand(0);
    Offset = CurDAG->getTargetConstant(C->getZExtValue(), MVT::i32);
  } else {
    Base = Addr;
    Offset = CurDAG->getTargetConstant(0, MVT::i32);
  }

  return true;
}

SDNode *AMDGPUDAGToDAGISel::SelectADD_SUB_I64(SDNode *N) {
  SDLoc DL(N);
  SDValue LHS = N->getOperand(0);
  SDValue RHS = N->getOperand(1);

  bool IsAdd = (N->getOpcode() == ISD::ADD);

  SDValue Sub0 = CurDAG->getTargetConstant(AMDGPU::sub0, MVT::i32);
  SDValue Sub1 = CurDAG->getTargetConstant(AMDGPU::sub1, MVT::i32);

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
    CurDAG->getTargetConstant(AMDGPU::SReg_64RegClassID, MVT::i32),
    SDValue(AddLo,0),
    Sub0,
    SDValue(AddHi,0),
    Sub1,
  };
  return CurDAG->SelectNodeTo(N, AMDGPU::REG_SEQUENCE, MVT::i64, Args);
}

SDNode *AMDGPUDAGToDAGISel::SelectDIV_SCALE(SDNode *N) {
  SDLoc SL(N);
  EVT VT = N->getValueType(0);

  assert(VT == MVT::f32 || VT == MVT::f64);

  unsigned Opc
    = (VT == MVT::f64) ? AMDGPU::V_DIV_SCALE_F64 : AMDGPU::V_DIV_SCALE_F32;

  const SDValue Zero = CurDAG->getTargetConstant(0, MVT::i32);
  const SDValue False = CurDAG->getTargetConstant(0, MVT::i1);
  SDValue Ops[] = {
    Zero,             // src0_modifiers
    N->getOperand(0), // src0
    Zero,             // src1_modifiers
    N->getOperand(1), // src1
    Zero,             // src2_modifiers
    N->getOperand(2), // src2
    False,            // clamp
    Zero              // omod
  };

  return CurDAG->SelectNodeTo(N, Opc, VT, MVT::i1, Ops);
}

bool AMDGPUDAGToDAGISel::isDSOffsetLegal(const SDValue &Base, unsigned Offset,
                                         unsigned OffsetBits) const {
  const AMDGPUSubtarget &ST = TM.getSubtarget<AMDGPUSubtarget>();
  if ((OffsetBits == 16 && !isUInt<16>(Offset)) ||
      (OffsetBits == 8 && !isUInt<8>(Offset)))
    return false;

  if (ST.getGeneration() >= AMDGPUSubtarget::SEA_ISLANDS)
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
  }

  // If we have a constant address, prefer to put the constant into the
  // offset. This can save moves to load the constant address since multiple
  // operations can share the zero base address register, and enables merging
  // into read2 / write2 instructions.
  if (const ConstantSDNode *CAddr = dyn_cast<ConstantSDNode>(Addr)) {
    if (isUInt<16>(CAddr->getZExtValue())) {
      SDValue Zero = CurDAG->getTargetConstant(0, MVT::i32);
      MachineSDNode *MovZero = CurDAG->getMachineNode(AMDGPU::V_MOV_B32_e32,
                                 SDLoc(Addr), MVT::i32, Zero);
      Base = SDValue(MovZero, 0);
      Offset = Addr;
      return true;
    }
  }

  // default case
  Base = Addr;
  Offset = CurDAG->getTargetConstant(0, MVT::i16);
  return true;
}

bool AMDGPUDAGToDAGISel::SelectDS64Bit4ByteAligned(SDValue Addr, SDValue &Base,
                                                   SDValue &Offset0,
                                                   SDValue &Offset1) const {
  if (CurDAG->isBaseWithConstantOffset(Addr)) {
    SDValue N0 = Addr.getOperand(0);
    SDValue N1 = Addr.getOperand(1);
    ConstantSDNode *C1 = cast<ConstantSDNode>(N1);
    unsigned DWordOffset0 = C1->getZExtValue() / 4;
    unsigned DWordOffset1 = DWordOffset0 + 1;
    // (add n0, c0)
    if (isDSOffsetLegal(N0, DWordOffset1, 8)) {
      Base = N0;
      Offset0 = CurDAG->getTargetConstant(DWordOffset0, MVT::i8);
      Offset1 = CurDAG->getTargetConstant(DWordOffset1, MVT::i8);
      return true;
    }
  }

  if (const ConstantSDNode *CAddr = dyn_cast<ConstantSDNode>(Addr)) {
    unsigned DWordOffset0 = CAddr->getZExtValue() / 4;
    unsigned DWordOffset1 = DWordOffset0 + 1;
    assert(4 * DWordOffset0 == CAddr->getZExtValue());

    if (isUInt<8>(DWordOffset0) && isUInt<8>(DWordOffset1)) {
      SDValue Zero = CurDAG->getTargetConstant(0, MVT::i32);
      MachineSDNode *MovZero
        = CurDAG->getMachineNode(AMDGPU::V_MOV_B32_e32,
                                 SDLoc(Addr), MVT::i32, Zero);
      Base = SDValue(MovZero, 0);
      Offset0 = CurDAG->getTargetConstant(DWordOffset0, MVT::i8);
      Offset1 = CurDAG->getTargetConstant(DWordOffset1, MVT::i8);
      return true;
    }
  }

  // default case
  Base = Addr;
  Offset0 = CurDAG->getTargetConstant(0, MVT::i8);
  Offset1 = CurDAG->getTargetConstant(1, MVT::i8);
  return true;
}

static bool isLegalMUBUFImmOffset(const ConstantSDNode *Imm) {
  return isUInt<12>(Imm->getZExtValue());
}

void AMDGPUDAGToDAGISel::SelectMUBUF(SDValue Addr, SDValue &Ptr,
                                     SDValue &VAddr, SDValue &SOffset,
                                     SDValue &Offset, SDValue &Offen,
                                     SDValue &Idxen, SDValue &Addr64,
                                     SDValue &GLC, SDValue &SLC,
                                     SDValue &TFE) const {
  SDLoc DL(Addr);

  GLC = CurDAG->getTargetConstant(0, MVT::i1);
  SLC = CurDAG->getTargetConstant(0, MVT::i1);
  TFE = CurDAG->getTargetConstant(0, MVT::i1);

  Idxen = CurDAG->getTargetConstant(0, MVT::i1);
  Offen = CurDAG->getTargetConstant(0, MVT::i1);
  Addr64 = CurDAG->getTargetConstant(0, MVT::i1);
  SOffset = CurDAG->getTargetConstant(0, MVT::i32);

  if (CurDAG->isBaseWithConstantOffset(Addr)) {
    SDValue N0 = Addr.getOperand(0);
    SDValue N1 = Addr.getOperand(1);
    ConstantSDNode *C1 = cast<ConstantSDNode>(N1);

    if (isLegalMUBUFImmOffset(C1)) {

      if (N0.getOpcode() == ISD::ADD) {
        // (add (add N2, N3), C1) -> addr64
        SDValue N2 = N0.getOperand(0);
        SDValue N3 = N0.getOperand(1);
        Addr64 = CurDAG->getTargetConstant(1, MVT::i1);
        Ptr = N2;
        VAddr = N3;
        Offset = CurDAG->getTargetConstant(C1->getZExtValue(), MVT::i16);
        return;
      }

      // (add N0, C1) -> offset
      VAddr = CurDAG->getTargetConstant(0, MVT::i32);
      Ptr = N0;
      Offset = CurDAG->getTargetConstant(C1->getZExtValue(), MVT::i16);
      return;
    }
  }
  if (Addr.getOpcode() == ISD::ADD) {
    // (add N0, N1) -> addr64
    SDValue N0 = Addr.getOperand(0);
    SDValue N1 = Addr.getOperand(1);
    Addr64 = CurDAG->getTargetConstant(1, MVT::i1);
    Ptr = N0;
    VAddr = N1;
    Offset = CurDAG->getTargetConstant(0, MVT::i16);
    return;
  }

  // default case -> offset
  VAddr = CurDAG->getTargetConstant(0, MVT::i32);
  Ptr = Addr;
  Offset = CurDAG->getTargetConstant(0, MVT::i16);

}

bool AMDGPUDAGToDAGISel::SelectMUBUFAddr64(SDValue Addr, SDValue &SRsrc,
                                           SDValue &VAddr,
                                           SDValue &Offset) const {
  SDValue Ptr, SOffset, Offen, Idxen, Addr64, GLC, SLC, TFE;

  SelectMUBUF(Addr, Ptr, VAddr, SOffset, Offset, Offen, Idxen, Addr64,
              GLC, SLC, TFE);

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
                                           SDValue &VAddr, SDValue &Offset,
                                           SDValue &SLC) const {
  SLC = CurDAG->getTargetConstant(0, MVT::i1);

  return SelectMUBUFAddr64(Addr, SRsrc, VAddr, Offset);
}

bool AMDGPUDAGToDAGISel::SelectMUBUFScratch(SDValue Addr, SDValue &Rsrc,
                                            SDValue &VAddr, SDValue &SOffset,
                                            SDValue &ImmOffset) const {

  SDLoc DL(Addr);
  MachineFunction &MF = CurDAG->getMachineFunction();
  const SIRegisterInfo *TRI =
      static_cast<const SIRegisterInfo *>(MF.getSubtarget().getRegisterInfo());
  MachineRegisterInfo &MRI = MF.getRegInfo();
  const SITargetLowering& Lowering =
    *static_cast<const SITargetLowering*>(getTargetLowering());

  unsigned ScratchOffsetReg =
      TRI->getPreloadedValue(MF, SIRegisterInfo::SCRATCH_WAVE_OFFSET);
  Lowering.CreateLiveInRegister(*CurDAG, &AMDGPU::SReg_32RegClass,
                                ScratchOffsetReg, MVT::i32);
  SDValue Sym0 = CurDAG->getExternalSymbol("SCRATCH_RSRC_DWORD0", MVT::i32);
  SDValue ScratchRsrcDword0 =
      SDValue(CurDAG->getMachineNode(AMDGPU::S_MOV_B32, DL, MVT::i32, Sym0), 0);

  SDValue Sym1 = CurDAG->getExternalSymbol("SCRATCH_RSRC_DWORD1", MVT::i32);
  SDValue ScratchRsrcDword1 =
      SDValue(CurDAG->getMachineNode(AMDGPU::S_MOV_B32, DL, MVT::i32, Sym1), 0);

  const SDValue RsrcOps[] = {
      CurDAG->getTargetConstant(AMDGPU::SReg_64RegClassID, MVT::i32),
      ScratchRsrcDword0,
      CurDAG->getTargetConstant(AMDGPU::sub0, MVT::i32),
      ScratchRsrcDword1,
      CurDAG->getTargetConstant(AMDGPU::sub1, MVT::i32),
  };
  SDValue ScratchPtr = SDValue(CurDAG->getMachineNode(AMDGPU::REG_SEQUENCE, DL,
                                              MVT::v2i32, RsrcOps), 0);
  Rsrc = SDValue(Lowering.buildScratchRSRC(*CurDAG, DL, ScratchPtr), 0);
  SOffset = CurDAG->getCopyFromReg(CurDAG->getEntryNode(), DL,
      MRI.getLiveInVirtReg(ScratchOffsetReg), MVT::i32);

  // (add n0, c1)
  if (CurDAG->isBaseWithConstantOffset(Addr)) {
    SDValue N1 = Addr.getOperand(1);
    ConstantSDNode *C1 = cast<ConstantSDNode>(N1);

    if (isLegalMUBUFImmOffset(C1)) {
      VAddr = Addr.getOperand(0);
      ImmOffset = CurDAG->getTargetConstant(C1->getZExtValue(), MVT::i16);
      return true;
    }
  }

  // (node)
  VAddr = Addr;
  ImmOffset = CurDAG->getTargetConstant(0, MVT::i16);
  return true;
}

bool AMDGPUDAGToDAGISel::SelectMUBUFOffset(SDValue Addr, SDValue &SRsrc,
                                           SDValue &SOffset, SDValue &Offset,
                                           SDValue &GLC, SDValue &SLC,
                                           SDValue &TFE) const {
  SDValue Ptr, VAddr, Offen, Idxen, Addr64;
  const SIInstrInfo *TII =
    static_cast<const SIInstrInfo *>(Subtarget.getInstrInfo());

  SelectMUBUF(Addr, Ptr, VAddr, SOffset, Offset, Offen, Idxen, Addr64,
              GLC, SLC, TFE);

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

// FIXME: This is incorrect and only enough to be able to compile.
SDNode *AMDGPUDAGToDAGISel::SelectAddrSpaceCast(SDNode *N) {
  AddrSpaceCastSDNode *ASC = cast<AddrSpaceCastSDNode>(N);
  SDLoc DL(N);

  assert(Subtarget.hasFlatAddressSpace() &&
         "addrspacecast only supported with flat address space!");

  assert((ASC->getSrcAddressSpace() != AMDGPUAS::CONSTANT_ADDRESS &&
          ASC->getDestAddressSpace() != AMDGPUAS::CONSTANT_ADDRESS) &&
         "Cannot cast address space to / from constant address!");

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
      CurDAG->getTargetConstant(AMDGPU::sub0, MVT::i32));
  }


  if (DestSize > SrcSize) {
    assert(SrcSize == 32 && DestSize == 64);

    // FIXME: This is probably wrong, we should never be defining
    // a register class with both VGPRs and SGPRs
    SDValue RC = CurDAG->getTargetConstant(AMDGPU::VS_64RegClassID, MVT::i32);

    const SDValue Ops[] = {
      RC,
      Src,
      CurDAG->getTargetConstant(AMDGPU::sub0, MVT::i32),
      SDValue(CurDAG->getMachineNode(AMDGPU::S_MOV_B32, SDLoc(N), MVT::i32,
                                     CurDAG->getConstant(0, MVT::i32)), 0),
      CurDAG->getTargetConstant(AMDGPU::sub1, MVT::i32)
    };

    return CurDAG->getMachineNode(TargetOpcode::REG_SEQUENCE,
                                  SDLoc(N), N->getValueType(0), Ops);
  }

  assert(SrcSize == 64 && DestSize == 64);
  return CurDAG->getNode(ISD::BITCAST, DL, DestVT, Src).getNode();
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

  SrcMods = CurDAG->getTargetConstant(Mods, MVT::i32);

  return true;
}

bool AMDGPUDAGToDAGISel::SelectVOP3Mods0(SDValue In, SDValue &Src,
                                         SDValue &SrcMods, SDValue &Clamp,
                                         SDValue &Omod) const {
  // FIXME: Handle Clamp and Omod
  Clamp = CurDAG->getTargetConstant(0, MVT::i32);
  Omod = CurDAG->getTargetConstant(0, MVT::i32);

  return SelectVOP3Mods(In, Src, SrcMods);
}

bool AMDGPUDAGToDAGISel::SelectVOP3Mods0Clamp(SDValue In, SDValue &Src,
                                              SDValue &SrcMods,
                                              SDValue &Omod) const {
  // FIXME: Handle Omod
  Omod = CurDAG->getTargetConstant(0, MVT::i32);

  return SelectVOP3Mods(In, Src, SrcMods);
}

bool AMDGPUDAGToDAGISel::SelectVOP3Mods0Clamp0OMod(SDValue In, SDValue &Src,
                                                   SDValue &SrcMods,
                                                   SDValue &Clamp,
                                                   SDValue &Omod) const {
  Clamp = Omod = CurDAG->getTargetConstant(0, MVT::i32);
  return SelectVOP3Mods(In, Src, SrcMods);
}

void AMDGPUDAGToDAGISel::PostprocessISelDAG() {
  const AMDGPUTargetLowering& Lowering =
    *static_cast<const AMDGPUTargetLowering*>(getTargetLowering());
  bool IsModified = false;
  do {
    IsModified = false;
    // Go over all selected nodes and try to fold them a bit more
    for (SelectionDAG::allnodes_iterator I = CurDAG->allnodes_begin(),
         E = CurDAG->allnodes_end(); I != E; ++I) {

      SDNode *Node = I;

      MachineSDNode *MachineNode = dyn_cast<MachineSDNode>(I);
      if (!MachineNode)
        continue;

      SDNode *ResNode = Lowering.PostISelFolding(MachineNode, *CurDAG);
      if (ResNode != Node) {
        ReplaceUses(Node, ResNode);
        IsModified = true;
      }
    }
    CurDAG->RemoveDeadNodes();
  } while (IsModified);
}
