//===-- NVPTXISelDAGToDAG.cpp - A dag to dag inst selector for NVPTX ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines an instruction selector for the NVPTX target.
//
//===----------------------------------------------------------------------===//

#include "NVPTXISelDAGToDAG.h"
#include "NVPTX.h"
#include "NVPTXUtilities.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/CodeGen/ISDOpcodes.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/SelectionDAGNodes.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/IR/NVVMIntrinsicUtils.h"
#include "llvm/Support/AtomicOrdering.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"
#include <optional>

using namespace llvm;

#define DEBUG_TYPE "nvptx-isel"
#define PASS_NAME "NVPTX DAG->DAG Pattern Instruction Selection"

static cl::opt<bool>
    EnableRsqrtOpt("nvptx-rsqrt-approx-opt", cl::init(true), cl::Hidden,
                   cl::desc("Enable reciprocal sqrt optimization"));

/// createNVPTXISelDag - This pass converts a legalized DAG into a
/// NVPTX-specific DAG, ready for instruction scheduling.
FunctionPass *llvm::createNVPTXISelDag(NVPTXTargetMachine &TM,
                                       llvm::CodeGenOptLevel OptLevel) {
  return new NVPTXDAGToDAGISelLegacy(TM, OptLevel);
}

NVPTXDAGToDAGISelLegacy::NVPTXDAGToDAGISelLegacy(NVPTXTargetMachine &tm,
                                                 CodeGenOptLevel OptLevel)
    : SelectionDAGISelLegacy(
          ID, std::make_unique<NVPTXDAGToDAGISel>(tm, OptLevel)) {}

char NVPTXDAGToDAGISelLegacy::ID = 0;

INITIALIZE_PASS(NVPTXDAGToDAGISelLegacy, DEBUG_TYPE, PASS_NAME, false, false)

NVPTXDAGToDAGISel::NVPTXDAGToDAGISel(NVPTXTargetMachine &tm,
                                     CodeGenOptLevel OptLevel)
    : SelectionDAGISel(tm, OptLevel), TM(tm) {}

bool NVPTXDAGToDAGISel::runOnMachineFunction(MachineFunction &MF) {
  Subtarget = &MF.getSubtarget<NVPTXSubtarget>();
  Scopes = NVPTXScopes(MF.getFunction().getContext());
  return SelectionDAGISel::runOnMachineFunction(MF);
}

NVPTX::DivPrecisionLevel
NVPTXDAGToDAGISel::getDivF32Level(const SDNode *N) const {
  return Subtarget->getTargetLowering()->getDivF32Level(*MF, *N);
}

bool NVPTXDAGToDAGISel::usePrecSqrtF32(const SDNode *N) const {
  return Subtarget->getTargetLowering()->usePrecSqrtF32(*MF, N);
}

bool NVPTXDAGToDAGISel::useF32FTZ() const {
  return Subtarget->getTargetLowering()->useF32FTZ(*MF);
}

bool NVPTXDAGToDAGISel::allowFMA() const {
  const NVPTXTargetLowering *TL = Subtarget->getTargetLowering();
  return TL->allowFMA(*MF, OptLevel);
}

bool NVPTXDAGToDAGISel::allowUnsafeFPMath() const {
  const NVPTXTargetLowering *TL = Subtarget->getTargetLowering();
  return TL->allowUnsafeFPMath(*MF);
}

bool NVPTXDAGToDAGISel::doRsqrtOpt() const { return EnableRsqrtOpt; }

/// Select - Select instructions not customized! Used for
/// expanded, promoted and normal instructions.
void NVPTXDAGToDAGISel::Select(SDNode *N) {

  if (N->isMachineOpcode()) {
    N->setNodeId(-1);
    return; // Already selected.
  }

  switch (N->getOpcode()) {
  case ISD::LOAD:
  case ISD::ATOMIC_LOAD:
    if (tryLoad(N))
      return;
    break;
  case ISD::STORE:
  case ISD::ATOMIC_STORE:
    if (tryStore(N))
      return;
    break;
  case ISD::ATOMIC_FENCE:
    if (tryFence(N))
      return;
    break;
  case NVPTXISD::UNPACK_VECTOR:
    tryUNPACK_VECTOR(N);
    return;
  case ISD::EXTRACT_VECTOR_ELT:
    if (tryEXTRACT_VECTOR_ELEMENT(N))
      return;
    break;
  case NVPTXISD::SETP_F16X2:
    SelectSETP_F16X2(N);
    return;
  case NVPTXISD::SETP_BF16X2:
    SelectSETP_BF16X2(N);
    return;
  case NVPTXISD::LoadV2:
  case NVPTXISD::LoadV4:
  case NVPTXISD::LoadV8:
    if (tryLoadVector(N))
      return;
    break;
  case NVPTXISD::LDUV2:
  case NVPTXISD::LDUV4:
    if (tryLDU(N))
      return;
    break;
  case NVPTXISD::StoreV2:
  case NVPTXISD::StoreV4:
  case NVPTXISD::StoreV8:
    if (tryStoreVector(N))
      return;
    break;
  case ISD::INTRINSIC_W_CHAIN:
    if (tryIntrinsicChain(N))
      return;
    break;
  case ISD::INTRINSIC_VOID:
    if (tryIntrinsicVoid(N))
      return;
    break;
  case ISD::AND:
  case ISD::SRA:
  case ISD::SRL:
    // Try to select BFE
    if (tryBFE(N))
      return;
    break;
  case ISD::ADDRSPACECAST:
    SelectAddrSpaceCast(N);
    return;
  case ISD::CopyToReg: {
    if (N->getOperand(1).getValueType() == MVT::i128) {
      SelectV2I64toI128(N);
      return;
    }
    break;
  }
  case ISD::CopyFromReg: {
    if (N->getOperand(1).getValueType() == MVT::i128) {
      SelectI128toV2I64(N);
      return;
    }
    break;
  }
  case ISD::FADD:
  case ISD::FMUL:
  case ISD::FSUB:
    if (tryBF16ArithToFMA(N))
      return;
    break;
  default:
    break;
  }
  SelectCode(N);
}

#define TCGEN05_LD_OPCODE(SHAPE, NUM)                                          \
  (enablePack ? NVPTX::TCGEN05_LD_##SHAPE##_##NUM##_PACK                       \
              : NVPTX::TCGEN05_LD_##SHAPE##_##NUM)

static unsigned getTcgen05LdOpcode(unsigned IID, bool enablePack) {
  switch (IID) {
  case Intrinsic::nvvm_tcgen05_ld_16x64b_x1:
    return TCGEN05_LD_OPCODE(16x64b, x1);
  case Intrinsic::nvvm_tcgen05_ld_16x64b_x2:
    return TCGEN05_LD_OPCODE(16x64b, x2);
  case Intrinsic::nvvm_tcgen05_ld_16x64b_x4:
    return TCGEN05_LD_OPCODE(16x64b, x4);
  case Intrinsic::nvvm_tcgen05_ld_16x64b_x8:
    return TCGEN05_LD_OPCODE(16x64b, x8);
  case Intrinsic::nvvm_tcgen05_ld_16x64b_x16:
    return TCGEN05_LD_OPCODE(16x64b, x16);
  case Intrinsic::nvvm_tcgen05_ld_16x64b_x32:
    return TCGEN05_LD_OPCODE(16x64b, x32);
  case Intrinsic::nvvm_tcgen05_ld_16x64b_x64:
    return TCGEN05_LD_OPCODE(16x64b, x64);
  case Intrinsic::nvvm_tcgen05_ld_16x64b_x128:
    return TCGEN05_LD_OPCODE(16x64b, x128);
  case Intrinsic::nvvm_tcgen05_ld_16x128b_x1:
    return TCGEN05_LD_OPCODE(16x128b, x1);
  case Intrinsic::nvvm_tcgen05_ld_16x128b_x2:
    return TCGEN05_LD_OPCODE(16x128b, x2);
  case Intrinsic::nvvm_tcgen05_ld_16x128b_x4:
    return TCGEN05_LD_OPCODE(16x128b, x4);
  case Intrinsic::nvvm_tcgen05_ld_16x128b_x8:
    return TCGEN05_LD_OPCODE(16x128b, x8);
  case Intrinsic::nvvm_tcgen05_ld_16x128b_x16:
    return TCGEN05_LD_OPCODE(16x128b, x16);
  case Intrinsic::nvvm_tcgen05_ld_16x128b_x32:
    return TCGEN05_LD_OPCODE(16x128b, x32);
  case Intrinsic::nvvm_tcgen05_ld_16x128b_x64:
    return TCGEN05_LD_OPCODE(16x128b, x64);
  case Intrinsic::nvvm_tcgen05_ld_16x256b_x1:
    return TCGEN05_LD_OPCODE(16x256b, x1);
  case Intrinsic::nvvm_tcgen05_ld_16x256b_x2:
    return TCGEN05_LD_OPCODE(16x256b, x2);
  case Intrinsic::nvvm_tcgen05_ld_16x256b_x4:
    return TCGEN05_LD_OPCODE(16x256b, x4);
  case Intrinsic::nvvm_tcgen05_ld_16x256b_x8:
    return TCGEN05_LD_OPCODE(16x256b, x8);
  case Intrinsic::nvvm_tcgen05_ld_16x256b_x16:
    return TCGEN05_LD_OPCODE(16x256b, x16);
  case Intrinsic::nvvm_tcgen05_ld_16x256b_x32:
    return TCGEN05_LD_OPCODE(16x256b, x32);
  case Intrinsic::nvvm_tcgen05_ld_16x32bx2_x1:
    return TCGEN05_LD_OPCODE(16x32bx2, x1);
  case Intrinsic::nvvm_tcgen05_ld_16x32bx2_x2:
    return TCGEN05_LD_OPCODE(16x32bx2, x2);
  case Intrinsic::nvvm_tcgen05_ld_16x32bx2_x4:
    return TCGEN05_LD_OPCODE(16x32bx2, x4);
  case Intrinsic::nvvm_tcgen05_ld_16x32bx2_x8:
    return TCGEN05_LD_OPCODE(16x32bx2, x8);
  case Intrinsic::nvvm_tcgen05_ld_16x32bx2_x16:
    return TCGEN05_LD_OPCODE(16x32bx2, x16);
  case Intrinsic::nvvm_tcgen05_ld_16x32bx2_x32:
    return TCGEN05_LD_OPCODE(16x32bx2, x32);
  case Intrinsic::nvvm_tcgen05_ld_16x32bx2_x64:
    return TCGEN05_LD_OPCODE(16x32bx2, x64);
  case Intrinsic::nvvm_tcgen05_ld_16x32bx2_x128:
    return TCGEN05_LD_OPCODE(16x32bx2, x128);
  case Intrinsic::nvvm_tcgen05_ld_32x32b_x1:
    return TCGEN05_LD_OPCODE(32x32b, x1);
  case Intrinsic::nvvm_tcgen05_ld_32x32b_x2:
    return TCGEN05_LD_OPCODE(32x32b, x2);
  case Intrinsic::nvvm_tcgen05_ld_32x32b_x4:
    return TCGEN05_LD_OPCODE(32x32b, x4);
  case Intrinsic::nvvm_tcgen05_ld_32x32b_x8:
    return TCGEN05_LD_OPCODE(32x32b, x8);
  case Intrinsic::nvvm_tcgen05_ld_32x32b_x16:
    return TCGEN05_LD_OPCODE(32x32b, x16);
  case Intrinsic::nvvm_tcgen05_ld_32x32b_x32:
    return TCGEN05_LD_OPCODE(32x32b, x32);
  case Intrinsic::nvvm_tcgen05_ld_32x32b_x64:
    return TCGEN05_LD_OPCODE(32x32b, x64);
  case Intrinsic::nvvm_tcgen05_ld_32x32b_x128:
    return TCGEN05_LD_OPCODE(32x32b, x128);
  }
  llvm_unreachable("unhandled tcgen05.ld lowering");
}

void NVPTXDAGToDAGISel::SelectTcgen05Ld(SDNode *N, bool hasOffset) {
  SDLoc DL(N);
  unsigned IID = cast<ConstantSDNode>(N->getOperand(1))->getZExtValue();

  if (hasOffset) {
    bool enablePack = cast<ConstantSDNode>(N->getOperand(4))->getZExtValue();
    auto OffsetNode = CurDAG->getTargetConstant(
        cast<ConstantSDNode>(N->getOperand(3))->getZExtValue(), DL, MVT::i32);
    ReplaceNode(N, CurDAG->getMachineNode(
                       getTcgen05LdOpcode(IID, enablePack), DL, N->getVTList(),
                       {N->getOperand(2), OffsetNode, N->getOperand(0)}));
  } else {
    bool enablePack = cast<ConstantSDNode>(N->getOperand(3))->getZExtValue();
    ReplaceNode(N, CurDAG->getMachineNode(
                       getTcgen05LdOpcode(IID, enablePack), DL, N->getVTList(),
                       {N->getOperand(2), N->getOperand(0)}));
  }
}

bool NVPTXDAGToDAGISel::tryIntrinsicChain(SDNode *N) {
  unsigned IID = N->getConstantOperandVal(1);
  switch (IID) {
  default:
    return false;
  case Intrinsic::nvvm_ldu_global_f:
  case Intrinsic::nvvm_ldu_global_i:
  case Intrinsic::nvvm_ldu_global_p:
    return tryLDU(N);

  case Intrinsic::nvvm_tcgen05_ld_16x64b_x1:
  case Intrinsic::nvvm_tcgen05_ld_16x64b_x2:
  case Intrinsic::nvvm_tcgen05_ld_16x64b_x4:
  case Intrinsic::nvvm_tcgen05_ld_16x64b_x8:
  case Intrinsic::nvvm_tcgen05_ld_16x64b_x16:
  case Intrinsic::nvvm_tcgen05_ld_16x64b_x32:
  case Intrinsic::nvvm_tcgen05_ld_16x64b_x64:
  case Intrinsic::nvvm_tcgen05_ld_16x64b_x128:
  case Intrinsic::nvvm_tcgen05_ld_16x128b_x1:
  case Intrinsic::nvvm_tcgen05_ld_16x128b_x2:
  case Intrinsic::nvvm_tcgen05_ld_16x128b_x4:
  case Intrinsic::nvvm_tcgen05_ld_16x128b_x16:
  case Intrinsic::nvvm_tcgen05_ld_16x128b_x32:
  case Intrinsic::nvvm_tcgen05_ld_16x128b_x64:
  case Intrinsic::nvvm_tcgen05_ld_16x256b_x1:
  case Intrinsic::nvvm_tcgen05_ld_16x128b_x8:
  case Intrinsic::nvvm_tcgen05_ld_16x256b_x2:
  case Intrinsic::nvvm_tcgen05_ld_16x256b_x4:
  case Intrinsic::nvvm_tcgen05_ld_16x256b_x8:
  case Intrinsic::nvvm_tcgen05_ld_16x256b_x16:
  case Intrinsic::nvvm_tcgen05_ld_16x256b_x32:
  case Intrinsic::nvvm_tcgen05_ld_32x32b_x1:
  case Intrinsic::nvvm_tcgen05_ld_32x32b_x2:
  case Intrinsic::nvvm_tcgen05_ld_32x32b_x4:
  case Intrinsic::nvvm_tcgen05_ld_32x32b_x8:
  case Intrinsic::nvvm_tcgen05_ld_32x32b_x16:
  case Intrinsic::nvvm_tcgen05_ld_32x32b_x32:
  case Intrinsic::nvvm_tcgen05_ld_32x32b_x64:
  case Intrinsic::nvvm_tcgen05_ld_32x32b_x128: {
    SelectTcgen05Ld(N);
    return true;
  }

  case Intrinsic::nvvm_tcgen05_ld_16x32bx2_x1:
  case Intrinsic::nvvm_tcgen05_ld_16x32bx2_x2:
  case Intrinsic::nvvm_tcgen05_ld_16x32bx2_x4:
  case Intrinsic::nvvm_tcgen05_ld_16x32bx2_x8:
  case Intrinsic::nvvm_tcgen05_ld_16x32bx2_x16:
  case Intrinsic::nvvm_tcgen05_ld_16x32bx2_x32:
  case Intrinsic::nvvm_tcgen05_ld_16x32bx2_x64:
  case Intrinsic::nvvm_tcgen05_ld_16x32bx2_x128: {
    SelectTcgen05Ld(N, /* hasOffset */ true);
    return true;
  }
  }
}

// Map ISD:CONDCODE value to appropriate CmpMode expected by
// NVPTXInstPrinter::printCmpMode()
SDValue NVPTXDAGToDAGISel::getPTXCmpMode(const CondCodeSDNode &CondCode) {
  using NVPTX::PTXCmpMode::CmpMode;
  const unsigned PTXCmpMode = [](ISD::CondCode CC) {
    switch (CC) {
    default:
      llvm_unreachable("Unexpected condition code.");
    case ISD::SETOEQ:
    case ISD::SETEQ:
      return CmpMode::EQ;
    case ISD::SETOGT:
    case ISD::SETGT:
      return CmpMode::GT;
    case ISD::SETOGE:
    case ISD::SETGE:
      return CmpMode::GE;
    case ISD::SETOLT:
    case ISD::SETLT:
      return CmpMode::LT;
    case ISD::SETOLE:
    case ISD::SETLE:
      return CmpMode::LE;
    case ISD::SETONE:
    case ISD::SETNE:
      return CmpMode::NE;
    case ISD::SETO:
      return CmpMode::NUM;
    case ISD::SETUO:
      return CmpMode::NotANumber;
    case ISD::SETUEQ:
      return CmpMode::EQU;
    case ISD::SETUGT:
      return CmpMode::GTU;
    case ISD::SETUGE:
      return CmpMode::GEU;
    case ISD::SETULT:
      return CmpMode::LTU;
    case ISD::SETULE:
      return CmpMode::LEU;
    case ISD::SETUNE:
      return CmpMode::NEU;
    }
  }(CondCode.get());
  return CurDAG->getTargetConstant(PTXCmpMode, SDLoc(), MVT::i32);
}

bool NVPTXDAGToDAGISel::SelectSETP_F16X2(SDNode *N) {
  SDValue PTXCmpMode = getPTXCmpMode(*cast<CondCodeSDNode>(N->getOperand(2)));
  SDLoc DL(N);
  SDNode *SetP = CurDAG->getMachineNode(
      NVPTX::SETP_f16x2rr, DL, MVT::i1, MVT::i1,
      {N->getOperand(0), N->getOperand(1), PTXCmpMode,
       CurDAG->getTargetConstant(useF32FTZ() ? 1 : 0, DL, MVT::i1)});
  ReplaceNode(N, SetP);
  return true;
}

bool NVPTXDAGToDAGISel::SelectSETP_BF16X2(SDNode *N) {
  SDValue PTXCmpMode = getPTXCmpMode(*cast<CondCodeSDNode>(N->getOperand(2)));
  SDLoc DL(N);
  SDNode *SetP = CurDAG->getMachineNode(
      NVPTX::SETP_bf16x2rr, DL, MVT::i1, MVT::i1,
      {N->getOperand(0), N->getOperand(1), PTXCmpMode,
       CurDAG->getTargetConstant(useF32FTZ() ? 1 : 0, DL, MVT::i1)});
  ReplaceNode(N, SetP);
  return true;
}

bool NVPTXDAGToDAGISel::tryUNPACK_VECTOR(SDNode *N) {
  SDValue Vector = N->getOperand(0);
  MVT EltVT = N->getSimpleValueType(0);

  MachineSDNode *N2 =
      CurDAG->getMachineNode(NVPTX::I64toV2I32, SDLoc(N), EltVT, EltVT, Vector);

  ReplaceNode(N, N2);
  return true;
}

// Find all instances of extract_vector_elt that use this v2f16 vector
// and coalesce them into a scattering move instruction.
bool NVPTXDAGToDAGISel::tryEXTRACT_VECTOR_ELEMENT(SDNode *N) {
  SDValue Vector = N->getOperand(0);

  MVT VT = Vector.getSimpleValueType();
  if (!(NVPTX::isPackedVectorTy(VT) && VT.getVectorNumElements() == 2))
    return false;

  unsigned Opcode;
  if (VT.is32BitVector())
    Opcode = NVPTX::I32toV2I16;
  else if (VT.is64BitVector())
    Opcode = NVPTX::I64toV2I32;
  else
    llvm_unreachable("Unhandled packed type");

  // Find and record all uses of this vector that extract element 0 or 1.
  SmallVector<SDNode *, 4> E0, E1;
  for (auto *U : Vector.getNode()->users()) {
    if (U->getOpcode() != ISD::EXTRACT_VECTOR_ELT)
      continue;
    if (U->getOperand(0) != Vector)
      continue;
    if (const ConstantSDNode *IdxConst =
            dyn_cast<ConstantSDNode>(U->getOperand(1))) {
      if (IdxConst->getZExtValue() == 0)
        E0.push_back(U);
      else if (IdxConst->getZExtValue() == 1)
        E1.push_back(U);
      else
        llvm_unreachable("Invalid vector index.");
    }
  }

  // There's no point scattering f16x2 if we only ever access one
  // element of it.
  if (E0.empty() || E1.empty())
    return false;

  // Merge (EltTy extractelt(V, 0), EltTy extractelt(V,1))
  // into EltTy,EltTy Split[EltTy]x2(V)
  MVT EltVT = VT.getVectorElementType();
  SDNode *ScatterOp =
      CurDAG->getMachineNode(Opcode, SDLoc(N), EltVT, EltVT, Vector);
  for (auto *Node : E0)
    ReplaceUses(SDValue(Node, 0), SDValue(ScatterOp, 0));
  for (auto *Node : E1)
    ReplaceUses(SDValue(Node, 0), SDValue(ScatterOp, 1));

  return true;
}

static std::optional<NVPTX::AddressSpace> convertAS(unsigned AS) {
  switch (AS) {
  case llvm::ADDRESS_SPACE_LOCAL:
    return NVPTX::AddressSpace::Local;
  case llvm::ADDRESS_SPACE_GLOBAL:
    return NVPTX::AddressSpace::Global;
  case llvm::ADDRESS_SPACE_SHARED:
    return NVPTX::AddressSpace::Shared;
  case llvm::ADDRESS_SPACE_SHARED_CLUSTER:
    return NVPTX::AddressSpace::SharedCluster;
  case llvm::ADDRESS_SPACE_GENERIC:
    return NVPTX::AddressSpace::Generic;
  case llvm::ADDRESS_SPACE_PARAM:
    return NVPTX::AddressSpace::Param;
  case llvm::ADDRESS_SPACE_CONST:
    return NVPTX::AddressSpace::Const;
  default:
    return std::nullopt;
  }
}

NVPTX::AddressSpace NVPTXDAGToDAGISel::getAddrSpace(const MemSDNode *N) {
  return convertAS(N->getMemOperand()->getAddrSpace())
      .value_or(NVPTX::AddressSpace::Generic);
}

NVPTX::Ordering NVPTXDAGToDAGISel::getMemOrder(const MemSDNode *N) const {
  // No "sem" orderings for SM/PTX versions which do not support memory ordering
  if (!Subtarget->hasMemoryOrdering())
    return NVPTX::Ordering::NotAtomic;
  auto Ordering = N->getMergedOrdering();
  switch (Ordering) {
  case AtomicOrdering::NotAtomic:
    return NVPTX::Ordering::NotAtomic;
  case AtomicOrdering::Unordered:
  case AtomicOrdering::Monotonic:
    return NVPTX::Ordering::Relaxed;
  case AtomicOrdering::Acquire:
    return NVPTX::Ordering::Acquire;
  case AtomicOrdering::Release:
    return NVPTX::Ordering::Release;
  case AtomicOrdering::AcquireRelease:
    return NVPTX::Ordering::AcquireRelease;
  case AtomicOrdering::SequentiallyConsistent:
    return NVPTX::Ordering::SequentiallyConsistent;
  }
  llvm_unreachable("Invalid atomic ordering");
}

NVPTX::Scope NVPTXDAGToDAGISel::getAtomicScope(const MemSDNode *N) const {
  // No "scope" modifier for SM/PTX versions which do not support scoped atomics
  // Functionally, these atomics are at device scope
  if (!Subtarget->hasAtomScope())
    return NVPTX::Scope::DefaultDevice;
  return Scopes[N->getSyncScopeID()];
}

namespace {

struct OperationOrderings {
  NVPTX::Ordering InstructionOrdering, FenceOrdering;
  OperationOrderings(NVPTX::Ordering IO = NVPTX::Ordering::NotAtomic,
                     NVPTX::Ordering FO = NVPTX::Ordering::NotAtomic)
      : InstructionOrdering(IO), FenceOrdering(FO) {}
};

static OperationOrderings
getOperationOrderings(MemSDNode *N, const NVPTXSubtarget *Subtarget) {
  AtomicOrdering Ordering = N->getSuccessOrdering();
  auto CodeAddrSpace = NVPTXDAGToDAGISel::getAddrSpace(N);

  bool HasMemoryOrdering = Subtarget->hasMemoryOrdering();
  bool HasRelaxedMMIO = Subtarget->hasRelaxedMMIO();

  // clang-format off

  // Lowering for Load/Store Operations (note: AcquireRelease Loads or Stores error).
  // Note: uses of Relaxed in the Atomic column of this table refer
  // to LLVM AtomicOrdering::Monotonic.
  //
  // | Atomic  | Volatile | Statespace         | PTX sm_60- | PTX sm_70+                   |
  // |---------|----------|--------------------|------------|------------------------------|
  // | No      | No       | All                | plain      | .weak                        |
  // | No      | Yes      | Generic,Shared,    | .volatile  | .volatile                    |
  // |         |          | Global [0]         |            |                              |
  // | No      | Yes      | Local,Const,Param  | plain [1]  | .weak [1]                    |
  // | Unorder | Yes/No   | All                | == Relaxed | == Relaxed                   |
  // | Relaxed | No       | Generic,Shared,    | .volatile  | <atomic sem>                 |
  // |         |          | Global [0]         |            |                              |
  // | Other   | No       | Generic,Shared,    | Error [2]  | <atomic sem>                 |
  // |         |          | Global [0]         |            |                              |
  // | Yes     | No       | Local,Const,Param  | plain [1]  | .weak [1]                    |
  // | Relaxed | Yes      | Generic,Shared [0] | .volatile  | .volatile                    |
  // | Relaxed | Yes      | Global [0]         | .volatile  | .mmio.relaxed.sys (PTX 8.2+) |
  // |         |          |                    |            |  or .volatile (PTX 8.1-)     |
  // | Relaxed | Yes      | Local,Const,Param  | plain [1]  | .weak [1]                    |
  // | Other   | Yes      | Generic, Shared,   | Error [2]  | <atomic sem> [3]             |
  // |         |          | / Global [0]       |            |                              |

  // Lowering of CUDA C++ SequentiallyConsistent Operations and Fences to PTX
  // by following the ABI proven sound in:
  //   Lustig et al, A Formal Analysis of the NVIDIA PTX Memory Consistency Model, ASPLOS’19.
  //   https://dl.acm.org/doi/pdf/10.1145/3297858.3304043
  //
  // | CUDA C++ Atomic Operation or Atomic Fence            | PTX Atomic Operation or Fence |
  // |------------------------------------------------------|-------------------------------|
  // | cuda::atomic_thread_fence                            | fence.sc.<scope>;             |
  // |   (memory_order_seq_cst, cuda::thread_scope_<scope>) |                               |
  // |------------------------------------------------------|-------------------------------|
  // | cuda::atomic_load                                    | fence.sc.<scope>;             |
  // |   (memory_order_seq_cst, cuda::thread_scope_<scope>) | ld.acquire.<scope>;           |
  // |------------------------------------------------------|-------------------------------|  
  // | cuda::atomic_store                                   | fence.sc.<scope>;             |
  // |   (memory_order_seq_cst, cuda::thread_scope_<scope>) | st.release.<scope>;           |
  // |------------------------------------------------------|-------------------------------|
  // | cuda::atomic_fetch_<op>                              | fence.sc.<scope>;             |
  // |   (memory_order_seq_cst, cuda::thread_scope_<scope>) | atom.acq_rel.<scope>;         |

  // clang-format on

  // [0]: volatile and atomics are only supported on global or shared
  //      memory locations, accessed via generic/shared/global pointers.
  //      MMIO is only supported on global memory locations,
  //      accessed via generic/global pointers.
  // TODO: Implement MMIO access via generic pointer to global.
  //       Currently implemented for global pointers only.

  // [1]: Lowering volatile/atomic operations to non-volatile/non-atomic
  //      PTX instructions fails to preserve their C++ side-effects.
  //
  //      Example (https://github.com/llvm/llvm-project/issues/62057):
  //
  //          void example() {
  //              std::atomic<bool> True = true;
  //              while (True.load(std::memory_order_relaxed));
  //          }
  //
  //      A C++ program that calls "example" is well-defined: the infinite loop
  //      performs an atomic operation. By lowering volatile/atomics to
  //      "weak" memory operations, we are transforming the above into:
  //
  //          void undefined_behavior() {
  //              bool True = true;
  //              while (True);
  //          }
  //
  //      which exhibits undefined behavior in both C++ and PTX.
  //
  //      Calling "example" in CUDA C++ compiled for sm_60- exhibits undefined
  //      behavior due to lack of Independent Forward Progress. Lowering these
  //      to weak memory operations in sm_60- is therefore fine.
  //
  //      TODO: lower atomic and volatile operations to memory locations
  //      in local, const, and param to two PTX instructions in sm_70+:
  //        - the "weak" memory instruction we are currently lowering to, and
  //        - some other instruction that preserves the side-effect, e.g.,
  //          a dead dummy volatile load.
  if (CodeAddrSpace == NVPTX::AddressSpace::Local ||
      CodeAddrSpace == NVPTX::AddressSpace::Const ||
      CodeAddrSpace == NVPTX::AddressSpace::Param) {
    return NVPTX::Ordering::NotAtomic;
  }

  // [2]: Atomics with Ordering different than Unordered or Relaxed are not
  //      supported on sm_60 and older; this includes volatile atomics.
  if (!(Ordering == AtomicOrdering::NotAtomic ||
        Ordering == AtomicOrdering::Unordered ||
        Ordering == AtomicOrdering::Monotonic) &&
      !HasMemoryOrdering) {
    report_fatal_error(
        formatv("PTX does not support \"atomic\" for orderings different than"
                "\"NotAtomic\" or \"Monotonic\" for sm_60 or older, but order "
                "is: \"{}\".",
                toIRString(Ordering)));
  }

  // [3]: TODO: these should eventually use .mmio<.atomic sem>; for now we drop
  // the volatile semantics and preserve the atomic ones.

  // PTX volatile and PTX atomics are not available for statespace that differ
  // from .generic, .global, or .shared. The behavior of PTX volatile and PTX
  // atomics is undefined if the generic address does not refer to a .global or
  // .shared memory location.
  bool AddrGenericOrGlobalOrShared =
      (CodeAddrSpace == NVPTX::AddressSpace::Generic ||
       CodeAddrSpace == NVPTX::AddressSpace::Global ||
       CodeAddrSpace == NVPTX::AddressSpace::Shared ||
       CodeAddrSpace == NVPTX::AddressSpace::SharedCluster);
  if (!AddrGenericOrGlobalOrShared)
    return NVPTX::Ordering::NotAtomic;

  bool UseRelaxedMMIO =
      HasRelaxedMMIO && CodeAddrSpace == NVPTX::AddressSpace::Global;

  switch (Ordering) {
  case AtomicOrdering::NotAtomic:
    return N->isVolatile() ? NVPTX::Ordering::Volatile
                           : NVPTX::Ordering::NotAtomic;
  case AtomicOrdering::Unordered:
    // We lower unordered in the exact same way as 'monotonic' to respect
    // LLVM IR atomicity requirements.
  case AtomicOrdering::Monotonic:
    if (N->isVolatile())
      return UseRelaxedMMIO ? NVPTX::Ordering::RelaxedMMIO
                            : NVPTX::Ordering::Volatile;
    else
      return HasMemoryOrdering ? NVPTX::Ordering::Relaxed
                               : NVPTX::Ordering::Volatile;
  // case AtomicOrdering::Consume: // If LLVM ever provides this, lower it to
  // Acquire.
  case AtomicOrdering::Acquire:
    if (!N->readMem())
      report_fatal_error(
          formatv("PTX only supports Acquire Ordering on reads: {}",
                  N->getOperationName()));
    return NVPTX::Ordering::Acquire;
  case AtomicOrdering::Release:
    if (!N->writeMem())
      report_fatal_error(
          formatv("PTX only supports Release Ordering on writes: {}",
                  N->getOperationName()));
    return NVPTX::Ordering::Release;
  case AtomicOrdering::AcquireRelease: {
    report_fatal_error(
        formatv("NVPTX does not support AcquireRelease Ordering on "
                "read-modify-write "
                "yet and PTX does not support it on loads or stores: {}",
                N->getOperationName()));
  }
  case AtomicOrdering::SequentiallyConsistent: {
    // LLVM-IR SequentiallyConsistent atomics map to a two-instruction PTX
    // sequence including a "fence.sc.sco" and the memory instruction with an
    // Ordering that differs from "sc": acq, rel, or acq_rel, depending on
    // whether the memory operation is a read, write, or read-modify-write.
    //
    // This sets the ordering of the fence to SequentiallyConsistent, and
    // sets the corresponding ordering for the instruction.
    NVPTX::Ordering InstrOrder;
    if (N->readMem())
      InstrOrder = NVPTX::Ordering::Acquire;
    else if (N->writeMem())
      InstrOrder = NVPTX::Ordering::Release;
    else
      report_fatal_error(
          formatv("NVPTX does not support SequentiallyConsistent Ordering on "
                  "read-modify-writes yet: {}",
                  N->getOperationName()));
    return OperationOrderings(InstrOrder,
                              NVPTX::Ordering::SequentiallyConsistent);
  }
  }
  report_fatal_error(
      formatv("NVPTX backend does not support AtomicOrdering \"{}\" yet.",
              toIRString(Ordering)));
}

} // namespace

NVPTX::Scope NVPTXDAGToDAGISel::getOperationScope(MemSDNode *N,
                                                  NVPTX::Ordering O) const {
  switch (O) {
  case NVPTX::Ordering::NotAtomic:
  case NVPTX::Ordering::Volatile: // Non-atomic volatile operations
    // NVPTX uses Thread scope as the scope of non-atomic operations.
    return NVPTX::Scope::Thread;
  case NVPTX::Ordering::RelaxedMMIO:
    // RelaxedMMIO operations are always system scope.
    // If a RelaxedMMIO order was generated from an atomic volatile operation
    // with a smaller thread scope, we bump it here to system scope.
    return NVPTX::Scope::System;
  case NVPTX::Ordering::Relaxed:
  case NVPTX::Ordering::Acquire:
  case NVPTX::Ordering::Release:
  case NVPTX::Ordering::AcquireRelease:
  case NVPTX::Ordering::SequentiallyConsistent:
    auto S = Scopes[N->getSyncScopeID()];

    // Atomic operations must have a scope greater than thread.
    if (S == NVPTX::Scope::Thread)
      report_fatal_error(
          formatv("Atomics need scope > \"{}\".", ScopeToString(S)));

    // If scope is cluster, clusters must be supported.
    if (S == NVPTX::Scope::Cluster)
      Subtarget->failIfClustersUnsupported("cluster scope");

    // If operation is volatile, then its scope is system.
    return N->isVolatile() ? NVPTX::Scope::System : S;
  }
  llvm_unreachable("unhandled ordering");
}

static bool canLowerToLDG(const MemSDNode &N, const NVPTXSubtarget &Subtarget,
                          NVPTX::AddressSpace CodeAddrSpace) {
  // We use ldg (i.e. ld.global.nc) for invariant loads from the global address
  // space.
  return Subtarget.hasLDG() && CodeAddrSpace == NVPTX::AddressSpace::Global &&
         N.isInvariant();
}

static unsigned int getFenceOp(NVPTX::Ordering O, NVPTX::Scope S,
                               NVPTXSubtarget const *T) {
  if (S == NVPTX::Scope::Cluster)
    T->failIfClustersUnsupported(".cluster scope fence");

  // Fall back to .acq_rel if .acquire, .release is not supported.
  if (!T->hasSplitAcquireAndReleaseFences() &&
      (O == NVPTX::Ordering::Acquire || O == NVPTX::Ordering::Release))
    O = NVPTX::Ordering::AcquireRelease;

  switch (O) {
  case NVPTX::Ordering::Acquire:
    switch (S) {
    case NVPTX::Scope::System:
      return T->hasMemoryOrdering() ? NVPTX::atomic_thread_fence_acquire_sys
                                    : NVPTX::INT_MEMBAR_SYS;
    case NVPTX::Scope::Block:
      return T->hasMemoryOrdering() ? NVPTX::atomic_thread_fence_acquire_cta
                                    : NVPTX::INT_MEMBAR_CTA;
    case NVPTX::Scope::Cluster:
      return NVPTX::atomic_thread_fence_acquire_cluster;
    case NVPTX::Scope::Device:
      return T->hasMemoryOrdering() ? NVPTX::atomic_thread_fence_acquire_gpu
                                    : NVPTX::INT_MEMBAR_GL;
    case NVPTX::Scope::Thread:
    case NVPTX::Scope::DefaultDevice:
      report_fatal_error(
          formatv("Unsupported scope \"{}\" for acquire/release/acq_rel fence.",
                  ScopeToString(S)));
    }
    break;
  case NVPTX::Ordering::Release:
    switch (S) {
    case NVPTX::Scope::System:
      return T->hasMemoryOrdering() ? NVPTX::atomic_thread_fence_release_sys
                                    : NVPTX::INT_MEMBAR_SYS;
    case NVPTX::Scope::Block:
      return T->hasMemoryOrdering() ? NVPTX::atomic_thread_fence_release_cta
                                    : NVPTX::INT_MEMBAR_CTA;
    case NVPTX::Scope::Cluster:
      return NVPTX::atomic_thread_fence_release_cluster;
    case NVPTX::Scope::Device:
      return T->hasMemoryOrdering() ? NVPTX::atomic_thread_fence_release_gpu
                                    : NVPTX::INT_MEMBAR_GL;
    case NVPTX::Scope::Thread:
    case NVPTX::Scope::DefaultDevice:
      report_fatal_error(
          formatv("Unsupported scope \"{}\" for acquire/release/acq_rel fence.",
                  ScopeToString(S)));
    }
    break;
  case NVPTX::Ordering::AcquireRelease: {
    switch (S) {
    case NVPTX::Scope::System:
      return T->hasMemoryOrdering() ? NVPTX::atomic_thread_fence_acq_rel_sys
                                    : NVPTX::INT_MEMBAR_SYS;
    case NVPTX::Scope::Block:
      return T->hasMemoryOrdering() ? NVPTX::atomic_thread_fence_acq_rel_cta
                                    : NVPTX::INT_MEMBAR_CTA;
    case NVPTX::Scope::Cluster:
      return NVPTX::atomic_thread_fence_acq_rel_cluster;
    case NVPTX::Scope::Device:
      return T->hasMemoryOrdering() ? NVPTX::atomic_thread_fence_acq_rel_gpu
                                    : NVPTX::INT_MEMBAR_GL;
    case NVPTX::Scope::Thread:
    case NVPTX::Scope::DefaultDevice:
      report_fatal_error(
          formatv("Unsupported scope \"{}\" for acquire/release/acq_rel fence.",
                  ScopeToString(S)));
    }
    break;
  }
  case NVPTX::Ordering::SequentiallyConsistent: {
    switch (S) {
    case NVPTX::Scope::System:
      return T->hasMemoryOrdering() ? NVPTX::atomic_thread_fence_seq_cst_sys
                                    : NVPTX::INT_MEMBAR_SYS;
    case NVPTX::Scope::Block:
      return T->hasMemoryOrdering() ? NVPTX::atomic_thread_fence_seq_cst_cta
                                    : NVPTX::INT_MEMBAR_CTA;
    case NVPTX::Scope::Cluster:
      return NVPTX::atomic_thread_fence_seq_cst_cluster;
    case NVPTX::Scope::Device:
      return T->hasMemoryOrdering() ? NVPTX::atomic_thread_fence_seq_cst_gpu
                                    : NVPTX::INT_MEMBAR_GL;
    case NVPTX::Scope::Thread:
    case NVPTX::Scope::DefaultDevice:
      report_fatal_error(formatv("Unsupported scope \"{}\" for seq_cst fence.",
                                 ScopeToString(S)));
    }
    break;
  }
  case NVPTX::Ordering::NotAtomic:
  case NVPTX::Ordering::Relaxed:
  case NVPTX::Ordering::Volatile:
  case NVPTX::Ordering::RelaxedMMIO:
    report_fatal_error(
        formatv("Unsupported \"{}\" ordering and \"{}\" scope for fence.",
                OrderingToString(O), ScopeToString(S)));
  }
  llvm_unreachable("unhandled ordering");
}

// Returns Memory Order and Scope of a memory instruction, and
// inserts any fence before the instruction that's required to
// implement its memory ordering.
std::pair<NVPTX::Ordering, NVPTX::Scope>
NVPTXDAGToDAGISel::insertMemoryInstructionFence(SDLoc DL, SDValue &Chain,
                                                MemSDNode *N) {
  auto [InstructionOrdering, FenceOrdering] =
      getOperationOrderings(N, Subtarget);
  auto Scope = getOperationScope(N, InstructionOrdering);

  // If a fence is required before the operation, insert it:
  switch (NVPTX::Ordering(FenceOrdering)) {
  case NVPTX::Ordering::NotAtomic:
    break;
  case NVPTX::Ordering::SequentiallyConsistent: {
    auto Op = getFenceOp(FenceOrdering, Scope, Subtarget);
    Chain = SDValue(CurDAG->getMachineNode(Op, DL, MVT::Other, Chain), 0);
    break;
  }
  default:
    report_fatal_error(
        formatv("Unexpected fence ordering: \"{}\".",
                OrderingToString(NVPTX::Ordering(FenceOrdering))));
  }
  return {InstructionOrdering, Scope};
}

void NVPTXDAGToDAGISel::SelectAddrSpaceCast(SDNode *N) {
  SDValue Src = N->getOperand(0);
  AddrSpaceCastSDNode *CastN = cast<AddrSpaceCastSDNode>(N);
  unsigned SrcAddrSpace = CastN->getSrcAddressSpace();
  unsigned DstAddrSpace = CastN->getDestAddressSpace();
  SDLoc DL(N);
  assert(SrcAddrSpace != DstAddrSpace &&
         "addrspacecast must be between different address spaces");

  if (DstAddrSpace == ADDRESS_SPACE_GENERIC) {
    // Specific to generic

    if (TM.is64Bit() && TM.getPointerSizeInBits(SrcAddrSpace) == 32) {
      SDValue CvtNone =
          CurDAG->getTargetConstant(NVPTX::PTXCvtMode::NONE, DL, MVT::i32);
      SDNode *Cvt = CurDAG->getMachineNode(NVPTX::CVT_u64_u32, DL, MVT::i64,
                                           Src, CvtNone);
      Src = SDValue(Cvt, 0);
    }

    unsigned Opc;
    switch (SrcAddrSpace) {
    default: report_fatal_error("Bad address space in addrspacecast");
    case ADDRESS_SPACE_GLOBAL:
      Opc = TM.is64Bit() ? NVPTX::cvta_global_64 : NVPTX::cvta_global;
      break;
    case ADDRESS_SPACE_SHARED:
      Opc = TM.is64Bit() ? NVPTX::cvta_shared_64 : NVPTX::cvta_shared;
      break;
    case ADDRESS_SPACE_SHARED_CLUSTER:
      if (!TM.is64Bit())
        report_fatal_error(
            "Shared cluster address space is only supported in 64-bit mode");
      Opc = NVPTX::cvta_shared_cluster_64;
      break;
    case ADDRESS_SPACE_CONST:
      Opc = TM.is64Bit() ? NVPTX::cvta_const_64 : NVPTX::cvta_const;
      break;
    case ADDRESS_SPACE_LOCAL:
      Opc = TM.is64Bit() ? NVPTX::cvta_local_64 : NVPTX::cvta_local;
      break;
    case ADDRESS_SPACE_PARAM:
      Opc = TM.is64Bit() ? NVPTX::cvta_param_64 : NVPTX::cvta_param;
      break;
    }
    ReplaceNode(N, CurDAG->getMachineNode(Opc, DL, N->getValueType(0), Src));
    return;
  } else {
    // Generic to specific
    if (SrcAddrSpace != 0)
      report_fatal_error("Cannot cast between two non-generic address spaces");
    unsigned Opc;
    switch (DstAddrSpace) {
    default: report_fatal_error("Bad address space in addrspacecast");
    case ADDRESS_SPACE_GLOBAL:
      Opc = TM.is64Bit() ? NVPTX::cvta_to_global_64 : NVPTX::cvta_to_global;
      break;
    case ADDRESS_SPACE_SHARED:
      Opc = TM.is64Bit() ? NVPTX::cvta_to_shared_64 : NVPTX::cvta_to_shared;
      break;
    case ADDRESS_SPACE_SHARED_CLUSTER:
      if (!TM.is64Bit())
        report_fatal_error(
            "Shared cluster address space is only supported in 64-bit mode");
      Opc = NVPTX::cvta_to_shared_cluster_64;
      break;
    case ADDRESS_SPACE_CONST:
      Opc = TM.is64Bit() ? NVPTX::cvta_to_const_64 : NVPTX::cvta_to_const;
      break;
    case ADDRESS_SPACE_LOCAL:
      Opc = TM.is64Bit() ? NVPTX::cvta_to_local_64 : NVPTX::cvta_to_local;
      break;
    case ADDRESS_SPACE_PARAM:
      Opc = TM.is64Bit() ? NVPTX::cvta_to_param_64 : NVPTX::cvta_to_param;
      break;
    }

    SDNode *CVTA = CurDAG->getMachineNode(Opc, DL, N->getValueType(0), Src);
    if (TM.is64Bit() && TM.getPointerSizeInBits(DstAddrSpace) == 32) {
      SDValue CvtNone =
          CurDAG->getTargetConstant(NVPTX::PTXCvtMode::NONE, DL, MVT::i32);
      CVTA = CurDAG->getMachineNode(NVPTX::CVT_u32_u64, DL, MVT::i32,
                                    SDValue(CVTA, 0), CvtNone);
    }

    ReplaceNode(N, CVTA);
    return;
  }
}

// Helper function template to reduce amount of boilerplate code for
// opcode selection.
static std::optional<unsigned>
pickOpcodeForVT(MVT::SimpleValueType VT, std::optional<unsigned> Opcode_i16,
                std::optional<unsigned> Opcode_i32,
                std::optional<unsigned> Opcode_i64) {
  switch (VT) {
  case MVT::f16:
  case MVT::i16:
  case MVT::bf16:
    return Opcode_i16;
  case MVT::v2f16:
  case MVT::v2bf16:
  case MVT::v2i16:
  case MVT::v4i8:
  case MVT::i32:
  case MVT::f32:
    return Opcode_i32;
  case MVT::v2f32:
  case MVT::i64:
  case MVT::f64:
    return Opcode_i64;
  default:
    return std::nullopt;
  }
}

static inline bool isAddLike(const SDValue V) {
  return V.getOpcode() == ISD::ADD ||
         (V->getOpcode() == ISD::OR && V->getFlags().hasDisjoint());
}

// selectBaseADDR - Match a dag node which will serve as the base address for an
// ADDR operand pair.
static SDValue selectBaseADDR(SDValue N, SelectionDAG *DAG) {
  if (const auto *GA = dyn_cast<GlobalAddressSDNode>(N))
    return DAG->getTargetGlobalAddress(GA->getGlobal(), SDLoc(N),
                                       GA->getValueType(0), GA->getOffset(),
                                       GA->getTargetFlags());
  if (const auto *ES = dyn_cast<ExternalSymbolSDNode>(N))
    return DAG->getTargetExternalSymbol(ES->getSymbol(), ES->getValueType(0),
                                        ES->getTargetFlags());
  if (const auto *FIN = dyn_cast<FrameIndexSDNode>(N))
    return DAG->getTargetFrameIndex(FIN->getIndex(), FIN->getValueType(0));

  return N;
}

static SDValue accumulateOffset(SDValue &Addr, SDLoc DL, SelectionDAG *DAG) {
  APInt AccumulatedOffset(64u, 0);
  while (isAddLike(Addr)) {
    const auto *CN = dyn_cast<ConstantSDNode>(Addr.getOperand(1));
    if (!CN)
      break;

    const APInt CI = CN->getAPIntValue().sext(64);
    if (!(CI + AccumulatedOffset).isSignedIntN(32))
      break;

    AccumulatedOffset += CI;
    Addr = Addr->getOperand(0);
  }
  return DAG->getSignedTargetConstant(AccumulatedOffset.getSExtValue(), DL,
                                      MVT::i32);
}

static std::pair<SDValue, SDValue> selectADDR(SDValue Addr, SelectionDAG *DAG) {
  SDValue Offset = accumulateOffset(Addr, SDLoc(Addr), DAG);
  SDValue Base = selectBaseADDR(Addr, DAG);
  return {Base, Offset};
}

// Select a pair of operands which represent a valid PTX address, this could be
// one of the following things:
//  - [var] - Offset is simply set to 0
//  - [reg] - Offset is simply set to 0
//  - [reg+immOff]
//  - [var+immOff]
// Note that immOff must fit into a 32-bit signed integer.
bool NVPTXDAGToDAGISel::SelectADDR(SDValue Addr, SDValue &Base,
                                   SDValue &Offset) {
  std::tie(Base, Offset) = selectADDR(Addr, CurDAG);
  return true;
}

bool NVPTXDAGToDAGISel::tryLoad(SDNode *N) {
  MemSDNode *LD = cast<MemSDNode>(N);
  assert(LD->readMem() && "Expected load");

  // do not support pre/post inc/dec
  const LoadSDNode *PlainLoad = dyn_cast<LoadSDNode>(LD);
  if (PlainLoad && PlainLoad->isIndexed())
    return false;

  const EVT LoadedEVT = LD->getMemoryVT();
  if (!LoadedEVT.isSimple())
    return false;
  const MVT LoadedVT = LoadedEVT.getSimpleVT();

  // Address Space Setting
  const auto CodeAddrSpace = getAddrSpace(LD);
  if (canLowerToLDG(*LD, *Subtarget, CodeAddrSpace))
    return tryLDG(LD);

  SDLoc DL(LD);
  SDValue Chain = N->getOperand(0);
  const auto [Ordering, Scope] = insertMemoryInstructionFence(DL, Chain, LD);

  const unsigned FromTypeWidth = LoadedVT.getSizeInBits();

  // Vector Setting
  const unsigned FromType =
      (PlainLoad && (PlainLoad->getExtensionType() == ISD::SEXTLOAD))
          ? NVPTX::PTXLdStInstCode::Signed
          : NVPTX::PTXLdStInstCode::Untyped;

  assert(isPowerOf2_32(FromTypeWidth) && FromTypeWidth >= 8 &&
         FromTypeWidth <= 128 && "Invalid width for load");

  // Create the machine instruction DAG
  const auto [Base, Offset] = selectADDR(N->getOperand(1), CurDAG);
  SDValue Ops[] = {getI32Imm(Ordering, DL),
                   getI32Imm(Scope, DL),
                   getI32Imm(CodeAddrSpace, DL),
                   getI32Imm(FromType, DL),
                   getI32Imm(FromTypeWidth, DL),
                   Base,
                   Offset,
                   Chain};

  const MVT::SimpleValueType TargetVT = LD->getSimpleValueType(0).SimpleTy;
  const std::optional<unsigned> Opcode =
      pickOpcodeForVT(TargetVT, NVPTX::LD_i16, NVPTX::LD_i32, NVPTX::LD_i64);
  if (!Opcode)
    return false;

  SDNode *NVPTXLD = CurDAG->getMachineNode(*Opcode, DL, LD->getVTList(), Ops);
  if (!NVPTXLD)
    return false;

  MachineMemOperand *MemRef = LD->getMemOperand();
  CurDAG->setNodeMemRefs(cast<MachineSDNode>(NVPTXLD), {MemRef});

  ReplaceNode(LD, NVPTXLD);
  return true;
}

static unsigned getLoadStoreVectorNumElts(SDNode *N) {
  switch (N->getOpcode()) {
  case NVPTXISD::LoadV2:
  case NVPTXISD::StoreV2:
    return 2;
  case NVPTXISD::LoadV4:
  case NVPTXISD::StoreV4:
    return 4;
  case NVPTXISD::LoadV8:
  case NVPTXISD::StoreV8:
    return 8;
  default:
    llvm_unreachable("Unexpected opcode");
  }
}

bool NVPTXDAGToDAGISel::tryLoadVector(SDNode *N) {
  MemSDNode *LD = cast<MemSDNode>(N);
  const EVT MemEVT = LD->getMemoryVT();
  if (!MemEVT.isSimple())
    return false;
  const MVT MemVT = MemEVT.getSimpleVT();

  // Address Space Setting
  const auto CodeAddrSpace = getAddrSpace(LD);
  if (canLowerToLDG(*LD, *Subtarget, CodeAddrSpace))
    return tryLDG(LD);

  const MVT EltVT = LD->getSimpleValueType(0);
  SDLoc DL(LD);
  SDValue Chain = LD->getChain();
  const auto [Ordering, Scope] = insertMemoryInstructionFence(DL, Chain, LD);

  // Type Setting: fromType + fromTypeWidth
  //
  // Sign   : ISD::SEXTLOAD
  // Unsign : ISD::ZEXTLOAD, ISD::NON_EXTLOAD or ISD::EXTLOAD and the
  //          type is integer
  // Float  : ISD::NON_EXTLOAD or ISD::EXTLOAD and the type is float
  // Read at least 8 bits (predicates are stored as 8-bit values)
  // The last operand holds the original LoadSDNode::getExtensionType() value
  const unsigned TotalWidth = MemVT.getSizeInBits();
  const unsigned ExtensionType =
      N->getConstantOperandVal(N->getNumOperands() - 1);
  const unsigned FromType = (ExtensionType == ISD::SEXTLOAD)
                                ? NVPTX::PTXLdStInstCode::Signed
                                : NVPTX::PTXLdStInstCode::Untyped;

  const unsigned FromTypeWidth = TotalWidth / getLoadStoreVectorNumElts(N);

  assert(!(EltVT.isVector() && ExtensionType != ISD::NON_EXTLOAD));
  assert(isPowerOf2_32(FromTypeWidth) && FromTypeWidth >= 8 &&
         FromTypeWidth <= 128 && TotalWidth <= 256 && "Invalid width for load");

  const auto [Base, Offset] = selectADDR(N->getOperand(1), CurDAG);
  SDValue Ops[] = {getI32Imm(Ordering, DL),
                   getI32Imm(Scope, DL),
                   getI32Imm(CodeAddrSpace, DL),
                   getI32Imm(FromType, DL),
                   getI32Imm(FromTypeWidth, DL),
                   Base,
                   Offset,
                   Chain};

  std::optional<unsigned> Opcode;
  switch (N->getOpcode()) {
  default:
    llvm_unreachable("Unexpected opcode");
  case NVPTXISD::LoadV2:
    Opcode = pickOpcodeForVT(EltVT.SimpleTy, NVPTX::LDV_i16_v2,
                             NVPTX::LDV_i32_v2, NVPTX::LDV_i64_v2);
    break;
  case NVPTXISD::LoadV4:
    Opcode = pickOpcodeForVT(EltVT.SimpleTy, NVPTX::LDV_i16_v4,
                             NVPTX::LDV_i32_v4, NVPTX::LDV_i64_v4);
    break;
  case NVPTXISD::LoadV8:
    Opcode = pickOpcodeForVT(EltVT.SimpleTy, {/* no v8i16 */},
                             NVPTX::LDV_i32_v8, {/* no v8i64 */});
    break;
  }
  if (!Opcode)
    return false;

  SDNode *NVPTXLD = CurDAG->getMachineNode(*Opcode, DL, LD->getVTList(), Ops);

  MachineMemOperand *MemRef = LD->getMemOperand();
  CurDAG->setNodeMemRefs(cast<MachineSDNode>(NVPTXLD), {MemRef});

  ReplaceNode(LD, NVPTXLD);
  return true;
}

bool NVPTXDAGToDAGISel::tryLDG(MemSDNode *LD) {
  const EVT LoadedEVT = LD->getMemoryVT();
  if (!LoadedEVT.isSimple())
    return false;
  const MVT LoadedVT = LoadedEVT.getSimpleVT();

  SDLoc DL(LD);

  const unsigned TotalWidth = LoadedVT.getSizeInBits();
  unsigned ExtensionType;
  unsigned NumElts;
  if (const auto *Load = dyn_cast<LoadSDNode>(LD)) {
    ExtensionType = Load->getExtensionType();
    NumElts = 1;
  } else {
    ExtensionType = LD->getConstantOperandVal(LD->getNumOperands() - 1);
    NumElts = getLoadStoreVectorNumElts(LD);
  }
  const unsigned FromType = (ExtensionType == ISD::SEXTLOAD)
                                ? NVPTX::PTXLdStInstCode::Signed
                                : NVPTX::PTXLdStInstCode::Untyped;

  const unsigned FromTypeWidth = TotalWidth / NumElts;

  assert(!(LD->getSimpleValueType(0).isVector() &&
           ExtensionType != ISD::NON_EXTLOAD));
  assert(isPowerOf2_32(FromTypeWidth) && FromTypeWidth >= 8 &&
         FromTypeWidth <= 128 && TotalWidth <= 256 && "Invalid width for load");

  const auto [Base, Offset] = selectADDR(LD->getOperand(1), CurDAG);
  SDValue Ops[] = {getI32Imm(FromType, DL), getI32Imm(FromTypeWidth, DL), Base,
                   Offset, LD->getChain()};

  const MVT::SimpleValueType TargetVT = LD->getSimpleValueType(0).SimpleTy;
  std::optional<unsigned> Opcode;
  switch (LD->getOpcode()) {
  default:
    llvm_unreachable("Unexpected opcode");
  case ISD::LOAD:
    Opcode = pickOpcodeForVT(TargetVT, NVPTX::LD_GLOBAL_NC_i16,
                             NVPTX::LD_GLOBAL_NC_i32, NVPTX::LD_GLOBAL_NC_i64);
    break;
  case NVPTXISD::LoadV2:
    Opcode =
        pickOpcodeForVT(TargetVT, NVPTX::LD_GLOBAL_NC_v2i16,
                        NVPTX::LD_GLOBAL_NC_v2i32, NVPTX::LD_GLOBAL_NC_v2i64);
    break;
  case NVPTXISD::LoadV4:
    Opcode =
        pickOpcodeForVT(TargetVT, NVPTX::LD_GLOBAL_NC_v4i16,
                        NVPTX::LD_GLOBAL_NC_v4i32, NVPTX::LD_GLOBAL_NC_v4i64);
    break;
  case NVPTXISD::LoadV8:
    Opcode = pickOpcodeForVT(TargetVT, {/* no v8i16 */},
                             NVPTX::LD_GLOBAL_NC_v8i32, {/* no v8i64 */});
    break;
  }
  if (!Opcode)
    return false;

  SDNode *NVPTXLDG = CurDAG->getMachineNode(*Opcode, DL, LD->getVTList(), Ops);

  ReplaceNode(LD, NVPTXLDG);
  return true;
}

bool NVPTXDAGToDAGISel::tryLDU(SDNode *N) {
  auto *LD = cast<MemSDNode>(N);

  unsigned NumElts;
  switch (N->getOpcode()) {
  default:
    llvm_unreachable("Unexpected opcode");
  case ISD::INTRINSIC_W_CHAIN:
    NumElts = 1;
    break;
  case NVPTXISD::LDUV2:
    NumElts = 2;
    break;
  case NVPTXISD::LDUV4:
    NumElts = 4;
    break;
  }

  SDLoc DL(N);
  const unsigned FromTypeWidth = LD->getMemoryVT().getSizeInBits() / NumElts;
  const MVT::SimpleValueType TargetVT = LD->getSimpleValueType(0).SimpleTy;

  // If this is an LDU intrinsic, the address is the third operand. If its an
  // LDU SD node (from custom vector handling), then its the second operand
  SDValue Addr =
      LD->getOperand(LD->getOpcode() == ISD::INTRINSIC_W_CHAIN ? 2 : 1);

  const auto [Base, Offset] = selectADDR(Addr, CurDAG);
  SDValue Ops[] = {getI32Imm(FromTypeWidth, DL), Base, Offset, LD->getChain()};

  std::optional<unsigned> Opcode;
  switch (N->getOpcode()) {
  default:
    llvm_unreachable("Unexpected opcode");
  case ISD::INTRINSIC_W_CHAIN:
    Opcode = pickOpcodeForVT(TargetVT, NVPTX::LDU_GLOBAL_i16,
                             NVPTX::LDU_GLOBAL_i32, NVPTX::LDU_GLOBAL_i64);
    break;
  case NVPTXISD::LDUV2:
    Opcode = pickOpcodeForVT(TargetVT, NVPTX::LDU_GLOBAL_v2i16,
                             NVPTX::LDU_GLOBAL_v2i32, NVPTX::LDU_GLOBAL_v2i64);
    break;
  case NVPTXISD::LDUV4:
    Opcode = pickOpcodeForVT(TargetVT, NVPTX::LDU_GLOBAL_v4i16,
                             NVPTX::LDU_GLOBAL_v4i32, {/* no v4i64 */});
    break;
  }
  if (!Opcode)
    return false;

  SDNode *NVPTXLDU = CurDAG->getMachineNode(*Opcode, DL, LD->getVTList(), Ops);

  ReplaceNode(LD, NVPTXLDU);
  return true;
}

bool NVPTXDAGToDAGISel::tryStore(SDNode *N) {
  MemSDNode *ST = cast<MemSDNode>(N);
  assert(ST->writeMem() && "Expected store");
  StoreSDNode *PlainStore = dyn_cast<StoreSDNode>(ST);
  AtomicSDNode *AtomicStore = dyn_cast<AtomicSDNode>(ST);
  assert((PlainStore || AtomicStore) && "Expected store");

  // do not support pre/post inc/dec
  if (PlainStore && PlainStore->isIndexed())
    return false;

  const EVT StoreVT = ST->getMemoryVT();
  if (!StoreVT.isSimple())
    return false;

  // Address Space Setting
  const auto CodeAddrSpace = getAddrSpace(ST);

  SDLoc DL(ST);
  SDValue Chain = ST->getChain();
  const auto [Ordering, Scope] = insertMemoryInstructionFence(DL, Chain, ST);

  // Vector Setting
  const unsigned ToTypeWidth = StoreVT.getSimpleVT().getSizeInBits();

  // Create the machine instruction DAG
  SDValue Value = PlainStore ? PlainStore->getValue() : AtomicStore->getVal();

  assert(isPowerOf2_32(ToTypeWidth) && ToTypeWidth >= 8 && ToTypeWidth <= 128 &&
         "Invalid width for store");

  const auto [Base, Offset] = selectADDR(ST->getBasePtr(), CurDAG);
  SDValue Ops[] = {selectPossiblyImm(Value),
                   getI32Imm(Ordering, DL),
                   getI32Imm(Scope, DL),
                   getI32Imm(CodeAddrSpace, DL),
                   getI32Imm(ToTypeWidth, DL),
                   Base,
                   Offset,
                   Chain};

  const std::optional<unsigned> Opcode =
      pickOpcodeForVT(Value.getSimpleValueType().SimpleTy, NVPTX::ST_i16,
                      NVPTX::ST_i32, NVPTX::ST_i64);
  if (!Opcode)
    return false;

  SDNode *NVPTXST = CurDAG->getMachineNode(*Opcode, DL, MVT::Other, Ops);

  if (!NVPTXST)
    return false;

  MachineMemOperand *MemRef = ST->getMemOperand();
  CurDAG->setNodeMemRefs(cast<MachineSDNode>(NVPTXST), {MemRef});
  ReplaceNode(ST, NVPTXST);
  return true;
}

bool NVPTXDAGToDAGISel::tryStoreVector(SDNode *N) {
  MemSDNode *ST = cast<MemSDNode>(N);
  const EVT StoreVT = ST->getMemoryVT();
  assert(StoreVT.isSimple() && "Store value is not simple");

  // Address Space Setting
  const auto CodeAddrSpace = getAddrSpace(ST);
  if (CodeAddrSpace == NVPTX::AddressSpace::Const) {
    report_fatal_error("Cannot store to pointer that points to constant "
                       "memory space");
  }

  SDLoc DL(ST);
  SDValue Chain = ST->getChain();
  const auto [Ordering, Scope] = insertMemoryInstructionFence(DL, Chain, ST);

  // Type Setting: toType + toTypeWidth
  // - for integer type, always use 'u'
  const unsigned TotalWidth = StoreVT.getSimpleVT().getSizeInBits();

  const unsigned NumElts = getLoadStoreVectorNumElts(ST);

  SmallVector<SDValue, 16> Ops;
  for (auto &V : ST->ops().slice(1, NumElts))
    Ops.push_back(selectPossiblyImm(V));
  SDValue Addr = N->getOperand(NumElts + 1);
  const unsigned ToTypeWidth = TotalWidth / NumElts;

  assert(isPowerOf2_32(ToTypeWidth) && ToTypeWidth >= 8 && ToTypeWidth <= 128 &&
         TotalWidth <= 256 && "Invalid width for store");

  const auto [Base, Offset] = selectADDR(Addr, CurDAG);
  Ops.append({getI32Imm(Ordering, DL), getI32Imm(Scope, DL),
              getI32Imm(CodeAddrSpace, DL), getI32Imm(ToTypeWidth, DL), Base,
              Offset, Chain});

  const MVT::SimpleValueType EltVT =
      ST->getOperand(1).getSimpleValueType().SimpleTy;
  std::optional<unsigned> Opcode;
  switch (ST->getOpcode()) {
  default:
    return false;
  case NVPTXISD::StoreV2:
    Opcode = pickOpcodeForVT(EltVT, NVPTX::STV_i16_v2, NVPTX::STV_i32_v2,
                             NVPTX::STV_i64_v2);
    break;
  case NVPTXISD::StoreV4:
    Opcode = pickOpcodeForVT(EltVT, NVPTX::STV_i16_v4, NVPTX::STV_i32_v4,
                             NVPTX::STV_i64_v4);
    break;
  case NVPTXISD::StoreV8:
    Opcode = pickOpcodeForVT(EltVT, {/* no v8i16 */}, NVPTX::STV_i32_v8,
                             {/* no v8i64 */});
    break;
  }

  if (!Opcode)
    return false;

  SDNode *NVPTXST = CurDAG->getMachineNode(*Opcode, DL, MVT::Other, Ops);

  MachineMemOperand *MemRef = ST->getMemOperand();
  CurDAG->setNodeMemRefs(cast<MachineSDNode>(NVPTXST), {MemRef});

  ReplaceNode(ST, NVPTXST);
  return true;
}

/// SelectBFE - Look for instruction sequences that can be made more efficient
/// by using the 'bfe' (bit-field extract) PTX instruction
bool NVPTXDAGToDAGISel::tryBFE(SDNode *N) {
  SDLoc DL(N);
  SDValue LHS = N->getOperand(0);
  SDValue RHS = N->getOperand(1);
  SDValue Len;
  SDValue Start;
  SDValue Val;
  bool IsSigned = false;

  if (N->getOpcode() == ISD::AND) {
    // Canonicalize the operands
    // We want 'and %val, %mask'
    if (isa<ConstantSDNode>(LHS) && !isa<ConstantSDNode>(RHS)) {
      std::swap(LHS, RHS);
    }

    ConstantSDNode *Mask = dyn_cast<ConstantSDNode>(RHS);
    if (!Mask) {
      // We need a constant mask on the RHS of the AND
      return false;
    }

    // Extract the mask bits
    uint64_t MaskVal = Mask->getZExtValue();
    if (!isMask_64(MaskVal)) {
      // We *could* handle shifted masks here, but doing so would require an
      // 'and' operation to fix up the low-order bits so we would trade
      // shr+and for bfe+and, which has the same throughput
      return false;
    }

    // How many bits are in our mask?
    int64_t NumBits = countr_one(MaskVal);
    Len = CurDAG->getTargetConstant(NumBits, DL, MVT::i32);

    if (LHS.getOpcode() == ISD::SRL || LHS.getOpcode() == ISD::SRA) {
      // We have a 'srl/and' pair, extract the effective start bit and length
      Val = LHS.getNode()->getOperand(0);
      Start = LHS.getNode()->getOperand(1);
      ConstantSDNode *StartConst = dyn_cast<ConstantSDNode>(Start);
      if (StartConst) {
        uint64_t StartVal = StartConst->getZExtValue();
        // How many "good" bits do we have left?  "good" is defined here as bits
        // that exist in the original value, not shifted in.
        int64_t GoodBits = Start.getValueSizeInBits() - StartVal;
        if (NumBits > GoodBits) {
          // Do not handle the case where bits have been shifted in. In theory
          // we could handle this, but the cost is likely higher than just
          // emitting the srl/and pair.
          return false;
        }
        Start = CurDAG->getTargetConstant(StartVal, DL, MVT::i32);
      } else {
        // Do not handle the case where the shift amount (can be zero if no srl
        // was found) is not constant. We could handle this case, but it would
        // require run-time logic that would be more expensive than just
        // emitting the srl/and pair.
        return false;
      }
    } else {
      // Do not handle the case where the LHS of the and is not a shift. While
      // it would be trivial to handle this case, it would just transform
      // 'and' -> 'bfe', but 'and' has higher-throughput.
      return false;
    }
  } else if (N->getOpcode() == ISD::SRL || N->getOpcode() == ISD::SRA) {
    if (LHS->getOpcode() == ISD::AND) {
      ConstantSDNode *ShiftCnst = dyn_cast<ConstantSDNode>(RHS);
      if (!ShiftCnst) {
        // Shift amount must be constant
        return false;
      }

      uint64_t ShiftAmt = ShiftCnst->getZExtValue();

      SDValue AndLHS = LHS->getOperand(0);
      SDValue AndRHS = LHS->getOperand(1);

      // Canonicalize the AND to have the mask on the RHS
      if (isa<ConstantSDNode>(AndLHS)) {
        std::swap(AndLHS, AndRHS);
      }

      ConstantSDNode *MaskCnst = dyn_cast<ConstantSDNode>(AndRHS);
      if (!MaskCnst) {
        // Mask must be constant
        return false;
      }

      uint64_t MaskVal = MaskCnst->getZExtValue();
      uint64_t NumZeros;
      uint64_t NumBits;
      if (isMask_64(MaskVal)) {
        NumZeros = 0;
        // The number of bits in the result bitfield will be the number of
        // trailing ones (the AND) minus the number of bits we shift off
        NumBits = llvm::countr_one(MaskVal) - ShiftAmt;
      } else if (isShiftedMask_64(MaskVal)) {
        NumZeros = llvm::countr_zero(MaskVal);
        unsigned NumOnes = llvm::countr_one(MaskVal >> NumZeros);
        // The number of bits in the result bitfield will be the number of
        // trailing zeros plus the number of set bits in the mask minus the
        // number of bits we shift off
        NumBits = NumZeros + NumOnes - ShiftAmt;
      } else {
        // This is not a mask we can handle
        return false;
      }

      if (ShiftAmt < NumZeros) {
        // Handling this case would require extra logic that would make this
        // transformation non-profitable
        return false;
      }

      Val = AndLHS;
      Start = CurDAG->getTargetConstant(ShiftAmt, DL, MVT::i32);
      Len = CurDAG->getTargetConstant(NumBits, DL, MVT::i32);

      // If pre-shift AND includes the sign bit in the bitfield, we must use
      // signed BFE to replicate that bit during bitfield extraction. If the
      // sign bit is not part of the mask, unsigned BFE will zero out upper bits
      // of the result
      if (N->getOpcode() == ISD::SRA)
        IsSigned = (ShiftAmt + NumBits) == Val.getValueSizeInBits();
    } else if (LHS->getOpcode() == ISD::SHL) {
      // Here, we have a pattern like:
      //
      // (sra (shl val, NN), MM)
      // or
      // (srl (shl val, NN), MM)
      //
      // If MM >= NN, we can efficiently optimize this with bfe
      Val = LHS->getOperand(0);

      SDValue ShlRHS = LHS->getOperand(1);
      ConstantSDNode *ShlCnst = dyn_cast<ConstantSDNode>(ShlRHS);
      if (!ShlCnst) {
        // Shift amount must be constant
        return false;
      }
      uint64_t InnerShiftAmt = ShlCnst->getZExtValue();

      SDValue ShrRHS = RHS;
      ConstantSDNode *ShrCnst = dyn_cast<ConstantSDNode>(ShrRHS);
      if (!ShrCnst) {
        // Shift amount must be constant
        return false;
      }
      uint64_t OuterShiftAmt = ShrCnst->getZExtValue();

      // To avoid extra codegen and be profitable, we need Outer >= Inner
      if (OuterShiftAmt < InnerShiftAmt) {
        return false;
      }

      // If the outer shift is more than the type size, we have no bitfield to
      // extract (since we also check that the inner shift is <= the outer shift
      // then this also implies that the inner shift is < the type size)
      if (OuterShiftAmt >= Val.getValueSizeInBits()) {
        return false;
      }

      Start = CurDAG->getTargetConstant(OuterShiftAmt - InnerShiftAmt, DL,
                                        MVT::i32);
      Len = CurDAG->getTargetConstant(Val.getValueSizeInBits() - OuterShiftAmt,
                                      DL, MVT::i32);

      if (N->getOpcode() == ISD::SRA) {
        // If we have a arithmetic right shift, we need to use the signed bfe
        // variant
        IsSigned = true;
      }
    } else {
      // No can do...
      return false;
    }
  } else {
    // No can do...
    return false;
  }


  unsigned Opc;
  // For the BFE operations we form here from "and" and "srl", always use the
  // unsigned variants.
  if (Val.getValueType() == MVT::i32) {
    if (IsSigned) {
      Opc = NVPTX::BFE_S32rii;
    } else {
      Opc = NVPTX::BFE_U32rii;
    }
  } else if (Val.getValueType() == MVT::i64) {
    if (IsSigned) {
      Opc = NVPTX::BFE_S64rii;
    } else {
      Opc = NVPTX::BFE_U64rii;
    }
  } else {
    // We cannot handle this type
    return false;
  }

  SDValue Ops[] = {
    Val, Start, Len
  };

  ReplaceNode(N, CurDAG->getMachineNode(Opc, DL, N->getVTList(), Ops));
  return true;
}

// Select bf16/bf16v2 FADD, FSUB, FMUL as fma on targets with only fma
bool NVPTXDAGToDAGISel::tryBF16ArithToFMA(SDNode *N) {
  EVT VT = SDValue(N, 0).getValueType();
  if (VT.getScalarType() != MVT::bf16)
    return false;

  const NVPTXSubtarget *STI = TM.getSubtargetImpl();
  if (STI->hasNativeBF16Support(N->getOpcode()))
    return false;

  const bool IsVec = VT.isVector();
  assert(!IsVec || VT.getVectorNumElements() == 2);
  SDLoc DL(N);
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  SmallVector<SDValue, 3> Operands;
  auto GetConstant = [&](float Value) -> SDValue {
    // BF16 immediates must be legalized to integer register values
    APFloat APF(Value);
    bool LosesInfo;
    APF.convert(APFloat::BFloat(), APFloat::rmNearestTiesToEven, &LosesInfo);
    assert(!LosesInfo);
    if (IsVec) {
      auto API = APF.bitcastToAPInt();
      API = API.concat(API);
      auto Const = CurDAG->getTargetConstant(API, DL, MVT::i32);
      return SDValue(CurDAG->getMachineNode(NVPTX::MOV_B32_i, DL, VT, Const),
                     0);
    }
    auto Const = CurDAG->getTargetConstantFP(APF, DL, VT);
    return SDValue(CurDAG->getMachineNode(NVPTX::MOV_BF16_i, DL, VT, Const), 0);
  };

  switch (N->getOpcode()) {
  case ISD::FADD:
    // add(a, b) -> fma(a, 1.0, b)
    Operands = {N0, GetConstant(1.0), N1};
    break;
  case ISD::FSUB:
    // sub(a, b) -> fma(b, -1.0, a)
    Operands = {N1, GetConstant(-1.0), N0};
    break;
  case ISD::FMUL:
    // mul(a, b) -> fma(a, b, -0.0)
    // NOTE: The identity is -0, not 0, because -0 + 0 == 0 for floats
    Operands = {N0, N1, GetConstant(-0.0)};
    break;
  default:
    llvm_unreachable("Unexpected opcode");
  };

  int Opcode = IsVec ? NVPTX::FMA_BF16x2rrr : NVPTX::FMA_BF16rrr;
  MachineSDNode *FMA = CurDAG->getMachineNode(Opcode, DL, VT, Operands);
  ReplaceNode(N, FMA);
  return true;
}

SDValue NVPTXDAGToDAGISel::selectPossiblyImm(SDValue V) {
  if (V.getOpcode() == ISD::BITCAST)
    V = V.getOperand(0);

  if (auto *CN = dyn_cast<ConstantSDNode>(V))
    return CurDAG->getTargetConstant(CN->getAPIntValue(), SDLoc(V),
                                     V.getValueType());
  if (auto *CN = dyn_cast<ConstantFPSDNode>(V))
    return CurDAG->getTargetConstantFP(CN->getValueAPF(), SDLoc(V),
                                       V.getValueType());
  return V;
}

/// SelectInlineAsmMemoryOperand - Implement addressing mode selection for
/// inline asm expressions.
bool NVPTXDAGToDAGISel::SelectInlineAsmMemoryOperand(
    const SDValue &Op, InlineAsm::ConstraintCode ConstraintID,
    std::vector<SDValue> &OutOps) {
  switch (ConstraintID) {
  default:
    return true;
  case InlineAsm::ConstraintCode::m: { // memory
    const auto [Base, Offset] = selectADDR(Op, CurDAG);
    OutOps.push_back(Base);
    OutOps.push_back(Offset);
    return false;
  }
  }
  return true;
}

void NVPTXDAGToDAGISel::SelectV2I64toI128(SDNode *N) {
  // Lower a CopyToReg with two 64-bit inputs
  // Dst:i128, lo:i64, hi:i64
  //
  // CopyToReg Dst, lo, hi;
  //
  // ==>
  //
  // tmp = V2I64toI128 {lo, hi};
  // CopyToReg Dst, tmp;
  SDValue Dst = N->getOperand(1);
  SDValue Lo = N->getOperand(2);
  SDValue Hi = N->getOperand(3);

  SDLoc DL(N);
  SDNode *Mov =
      CurDAG->getMachineNode(NVPTX::V2I64toI128, DL, MVT::i128, {Lo, Hi});

  SmallVector<SDValue, 4> NewOps(N->getNumOperands() - 1);
  NewOps[0] = N->getOperand(0);
  NewOps[1] = Dst;
  NewOps[2] = SDValue(Mov, 0);
  if (N->getNumOperands() == 5)
    NewOps[3] = N->getOperand(4);
  SDValue NewValue = CurDAG->getNode(ISD::CopyToReg, DL, SmallVector<EVT>(N->values()), NewOps);

  ReplaceNode(N, NewValue.getNode());
}

void NVPTXDAGToDAGISel::SelectI128toV2I64(SDNode *N) {
  // Lower CopyFromReg from a 128-bit regs to two 64-bit regs
  // Dst:i128, Src:i128
  //
  // {lo, hi} = CopyFromReg Src
  //
  // ==>
  //
  // {lo, hi} = I128toV2I64 Src
  //
  SDValue Ch = N->getOperand(0);
  SDValue Src = N->getOperand(1);
  SDValue Glue = N->getOperand(2);
  SDLoc DL(N);

  // Add Glue and Ch to the operands and results to avoid break the execution
  // order
  SDNode *Mov = CurDAG->getMachineNode(
      NVPTX::I128toV2I64, DL,
      {MVT::i64, MVT::i64, Ch.getValueType(), Glue.getValueType()},
      {Src, Ch, Glue});

  ReplaceNode(N, Mov);
}

bool NVPTXDAGToDAGISel::tryFence(SDNode *N) {
  SDLoc DL(N);
  assert(N->getOpcode() == ISD::ATOMIC_FENCE);
  unsigned int FenceOp =
      getFenceOp(NVPTX::Ordering(N->getConstantOperandVal(1)),
                 Scopes[N->getConstantOperandVal(2)], Subtarget);
  SDValue Chain = N->getOperand(0);
  SDNode *FenceNode = CurDAG->getMachineNode(FenceOp, DL, MVT::Other, Chain);
  ReplaceNode(N, FenceNode);
  return true;
}

NVPTXScopes::NVPTXScopes(LLVMContext &C) {
  Scopes[C.getOrInsertSyncScopeID("singlethread")] = NVPTX::Scope::Thread;
  Scopes[C.getOrInsertSyncScopeID("")] = NVPTX::Scope::System;
  Scopes[C.getOrInsertSyncScopeID("block")] = NVPTX::Scope::Block;
  Scopes[C.getOrInsertSyncScopeID("cluster")] = NVPTX::Scope::Cluster;
  Scopes[C.getOrInsertSyncScopeID("device")] = NVPTX::Scope::Device;
}

NVPTX::Scope NVPTXScopes::operator[](SyncScope::ID ID) const {
  if (Scopes.empty())
    llvm_unreachable("NVPTX Scopes must be initialized before calling "
                     "NVPTXScopes::operator[]");

  auto S = Scopes.find(ID);
  if (S == Scopes.end()) {
    // TODO:
    // - Add API to LLVMContext to get the name of a single scope.
    // - Use that API here to print an error containing the name
    //   of this Unknown ID.
    report_fatal_error(formatv("Could not find scope ID={}.", int(ID)));
  }
  return S->second;
}

bool NVPTXScopes::empty() const { return Scopes.size() == 0; }

#define CP_ASYNC_BULK_TENSOR_OPCODE(dir, dim, mode, is_s32, suffix)            \
  (is_s32                                                                      \
       ? NVPTX::CP_ASYNC_BULK_TENSOR_##dir##_##dim##_SHARED32_##mode##suffix   \
       : NVPTX::CP_ASYNC_BULK_TENSOR_##dir##_##dim##_##mode##suffix)

#define GET_CP_ASYNC_BULK_TENSOR_OPCODE_S2G_RED(dim, mode, is_ch, is_s32)      \
  (is_ch ? (CP_ASYNC_BULK_TENSOR_OPCODE(RED, dim, mode, is_s32, _CH))          \
         : (CP_ASYNC_BULK_TENSOR_OPCODE(RED, dim, mode, is_s32, )))

#define GET_CP_ASYNC_BULK_TENSOR_OPCODE_G2S(dim, mode, is_mc, is_ch, is_s32)   \
  [&]() -> auto {                                                              \
    if (is_mc && is_ch)                                                        \
      return CP_ASYNC_BULK_TENSOR_OPCODE(G2S, dim, mode, is_s32, _MC_CH);      \
    if (is_ch)                                                                 \
      return CP_ASYNC_BULK_TENSOR_OPCODE(G2S, dim, mode, is_s32, _CH);         \
    if (is_mc)                                                                 \
      return CP_ASYNC_BULK_TENSOR_OPCODE(G2S, dim, mode, is_s32, _MC);         \
    return CP_ASYNC_BULK_TENSOR_OPCODE(G2S, dim, mode, is_s32, );              \
  }()

static unsigned GetCpAsyncBulkTensorS2GReductionOpcode(size_t Dim,
                                                       bool IsShared32,
                                                       bool IsCacheHint,
                                                       bool IsIm2Col) {
  if (IsIm2Col) {
    switch (Dim) {
    case 3:
      return GET_CP_ASYNC_BULK_TENSOR_OPCODE_S2G_RED(3D, IM2COL, IsCacheHint,
                                                     IsShared32);
    case 4:
      return GET_CP_ASYNC_BULK_TENSOR_OPCODE_S2G_RED(4D, IM2COL, IsCacheHint,
                                                     IsShared32);
    case 5:
      return GET_CP_ASYNC_BULK_TENSOR_OPCODE_S2G_RED(5D, IM2COL, IsCacheHint,
                                                     IsShared32);
    default:
      llvm_unreachable("Invalid Dimension in im2col mode for "
                       "GetCpAsyncBulkTensorS2GReductionOpcode.");
    }
  } else {
    switch (Dim) {
    case 1:
      return GET_CP_ASYNC_BULK_TENSOR_OPCODE_S2G_RED(1D, TILE, IsCacheHint,
                                                     IsShared32);
    case 2:
      return GET_CP_ASYNC_BULK_TENSOR_OPCODE_S2G_RED(2D, TILE, IsCacheHint,
                                                     IsShared32);
    case 3:
      return GET_CP_ASYNC_BULK_TENSOR_OPCODE_S2G_RED(3D, TILE, IsCacheHint,
                                                     IsShared32);
    case 4:
      return GET_CP_ASYNC_BULK_TENSOR_OPCODE_S2G_RED(4D, TILE, IsCacheHint,
                                                     IsShared32);
    case 5:
      return GET_CP_ASYNC_BULK_TENSOR_OPCODE_S2G_RED(5D, TILE, IsCacheHint,
                                                     IsShared32);
    default:
      llvm_unreachable("Invalid Dimension in tile mode for "
                       "GetCpAsyncBulkTensorS2GReductionOpcode.");
    }
  }
}

static unsigned GetCpAsyncBulkTensorG2SOpcode(size_t Dim, bool IsShared32,
                                              bool IsMultiCast,
                                              bool IsCacheHint, bool IsIm2Col) {
  if (IsIm2Col) {
    switch (Dim) {
    case 3:
      return GET_CP_ASYNC_BULK_TENSOR_OPCODE_G2S(3D, IM2COL, IsMultiCast,
                                                 IsCacheHint, IsShared32);
    case 4:
      return GET_CP_ASYNC_BULK_TENSOR_OPCODE_G2S(4D, IM2COL, IsMultiCast,
                                                 IsCacheHint, IsShared32);
    case 5:
      return GET_CP_ASYNC_BULK_TENSOR_OPCODE_G2S(5D, IM2COL, IsMultiCast,
                                                 IsCacheHint, IsShared32);
    default:
      llvm_unreachable("Invalid Dimension in im2col mode for "
                       "GetCpAsyncBulkTensorG2SOpcode.");
    }
  } else {
    switch (Dim) {
    case 1:
      return GET_CP_ASYNC_BULK_TENSOR_OPCODE_G2S(1D, TILE, IsMultiCast,
                                                 IsCacheHint, IsShared32);
    case 2:
      return GET_CP_ASYNC_BULK_TENSOR_OPCODE_G2S(2D, TILE, IsMultiCast,
                                                 IsCacheHint, IsShared32);
    case 3:
      return GET_CP_ASYNC_BULK_TENSOR_OPCODE_G2S(3D, TILE, IsMultiCast,
                                                 IsCacheHint, IsShared32);
    case 4:
      return GET_CP_ASYNC_BULK_TENSOR_OPCODE_G2S(4D, TILE, IsMultiCast,
                                                 IsCacheHint, IsShared32);
    case 5:
      return GET_CP_ASYNC_BULK_TENSOR_OPCODE_G2S(5D, TILE, IsMultiCast,
                                                 IsCacheHint, IsShared32);
    default:
      llvm_unreachable(
          "Invalid Dimension in tile mode for GetCpAsyncBulkTensorG2SOpcode.");
    }
  }
}

static size_t GetDimsFromIntrinsic(unsigned IID) {
  switch (IID) {
  case Intrinsic::nvvm_cp_async_bulk_tensor_g2s_im2col_3d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_prefetch_im2col_3d:
    return 3;
  case Intrinsic::nvvm_cp_async_bulk_tensor_g2s_im2col_4d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_prefetch_im2col_4d:
    return 4;
  case Intrinsic::nvvm_cp_async_bulk_tensor_g2s_im2col_5d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_prefetch_im2col_5d:
    return 5;
  default:
    llvm_unreachable("Invalid im2col intrinsic in GetDimsFromIntrinsic.");
  }
}

void NVPTXDAGToDAGISel::SelectCpAsyncBulkTensorG2SCommon(SDNode *N,
                                                         bool IsIm2Col) {
  // We have {Chain, Intrinsic-ID} followed by the actual intrisic args:
  // {dst, mbar, src, dims{d0...dN}, im2col_offsets{dims-2}
  // multicast, cache_hint,
  // multicast_flag, cache_hint_flag, cta_group_flag}
  // NumOperands = {Chain, IID} + {Actual intrinsic args}
  //             = {2}          + {8 + dims + im2col_offsets}
  size_t NumOps = N->getNumOperands();
  size_t NumDims = IsIm2Col ? GetDimsFromIntrinsic(N->getConstantOperandVal(1))
                            : (NumOps - 10);
  // Offsets is always 'NumDims - 2' and only for im2col mode
  size_t NumOffsets = IsIm2Col ? (NumDims - 2) : 0;
  bool IsCacheHint = N->getConstantOperandVal(NumOps - 2) == 1;
  bool IsMultiCast = N->getConstantOperandVal(NumOps - 3) == 1;
  size_t NumBaseArgs = NumDims + NumOffsets + 3; // for {dst, mbar, src}
  size_t MultiCastIdx = NumBaseArgs + 2;         // for Chain and IID

  unsigned CTAGroupVal = N->getConstantOperandVal(NumOps - 1);
  if ((CTAGroupVal > 0) && !Subtarget->hasCpAsyncBulkTensorCTAGroupSupport())
    report_fatal_error(
        formatv("CpAsyncBulkTensorG2S cta_group::1/2 is not supported on sm_{}",
                Subtarget->getSmVersion()));

  SDLoc DL(N);
  SmallVector<SDValue, 8> Ops(N->ops().slice(2, NumBaseArgs));

  // Push MultiCast operand, if available
  if (IsMultiCast)
    Ops.push_back(N->getOperand(MultiCastIdx));

  // Push CacheHint operand, if available
  if (IsCacheHint)
    Ops.push_back(N->getOperand(MultiCastIdx + 1));

  // Flag for CTA Group
  Ops.push_back(getI32Imm(CTAGroupVal, DL));

  // Finally, the chain operand
  Ops.push_back(N->getOperand(0));

  bool IsShared32 =
      CurDAG->getDataLayout().getPointerSizeInBits(ADDRESS_SPACE_SHARED) == 32;
  unsigned Opcode = GetCpAsyncBulkTensorG2SOpcode(
      NumDims, IsShared32, IsMultiCast, IsCacheHint, IsIm2Col);
  ReplaceNode(N, CurDAG->getMachineNode(Opcode, DL, N->getVTList(), Ops));
}

void NVPTXDAGToDAGISel::SelectCpAsyncBulkTensorReduceCommon(SDNode *N,
                                                            unsigned RedOp,
                                                            bool IsIm2Col) {
  // We have {Chain, Intrinsic-ID} followed by the actual intrisic args:
  // src, dst, dims{d0...dN}, cache_hint, cache_hint_flag
  // NumOperands = {Chain, IID} + {Actual intrinsic args}
  //             = {2}          + {4 + dims}
  size_t NumOps = N->getNumOperands();
  size_t NumDims = NumOps - 6;
  bool IsCacheHint = N->getConstantOperandVal(NumOps - 1) == 1;
  size_t NumArgs = NumDims + (IsCacheHint ? 3 : 2); // src, dst, cache_hint

  SDLoc DL(N);
  SmallVector<SDValue, 12> Ops(N->ops().slice(2, NumArgs));
  Ops.push_back(getI32Imm(RedOp, DL)); // Reduction Op
  Ops.push_back(N->getOperand(0));     // Chain operand

  bool IsShared32 =
      CurDAG->getDataLayout().getPointerSizeInBits(ADDRESS_SPACE_SHARED) == 32;
  unsigned Opcode = GetCpAsyncBulkTensorS2GReductionOpcode(
      NumDims, IsShared32, IsCacheHint, IsIm2Col);
  ReplaceNode(N, CurDAG->getMachineNode(Opcode, DL, N->getVTList(), Ops));
}

#define TCGEN05_ST_OPCODE(SHAPE, NUM)                                          \
  (enableUnpack ? NVPTX::TCGEN05_ST_##SHAPE##_##NUM##_UNPACK                   \
                : NVPTX::TCGEN05_ST_##SHAPE##_##NUM)

static unsigned getTcgen05StOpcode(unsigned IID, bool enableUnpack) {
  switch (IID) {
  case Intrinsic::nvvm_tcgen05_st_16x64b_x1:
    return TCGEN05_ST_OPCODE(16x64b, x1);
  case Intrinsic::nvvm_tcgen05_st_16x64b_x2:
    return TCGEN05_ST_OPCODE(16x64b, x2);
  case Intrinsic::nvvm_tcgen05_st_16x64b_x4:
    return TCGEN05_ST_OPCODE(16x64b, x4);
  case Intrinsic::nvvm_tcgen05_st_16x64b_x8:
    return TCGEN05_ST_OPCODE(16x64b, x8);
  case Intrinsic::nvvm_tcgen05_st_16x64b_x16:
    return TCGEN05_ST_OPCODE(16x64b, x16);
  case Intrinsic::nvvm_tcgen05_st_16x64b_x32:
    return TCGEN05_ST_OPCODE(16x64b, x32);
  case Intrinsic::nvvm_tcgen05_st_16x64b_x64:
    return TCGEN05_ST_OPCODE(16x64b, x64);
  case Intrinsic::nvvm_tcgen05_st_16x64b_x128:
    return TCGEN05_ST_OPCODE(16x64b, x128);
  case Intrinsic::nvvm_tcgen05_st_16x128b_x1:
    return TCGEN05_ST_OPCODE(16x128b, x1);
  case Intrinsic::nvvm_tcgen05_st_16x128b_x2:
    return TCGEN05_ST_OPCODE(16x128b, x2);
  case Intrinsic::nvvm_tcgen05_st_16x128b_x4:
    return TCGEN05_ST_OPCODE(16x128b, x4);
  case Intrinsic::nvvm_tcgen05_st_16x128b_x8:
    return TCGEN05_ST_OPCODE(16x128b, x8);
  case Intrinsic::nvvm_tcgen05_st_16x128b_x16:
    return TCGEN05_ST_OPCODE(16x128b, x16);
  case Intrinsic::nvvm_tcgen05_st_16x128b_x32:
    return TCGEN05_ST_OPCODE(16x128b, x32);
  case Intrinsic::nvvm_tcgen05_st_16x128b_x64:
    return TCGEN05_ST_OPCODE(16x128b, x64);
  case Intrinsic::nvvm_tcgen05_st_16x256b_x1:
    return TCGEN05_ST_OPCODE(16x256b, x1);
  case Intrinsic::nvvm_tcgen05_st_16x256b_x2:
    return TCGEN05_ST_OPCODE(16x256b, x2);
  case Intrinsic::nvvm_tcgen05_st_16x256b_x4:
    return TCGEN05_ST_OPCODE(16x256b, x4);
  case Intrinsic::nvvm_tcgen05_st_16x256b_x8:
    return TCGEN05_ST_OPCODE(16x256b, x8);
  case Intrinsic::nvvm_tcgen05_st_16x256b_x16:
    return TCGEN05_ST_OPCODE(16x256b, x16);
  case Intrinsic::nvvm_tcgen05_st_16x256b_x32:
    return TCGEN05_ST_OPCODE(16x256b, x32);
  case Intrinsic::nvvm_tcgen05_st_16x32bx2_x1:
    return TCGEN05_ST_OPCODE(16x32bx2, x1);
  case Intrinsic::nvvm_tcgen05_st_16x32bx2_x2:
    return TCGEN05_ST_OPCODE(16x32bx2, x2);
  case Intrinsic::nvvm_tcgen05_st_16x32bx2_x4:
    return TCGEN05_ST_OPCODE(16x32bx2, x4);
  case Intrinsic::nvvm_tcgen05_st_16x32bx2_x8:
    return TCGEN05_ST_OPCODE(16x32bx2, x8);
  case Intrinsic::nvvm_tcgen05_st_16x32bx2_x16:
    return TCGEN05_ST_OPCODE(16x32bx2, x16);
  case Intrinsic::nvvm_tcgen05_st_16x32bx2_x32:
    return TCGEN05_ST_OPCODE(16x32bx2, x32);
  case Intrinsic::nvvm_tcgen05_st_16x32bx2_x64:
    return TCGEN05_ST_OPCODE(16x32bx2, x64);
  case Intrinsic::nvvm_tcgen05_st_16x32bx2_x128:
    return TCGEN05_ST_OPCODE(16x32bx2, x128);
  case Intrinsic::nvvm_tcgen05_st_32x32b_x1:
    return TCGEN05_ST_OPCODE(32x32b, x1);
  case Intrinsic::nvvm_tcgen05_st_32x32b_x2:
    return TCGEN05_ST_OPCODE(32x32b, x2);
  case Intrinsic::nvvm_tcgen05_st_32x32b_x4:
    return TCGEN05_ST_OPCODE(32x32b, x4);
  case Intrinsic::nvvm_tcgen05_st_32x32b_x8:
    return TCGEN05_ST_OPCODE(32x32b, x8);
  case Intrinsic::nvvm_tcgen05_st_32x32b_x16:
    return TCGEN05_ST_OPCODE(32x32b, x16);
  case Intrinsic::nvvm_tcgen05_st_32x32b_x32:
    return TCGEN05_ST_OPCODE(32x32b, x32);
  case Intrinsic::nvvm_tcgen05_st_32x32b_x64:
    return TCGEN05_ST_OPCODE(32x32b, x64);
  case Intrinsic::nvvm_tcgen05_st_32x32b_x128:
    return TCGEN05_ST_OPCODE(32x32b, x128);
  }
  llvm_unreachable("unhandled tcgen05.st lowering");
}

void NVPTXDAGToDAGISel::SelectTcgen05St(SDNode *N, bool hasOffset) {
  SDLoc DL(N);
  unsigned IID = cast<ConstantSDNode>(N->getOperand(1))->getZExtValue();

  SmallVector<SDValue, 128> Operands = {
      N->getOperand(2) // taddr
  };

  if (hasOffset)
    Operands.push_back(CurDAG->getTargetConstant(
        cast<ConstantSDNode>(N->getOperand(3))->getZExtValue(), DL,
        MVT::i32)); // Offset

  for (unsigned I = hasOffset ? 4 : 3; I < (N->getNumOperands() - 1); I++)
    Operands.push_back(N->getOperand(I));

  bool enableUnpack =
      cast<ConstantSDNode>(N->getOperand(N->getNumOperands() - 1))
          ->getZExtValue();

  Operands.push_back(N->getOperand(0)); // Chain
  ReplaceNode(N, CurDAG->getMachineNode(getTcgen05StOpcode(IID, enableUnpack),
                                        DL, N->getVTList(), Operands));
}

bool NVPTXDAGToDAGISel::tryIntrinsicVoid(SDNode *N) {
  unsigned IID = N->getConstantOperandVal(1);
  using TMARedTy = llvm::nvvm::TMAReductionOp;
  auto CastTy = [](TMARedTy Op) { return static_cast<unsigned>(Op); };
  switch (IID) {
  default:
    return false;
  case Intrinsic::nvvm_cp_async_bulk_tensor_g2s_tile_1d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_g2s_tile_2d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_g2s_tile_3d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_g2s_tile_4d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_g2s_tile_5d:
    SelectCpAsyncBulkTensorG2SCommon(N);
    return true;
  case Intrinsic::nvvm_cp_async_bulk_tensor_g2s_im2col_3d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_g2s_im2col_4d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_g2s_im2col_5d:
    SelectCpAsyncBulkTensorG2SCommon(N, /*IsIm2Col=*/true);
    return true;
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_add_tile_1d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_add_tile_2d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_add_tile_3d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_add_tile_4d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_add_tile_5d:
    SelectCpAsyncBulkTensorReduceCommon(N, CastTy(TMARedTy::ADD));
    return true;
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_add_im2col_3d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_add_im2col_4d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_add_im2col_5d:
    SelectCpAsyncBulkTensorReduceCommon(N, CastTy(TMARedTy::ADD),
                                        /*IsIm2Col=*/true);
    return true;
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_min_tile_1d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_min_tile_2d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_min_tile_3d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_min_tile_4d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_min_tile_5d:
    SelectCpAsyncBulkTensorReduceCommon(N, CastTy(TMARedTy::MIN));
    return true;
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_min_im2col_3d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_min_im2col_4d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_min_im2col_5d:
    SelectCpAsyncBulkTensorReduceCommon(N, CastTy(TMARedTy::MIN),
                                        /*IsIm2Col=*/true);
    return true;
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_max_tile_1d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_max_tile_2d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_max_tile_3d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_max_tile_4d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_max_tile_5d:
    SelectCpAsyncBulkTensorReduceCommon(N, CastTy(TMARedTy::MAX));
    return true;
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_max_im2col_3d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_max_im2col_4d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_max_im2col_5d:
    SelectCpAsyncBulkTensorReduceCommon(N, CastTy(TMARedTy::MAX),
                                        /*IsIm2Col=*/true);
    return true;
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_inc_tile_1d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_inc_tile_2d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_inc_tile_3d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_inc_tile_4d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_inc_tile_5d:
    SelectCpAsyncBulkTensorReduceCommon(N, CastTy(TMARedTy::INC));
    return true;
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_inc_im2col_3d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_inc_im2col_4d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_inc_im2col_5d:
    SelectCpAsyncBulkTensorReduceCommon(N, CastTy(TMARedTy::INC),
                                        /*IsIm2Col=*/true);
    return true;
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_dec_tile_1d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_dec_tile_2d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_dec_tile_3d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_dec_tile_4d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_dec_tile_5d:
    SelectCpAsyncBulkTensorReduceCommon(N, CastTy(TMARedTy::DEC));
    return true;
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_dec_im2col_3d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_dec_im2col_4d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_dec_im2col_5d:
    SelectCpAsyncBulkTensorReduceCommon(N, CastTy(TMARedTy::DEC),
                                        /*IsIm2Col=*/true);
    return true;
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_and_tile_1d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_and_tile_2d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_and_tile_3d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_and_tile_4d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_and_tile_5d:
    SelectCpAsyncBulkTensorReduceCommon(N, CastTy(TMARedTy::AND));
    return true;
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_and_im2col_3d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_and_im2col_4d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_and_im2col_5d:
    SelectCpAsyncBulkTensorReduceCommon(N, CastTy(TMARedTy::AND),
                                        /*IsIm2Col=*/true);
    return true;
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_or_tile_1d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_or_tile_2d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_or_tile_3d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_or_tile_4d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_or_tile_5d:
    SelectCpAsyncBulkTensorReduceCommon(N, CastTy(TMARedTy::OR));
    return true;
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_or_im2col_3d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_or_im2col_4d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_or_im2col_5d:
    SelectCpAsyncBulkTensorReduceCommon(N, CastTy(TMARedTy::OR),
                                        /*IsIm2Col=*/true);
    return true;
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_xor_tile_1d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_xor_tile_2d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_xor_tile_3d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_xor_tile_4d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_xor_tile_5d:
    SelectCpAsyncBulkTensorReduceCommon(N, CastTy(TMARedTy::XOR));
    return true;
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_xor_im2col_3d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_xor_im2col_4d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_xor_im2col_5d:
    SelectCpAsyncBulkTensorReduceCommon(N, CastTy(TMARedTy::XOR),
                                        /*IsIm2Col=*/true);
    return true;

  case Intrinsic::nvvm_tcgen05_st_16x64b_x1:
  case Intrinsic::nvvm_tcgen05_st_16x64b_x2:
  case Intrinsic::nvvm_tcgen05_st_16x64b_x4:
  case Intrinsic::nvvm_tcgen05_st_16x64b_x8:
  case Intrinsic::nvvm_tcgen05_st_16x64b_x16:
  case Intrinsic::nvvm_tcgen05_st_16x64b_x32:
  case Intrinsic::nvvm_tcgen05_st_16x64b_x64:
  case Intrinsic::nvvm_tcgen05_st_16x64b_x128:
  case Intrinsic::nvvm_tcgen05_st_32x32b_x1:
  case Intrinsic::nvvm_tcgen05_st_32x32b_x2:
  case Intrinsic::nvvm_tcgen05_st_32x32b_x4:
  case Intrinsic::nvvm_tcgen05_st_32x32b_x8:
  case Intrinsic::nvvm_tcgen05_st_32x32b_x16:
  case Intrinsic::nvvm_tcgen05_st_32x32b_x32:
  case Intrinsic::nvvm_tcgen05_st_32x32b_x64:
  case Intrinsic::nvvm_tcgen05_st_32x32b_x128:
  case Intrinsic::nvvm_tcgen05_st_16x128b_x1:
  case Intrinsic::nvvm_tcgen05_st_16x128b_x2:
  case Intrinsic::nvvm_tcgen05_st_16x128b_x4:
  case Intrinsic::nvvm_tcgen05_st_16x128b_x8:
  case Intrinsic::nvvm_tcgen05_st_16x128b_x16:
  case Intrinsic::nvvm_tcgen05_st_16x128b_x32:
  case Intrinsic::nvvm_tcgen05_st_16x128b_x64:
  case Intrinsic::nvvm_tcgen05_st_16x256b_x1:
  case Intrinsic::nvvm_tcgen05_st_16x256b_x2:
  case Intrinsic::nvvm_tcgen05_st_16x256b_x4:
  case Intrinsic::nvvm_tcgen05_st_16x256b_x8:
  case Intrinsic::nvvm_tcgen05_st_16x256b_x16:
  case Intrinsic::nvvm_tcgen05_st_16x256b_x32: {
    SelectTcgen05St(N);
    return true;
  }

  case Intrinsic::nvvm_tcgen05_st_16x32bx2_x1:
  case Intrinsic::nvvm_tcgen05_st_16x32bx2_x2:
  case Intrinsic::nvvm_tcgen05_st_16x32bx2_x4:
  case Intrinsic::nvvm_tcgen05_st_16x32bx2_x8:
  case Intrinsic::nvvm_tcgen05_st_16x32bx2_x16:
  case Intrinsic::nvvm_tcgen05_st_16x32bx2_x32:
  case Intrinsic::nvvm_tcgen05_st_16x32bx2_x64:
  case Intrinsic::nvvm_tcgen05_st_16x32bx2_x128: {
    SelectTcgen05St(N, /*  hasOffset */ true);
    return true;
  }
  }
}
