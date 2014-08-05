//===-- AMDGPUISelLowering.cpp - AMDGPU Common DAG lowering functions -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// \brief This is the parent TargetLowering class for hardware code gen
/// targets.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUISelLowering.h"
#include "AMDGPU.h"
#include "AMDGPUFrameLowering.h"
#include "AMDGPUIntrinsicInfo.h"
#include "AMDGPURegisterInfo.h"
#include "AMDGPUSubtarget.h"
#include "R600MachineFunctionInfo.h"
#include "SIMachineFunctionInfo.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"

using namespace llvm;

namespace {

/// Diagnostic information for unimplemented or unsupported feature reporting.
class DiagnosticInfoUnsupported : public DiagnosticInfo {
private:
  const Twine &Description;
  const Function &Fn;

  static int KindID;

  static int getKindID() {
    if (KindID == 0)
      KindID = llvm::getNextAvailablePluginDiagnosticKind();
    return KindID;
  }

public:
  DiagnosticInfoUnsupported(const Function &Fn, const Twine &Desc,
                          DiagnosticSeverity Severity = DS_Error)
    : DiagnosticInfo(getKindID(), Severity),
      Description(Desc),
      Fn(Fn) { }

  const Function &getFunction() const { return Fn; }
  const Twine &getDescription() const { return Description; }

  void print(DiagnosticPrinter &DP) const override {
    DP << "unsupported " << getDescription() << " in " << Fn.getName();
  }

  static bool classof(const DiagnosticInfo *DI) {
    return DI->getKind() == getKindID();
  }
};

int DiagnosticInfoUnsupported::KindID = 0;
}


static bool allocateStack(unsigned ValNo, MVT ValVT, MVT LocVT,
                      CCValAssign::LocInfo LocInfo,
                      ISD::ArgFlagsTy ArgFlags, CCState &State) {
  unsigned Offset = State.AllocateStack(ValVT.getStoreSize(),
                                        ArgFlags.getOrigAlign());
  State.addLoc(CCValAssign::getMem(ValNo, ValVT, Offset, LocVT, LocInfo));

  return true;
}

#include "AMDGPUGenCallingConv.inc"

// Find a larger type to do a load / store of a vector with.
EVT AMDGPUTargetLowering::getEquivalentMemType(LLVMContext &Ctx, EVT VT) {
  unsigned StoreSize = VT.getStoreSizeInBits();
  if (StoreSize <= 32)
    return EVT::getIntegerVT(Ctx, StoreSize);

  assert(StoreSize % 32 == 0 && "Store size not a multiple of 32");
  return EVT::getVectorVT(Ctx, MVT::i32, StoreSize / 32);
}

// Type for a vector that will be loaded to.
EVT AMDGPUTargetLowering::getEquivalentLoadRegType(LLVMContext &Ctx, EVT VT) {
  unsigned StoreSize = VT.getStoreSizeInBits();
  if (StoreSize <= 32)
    return EVT::getIntegerVT(Ctx, 32);

  return EVT::getVectorVT(Ctx, MVT::i32, StoreSize / 32);
}

AMDGPUTargetLowering::AMDGPUTargetLowering(TargetMachine &TM) :
  TargetLowering(TM, new TargetLoweringObjectFileELF()) {

  Subtarget = &TM.getSubtarget<AMDGPUSubtarget>();

  setOperationAction(ISD::Constant, MVT::i32, Legal);
  setOperationAction(ISD::Constant, MVT::i64, Legal);
  setOperationAction(ISD::ConstantFP, MVT::f32, Legal);
  setOperationAction(ISD::ConstantFP, MVT::f64, Legal);

  setOperationAction(ISD::BR_JT, MVT::Other, Expand);
  setOperationAction(ISD::BRIND, MVT::Other, Expand);

  // We need to custom lower some of the intrinsics
  setOperationAction(ISD::INTRINSIC_WO_CHAIN, MVT::Other, Custom);

  // Library functions.  These default to Expand, but we have instructions
  // for them.
  setOperationAction(ISD::FCEIL,  MVT::f32, Legal);
  setOperationAction(ISD::FEXP2,  MVT::f32, Legal);
  setOperationAction(ISD::FPOW,   MVT::f32, Legal);
  setOperationAction(ISD::FLOG2,  MVT::f32, Legal);
  setOperationAction(ISD::FABS,   MVT::f32, Legal);
  setOperationAction(ISD::FFLOOR, MVT::f32, Legal);
  setOperationAction(ISD::FRINT,  MVT::f32, Legal);
  setOperationAction(ISD::FROUND, MVT::f32, Legal);
  setOperationAction(ISD::FTRUNC, MVT::f32, Legal);

  // Lower floating point store/load to integer store/load to reduce the number
  // of patterns in tablegen.
  setOperationAction(ISD::STORE, MVT::f32, Promote);
  AddPromotedToType(ISD::STORE, MVT::f32, MVT::i32);

  setOperationAction(ISD::STORE, MVT::v2f32, Promote);
  AddPromotedToType(ISD::STORE, MVT::v2f32, MVT::v2i32);

  setOperationAction(ISD::STORE, MVT::i64, Promote);
  AddPromotedToType(ISD::STORE, MVT::i64, MVT::v2i32);

  setOperationAction(ISD::STORE, MVT::v4f32, Promote);
  AddPromotedToType(ISD::STORE, MVT::v4f32, MVT::v4i32);

  setOperationAction(ISD::STORE, MVT::v8f32, Promote);
  AddPromotedToType(ISD::STORE, MVT::v8f32, MVT::v8i32);

  setOperationAction(ISD::STORE, MVT::v16f32, Promote);
  AddPromotedToType(ISD::STORE, MVT::v16f32, MVT::v16i32);

  setOperationAction(ISD::STORE, MVT::f64, Promote);
  AddPromotedToType(ISD::STORE, MVT::f64, MVT::i64);

  setOperationAction(ISD::STORE, MVT::v2f64, Promote);
  AddPromotedToType(ISD::STORE, MVT::v2f64, MVT::v2i64);

  // Custom lowering of vector stores is required for local address space
  // stores.
  setOperationAction(ISD::STORE, MVT::v4i32, Custom);
  // XXX: Native v2i32 local address space stores are possible, but not
  // currently implemented.
  setOperationAction(ISD::STORE, MVT::v2i32, Custom);

  setTruncStoreAction(MVT::v2i32, MVT::v2i16, Custom);
  setTruncStoreAction(MVT::v2i32, MVT::v2i8, Custom);
  setTruncStoreAction(MVT::v4i32, MVT::v4i8, Custom);

  // XXX: This can be change to Custom, once ExpandVectorStores can
  // handle 64-bit stores.
  setTruncStoreAction(MVT::v4i32, MVT::v4i16, Expand);

  setTruncStoreAction(MVT::i64, MVT::i16, Expand);
  setTruncStoreAction(MVT::i64, MVT::i8, Expand);
  setTruncStoreAction(MVT::i64, MVT::i1, Expand);
  setTruncStoreAction(MVT::v2i64, MVT::v2i1, Expand);
  setTruncStoreAction(MVT::v4i64, MVT::v4i1, Expand);


  setOperationAction(ISD::LOAD, MVT::f32, Promote);
  AddPromotedToType(ISD::LOAD, MVT::f32, MVT::i32);

  setOperationAction(ISD::LOAD, MVT::v2f32, Promote);
  AddPromotedToType(ISD::LOAD, MVT::v2f32, MVT::v2i32);

  setOperationAction(ISD::LOAD, MVT::i64, Promote);
  AddPromotedToType(ISD::LOAD, MVT::i64, MVT::v2i32);

  setOperationAction(ISD::LOAD, MVT::v4f32, Promote);
  AddPromotedToType(ISD::LOAD, MVT::v4f32, MVT::v4i32);

  setOperationAction(ISD::LOAD, MVT::v8f32, Promote);
  AddPromotedToType(ISD::LOAD, MVT::v8f32, MVT::v8i32);

  setOperationAction(ISD::LOAD, MVT::v16f32, Promote);
  AddPromotedToType(ISD::LOAD, MVT::v16f32, MVT::v16i32);

  setOperationAction(ISD::LOAD, MVT::f64, Promote);
  AddPromotedToType(ISD::LOAD, MVT::f64, MVT::i64);

  setOperationAction(ISD::LOAD, MVT::v2f64, Promote);
  AddPromotedToType(ISD::LOAD, MVT::v2f64, MVT::v2i64);

  setOperationAction(ISD::CONCAT_VECTORS, MVT::v4i32, Custom);
  setOperationAction(ISD::CONCAT_VECTORS, MVT::v4f32, Custom);
  setOperationAction(ISD::CONCAT_VECTORS, MVT::v8i32, Custom);
  setOperationAction(ISD::CONCAT_VECTORS, MVT::v8f32, Custom);
  setOperationAction(ISD::EXTRACT_SUBVECTOR, MVT::v2f32, Custom);
  setOperationAction(ISD::EXTRACT_SUBVECTOR, MVT::v2i32, Custom);
  setOperationAction(ISD::EXTRACT_SUBVECTOR, MVT::v4f32, Custom);
  setOperationAction(ISD::EXTRACT_SUBVECTOR, MVT::v4i32, Custom);
  setOperationAction(ISD::EXTRACT_SUBVECTOR, MVT::v8f32, Custom);
  setOperationAction(ISD::EXTRACT_SUBVECTOR, MVT::v8i32, Custom);

  setLoadExtAction(ISD::EXTLOAD, MVT::v2i8, Expand);
  setLoadExtAction(ISD::SEXTLOAD, MVT::v2i8, Expand);
  setLoadExtAction(ISD::ZEXTLOAD, MVT::v2i8, Expand);
  setLoadExtAction(ISD::EXTLOAD, MVT::v4i8, Expand);
  setLoadExtAction(ISD::SEXTLOAD, MVT::v4i8, Expand);
  setLoadExtAction(ISD::ZEXTLOAD, MVT::v4i8, Expand);
  setLoadExtAction(ISD::EXTLOAD, MVT::v2i16, Expand);
  setLoadExtAction(ISD::SEXTLOAD, MVT::v2i16, Expand);
  setLoadExtAction(ISD::ZEXTLOAD, MVT::v2i16, Expand);
  setLoadExtAction(ISD::EXTLOAD, MVT::v4i16, Expand);
  setLoadExtAction(ISD::SEXTLOAD, MVT::v4i16, Expand);
  setLoadExtAction(ISD::ZEXTLOAD, MVT::v4i16, Expand);

  setOperationAction(ISD::BR_CC, MVT::i1, Expand);

  if (Subtarget->getGeneration() < AMDGPUSubtarget::SEA_ISLANDS) {
    setOperationAction(ISD::FCEIL, MVT::f64, Custom);
    setOperationAction(ISD::FTRUNC, MVT::f64, Custom);
    setOperationAction(ISD::FRINT, MVT::f64, Custom);
    setOperationAction(ISD::FFLOOR, MVT::f64, Custom);
  }

  if (!Subtarget->hasBFI()) {
    // fcopysign can be done in a single instruction with BFI.
    setOperationAction(ISD::FCOPYSIGN, MVT::f32, Expand);
    setOperationAction(ISD::FCOPYSIGN, MVT::f64, Expand);
  }

  setOperationAction(ISD::FP16_TO_FP, MVT::f64, Expand);

  setLoadExtAction(ISD::EXTLOAD, MVT::f16, Expand);
  setTruncStoreAction(MVT::f32, MVT::f16, Expand);
  setTruncStoreAction(MVT::f64, MVT::f16, Expand);

  const MVT ScalarIntVTs[] = { MVT::i32, MVT::i64 };
  for (MVT VT : ScalarIntVTs) {
    setOperationAction(ISD::SREM, VT, Expand);
    setOperationAction(ISD::SDIV, VT, Expand);

    // GPU does not have divrem function for signed or unsigned.
    setOperationAction(ISD::SDIVREM, VT, Custom);
    setOperationAction(ISD::UDIVREM, VT, Custom);

    // GPU does not have [S|U]MUL_LOHI functions as a single instruction.
    setOperationAction(ISD::SMUL_LOHI, VT, Expand);
    setOperationAction(ISD::UMUL_LOHI, VT, Expand);

    setOperationAction(ISD::BSWAP, VT, Expand);
    setOperationAction(ISD::CTTZ, VT, Expand);
    setOperationAction(ISD::CTLZ, VT, Expand);
  }

  if (!Subtarget->hasBCNT(32))
    setOperationAction(ISD::CTPOP, MVT::i32, Expand);

  if (!Subtarget->hasBCNT(64))
    setOperationAction(ISD::CTPOP, MVT::i64, Expand);

  // The hardware supports 32-bit ROTR, but not ROTL.
  setOperationAction(ISD::ROTL, MVT::i32, Expand);
  setOperationAction(ISD::ROTL, MVT::i64, Expand);
  setOperationAction(ISD::ROTR, MVT::i64, Expand);

  setOperationAction(ISD::MUL, MVT::i64, Expand);
  setOperationAction(ISD::MULHU, MVT::i64, Expand);
  setOperationAction(ISD::MULHS, MVT::i64, Expand);
  setOperationAction(ISD::UDIV, MVT::i32, Expand);
  setOperationAction(ISD::UREM, MVT::i32, Expand);
  setOperationAction(ISD::UINT_TO_FP, MVT::i64, Custom);
  setOperationAction(ISD::SELECT_CC, MVT::i64, Expand);

  if (!Subtarget->hasFFBH())
    setOperationAction(ISD::CTLZ_ZERO_UNDEF, MVT::i32, Expand);

  if (!Subtarget->hasFFBL())
    setOperationAction(ISD::CTTZ_ZERO_UNDEF, MVT::i32, Expand);

  static const MVT::SimpleValueType VectorIntTypes[] = {
    MVT::v2i32, MVT::v4i32
  };

  for (MVT VT : VectorIntTypes) {
    // Expand the following operations for the current type by default.
    setOperationAction(ISD::ADD,  VT, Expand);
    setOperationAction(ISD::AND,  VT, Expand);
    setOperationAction(ISD::FP_TO_SINT, VT, Expand);
    setOperationAction(ISD::FP_TO_UINT, VT, Expand);
    setOperationAction(ISD::MUL,  VT, Expand);
    setOperationAction(ISD::OR,   VT, Expand);
    setOperationAction(ISD::SHL,  VT, Expand);
    setOperationAction(ISD::SRA,  VT, Expand);
    setOperationAction(ISD::SRL,  VT, Expand);
    setOperationAction(ISD::ROTL, VT, Expand);
    setOperationAction(ISD::ROTR, VT, Expand);
    setOperationAction(ISD::SUB,  VT, Expand);
    setOperationAction(ISD::SINT_TO_FP, VT, Expand);
    setOperationAction(ISD::UINT_TO_FP, VT, Expand);
    // TODO: Implement custom UREM / SREM routines.
    setOperationAction(ISD::SDIV, VT, Expand);
    setOperationAction(ISD::UDIV, VT, Expand);
    setOperationAction(ISD::SREM, VT, Expand);
    setOperationAction(ISD::UREM, VT, Expand);
    setOperationAction(ISD::SMUL_LOHI, VT, Expand);
    setOperationAction(ISD::UMUL_LOHI, VT, Expand);
    setOperationAction(ISD::SDIVREM, VT, Custom);
    setOperationAction(ISD::UDIVREM, VT, Custom);
    setOperationAction(ISD::ADDC, VT, Expand);
    setOperationAction(ISD::SUBC, VT, Expand);
    setOperationAction(ISD::ADDE, VT, Expand);
    setOperationAction(ISD::SUBE, VT, Expand);
    setOperationAction(ISD::SELECT, VT, Expand);
    setOperationAction(ISD::VSELECT, VT, Expand);
    setOperationAction(ISD::SELECT_CC, VT, Expand);
    setOperationAction(ISD::XOR,  VT, Expand);
    setOperationAction(ISD::BSWAP, VT, Expand);
    setOperationAction(ISD::CTPOP, VT, Expand);
    setOperationAction(ISD::CTTZ, VT, Expand);
    setOperationAction(ISD::CTTZ_ZERO_UNDEF, VT, Expand);
    setOperationAction(ISD::CTLZ, VT, Expand);
    setOperationAction(ISD::CTLZ_ZERO_UNDEF, VT, Expand);
    setOperationAction(ISD::VECTOR_SHUFFLE, VT, Expand);
  }

  static const MVT::SimpleValueType FloatVectorTypes[] = {
    MVT::v2f32, MVT::v4f32
  };

  for (MVT VT : FloatVectorTypes) {
    setOperationAction(ISD::FABS, VT, Expand);
    setOperationAction(ISD::FADD, VT, Expand);
    setOperationAction(ISD::FCEIL, VT, Expand);
    setOperationAction(ISD::FCOS, VT, Expand);
    setOperationAction(ISD::FDIV, VT, Expand);
    setOperationAction(ISD::FEXP2, VT, Expand);
    setOperationAction(ISD::FLOG2, VT, Expand);
    setOperationAction(ISD::FPOW, VT, Expand);
    setOperationAction(ISD::FFLOOR, VT, Expand);
    setOperationAction(ISD::FTRUNC, VT, Expand);
    setOperationAction(ISD::FMUL, VT, Expand);
    setOperationAction(ISD::FMA, VT, Expand);
    setOperationAction(ISD::FRINT, VT, Expand);
    setOperationAction(ISD::FNEARBYINT, VT, Expand);
    setOperationAction(ISD::FSQRT, VT, Expand);
    setOperationAction(ISD::FSIN, VT, Expand);
    setOperationAction(ISD::FSUB, VT, Expand);
    setOperationAction(ISD::FNEG, VT, Expand);
    setOperationAction(ISD::SELECT, VT, Expand);
    setOperationAction(ISD::VSELECT, VT, Expand);
    setOperationAction(ISD::SELECT_CC, VT, Expand);
    setOperationAction(ISD::FCOPYSIGN, VT, Expand);
    setOperationAction(ISD::VECTOR_SHUFFLE, VT, Expand);
  }

  setOperationAction(ISD::FNEARBYINT, MVT::f32, Custom);
  setOperationAction(ISD::FNEARBYINT, MVT::f64, Custom);

  setTargetDAGCombine(ISD::MUL);
  setTargetDAGCombine(ISD::SELECT_CC);
  setTargetDAGCombine(ISD::STORE);

  setSchedulingPreference(Sched::RegPressure);
  setJumpIsExpensive(true);

  setSelectIsExpensive(false);
  PredictableSelectIsExpensive = false;

  // There are no integer divide instructions, and these expand to a pretty
  // large sequence of instructions.
  setIntDivIsCheap(false);
  setPow2DivIsCheap(false);

  // TODO: Investigate this when 64-bit divides are implemented.
  addBypassSlowDiv(64, 32);

  // FIXME: Need to really handle these.
  MaxStoresPerMemcpy  = 4096;
  MaxStoresPerMemmove = 4096;
  MaxStoresPerMemset  = 4096;
}

//===----------------------------------------------------------------------===//
// Target Information
//===----------------------------------------------------------------------===//

MVT AMDGPUTargetLowering::getVectorIdxTy() const {
  return MVT::i32;
}

bool AMDGPUTargetLowering::isSelectSupported(SelectSupportKind SelType) const {
  return true;
}

// The backend supports 32 and 64 bit floating point immediates.
// FIXME: Why are we reporting vectors of FP immediates as legal?
bool AMDGPUTargetLowering::isFPImmLegal(const APFloat &Imm, EVT VT) const {
  EVT ScalarVT = VT.getScalarType();
  return (ScalarVT == MVT::f32 || ScalarVT == MVT::f64);
}

// We don't want to shrink f64 / f32 constants.
bool AMDGPUTargetLowering::ShouldShrinkFPConstant(EVT VT) const {
  EVT ScalarVT = VT.getScalarType();
  return (ScalarVT != MVT::f32 && ScalarVT != MVT::f64);
}

bool AMDGPUTargetLowering::isLoadBitCastBeneficial(EVT LoadTy,
                                                   EVT CastTy) const {
  if (LoadTy.getSizeInBits() != CastTy.getSizeInBits())
    return true;

  unsigned LScalarSize = LoadTy.getScalarType().getSizeInBits();
  unsigned CastScalarSize = CastTy.getScalarType().getSizeInBits();

  return ((LScalarSize <= CastScalarSize) ||
          (CastScalarSize >= 32) ||
          (LScalarSize < 32));
}

//===---------------------------------------------------------------------===//
// Target Properties
//===---------------------------------------------------------------------===//

bool AMDGPUTargetLowering::isFAbsFree(EVT VT) const {
  assert(VT.isFloatingPoint());
  return VT == MVT::f32;
}

bool AMDGPUTargetLowering::isFNegFree(EVT VT) const {
  assert(VT.isFloatingPoint());
  return VT == MVT::f32;
}

bool AMDGPUTargetLowering::isTruncateFree(EVT Source, EVT Dest) const {
  // Truncate is just accessing a subregister.
  return Dest.bitsLT(Source) && (Dest.getSizeInBits() % 32 == 0);
}

bool AMDGPUTargetLowering::isTruncateFree(Type *Source, Type *Dest) const {
  // Truncate is just accessing a subregister.
  return Dest->getPrimitiveSizeInBits() < Source->getPrimitiveSizeInBits() &&
         (Dest->getPrimitiveSizeInBits() % 32 == 0);
}

bool AMDGPUTargetLowering::isZExtFree(Type *Src, Type *Dest) const {
  const DataLayout *DL = getDataLayout();
  unsigned SrcSize = DL->getTypeSizeInBits(Src->getScalarType());
  unsigned DestSize = DL->getTypeSizeInBits(Dest->getScalarType());

  return SrcSize == 32 && DestSize == 64;
}

bool AMDGPUTargetLowering::isZExtFree(EVT Src, EVT Dest) const {
  // Any register load of a 64-bit value really requires 2 32-bit moves. For all
  // practical purposes, the extra mov 0 to load a 64-bit is free.  As used,
  // this will enable reducing 64-bit operations the 32-bit, which is always
  // good.
  return Src == MVT::i32 && Dest == MVT::i64;
}

bool AMDGPUTargetLowering::isZExtFree(SDValue Val, EVT VT2) const {
  return isZExtFree(Val.getValueType(), VT2);
}

bool AMDGPUTargetLowering::isNarrowingProfitable(EVT SrcVT, EVT DestVT) const {
  // There aren't really 64-bit registers, but pairs of 32-bit ones and only a
  // limited number of native 64-bit operations. Shrinking an operation to fit
  // in a single 32-bit register should always be helpful. As currently used,
  // this is much less general than the name suggests, and is only used in
  // places trying to reduce the sizes of loads. Shrinking loads to < 32-bits is
  // not profitable, and may actually be harmful.
  return SrcVT.getSizeInBits() > 32 && DestVT.getSizeInBits() == 32;
}

//===---------------------------------------------------------------------===//
// TargetLowering Callbacks
//===---------------------------------------------------------------------===//

void AMDGPUTargetLowering::AnalyzeFormalArguments(CCState &State,
                             const SmallVectorImpl<ISD::InputArg> &Ins) const {

  State.AnalyzeFormalArguments(Ins, CC_AMDGPU);
}

SDValue AMDGPUTargetLowering::LowerReturn(
                                     SDValue Chain,
                                     CallingConv::ID CallConv,
                                     bool isVarArg,
                                     const SmallVectorImpl<ISD::OutputArg> &Outs,
                                     const SmallVectorImpl<SDValue> &OutVals,
                                     SDLoc DL, SelectionDAG &DAG) const {
  return DAG.getNode(AMDGPUISD::RET_FLAG, DL, MVT::Other, Chain);
}

//===---------------------------------------------------------------------===//
// Target specific lowering
//===---------------------------------------------------------------------===//

SDValue AMDGPUTargetLowering::LowerCall(CallLoweringInfo &CLI,
                                        SmallVectorImpl<SDValue> &InVals) const {
  SDValue Callee = CLI.Callee;
  SelectionDAG &DAG = CLI.DAG;

  const Function &Fn = *DAG.getMachineFunction().getFunction();

  StringRef FuncName("<unknown>");

  if (const ExternalSymbolSDNode *G = dyn_cast<ExternalSymbolSDNode>(Callee))
    FuncName = G->getSymbol();
  else if (const GlobalAddressSDNode *G = dyn_cast<GlobalAddressSDNode>(Callee))
    FuncName = G->getGlobal()->getName();

  DiagnosticInfoUnsupported NoCalls(Fn, "call to function " + FuncName);
  DAG.getContext()->diagnose(NoCalls);
  return SDValue();
}

SDValue AMDGPUTargetLowering::LowerOperation(SDValue Op,
                                             SelectionDAG &DAG) const {
  switch (Op.getOpcode()) {
  default:
    Op.getNode()->dump();
    llvm_unreachable("Custom lowering code for this"
                     "instruction is not implemented yet!");
    break;
  case ISD::SIGN_EXTEND_INREG: return LowerSIGN_EXTEND_INREG(Op, DAG);
  case ISD::CONCAT_VECTORS: return LowerCONCAT_VECTORS(Op, DAG);
  case ISD::EXTRACT_SUBVECTOR: return LowerEXTRACT_SUBVECTOR(Op, DAG);
  case ISD::FrameIndex: return LowerFrameIndex(Op, DAG);
  case ISD::INTRINSIC_WO_CHAIN: return LowerINTRINSIC_WO_CHAIN(Op, DAG);
  case ISD::SDIV: return LowerSDIV(Op, DAG);
  case ISD::SREM: return LowerSREM(Op, DAG);
  case ISD::UDIVREM: return LowerUDIVREM(Op, DAG);
  case ISD::SDIVREM: return LowerSDIVREM(Op, DAG);
  case ISD::FCEIL: return LowerFCEIL(Op, DAG);
  case ISD::FTRUNC: return LowerFTRUNC(Op, DAG);
  case ISD::FRINT: return LowerFRINT(Op, DAG);
  case ISD::FNEARBYINT: return LowerFNEARBYINT(Op, DAG);
  case ISD::FFLOOR: return LowerFFLOOR(Op, DAG);
  case ISD::UINT_TO_FP: return LowerUINT_TO_FP(Op, DAG);
  }
  return Op;
}

void AMDGPUTargetLowering::ReplaceNodeResults(SDNode *N,
                                              SmallVectorImpl<SDValue> &Results,
                                              SelectionDAG &DAG) const {
  switch (N->getOpcode()) {
  case ISD::SIGN_EXTEND_INREG:
    // Different parts of legalization seem to interpret which type of
    // sign_extend_inreg is the one to check for custom lowering. The extended
    // from type is what really matters, but some places check for custom
    // lowering of the result type. This results in trying to use
    // ReplaceNodeResults to sext_in_reg to an illegal type, so we'll just do
    // nothing here and let the illegal result integer be handled normally.
    return;
  case ISD::LOAD: {
    SDNode *Node = LowerLOAD(SDValue(N, 0), DAG).getNode();
    if (!Node)
      return;

    Results.push_back(SDValue(Node, 0));
    Results.push_back(SDValue(Node, 1));
    // XXX: LLVM seems not to replace Chain Value inside CustomWidenLowerNode
    // function
    DAG.ReplaceAllUsesOfValueWith(SDValue(N,1), SDValue(Node, 1));
    return;
  }
  case ISD::STORE: {
    SDValue Lowered = LowerSTORE(SDValue(N, 0), DAG);
    if (Lowered.getNode())
      Results.push_back(Lowered);
    return;
  }
  default:
    return;
  }
}

// FIXME: This implements accesses to initialized globals in the constant
// address space by copying them to private and accessing that. It does not
// properly handle illegal types or vectors. The private vector loads are not
// scalarized, and the illegal scalars hit an assertion. This technique will not
// work well with large initializers, and this should eventually be
// removed. Initialized globals should be placed into a data section that the
// runtime will load into a buffer before the kernel is executed. Uses of the
// global need to be replaced with a pointer loaded from an implicit kernel
// argument into this buffer holding the copy of the data, which will remove the
// need for any of this.
SDValue AMDGPUTargetLowering::LowerConstantInitializer(const Constant* Init,
                                                       const GlobalValue *GV,
                                                       const SDValue &InitPtr,
                                                       SDValue Chain,
                                                       SelectionDAG &DAG) const {
  const DataLayout *TD = getTargetMachine().getDataLayout();
  SDLoc DL(InitPtr);
  Type *InitTy = Init->getType();

  if (const ConstantInt *CI = dyn_cast<ConstantInt>(Init)) {
    EVT VT = EVT::getEVT(InitTy);
    PointerType *PtrTy = PointerType::get(InitTy, AMDGPUAS::PRIVATE_ADDRESS);
    return DAG.getStore(Chain, DL, DAG.getConstant(*CI, VT), InitPtr,
                        MachinePointerInfo(UndefValue::get(PtrTy)), false, false,
                        TD->getPrefTypeAlignment(InitTy));
  }

  if (const ConstantFP *CFP = dyn_cast<ConstantFP>(Init)) {
    EVT VT = EVT::getEVT(CFP->getType());
    PointerType *PtrTy = PointerType::get(CFP->getType(), 0);
    return DAG.getStore(Chain, DL, DAG.getConstantFP(*CFP, VT), InitPtr,
                 MachinePointerInfo(UndefValue::get(PtrTy)), false, false,
                 TD->getPrefTypeAlignment(CFP->getType()));
  }

  if (StructType *ST = dyn_cast<StructType>(InitTy)) {
    const StructLayout *SL = TD->getStructLayout(ST);

    EVT PtrVT = InitPtr.getValueType();
    SmallVector<SDValue, 8> Chains;

    for (unsigned I = 0, N = ST->getNumElements(); I != N; ++I) {
      SDValue Offset = DAG.getConstant(SL->getElementOffset(I), PtrVT);
      SDValue Ptr = DAG.getNode(ISD::ADD, DL, PtrVT, InitPtr, Offset);

      Constant *Elt = Init->getAggregateElement(I);
      Chains.push_back(LowerConstantInitializer(Elt, GV, Ptr, Chain, DAG));
    }

    return DAG.getNode(ISD::TokenFactor, DL, MVT::Other, Chains);
  }

  if (SequentialType *SeqTy = dyn_cast<SequentialType>(InitTy)) {
    EVT PtrVT = InitPtr.getValueType();

    unsigned NumElements;
    if (ArrayType *AT = dyn_cast<ArrayType>(SeqTy))
      NumElements = AT->getNumElements();
    else if (VectorType *VT = dyn_cast<VectorType>(SeqTy))
      NumElements = VT->getNumElements();
    else
      llvm_unreachable("Unexpected type");

    unsigned EltSize = TD->getTypeAllocSize(SeqTy->getElementType());
    SmallVector<SDValue, 8> Chains;
    for (unsigned i = 0; i < NumElements; ++i) {
      SDValue Offset = DAG.getConstant(i * EltSize, PtrVT);
      SDValue Ptr = DAG.getNode(ISD::ADD, DL, PtrVT, InitPtr, Offset);

      Constant *Elt = Init->getAggregateElement(i);
      Chains.push_back(LowerConstantInitializer(Elt, GV, Ptr, Chain, DAG));
    }

    return DAG.getNode(ISD::TokenFactor, DL, MVT::Other, Chains);
  }

  if (isa<UndefValue>(Init)) {
    EVT VT = EVT::getEVT(InitTy);
    PointerType *PtrTy = PointerType::get(InitTy, AMDGPUAS::PRIVATE_ADDRESS);
    return DAG.getStore(Chain, DL, DAG.getUNDEF(VT), InitPtr,
                        MachinePointerInfo(UndefValue::get(PtrTy)), false, false,
                        TD->getPrefTypeAlignment(InitTy));
  }

  Init->dump();
  llvm_unreachable("Unhandled constant initializer");
}

SDValue AMDGPUTargetLowering::LowerGlobalAddress(AMDGPUMachineFunction* MFI,
                                                 SDValue Op,
                                                 SelectionDAG &DAG) const {

  const DataLayout *TD = getTargetMachine().getDataLayout();
  GlobalAddressSDNode *G = cast<GlobalAddressSDNode>(Op);
  const GlobalValue *GV = G->getGlobal();

  switch (G->getAddressSpace()) {
  default: llvm_unreachable("Global Address lowering not implemented for this "
                            "address space");
  case AMDGPUAS::LOCAL_ADDRESS: {
    // XXX: What does the value of G->getOffset() mean?
    assert(G->getOffset() == 0 &&
         "Do not know what to do with an non-zero offset");

    unsigned Offset;
    if (MFI->LocalMemoryObjects.count(GV) == 0) {
      uint64_t Size = TD->getTypeAllocSize(GV->getType()->getElementType());
      Offset = MFI->LDSSize;
      MFI->LocalMemoryObjects[GV] = Offset;
      // XXX: Account for alignment?
      MFI->LDSSize += Size;
    } else {
      Offset = MFI->LocalMemoryObjects[GV];
    }

    return DAG.getConstant(Offset, getPointerTy(G->getAddressSpace()));
  }
  case AMDGPUAS::CONSTANT_ADDRESS: {
    MachineFrameInfo *FrameInfo = DAG.getMachineFunction().getFrameInfo();
    Type *EltType = GV->getType()->getElementType();
    unsigned Size = TD->getTypeAllocSize(EltType);
    unsigned Alignment = TD->getPrefTypeAlignment(EltType);

    MVT PrivPtrVT = getPointerTy(AMDGPUAS::PRIVATE_ADDRESS);
    MVT ConstPtrVT = getPointerTy(AMDGPUAS::CONSTANT_ADDRESS);

    int FI = FrameInfo->CreateStackObject(Size, Alignment, false);
    SDValue InitPtr = DAG.getFrameIndex(FI, PrivPtrVT);

    const GlobalVariable *Var = cast<GlobalVariable>(GV);
    if (!Var->hasInitializer()) {
      // This has no use, but bugpoint will hit it.
      return DAG.getZExtOrTrunc(InitPtr, SDLoc(Op), ConstPtrVT);
    }

    const Constant *Init = Var->getInitializer();
    SmallVector<SDNode*, 8> WorkList;

    for (SDNode::use_iterator I = DAG.getEntryNode()->use_begin(),
                              E = DAG.getEntryNode()->use_end(); I != E; ++I) {
      if (I->getOpcode() != AMDGPUISD::REGISTER_LOAD && I->getOpcode() != ISD::LOAD)
        continue;
      WorkList.push_back(*I);
    }
    SDValue Chain = LowerConstantInitializer(Init, GV, InitPtr, DAG.getEntryNode(), DAG);
    for (SmallVector<SDNode*, 8>::iterator I = WorkList.begin(),
                                           E = WorkList.end(); I != E; ++I) {
      SmallVector<SDValue, 8> Ops;
      Ops.push_back(Chain);
      for (unsigned i = 1; i < (*I)->getNumOperands(); ++i) {
        Ops.push_back((*I)->getOperand(i));
      }
      DAG.UpdateNodeOperands(*I, Ops);
    }
    return DAG.getZExtOrTrunc(InitPtr, SDLoc(Op), ConstPtrVT);
  }
  }
}

SDValue AMDGPUTargetLowering::LowerCONCAT_VECTORS(SDValue Op,
                                                  SelectionDAG &DAG) const {
  SmallVector<SDValue, 8> Args;
  SDValue A = Op.getOperand(0);
  SDValue B = Op.getOperand(1);

  DAG.ExtractVectorElements(A, Args);
  DAG.ExtractVectorElements(B, Args);

  return DAG.getNode(ISD::BUILD_VECTOR, SDLoc(Op), Op.getValueType(), Args);
}

SDValue AMDGPUTargetLowering::LowerEXTRACT_SUBVECTOR(SDValue Op,
                                                     SelectionDAG &DAG) const {

  SmallVector<SDValue, 8> Args;
  unsigned Start = cast<ConstantSDNode>(Op.getOperand(1))->getZExtValue();
  EVT VT = Op.getValueType();
  DAG.ExtractVectorElements(Op.getOperand(0), Args, Start,
                            VT.getVectorNumElements());

  return DAG.getNode(ISD::BUILD_VECTOR, SDLoc(Op), Op.getValueType(), Args);
}

SDValue AMDGPUTargetLowering::LowerFrameIndex(SDValue Op,
                                              SelectionDAG &DAG) const {

  MachineFunction &MF = DAG.getMachineFunction();
  const AMDGPUFrameLowering *TFL =
   static_cast<const AMDGPUFrameLowering*>(getTargetMachine().getFrameLowering());

  FrameIndexSDNode *FIN = cast<FrameIndexSDNode>(Op);

  unsigned FrameIndex = FIN->getIndex();
  unsigned Offset = TFL->getFrameIndexOffset(MF, FrameIndex);
  return DAG.getConstant(Offset * 4 * TFL->getStackWidth(MF),
                         Op.getValueType());
}

SDValue AMDGPUTargetLowering::LowerINTRINSIC_WO_CHAIN(SDValue Op,
    SelectionDAG &DAG) const {
  unsigned IntrinsicID = cast<ConstantSDNode>(Op.getOperand(0))->getZExtValue();
  SDLoc DL(Op);
  EVT VT = Op.getValueType();

  switch (IntrinsicID) {
    default: return Op;
    case AMDGPUIntrinsic::AMDGPU_abs:
    case AMDGPUIntrinsic::AMDIL_abs: // Legacy name.
      return LowerIntrinsicIABS(Op, DAG);
    case AMDGPUIntrinsic::AMDGPU_lrp:
      return LowerIntrinsicLRP(Op, DAG);
    case AMDGPUIntrinsic::AMDGPU_fract:
    case AMDGPUIntrinsic::AMDIL_fraction: // Legacy name.
      return DAG.getNode(AMDGPUISD::FRACT, DL, VT, Op.getOperand(1));

    case AMDGPUIntrinsic::AMDGPU_clamp:
    case AMDGPUIntrinsic::AMDIL_clamp: // Legacy name.
      return DAG.getNode(AMDGPUISD::CLAMP, DL, VT,
                         Op.getOperand(1), Op.getOperand(2), Op.getOperand(3));

    case Intrinsic::AMDGPU_div_scale: {
      // 3rd parameter required to be a constant.
      const ConstantSDNode *Param = dyn_cast<ConstantSDNode>(Op.getOperand(3));
      if (!Param)
        return DAG.getUNDEF(VT);

      // Translate to the operands expected by the machine instruction. The
      // first parameter must be the same as the first instruction.
      SDValue Numerator = Op.getOperand(1);
      SDValue Denominator = Op.getOperand(2);
      SDValue Src0 = Param->isAllOnesValue() ? Numerator : Denominator;

      return DAG.getNode(AMDGPUISD::DIV_SCALE, DL, VT,
                         Src0, Denominator, Numerator);
    }

    case Intrinsic::AMDGPU_div_fmas:
      return DAG.getNode(AMDGPUISD::DIV_FMAS, DL, VT,
                         Op.getOperand(1), Op.getOperand(2), Op.getOperand(3));

    case Intrinsic::AMDGPU_div_fixup:
      return DAG.getNode(AMDGPUISD::DIV_FIXUP, DL, VT,
                         Op.getOperand(1), Op.getOperand(2), Op.getOperand(3));

    case Intrinsic::AMDGPU_trig_preop:
      return DAG.getNode(AMDGPUISD::TRIG_PREOP, DL, VT,
                         Op.getOperand(1), Op.getOperand(2));

    case Intrinsic::AMDGPU_rcp:
      return DAG.getNode(AMDGPUISD::RCP, DL, VT, Op.getOperand(1));

    case Intrinsic::AMDGPU_rsq:
      return DAG.getNode(AMDGPUISD::RSQ, DL, VT, Op.getOperand(1));

    case AMDGPUIntrinsic::AMDGPU_legacy_rsq:
      return DAG.getNode(AMDGPUISD::RSQ_LEGACY, DL, VT, Op.getOperand(1));

    case Intrinsic::AMDGPU_rsq_clamped:
      return DAG.getNode(AMDGPUISD::RSQ_CLAMPED, DL, VT, Op.getOperand(1));

    case AMDGPUIntrinsic::AMDGPU_imax:
      return DAG.getNode(AMDGPUISD::SMAX, DL, VT, Op.getOperand(1),
                                                  Op.getOperand(2));
    case AMDGPUIntrinsic::AMDGPU_umax:
      return DAG.getNode(AMDGPUISD::UMAX, DL, VT, Op.getOperand(1),
                                                  Op.getOperand(2));
    case AMDGPUIntrinsic::AMDGPU_imin:
      return DAG.getNode(AMDGPUISD::SMIN, DL, VT, Op.getOperand(1),
                                                  Op.getOperand(2));
    case AMDGPUIntrinsic::AMDGPU_umin:
      return DAG.getNode(AMDGPUISD::UMIN, DL, VT, Op.getOperand(1),
                                                  Op.getOperand(2));

    case AMDGPUIntrinsic::AMDGPU_umul24:
      return DAG.getNode(AMDGPUISD::MUL_U24, DL, VT,
                         Op.getOperand(1), Op.getOperand(2));

    case AMDGPUIntrinsic::AMDGPU_imul24:
      return DAG.getNode(AMDGPUISD::MUL_I24, DL, VT,
                         Op.getOperand(1), Op.getOperand(2));

    case AMDGPUIntrinsic::AMDGPU_umad24:
      return DAG.getNode(AMDGPUISD::MAD_U24, DL, VT,
                         Op.getOperand(1), Op.getOperand(2), Op.getOperand(3));

    case AMDGPUIntrinsic::AMDGPU_imad24:
      return DAG.getNode(AMDGPUISD::MAD_I24, DL, VT,
                         Op.getOperand(1), Op.getOperand(2), Op.getOperand(3));

    case AMDGPUIntrinsic::AMDGPU_cvt_f32_ubyte0:
      return DAG.getNode(AMDGPUISD::CVT_F32_UBYTE0, DL, VT, Op.getOperand(1));

    case AMDGPUIntrinsic::AMDGPU_cvt_f32_ubyte1:
      return DAG.getNode(AMDGPUISD::CVT_F32_UBYTE1, DL, VT, Op.getOperand(1));

    case AMDGPUIntrinsic::AMDGPU_cvt_f32_ubyte2:
      return DAG.getNode(AMDGPUISD::CVT_F32_UBYTE2, DL, VT, Op.getOperand(1));

    case AMDGPUIntrinsic::AMDGPU_cvt_f32_ubyte3:
      return DAG.getNode(AMDGPUISD::CVT_F32_UBYTE3, DL, VT, Op.getOperand(1));

    case AMDGPUIntrinsic::AMDGPU_bfe_i32:
      return DAG.getNode(AMDGPUISD::BFE_I32, DL, VT,
                         Op.getOperand(1),
                         Op.getOperand(2),
                         Op.getOperand(3));

    case AMDGPUIntrinsic::AMDGPU_bfe_u32:
      return DAG.getNode(AMDGPUISD::BFE_U32, DL, VT,
                         Op.getOperand(1),
                         Op.getOperand(2),
                         Op.getOperand(3));

    case AMDGPUIntrinsic::AMDGPU_bfi:
      return DAG.getNode(AMDGPUISD::BFI, DL, VT,
                         Op.getOperand(1),
                         Op.getOperand(2),
                         Op.getOperand(3));

    case AMDGPUIntrinsic::AMDGPU_bfm:
      return DAG.getNode(AMDGPUISD::BFM, DL, VT,
                         Op.getOperand(1),
                         Op.getOperand(2));

    case AMDGPUIntrinsic::AMDGPU_brev:
      return DAG.getNode(AMDGPUISD::BREV, DL, VT, Op.getOperand(1));

    case AMDGPUIntrinsic::AMDIL_exp: // Legacy name.
      return DAG.getNode(ISD::FEXP2, DL, VT, Op.getOperand(1));

    case AMDGPUIntrinsic::AMDIL_round_nearest: // Legacy name.
      return DAG.getNode(ISD::FRINT, DL, VT, Op.getOperand(1));
    case AMDGPUIntrinsic::AMDGPU_trunc: // Legacy name.
      return DAG.getNode(ISD::FTRUNC, DL, VT, Op.getOperand(1));
  }
}

///IABS(a) = SMAX(sub(0, a), a)
SDValue AMDGPUTargetLowering::LowerIntrinsicIABS(SDValue Op,
                                                 SelectionDAG &DAG) const {
  SDLoc DL(Op);
  EVT VT = Op.getValueType();
  SDValue Neg = DAG.getNode(ISD::SUB, DL, VT, DAG.getConstant(0, VT),
                                              Op.getOperand(1));

  return DAG.getNode(AMDGPUISD::SMAX, DL, VT, Neg, Op.getOperand(1));
}

/// Linear Interpolation
/// LRP(a, b, c) = muladd(a,  b, (1 - a) * c)
SDValue AMDGPUTargetLowering::LowerIntrinsicLRP(SDValue Op,
                                                SelectionDAG &DAG) const {
  SDLoc DL(Op);
  EVT VT = Op.getValueType();
  SDValue OneSubA = DAG.getNode(ISD::FSUB, DL, VT,
                                DAG.getConstantFP(1.0f, MVT::f32),
                                Op.getOperand(1));
  SDValue OneSubAC = DAG.getNode(ISD::FMUL, DL, VT, OneSubA,
                                                    Op.getOperand(3));
  return DAG.getNode(ISD::FADD, DL, VT,
      DAG.getNode(ISD::FMUL, DL, VT, Op.getOperand(1), Op.getOperand(2)),
      OneSubAC);
}

/// \brief Generate Min/Max node
SDValue AMDGPUTargetLowering::CombineMinMax(SDNode *N,
                                            SelectionDAG &DAG) const {
  SDLoc DL(N);
  EVT VT = N->getValueType(0);

  SDValue LHS = N->getOperand(0);
  SDValue RHS = N->getOperand(1);
  SDValue True = N->getOperand(2);
  SDValue False = N->getOperand(3);
  SDValue CC = N->getOperand(4);

  if (VT != MVT::f32 ||
      !((LHS == True && RHS == False) || (LHS == False && RHS == True))) {
    return SDValue();
  }

  ISD::CondCode CCOpcode = cast<CondCodeSDNode>(CC)->get();
  switch (CCOpcode) {
  case ISD::SETOEQ:
  case ISD::SETONE:
  case ISD::SETUNE:
  case ISD::SETNE:
  case ISD::SETUEQ:
  case ISD::SETEQ:
  case ISD::SETFALSE:
  case ISD::SETFALSE2:
  case ISD::SETTRUE:
  case ISD::SETTRUE2:
  case ISD::SETUO:
  case ISD::SETO:
    llvm_unreachable("Operation should already be optimised!");
  case ISD::SETULE:
  case ISD::SETULT:
  case ISD::SETOLE:
  case ISD::SETOLT:
  case ISD::SETLE:
  case ISD::SETLT: {
    unsigned Opc = (LHS == True) ? AMDGPUISD::FMIN : AMDGPUISD::FMAX;
    return DAG.getNode(Opc, DL, VT, LHS, RHS);
  }
  case ISD::SETGT:
  case ISD::SETGE:
  case ISD::SETUGE:
  case ISD::SETOGE:
  case ISD::SETUGT:
  case ISD::SETOGT: {
    unsigned Opc = (LHS == True) ? AMDGPUISD::FMAX : AMDGPUISD::FMIN;
    return DAG.getNode(Opc, DL, VT, LHS, RHS);
  }
  case ISD::SETCC_INVALID:
    llvm_unreachable("Invalid setcc condcode!");
  }
  return SDValue();
}

SDValue AMDGPUTargetLowering::SplitVectorLoad(const SDValue &Op,
                                              SelectionDAG &DAG) const {
  LoadSDNode *Load = dyn_cast<LoadSDNode>(Op);
  EVT MemEltVT = Load->getMemoryVT().getVectorElementType();
  EVT LoadVT = Op.getValueType();
  EVT EltVT = Op.getValueType().getVectorElementType();
  EVT PtrVT = Load->getBasePtr().getValueType();

  unsigned NumElts = Load->getMemoryVT().getVectorNumElements();
  SmallVector<SDValue, 8> Loads;
  SmallVector<SDValue, 8> Chains;

  SDLoc SL(Op);

  for (unsigned i = 0, e = NumElts; i != e; ++i) {
    SDValue Ptr = DAG.getNode(ISD::ADD, SL, PtrVT, Load->getBasePtr(),
                    DAG.getConstant(i * (MemEltVT.getSizeInBits() / 8), PtrVT));

    SDValue NewLoad
      = DAG.getExtLoad(Load->getExtensionType(), SL, EltVT,
                       Load->getChain(), Ptr,
                       MachinePointerInfo(Load->getMemOperand()->getValue()),
                       MemEltVT, Load->isVolatile(), Load->isNonTemporal(),
                       Load->getAlignment());
    Loads.push_back(NewLoad.getValue(0));
    Chains.push_back(NewLoad.getValue(1));
  }

  SDValue Ops[] = {
    DAG.getNode(ISD::BUILD_VECTOR, SL, LoadVT, Loads),
    DAG.getNode(ISD::TokenFactor, SL, MVT::Other, Chains)
  };

  return DAG.getMergeValues(Ops, SL);
}

SDValue AMDGPUTargetLowering::MergeVectorStore(const SDValue &Op,
                                               SelectionDAG &DAG) const {
  StoreSDNode *Store = cast<StoreSDNode>(Op);
  EVT MemVT = Store->getMemoryVT();
  unsigned MemBits = MemVT.getSizeInBits();

  // Byte stores are really expensive, so if possible, try to pack 32-bit vector
  // truncating store into an i32 store.
  // XXX: We could also handle optimize other vector bitwidths.
  if (!MemVT.isVector() || MemBits > 32) {
    return SDValue();
  }

  SDLoc DL(Op);
  SDValue Value = Store->getValue();
  EVT VT = Value.getValueType();
  EVT ElemVT = VT.getVectorElementType();
  SDValue Ptr = Store->getBasePtr();
  EVT MemEltVT = MemVT.getVectorElementType();
  unsigned MemEltBits = MemEltVT.getSizeInBits();
  unsigned MemNumElements = MemVT.getVectorNumElements();
  unsigned PackedSize = MemVT.getStoreSizeInBits();
  SDValue Mask = DAG.getConstant((1 << MemEltBits) - 1, MVT::i32);

  assert(Value.getValueType().getScalarSizeInBits() >= 32);

  SDValue PackedValue;
  for (unsigned i = 0; i < MemNumElements; ++i) {
    SDValue Elt = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, ElemVT, Value,
                              DAG.getConstant(i, MVT::i32));
    Elt = DAG.getZExtOrTrunc(Elt, DL, MVT::i32);
    Elt = DAG.getNode(ISD::AND, DL, MVT::i32, Elt, Mask); // getZeroExtendInReg

    SDValue Shift = DAG.getConstant(MemEltBits * i, MVT::i32);
    Elt = DAG.getNode(ISD::SHL, DL, MVT::i32, Elt, Shift);

    if (i == 0) {
      PackedValue = Elt;
    } else {
      PackedValue = DAG.getNode(ISD::OR, DL, MVT::i32, PackedValue, Elt);
    }
  }

  if (PackedSize < 32) {
    EVT PackedVT = EVT::getIntegerVT(*DAG.getContext(), PackedSize);
    return DAG.getTruncStore(Store->getChain(), DL, PackedValue, Ptr,
                             Store->getMemOperand()->getPointerInfo(),
                             PackedVT,
                             Store->isNonTemporal(), Store->isVolatile(),
                             Store->getAlignment());
  }

  return DAG.getStore(Store->getChain(), DL, PackedValue, Ptr,
                      Store->getMemOperand()->getPointerInfo(),
                      Store->isVolatile(),  Store->isNonTemporal(),
                      Store->getAlignment());
}

SDValue AMDGPUTargetLowering::SplitVectorStore(SDValue Op,
                                            SelectionDAG &DAG) const {
  StoreSDNode *Store = cast<StoreSDNode>(Op);
  EVT MemEltVT = Store->getMemoryVT().getVectorElementType();
  EVT EltVT = Store->getValue().getValueType().getVectorElementType();
  EVT PtrVT = Store->getBasePtr().getValueType();
  unsigned NumElts = Store->getMemoryVT().getVectorNumElements();
  SDLoc SL(Op);

  SmallVector<SDValue, 8> Chains;

  for (unsigned i = 0, e = NumElts; i != e; ++i) {
    SDValue Val = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, SL, EltVT,
                              Store->getValue(), DAG.getConstant(i, MVT::i32));
    SDValue Ptr = DAG.getNode(ISD::ADD, SL, PtrVT,
                              Store->getBasePtr(),
                            DAG.getConstant(i * (MemEltVT.getSizeInBits() / 8),
                                            PtrVT));
    Chains.push_back(DAG.getTruncStore(Store->getChain(), SL, Val, Ptr,
                         MachinePointerInfo(Store->getMemOperand()->getValue()),
                         MemEltVT, Store->isVolatile(), Store->isNonTemporal(),
                         Store->getAlignment()));
  }
  return DAG.getNode(ISD::TokenFactor, SL, MVT::Other, Chains);
}

SDValue AMDGPUTargetLowering::LowerLOAD(SDValue Op, SelectionDAG &DAG) const {
  SDLoc DL(Op);
  LoadSDNode *Load = cast<LoadSDNode>(Op);
  ISD::LoadExtType ExtType = Load->getExtensionType();
  EVT VT = Op.getValueType();
  EVT MemVT = Load->getMemoryVT();

  if (ExtType != ISD::NON_EXTLOAD && !VT.isVector() && VT.getSizeInBits() > 32) {
    // We can do the extload to 32-bits, and then need to separately extend to
    // 64-bits.

    SDValue ExtLoad32 = DAG.getExtLoad(ExtType, DL, MVT::i32,
                                       Load->getChain(),
                                       Load->getBasePtr(),
                                       MemVT,
                                       Load->getMemOperand());

    SDValue Ops[] = {
      DAG.getNode(ISD::getExtForLoadExtType(ExtType), DL, VT, ExtLoad32),
      ExtLoad32.getValue(1)
    };

    return DAG.getMergeValues(Ops, DL);
  }

  if (ExtType == ISD::NON_EXTLOAD && VT.getSizeInBits() < 32) {
    assert(VT == MVT::i1 && "Only i1 non-extloads expected");
    // FIXME: Copied from PPC
    // First, load into 32 bits, then truncate to 1 bit.

    SDValue Chain = Load->getChain();
    SDValue BasePtr = Load->getBasePtr();
    MachineMemOperand *MMO = Load->getMemOperand();

    SDValue NewLD = DAG.getExtLoad(ISD::EXTLOAD, DL, MVT::i32, Chain,
                                   BasePtr, MVT::i8, MMO);

    SDValue Ops[] = {
      DAG.getNode(ISD::TRUNCATE, DL, VT, NewLD),
      NewLD.getValue(1)
    };

    return DAG.getMergeValues(Ops, DL);
  }

  if (Subtarget->getGeneration() >= AMDGPUSubtarget::SOUTHERN_ISLANDS ||
      Load->getAddressSpace() != AMDGPUAS::PRIVATE_ADDRESS ||
      ExtType == ISD::NON_EXTLOAD || Load->getMemoryVT().bitsGE(MVT::i32))
    return SDValue();


  SDValue Ptr = DAG.getNode(ISD::SRL, DL, MVT::i32, Load->getBasePtr(),
                            DAG.getConstant(2, MVT::i32));
  SDValue Ret = DAG.getNode(AMDGPUISD::REGISTER_LOAD, DL, Op.getValueType(),
                            Load->getChain(), Ptr,
                            DAG.getTargetConstant(0, MVT::i32),
                            Op.getOperand(2));
  SDValue ByteIdx = DAG.getNode(ISD::AND, DL, MVT::i32,
                                Load->getBasePtr(),
                                DAG.getConstant(0x3, MVT::i32));
  SDValue ShiftAmt = DAG.getNode(ISD::SHL, DL, MVT::i32, ByteIdx,
                                 DAG.getConstant(3, MVT::i32));

  Ret = DAG.getNode(ISD::SRL, DL, MVT::i32, Ret, ShiftAmt);

  EVT MemEltVT = MemVT.getScalarType();
  if (ExtType == ISD::SEXTLOAD) {
    SDValue MemEltVTNode = DAG.getValueType(MemEltVT);

    SDValue Ops[] = {
      DAG.getNode(ISD::SIGN_EXTEND_INREG, DL, MVT::i32, Ret, MemEltVTNode),
      Load->getChain()
    };

    return DAG.getMergeValues(Ops, DL);
  }

  SDValue Ops[] = {
    DAG.getZeroExtendInReg(Ret, DL, MemEltVT),
    Load->getChain()
  };

  return DAG.getMergeValues(Ops, DL);
}

SDValue AMDGPUTargetLowering::LowerSTORE(SDValue Op, SelectionDAG &DAG) const {
  SDLoc DL(Op);
  SDValue Result = AMDGPUTargetLowering::MergeVectorStore(Op, DAG);
  if (Result.getNode()) {
    return Result;
  }

  StoreSDNode *Store = cast<StoreSDNode>(Op);
  SDValue Chain = Store->getChain();
  if ((Store->getAddressSpace() == AMDGPUAS::LOCAL_ADDRESS ||
       Store->getAddressSpace() == AMDGPUAS::PRIVATE_ADDRESS) &&
      Store->getValue().getValueType().isVector()) {
    return SplitVectorStore(Op, DAG);
  }

  EVT MemVT = Store->getMemoryVT();
  if (Store->getAddressSpace() == AMDGPUAS::PRIVATE_ADDRESS &&
      MemVT.bitsLT(MVT::i32)) {
    unsigned Mask = 0;
    if (Store->getMemoryVT() == MVT::i8) {
      Mask = 0xff;
    } else if (Store->getMemoryVT() == MVT::i16) {
      Mask = 0xffff;
    }
    SDValue BasePtr = Store->getBasePtr();
    SDValue Ptr = DAG.getNode(ISD::SRL, DL, MVT::i32, BasePtr,
                              DAG.getConstant(2, MVT::i32));
    SDValue Dst = DAG.getNode(AMDGPUISD::REGISTER_LOAD, DL, MVT::i32,
                              Chain, Ptr, DAG.getTargetConstant(0, MVT::i32));

    SDValue ByteIdx = DAG.getNode(ISD::AND, DL, MVT::i32, BasePtr,
                                  DAG.getConstant(0x3, MVT::i32));

    SDValue ShiftAmt = DAG.getNode(ISD::SHL, DL, MVT::i32, ByteIdx,
                                   DAG.getConstant(3, MVT::i32));

    SDValue SExtValue = DAG.getNode(ISD::SIGN_EXTEND, DL, MVT::i32,
                                    Store->getValue());

    SDValue MaskedValue = DAG.getZeroExtendInReg(SExtValue, DL, MemVT);

    SDValue ShiftedValue = DAG.getNode(ISD::SHL, DL, MVT::i32,
                                       MaskedValue, ShiftAmt);

    SDValue DstMask = DAG.getNode(ISD::SHL, DL, MVT::i32, DAG.getConstant(Mask, MVT::i32),
                                  ShiftAmt);
    DstMask = DAG.getNode(ISD::XOR, DL, MVT::i32, DstMask,
                          DAG.getConstant(0xffffffff, MVT::i32));
    Dst = DAG.getNode(ISD::AND, DL, MVT::i32, Dst, DstMask);

    SDValue Value = DAG.getNode(ISD::OR, DL, MVT::i32, Dst, ShiftedValue);
    return DAG.getNode(AMDGPUISD::REGISTER_STORE, DL, MVT::Other,
                       Chain, Value, Ptr, DAG.getTargetConstant(0, MVT::i32));
  }
  return SDValue();
}

SDValue AMDGPUTargetLowering::LowerSDIV24(SDValue Op, SelectionDAG &DAG) const {
  SDLoc DL(Op);
  EVT OVT = Op.getValueType();
  SDValue LHS = Op.getOperand(0);
  SDValue RHS = Op.getOperand(1);
  MVT INTTY;
  MVT FLTTY;
  if (!OVT.isVector()) {
    INTTY = MVT::i32;
    FLTTY = MVT::f32;
  } else if (OVT.getVectorNumElements() == 2) {
    INTTY = MVT::v2i32;
    FLTTY = MVT::v2f32;
  } else if (OVT.getVectorNumElements() == 4) {
    INTTY = MVT::v4i32;
    FLTTY = MVT::v4f32;
  }
  unsigned bitsize = OVT.getScalarType().getSizeInBits();
  // char|short jq = ia ^ ib;
  SDValue jq = DAG.getNode(ISD::XOR, DL, OVT, LHS, RHS);

  // jq = jq >> (bitsize - 2)
  jq = DAG.getNode(ISD::SRA, DL, OVT, jq, DAG.getConstant(bitsize - 2, OVT));

  // jq = jq | 0x1
  jq = DAG.getNode(ISD::OR, DL, OVT, jq, DAG.getConstant(1, OVT));

  // jq = (int)jq
  jq = DAG.getSExtOrTrunc(jq, DL, INTTY);

  // int ia = (int)LHS;
  SDValue ia = DAG.getSExtOrTrunc(LHS, DL, INTTY);

  // int ib, (int)RHS;
  SDValue ib = DAG.getSExtOrTrunc(RHS, DL, INTTY);

  // float fa = (float)ia;
  SDValue fa = DAG.getNode(ISD::SINT_TO_FP, DL, FLTTY, ia);

  // float fb = (float)ib;
  SDValue fb = DAG.getNode(ISD::SINT_TO_FP, DL, FLTTY, ib);

  // float fq = native_divide(fa, fb);
  SDValue fq = DAG.getNode(ISD::FMUL, DL, FLTTY,
                           fa, DAG.getNode(AMDGPUISD::RCP, DL, FLTTY, fb));

  // fq = trunc(fq);
  fq = DAG.getNode(ISD::FTRUNC, DL, FLTTY, fq);

  // float fqneg = -fq;
  SDValue fqneg = DAG.getNode(ISD::FNEG, DL, FLTTY, fq);

  // float fr = mad(fqneg, fb, fa);
  SDValue fr = DAG.getNode(ISD::FADD, DL, FLTTY,
      DAG.getNode(ISD::MUL, DL, FLTTY, fqneg, fb), fa);

  // int iq = (int)fq;
  SDValue iq = DAG.getNode(ISD::FP_TO_SINT, DL, INTTY, fq);

  // fr = fabs(fr);
  fr = DAG.getNode(ISD::FABS, DL, FLTTY, fr);

  // fb = fabs(fb);
  fb = DAG.getNode(ISD::FABS, DL, FLTTY, fb);

  // int cv = fr >= fb;
  SDValue cv;
  if (INTTY == MVT::i32) {
    cv = DAG.getSetCC(DL, INTTY, fr, fb, ISD::SETOGE);
  } else {
    cv = DAG.getSetCC(DL, INTTY, fr, fb, ISD::SETOGE);
  }
  // jq = (cv ? jq : 0);
  jq = DAG.getNode(ISD::SELECT, DL, OVT, cv, jq,
      DAG.getConstant(0, OVT));
  // dst = iq + jq;
  iq = DAG.getSExtOrTrunc(iq, DL, OVT);
  iq = DAG.getNode(ISD::ADD, DL, OVT, iq, jq);
  return iq;
}

SDValue AMDGPUTargetLowering::LowerSDIV32(SDValue Op, SelectionDAG &DAG) const {
  SDLoc DL(Op);
  EVT OVT = Op.getValueType();
  SDValue LHS = Op.getOperand(0);
  SDValue RHS = Op.getOperand(1);
  // The LowerSDIV32 function generates equivalent to the following IL.
  // mov r0, LHS
  // mov r1, RHS
  // ilt r10, r0, 0
  // ilt r11, r1, 0
  // iadd r0, r0, r10
  // iadd r1, r1, r11
  // ixor r0, r0, r10
  // ixor r1, r1, r11
  // udiv r0, r0, r1
  // ixor r10, r10, r11
  // iadd r0, r0, r10
  // ixor DST, r0, r10

  // mov r0, LHS
  SDValue r0 = LHS;

  // mov r1, RHS
  SDValue r1 = RHS;

  // ilt r10, r0, 0
  SDValue r10 = DAG.getSelectCC(DL,
      r0, DAG.getConstant(0, OVT),
      DAG.getConstant(-1, OVT),
      DAG.getConstant(0, OVT),
      ISD::SETLT);

  // ilt r11, r1, 0
  SDValue r11 = DAG.getSelectCC(DL,
      r1, DAG.getConstant(0, OVT),
      DAG.getConstant(-1, OVT),
      DAG.getConstant(0, OVT),
      ISD::SETLT);

  // iadd r0, r0, r10
  r0 = DAG.getNode(ISD::ADD, DL, OVT, r0, r10);

  // iadd r1, r1, r11
  r1 = DAG.getNode(ISD::ADD, DL, OVT, r1, r11);

  // ixor r0, r0, r10
  r0 = DAG.getNode(ISD::XOR, DL, OVT, r0, r10);

  // ixor r1, r1, r11
  r1 = DAG.getNode(ISD::XOR, DL, OVT, r1, r11);

  // udiv r0, r0, r1
  r0 = DAG.getNode(ISD::UDIV, DL, OVT, r0, r1);

  // ixor r10, r10, r11
  r10 = DAG.getNode(ISD::XOR, DL, OVT, r10, r11);

  // iadd r0, r0, r10
  r0 = DAG.getNode(ISD::ADD, DL, OVT, r0, r10);

  // ixor DST, r0, r10
  SDValue DST = DAG.getNode(ISD::XOR, DL, OVT, r0, r10);
  return DST;
}

SDValue AMDGPUTargetLowering::LowerSDIV64(SDValue Op, SelectionDAG &DAG) const {
  return SDValue(Op.getNode(), 0);
}

SDValue AMDGPUTargetLowering::LowerSDIV(SDValue Op, SelectionDAG &DAG) const {
  EVT OVT = Op.getValueType().getScalarType();

  if (OVT == MVT::i64)
    return LowerSDIV64(Op, DAG);

  if (OVT.getScalarType() == MVT::i32)
    return LowerSDIV32(Op, DAG);

  if (OVT == MVT::i16 || OVT == MVT::i8) {
    // FIXME: We should be checking for the masked bits. This isn't reached
    // because i8 and i16 are not legal types.
    return LowerSDIV24(Op, DAG);
  }

  return SDValue(Op.getNode(), 0);
}

SDValue AMDGPUTargetLowering::LowerSREM32(SDValue Op, SelectionDAG &DAG) const {
  SDLoc DL(Op);
  EVT OVT = Op.getValueType();
  SDValue LHS = Op.getOperand(0);
  SDValue RHS = Op.getOperand(1);
  // The LowerSREM32 function generates equivalent to the following IL.
  // mov r0, LHS
  // mov r1, RHS
  // ilt r10, r0, 0
  // ilt r11, r1, 0
  // iadd r0, r0, r10
  // iadd r1, r1, r11
  // ixor r0, r0, r10
  // ixor r1, r1, r11
  // udiv r20, r0, r1
  // umul r20, r20, r1
  // sub r0, r0, r20
  // iadd r0, r0, r10
  // ixor DST, r0, r10

  // mov r0, LHS
  SDValue r0 = LHS;

  // mov r1, RHS
  SDValue r1 = RHS;

  // ilt r10, r0, 0
  SDValue r10 = DAG.getSetCC(DL, OVT, r0, DAG.getConstant(0, OVT), ISD::SETLT);

  // ilt r11, r1, 0
  SDValue r11 = DAG.getSetCC(DL, OVT, r1, DAG.getConstant(0, OVT), ISD::SETLT);

  // iadd r0, r0, r10
  r0 = DAG.getNode(ISD::ADD, DL, OVT, r0, r10);

  // iadd r1, r1, r11
  r1 = DAG.getNode(ISD::ADD, DL, OVT, r1, r11);

  // ixor r0, r0, r10
  r0 = DAG.getNode(ISD::XOR, DL, OVT, r0, r10);

  // ixor r1, r1, r11
  r1 = DAG.getNode(ISD::XOR, DL, OVT, r1, r11);

  // udiv r20, r0, r1
  SDValue r20 = DAG.getNode(ISD::UREM, DL, OVT, r0, r1);

  // umul r20, r20, r1
  r20 = DAG.getNode(AMDGPUISD::UMUL, DL, OVT, r20, r1);

  // sub r0, r0, r20
  r0 = DAG.getNode(ISD::SUB, DL, OVT, r0, r20);

  // iadd r0, r0, r10
  r0 = DAG.getNode(ISD::ADD, DL, OVT, r0, r10);

  // ixor DST, r0, r10
  SDValue DST = DAG.getNode(ISD::XOR, DL, OVT, r0, r10);
  return DST;
}

SDValue AMDGPUTargetLowering::LowerSREM64(SDValue Op, SelectionDAG &DAG) const {
  return SDValue(Op.getNode(), 0);
}

SDValue AMDGPUTargetLowering::LowerSREM(SDValue Op, SelectionDAG &DAG) const {
  EVT OVT = Op.getValueType();

  if (OVT.getScalarType() == MVT::i64)
    return LowerSREM64(Op, DAG);

  if (OVT.getScalarType() == MVT::i32)
    return LowerSREM32(Op, DAG);

  return SDValue(Op.getNode(), 0);
}

SDValue AMDGPUTargetLowering::LowerUDIVREM(SDValue Op,
                                           SelectionDAG &DAG) const {
  SDLoc DL(Op);
  EVT VT = Op.getValueType();

  SDValue Num = Op.getOperand(0);
  SDValue Den = Op.getOperand(1);

  // RCP =  URECIP(Den) = 2^32 / Den + e
  // e is rounding error.
  SDValue RCP = DAG.getNode(AMDGPUISD::URECIP, DL, VT, Den);

  // RCP_LO = umulo(RCP, Den) */
  SDValue RCP_LO = DAG.getNode(ISD::UMULO, DL, VT, RCP, Den);

  // RCP_HI = mulhu (RCP, Den) */
  SDValue RCP_HI = DAG.getNode(ISD::MULHU, DL, VT, RCP, Den);

  // NEG_RCP_LO = -RCP_LO
  SDValue NEG_RCP_LO = DAG.getNode(ISD::SUB, DL, VT, DAG.getConstant(0, VT),
                                                     RCP_LO);

  // ABS_RCP_LO = (RCP_HI == 0 ? NEG_RCP_LO : RCP_LO)
  SDValue ABS_RCP_LO = DAG.getSelectCC(DL, RCP_HI, DAG.getConstant(0, VT),
                                           NEG_RCP_LO, RCP_LO,
                                           ISD::SETEQ);
  // Calculate the rounding error from the URECIP instruction
  // E = mulhu(ABS_RCP_LO, RCP)
  SDValue E = DAG.getNode(ISD::MULHU, DL, VT, ABS_RCP_LO, RCP);

  // RCP_A_E = RCP + E
  SDValue RCP_A_E = DAG.getNode(ISD::ADD, DL, VT, RCP, E);

  // RCP_S_E = RCP - E
  SDValue RCP_S_E = DAG.getNode(ISD::SUB, DL, VT, RCP, E);

  // Tmp0 = (RCP_HI == 0 ? RCP_A_E : RCP_SUB_E)
  SDValue Tmp0 = DAG.getSelectCC(DL, RCP_HI, DAG.getConstant(0, VT),
                                     RCP_A_E, RCP_S_E,
                                     ISD::SETEQ);
  // Quotient = mulhu(Tmp0, Num)
  SDValue Quotient = DAG.getNode(ISD::MULHU, DL, VT, Tmp0, Num);

  // Num_S_Remainder = Quotient * Den
  SDValue Num_S_Remainder = DAG.getNode(ISD::UMULO, DL, VT, Quotient, Den);

  // Remainder = Num - Num_S_Remainder
  SDValue Remainder = DAG.getNode(ISD::SUB, DL, VT, Num, Num_S_Remainder);

  // Remainder_GE_Den = (Remainder >= Den ? -1 : 0)
  SDValue Remainder_GE_Den = DAG.getSelectCC(DL, Remainder, Den,
                                                 DAG.getConstant(-1, VT),
                                                 DAG.getConstant(0, VT),
                                                 ISD::SETUGE);
  // Remainder_GE_Zero = (Num >= Num_S_Remainder ? -1 : 0)
  SDValue Remainder_GE_Zero = DAG.getSelectCC(DL, Num,
                                                  Num_S_Remainder,
                                                  DAG.getConstant(-1, VT),
                                                  DAG.getConstant(0, VT),
                                                  ISD::SETUGE);
  // Tmp1 = Remainder_GE_Den & Remainder_GE_Zero
  SDValue Tmp1 = DAG.getNode(ISD::AND, DL, VT, Remainder_GE_Den,
                                               Remainder_GE_Zero);

  // Calculate Division result:

  // Quotient_A_One = Quotient + 1
  SDValue Quotient_A_One = DAG.getNode(ISD::ADD, DL, VT, Quotient,
                                                         DAG.getConstant(1, VT));

  // Quotient_S_One = Quotient - 1
  SDValue Quotient_S_One = DAG.getNode(ISD::SUB, DL, VT, Quotient,
                                                         DAG.getConstant(1, VT));

  // Div = (Tmp1 == 0 ? Quotient : Quotient_A_One)
  SDValue Div = DAG.getSelectCC(DL, Tmp1, DAG.getConstant(0, VT),
                                     Quotient, Quotient_A_One, ISD::SETEQ);

  // Div = (Remainder_GE_Zero == 0 ? Quotient_S_One : Div)
  Div = DAG.getSelectCC(DL, Remainder_GE_Zero, DAG.getConstant(0, VT),
                            Quotient_S_One, Div, ISD::SETEQ);

  // Calculate Rem result:

  // Remainder_S_Den = Remainder - Den
  SDValue Remainder_S_Den = DAG.getNode(ISD::SUB, DL, VT, Remainder, Den);

  // Remainder_A_Den = Remainder + Den
  SDValue Remainder_A_Den = DAG.getNode(ISD::ADD, DL, VT, Remainder, Den);

  // Rem = (Tmp1 == 0 ? Remainder : Remainder_S_Den)
  SDValue Rem = DAG.getSelectCC(DL, Tmp1, DAG.getConstant(0, VT),
                                    Remainder, Remainder_S_Den, ISD::SETEQ);

  // Rem = (Remainder_GE_Zero == 0 ? Remainder_A_Den : Rem)
  Rem = DAG.getSelectCC(DL, Remainder_GE_Zero, DAG.getConstant(0, VT),
                            Remainder_A_Den, Rem, ISD::SETEQ);
  SDValue Ops[2] = {
    Div,
    Rem
  };
  return DAG.getMergeValues(Ops, DL);
}

SDValue AMDGPUTargetLowering::LowerSDIVREM(SDValue Op,
                                           SelectionDAG &DAG) const {
  SDLoc DL(Op);
  EVT VT = Op.getValueType();

  SDValue Zero = DAG.getConstant(0, VT);
  SDValue NegOne = DAG.getConstant(-1, VT);

  SDValue LHS = Op.getOperand(0);
  SDValue RHS = Op.getOperand(1);

  SDValue LHSign = DAG.getSelectCC(DL, LHS, Zero, NegOne, Zero, ISD::SETLT);
  SDValue RHSign = DAG.getSelectCC(DL, RHS, Zero, NegOne, Zero, ISD::SETLT);
  SDValue DSign = DAG.getNode(ISD::XOR, DL, VT, LHSign, RHSign);
  SDValue RSign = LHSign; // Remainder sign is the same as LHS

  LHS = DAG.getNode(ISD::ADD, DL, VT, LHS, LHSign);
  RHS = DAG.getNode(ISD::ADD, DL, VT, RHS, RHSign);

  LHS = DAG.getNode(ISD::XOR, DL, VT, LHS, LHSign);
  RHS = DAG.getNode(ISD::XOR, DL, VT, RHS, RHSign);

  SDValue Div = DAG.getNode(ISD::UDIVREM, DL, DAG.getVTList(VT, VT), LHS, RHS);
  SDValue Rem = Div.getValue(1);

  Div = DAG.getNode(ISD::XOR, DL, VT, Div, DSign);
  Rem = DAG.getNode(ISD::XOR, DL, VT, Rem, RSign);

  Div = DAG.getNode(ISD::SUB, DL, VT, Div, DSign);
  Rem = DAG.getNode(ISD::SUB, DL, VT, Rem, RSign);

  SDValue Res[2] = {
    Div,
    Rem
  };
  return DAG.getMergeValues(Res, DL);
}

SDValue AMDGPUTargetLowering::LowerFCEIL(SDValue Op, SelectionDAG &DAG) const {
  SDLoc SL(Op);
  SDValue Src = Op.getOperand(0);

  // result = trunc(src)
  // if (src > 0.0 && src != result)
  //   result += 1.0

  SDValue Trunc = DAG.getNode(ISD::FTRUNC, SL, MVT::f64, Src);

  const SDValue Zero = DAG.getConstantFP(0.0, MVT::f64);
  const SDValue One = DAG.getConstantFP(1.0, MVT::f64);

  EVT SetCCVT = getSetCCResultType(*DAG.getContext(), MVT::f64);

  SDValue Lt0 = DAG.getSetCC(SL, SetCCVT, Src, Zero, ISD::SETOGT);
  SDValue NeTrunc = DAG.getSetCC(SL, SetCCVT, Src, Trunc, ISD::SETONE);
  SDValue And = DAG.getNode(ISD::AND, SL, SetCCVT, Lt0, NeTrunc);

  SDValue Add = DAG.getNode(ISD::SELECT, SL, MVT::f64, And, One, Zero);
  return DAG.getNode(ISD::FADD, SL, MVT::f64, Trunc, Add);
}

SDValue AMDGPUTargetLowering::LowerFTRUNC(SDValue Op, SelectionDAG &DAG) const {
  SDLoc SL(Op);
  SDValue Src = Op.getOperand(0);

  assert(Op.getValueType() == MVT::f64);

  const SDValue Zero = DAG.getConstant(0, MVT::i32);
  const SDValue One = DAG.getConstant(1, MVT::i32);

  SDValue VecSrc = DAG.getNode(ISD::BITCAST, SL, MVT::v2i32, Src);

  // Extract the upper half, since this is where we will find the sign and
  // exponent.
  SDValue Hi = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, SL, MVT::i32, VecSrc, One);

  const unsigned FractBits = 52;
  const unsigned ExpBits = 11;

  // Extract the exponent.
  SDValue ExpPart = DAG.getNode(AMDGPUISD::BFE_I32, SL, MVT::i32,
                                Hi,
                                DAG.getConstant(FractBits - 32, MVT::i32),
                                DAG.getConstant(ExpBits, MVT::i32));
  SDValue Exp = DAG.getNode(ISD::SUB, SL, MVT::i32, ExpPart,
                            DAG.getConstant(1023, MVT::i32));

  // Extract the sign bit.
  const SDValue SignBitMask = DAG.getConstant(UINT32_C(1) << 31, MVT::i32);
  SDValue SignBit = DAG.getNode(ISD::AND, SL, MVT::i32, Hi, SignBitMask);

  // Extend back to to 64-bits.
  SDValue SignBit64 = DAG.getNode(ISD::BUILD_VECTOR, SL, MVT::v2i32,
                                  Zero, SignBit);
  SignBit64 = DAG.getNode(ISD::BITCAST, SL, MVT::i64, SignBit64);

  SDValue BcInt = DAG.getNode(ISD::BITCAST, SL, MVT::i64, Src);
  const SDValue FractMask
    = DAG.getConstant((UINT64_C(1) << FractBits) - 1, MVT::i64);

  SDValue Shr = DAG.getNode(ISD::SRA, SL, MVT::i64, FractMask, Exp);
  SDValue Not = DAG.getNOT(SL, Shr, MVT::i64);
  SDValue Tmp0 = DAG.getNode(ISD::AND, SL, MVT::i64, BcInt, Not);

  EVT SetCCVT = getSetCCResultType(*DAG.getContext(), MVT::i32);

  const SDValue FiftyOne = DAG.getConstant(FractBits - 1, MVT::i32);

  SDValue ExpLt0 = DAG.getSetCC(SL, SetCCVT, Exp, Zero, ISD::SETLT);
  SDValue ExpGt51 = DAG.getSetCC(SL, SetCCVT, Exp, FiftyOne, ISD::SETGT);

  SDValue Tmp1 = DAG.getNode(ISD::SELECT, SL, MVT::i64, ExpLt0, SignBit64, Tmp0);
  SDValue Tmp2 = DAG.getNode(ISD::SELECT, SL, MVT::i64, ExpGt51, BcInt, Tmp1);

  return DAG.getNode(ISD::BITCAST, SL, MVT::f64, Tmp2);
}

SDValue AMDGPUTargetLowering::LowerFRINT(SDValue Op, SelectionDAG &DAG) const {
  SDLoc SL(Op);
  SDValue Src = Op.getOperand(0);

  assert(Op.getValueType() == MVT::f64);

  APFloat C1Val(APFloat::IEEEdouble, "0x1.0p+52");
  SDValue C1 = DAG.getConstantFP(C1Val, MVT::f64);
  SDValue CopySign = DAG.getNode(ISD::FCOPYSIGN, SL, MVT::f64, C1, Src);

  SDValue Tmp1 = DAG.getNode(ISD::FADD, SL, MVT::f64, Src, CopySign);
  SDValue Tmp2 = DAG.getNode(ISD::FSUB, SL, MVT::f64, Tmp1, CopySign);

  SDValue Fabs = DAG.getNode(ISD::FABS, SL, MVT::f64, Src);

  APFloat C2Val(APFloat::IEEEdouble, "0x1.fffffffffffffp+51");
  SDValue C2 = DAG.getConstantFP(C2Val, MVT::f64);

  EVT SetCCVT = getSetCCResultType(*DAG.getContext(), MVT::f64);
  SDValue Cond = DAG.getSetCC(SL, SetCCVT, Fabs, C2, ISD::SETOGT);

  return DAG.getSelect(SL, MVT::f64, Cond, Src, Tmp2);
}

SDValue AMDGPUTargetLowering::LowerFNEARBYINT(SDValue Op, SelectionDAG &DAG) const {
  // FNEARBYINT and FRINT are the same, except in their handling of FP
  // exceptions. Those aren't really meaningful for us, and OpenCL only has
  // rint, so just treat them as equivalent.
  return DAG.getNode(ISD::FRINT, SDLoc(Op), Op.getValueType(), Op.getOperand(0));
}

SDValue AMDGPUTargetLowering::LowerFFLOOR(SDValue Op, SelectionDAG &DAG) const {
  SDLoc SL(Op);
  SDValue Src = Op.getOperand(0);

  // result = trunc(src);
  // if (src < 0.0 && src != result)
  //   result += -1.0.

  SDValue Trunc = DAG.getNode(ISD::FTRUNC, SL, MVT::f64, Src);

  const SDValue Zero = DAG.getConstantFP(0.0, MVT::f64);
  const SDValue NegOne = DAG.getConstantFP(-1.0, MVT::f64);

  EVT SetCCVT = getSetCCResultType(*DAG.getContext(), MVT::f64);

  SDValue Lt0 = DAG.getSetCC(SL, SetCCVT, Src, Zero, ISD::SETOLT);
  SDValue NeTrunc = DAG.getSetCC(SL, SetCCVT, Src, Trunc, ISD::SETONE);
  SDValue And = DAG.getNode(ISD::AND, SL, SetCCVT, Lt0, NeTrunc);

  SDValue Add = DAG.getNode(ISD::SELECT, SL, MVT::f64, And, NegOne, Zero);
  return DAG.getNode(ISD::FADD, SL, MVT::f64, Trunc, Add);
}

SDValue AMDGPUTargetLowering::LowerUINT_TO_FP(SDValue Op,
                                               SelectionDAG &DAG) const {
  SDValue S0 = Op.getOperand(0);
  SDLoc DL(Op);
  if (Op.getValueType() != MVT::f32 || S0.getValueType() != MVT::i64)
    return SDValue();

  // f32 uint_to_fp i64
  SDValue Lo = DAG.getNode(ISD::EXTRACT_ELEMENT, DL, MVT::i32, S0,
                           DAG.getConstant(0, MVT::i32));
  SDValue FloatLo = DAG.getNode(ISD::UINT_TO_FP, DL, MVT::f32, Lo);
  SDValue Hi = DAG.getNode(ISD::EXTRACT_ELEMENT, DL, MVT::i32, S0,
                           DAG.getConstant(1, MVT::i32));
  SDValue FloatHi = DAG.getNode(ISD::UINT_TO_FP, DL, MVT::f32, Hi);
  FloatHi = DAG.getNode(ISD::FMUL, DL, MVT::f32, FloatHi,
                        DAG.getConstantFP(4294967296.0f, MVT::f32)); // 2^32
  return DAG.getNode(ISD::FADD, DL, MVT::f32, FloatLo, FloatHi);
}

SDValue AMDGPUTargetLowering::ExpandSIGN_EXTEND_INREG(SDValue Op,
                                                      unsigned BitsDiff,
                                                      SelectionDAG &DAG) const {
  MVT VT = Op.getSimpleValueType();
  SDLoc DL(Op);
  SDValue Shift = DAG.getConstant(BitsDiff, VT);
  // Shift left by 'Shift' bits.
  SDValue Shl = DAG.getNode(ISD::SHL, DL, VT, Op.getOperand(0), Shift);
  // Signed shift Right by 'Shift' bits.
  return DAG.getNode(ISD::SRA, DL, VT, Shl, Shift);
}

SDValue AMDGPUTargetLowering::LowerSIGN_EXTEND_INREG(SDValue Op,
                                                     SelectionDAG &DAG) const {
  EVT ExtraVT = cast<VTSDNode>(Op.getOperand(1))->getVT();
  MVT VT = Op.getSimpleValueType();
  MVT ScalarVT = VT.getScalarType();

  if (!VT.isVector())
    return SDValue();

  SDValue Src = Op.getOperand(0);
  SDLoc DL(Op);

  // TODO: Don't scalarize on Evergreen?
  unsigned NElts = VT.getVectorNumElements();
  SmallVector<SDValue, 8> Args;
  DAG.ExtractVectorElements(Src, Args, 0, NElts);

  SDValue VTOp = DAG.getValueType(ExtraVT.getScalarType());
  for (unsigned I = 0; I < NElts; ++I)
    Args[I] = DAG.getNode(ISD::SIGN_EXTEND_INREG, DL, ScalarVT, Args[I], VTOp);

  return DAG.getNode(ISD::BUILD_VECTOR, DL, VT, Args);
}

//===----------------------------------------------------------------------===//
// Custom DAG optimizations
//===----------------------------------------------------------------------===//

static bool isU24(SDValue Op, SelectionDAG &DAG) {
  APInt KnownZero, KnownOne;
  EVT VT = Op.getValueType();
  DAG.computeKnownBits(Op, KnownZero, KnownOne);

  return (VT.getSizeInBits() - KnownZero.countLeadingOnes()) <= 24;
}

static bool isI24(SDValue Op, SelectionDAG &DAG) {
  EVT VT = Op.getValueType();

  // In order for this to be a signed 24-bit value, bit 23, must
  // be a sign bit.
  return VT.getSizeInBits() >= 24 && // Types less than 24-bit should be treated
                                     // as unsigned 24-bit values.
         (VT.getSizeInBits() - DAG.ComputeNumSignBits(Op)) < 24;
}

static void simplifyI24(SDValue Op, TargetLowering::DAGCombinerInfo &DCI) {

  SelectionDAG &DAG = DCI.DAG;
  const TargetLowering &TLI = DAG.getTargetLoweringInfo();
  EVT VT = Op.getValueType();

  APInt Demanded = APInt::getLowBitsSet(VT.getSizeInBits(), 24);
  APInt KnownZero, KnownOne;
  TargetLowering::TargetLoweringOpt TLO(DAG, true, true);
  if (TLI.SimplifyDemandedBits(Op, Demanded, KnownZero, KnownOne, TLO))
    DCI.CommitTargetLoweringOpt(TLO);
}

template <typename IntTy>
static SDValue constantFoldBFE(SelectionDAG &DAG, IntTy Src0,
                               uint32_t Offset, uint32_t Width) {
  if (Width + Offset < 32) {
    IntTy Result = (Src0 << (32 - Offset - Width)) >> (32 - Width);
    return DAG.getConstant(Result, MVT::i32);
  }

  return DAG.getConstant(Src0 >> Offset, MVT::i32);
}

static bool usesAllNormalStores(SDNode *LoadVal) {
  for (SDNode::use_iterator I = LoadVal->use_begin(); !I.atEnd(); ++I) {
    if (!ISD::isNormalStore(*I))
      return false;
  }

  return true;
}

// If we have a copy of an illegal type, replace it with a load / store of an
// equivalently sized legal type. This avoids intermediate bit pack / unpack
// instructions emitted when handling extloads and truncstores. Ideally we could
// recognize the pack / unpack pattern to eliminate it.
SDValue AMDGPUTargetLowering::performStoreCombine(SDNode *N,
                                                  DAGCombinerInfo &DCI) const {
  if (!DCI.isBeforeLegalize())
    return SDValue();

  StoreSDNode *SN = cast<StoreSDNode>(N);
  SDValue Value = SN->getValue();
  EVT VT = Value.getValueType();

  if (isTypeLegal(VT) || SN->isVolatile() || !ISD::isNormalLoad(Value.getNode()))
    return SDValue();

  LoadSDNode *LoadVal = cast<LoadSDNode>(Value);
  if (LoadVal->isVolatile() || !usesAllNormalStores(LoadVal))
    return SDValue();

  EVT MemVT = LoadVal->getMemoryVT();

  SDLoc SL(N);
  SelectionDAG &DAG = DCI.DAG;
  EVT LoadVT = getEquivalentMemType(*DAG.getContext(), MemVT);

  SDValue NewLoad = DAG.getLoad(ISD::UNINDEXED, ISD::NON_EXTLOAD,
                                LoadVT, SL,
                                LoadVal->getChain(),
                                LoadVal->getBasePtr(),
                                LoadVal->getOffset(),
                                LoadVT,
                                LoadVal->getMemOperand());

  SDValue CastLoad = DAG.getNode(ISD::BITCAST, SL, VT, NewLoad.getValue(0));
  DCI.CombineTo(LoadVal, CastLoad, NewLoad.getValue(1), false);

  return DAG.getStore(SN->getChain(), SL, NewLoad,
                      SN->getBasePtr(), SN->getMemOperand());
}

SDValue AMDGPUTargetLowering::performMulCombine(SDNode *N,
                                                DAGCombinerInfo &DCI) const {
  EVT VT = N->getValueType(0);

  if (VT.isVector() || VT.getSizeInBits() > 32)
    return SDValue();

  SelectionDAG &DAG = DCI.DAG;
  SDLoc DL(N);

  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  SDValue Mul;

  if (Subtarget->hasMulU24() && isU24(N0, DAG) && isU24(N1, DAG)) {
    N0 = DAG.getZExtOrTrunc(N0, DL, MVT::i32);
    N1 = DAG.getZExtOrTrunc(N1, DL, MVT::i32);
    Mul = DAG.getNode(AMDGPUISD::MUL_U24, DL, MVT::i32, N0, N1);
  } else if (Subtarget->hasMulI24() && isI24(N0, DAG) && isI24(N1, DAG)) {
    N0 = DAG.getSExtOrTrunc(N0, DL, MVT::i32);
    N1 = DAG.getSExtOrTrunc(N1, DL, MVT::i32);
    Mul = DAG.getNode(AMDGPUISD::MUL_I24, DL, MVT::i32, N0, N1);
  } else {
    return SDValue();
  }

  // We need to use sext even for MUL_U24, because MUL_U24 is used
  // for signed multiply of 8 and 16-bit types.
  return DAG.getSExtOrTrunc(Mul, DL, VT);
}

SDValue AMDGPUTargetLowering::PerformDAGCombine(SDNode *N,
                                                DAGCombinerInfo &DCI) const {
  SelectionDAG &DAG = DCI.DAG;
  SDLoc DL(N);

  switch(N->getOpcode()) {
    default: break;
    case ISD::MUL:
      return performMulCombine(N, DCI);
    case AMDGPUISD::MUL_I24:
    case AMDGPUISD::MUL_U24: {
      SDValue N0 = N->getOperand(0);
      SDValue N1 = N->getOperand(1);
      simplifyI24(N0, DCI);
      simplifyI24(N1, DCI);
      return SDValue();
    }
    case ISD::SELECT_CC: {
      return CombineMinMax(N, DAG);
    }
  case AMDGPUISD::BFE_I32:
  case AMDGPUISD::BFE_U32: {
    assert(!N->getValueType(0).isVector() &&
           "Vector handling of BFE not implemented");
    ConstantSDNode *Width = dyn_cast<ConstantSDNode>(N->getOperand(2));
    if (!Width)
      break;

    uint32_t WidthVal = Width->getZExtValue() & 0x1f;
    if (WidthVal == 0)
      return DAG.getConstant(0, MVT::i32);

    ConstantSDNode *Offset = dyn_cast<ConstantSDNode>(N->getOperand(1));
    if (!Offset)
      break;

    SDValue BitsFrom = N->getOperand(0);
    uint32_t OffsetVal = Offset->getZExtValue() & 0x1f;

    bool Signed = N->getOpcode() == AMDGPUISD::BFE_I32;

    if (OffsetVal == 0) {
      // This is already sign / zero extended, so try to fold away extra BFEs.
      unsigned SignBits =  Signed ? (32 - WidthVal + 1) : (32 - WidthVal);

      unsigned OpSignBits = DAG.ComputeNumSignBits(BitsFrom);
      if (OpSignBits >= SignBits)
        return BitsFrom;

      EVT SmallVT = EVT::getIntegerVT(*DAG.getContext(), WidthVal);
      if (Signed) {
        // This is a sign_extend_inreg. Replace it to take advantage of existing
        // DAG Combines. If not eliminated, we will match back to BFE during
        // selection.

        // TODO: The sext_inreg of extended types ends, although we can could
        // handle them in a single BFE.
        return DAG.getNode(ISD::SIGN_EXTEND_INREG, DL, MVT::i32, BitsFrom,
                           DAG.getValueType(SmallVT));
      }

      return DAG.getZeroExtendInReg(BitsFrom, DL, SmallVT);
    }

    if (ConstantSDNode *Val = dyn_cast<ConstantSDNode>(N->getOperand(0))) {
      if (Signed) {
        return constantFoldBFE<int32_t>(DAG,
                                        Val->getSExtValue(),
                                        OffsetVal,
                                        WidthVal);
      }

      return constantFoldBFE<uint32_t>(DAG,
                                       Val->getZExtValue(),
                                       OffsetVal,
                                       WidthVal);
    }

    APInt Demanded = APInt::getBitsSet(32,
                                       OffsetVal,
                                       OffsetVal + WidthVal);

    if ((OffsetVal + WidthVal) >= 32) {
      SDValue ShiftVal = DAG.getConstant(OffsetVal, MVT::i32);
      return DAG.getNode(Signed ? ISD::SRA : ISD::SRL, DL, MVT::i32,
                         BitsFrom, ShiftVal);
    }

    APInt KnownZero, KnownOne;
    TargetLowering::TargetLoweringOpt TLO(DAG, !DCI.isBeforeLegalize(),
                                          !DCI.isBeforeLegalizeOps());
    const TargetLowering &TLI = DAG.getTargetLoweringInfo();
    if (TLO.ShrinkDemandedConstant(BitsFrom, Demanded) ||
        TLI.SimplifyDemandedBits(BitsFrom, Demanded, KnownZero, KnownOne, TLO)) {
      DCI.CommitTargetLoweringOpt(TLO);
    }

    break;
  }

  case ISD::STORE:
    return performStoreCombine(N, DCI);
  }
  return SDValue();
}

//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//

void AMDGPUTargetLowering::getOriginalFunctionArgs(
                               SelectionDAG &DAG,
                               const Function *F,
                               const SmallVectorImpl<ISD::InputArg> &Ins,
                               SmallVectorImpl<ISD::InputArg> &OrigIns) const {

  for (unsigned i = 0, e = Ins.size(); i < e; ++i) {
    if (Ins[i].ArgVT == Ins[i].VT) {
      OrigIns.push_back(Ins[i]);
      continue;
    }

    EVT VT;
    if (Ins[i].ArgVT.isVector() && !Ins[i].VT.isVector()) {
      // Vector has been split into scalars.
      VT = Ins[i].ArgVT.getVectorElementType();
    } else if (Ins[i].VT.isVector() && Ins[i].ArgVT.isVector() &&
               Ins[i].ArgVT.getVectorElementType() !=
               Ins[i].VT.getVectorElementType()) {
      // Vector elements have been promoted
      VT = Ins[i].ArgVT;
    } else {
      // Vector has been spilt into smaller vectors.
      VT = Ins[i].VT;
    }

    ISD::InputArg Arg(Ins[i].Flags, VT, VT, Ins[i].Used,
                      Ins[i].OrigArgIndex, Ins[i].PartOffset);
    OrigIns.push_back(Arg);
  }
}

bool AMDGPUTargetLowering::isHWTrueValue(SDValue Op) const {
  if (ConstantFPSDNode * CFP = dyn_cast<ConstantFPSDNode>(Op)) {
    return CFP->isExactlyValue(1.0);
  }
  if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(Op)) {
    return C->isAllOnesValue();
  }
  return false;
}

bool AMDGPUTargetLowering::isHWFalseValue(SDValue Op) const {
  if (ConstantFPSDNode * CFP = dyn_cast<ConstantFPSDNode>(Op)) {
    return CFP->getValueAPF().isZero();
  }
  if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(Op)) {
    return C->isNullValue();
  }
  return false;
}

SDValue AMDGPUTargetLowering::CreateLiveInRegister(SelectionDAG &DAG,
                                                  const TargetRegisterClass *RC,
                                                   unsigned Reg, EVT VT) const {
  MachineFunction &MF = DAG.getMachineFunction();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  unsigned VirtualRegister;
  if (!MRI.isLiveIn(Reg)) {
    VirtualRegister = MRI.createVirtualRegister(RC);
    MRI.addLiveIn(Reg, VirtualRegister);
  } else {
    VirtualRegister = MRI.getLiveInVirtReg(Reg);
  }
  return DAG.getRegister(VirtualRegister, VT);
}

#define NODE_NAME_CASE(node) case AMDGPUISD::node: return #node;

const char* AMDGPUTargetLowering::getTargetNodeName(unsigned Opcode) const {
  switch (Opcode) {
  default: return nullptr;
  // AMDIL DAG nodes
  NODE_NAME_CASE(CALL);
  NODE_NAME_CASE(UMUL);
  NODE_NAME_CASE(RET_FLAG);
  NODE_NAME_CASE(BRANCH_COND);

  // AMDGPU DAG nodes
  NODE_NAME_CASE(DWORDADDR)
  NODE_NAME_CASE(FRACT)
  NODE_NAME_CASE(CLAMP)
  NODE_NAME_CASE(FMAX)
  NODE_NAME_CASE(SMAX)
  NODE_NAME_CASE(UMAX)
  NODE_NAME_CASE(FMIN)
  NODE_NAME_CASE(SMIN)
  NODE_NAME_CASE(UMIN)
  NODE_NAME_CASE(URECIP)
  NODE_NAME_CASE(DIV_SCALE)
  NODE_NAME_CASE(DIV_FMAS)
  NODE_NAME_CASE(DIV_FIXUP)
  NODE_NAME_CASE(TRIG_PREOP)
  NODE_NAME_CASE(RCP)
  NODE_NAME_CASE(RSQ)
  NODE_NAME_CASE(RSQ_LEGACY)
  NODE_NAME_CASE(RSQ_CLAMPED)
  NODE_NAME_CASE(DOT4)
  NODE_NAME_CASE(BFE_U32)
  NODE_NAME_CASE(BFE_I32)
  NODE_NAME_CASE(BFI)
  NODE_NAME_CASE(BFM)
  NODE_NAME_CASE(BREV)
  NODE_NAME_CASE(MUL_U24)
  NODE_NAME_CASE(MUL_I24)
  NODE_NAME_CASE(MAD_U24)
  NODE_NAME_CASE(MAD_I24)
  NODE_NAME_CASE(EXPORT)
  NODE_NAME_CASE(CONST_ADDRESS)
  NODE_NAME_CASE(REGISTER_LOAD)
  NODE_NAME_CASE(REGISTER_STORE)
  NODE_NAME_CASE(LOAD_CONSTANT)
  NODE_NAME_CASE(LOAD_INPUT)
  NODE_NAME_CASE(SAMPLE)
  NODE_NAME_CASE(SAMPLEB)
  NODE_NAME_CASE(SAMPLED)
  NODE_NAME_CASE(SAMPLEL)
  NODE_NAME_CASE(CVT_F32_UBYTE0)
  NODE_NAME_CASE(CVT_F32_UBYTE1)
  NODE_NAME_CASE(CVT_F32_UBYTE2)
  NODE_NAME_CASE(CVT_F32_UBYTE3)
  NODE_NAME_CASE(BUILD_VERTICAL_VECTOR)
  NODE_NAME_CASE(CONST_DATA_PTR)
  NODE_NAME_CASE(STORE_MSKOR)
  NODE_NAME_CASE(TBUFFER_STORE_FORMAT)
  }
}

static void computeKnownBitsForMinMax(const SDValue Op0,
                                      const SDValue Op1,
                                      APInt &KnownZero,
                                      APInt &KnownOne,
                                      const SelectionDAG &DAG,
                                      unsigned Depth) {
  APInt Op0Zero, Op0One;
  APInt Op1Zero, Op1One;
  DAG.computeKnownBits(Op0, Op0Zero, Op0One, Depth);
  DAG.computeKnownBits(Op1, Op1Zero, Op1One, Depth);

  KnownZero = Op0Zero & Op1Zero;
  KnownOne = Op0One & Op1One;
}

void AMDGPUTargetLowering::computeKnownBitsForTargetNode(
  const SDValue Op,
  APInt &KnownZero,
  APInt &KnownOne,
  const SelectionDAG &DAG,
  unsigned Depth) const {

  KnownZero = KnownOne = APInt(KnownOne.getBitWidth(), 0); // Don't know anything.

  APInt KnownZero2;
  APInt KnownOne2;
  unsigned Opc = Op.getOpcode();

  switch (Opc) {
  default:
    break;
  case ISD::INTRINSIC_WO_CHAIN: {
    // FIXME: The intrinsic should just use the node.
    switch (cast<ConstantSDNode>(Op.getOperand(0))->getZExtValue()) {
    case AMDGPUIntrinsic::AMDGPU_imax:
    case AMDGPUIntrinsic::AMDGPU_umax:
    case AMDGPUIntrinsic::AMDGPU_imin:
    case AMDGPUIntrinsic::AMDGPU_umin:
      computeKnownBitsForMinMax(Op.getOperand(1), Op.getOperand(2),
                                KnownZero, KnownOne, DAG, Depth);
      break;
    default:
      break;
    }

    break;
  }
  case AMDGPUISD::SMAX:
  case AMDGPUISD::UMAX:
  case AMDGPUISD::SMIN:
  case AMDGPUISD::UMIN:
    computeKnownBitsForMinMax(Op.getOperand(0), Op.getOperand(1),
                              KnownZero, KnownOne, DAG, Depth);
    break;

  case AMDGPUISD::BFE_I32:
  case AMDGPUISD::BFE_U32: {
    ConstantSDNode *CWidth = dyn_cast<ConstantSDNode>(Op.getOperand(2));
    if (!CWidth)
      return;

    unsigned BitWidth = 32;
    uint32_t Width = CWidth->getZExtValue() & 0x1f;
    if (Width == 0) {
      KnownZero = APInt::getAllOnesValue(BitWidth);
      KnownOne = APInt::getNullValue(BitWidth);
      return;
    }

    // FIXME: This could do a lot more. If offset is 0, should be the same as
    // sign_extend_inreg implementation, but that involves duplicating it.
    if (Opc == AMDGPUISD::BFE_I32)
      KnownOne = APInt::getHighBitsSet(BitWidth, BitWidth - Width);
    else
      KnownZero = APInt::getHighBitsSet(BitWidth, BitWidth - Width);

    break;
  }
  }
}

unsigned AMDGPUTargetLowering::ComputeNumSignBitsForTargetNode(
  SDValue Op,
  const SelectionDAG &DAG,
  unsigned Depth) const {
  switch (Op.getOpcode()) {
  case AMDGPUISD::BFE_I32: {
    ConstantSDNode *Width = dyn_cast<ConstantSDNode>(Op.getOperand(2));
    if (!Width)
      return 1;

    unsigned SignBits = 32 - Width->getZExtValue() + 1;
    ConstantSDNode *Offset = dyn_cast<ConstantSDNode>(Op.getOperand(1));
    if (!Offset || !Offset->isNullValue())
      return SignBits;

    // TODO: Could probably figure something out with non-0 offsets.
    unsigned Op0SignBits = DAG.ComputeNumSignBits(Op.getOperand(0), Depth + 1);
    return std::max(SignBits, Op0SignBits);
  }

  case AMDGPUISD::BFE_U32: {
    ConstantSDNode *Width = dyn_cast<ConstantSDNode>(Op.getOperand(2));
    return Width ? 32 - (Width->getZExtValue() & 0x1f) : 1;
  }

  default:
    return 1;
  }
}
