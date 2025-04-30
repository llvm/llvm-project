//===-- Next32ISelLowering.cpp - Next32 DAG Lowering Implementation -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the interfaces that Next32 uses to lower LLVM code into a
// selection DAG.
//
//===----------------------------------------------------------------------===//

#include "Next32ISelLowering.h"
#include "Next32.h"
#include "Next32CallingConv.h"
#include "Next32Subtarget.h"
#include "TargetInfo/Next32BaseInfo.h"

#include "llvm/ADT/StringSwitch.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/CodeGen/ISDOpcodes.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/CodeGen/SelectionDAGNodes.h"
#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/MC/MCSymbolELF.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/KnownBits.h"
#include <algorithm>
#include <vector>

using namespace llvm;

#define DEBUG_TYPE "next32-lower"

struct Next32Lea {
  SDNode *Node;

  Next32Lea(SDNode *N) : Node(N) {
    assert(N != nullptr && "Lea Node can't be nullptr.");
    assert(N->getOpcode() == Next32ISD::PSEUDO_LEA &&
           "Unexpected opcode for PSEUDO LEA.");
  }

  std::pair<SDValue, SDValue> getBase() {
    return std::make_pair(Node->getOperand(0), Node->getOperand(1));
  }

  std::pair<SDValue, SDValue> getIndex() {
    return std::make_pair(Node->getOperand(2), Node->getOperand(3));
  }

  std::pair<SDValue, SDValue> getScale() {
    return std::make_pair(Node->getOperand(4), Node->getOperand(5));
  }

  std::pair<SDValue, SDValue> getDisp() {
    return std::make_pair(Node->getOperand(6), Node->getOperand(7));
  }

  bool hasBase() {
    return !(isa<ConstantSDNode>(Node->getOperand(0)) &&
             isa<ConstantSDNode>(Node->getOperand(1)) &&
             cast<ConstantSDNode>(Node->getOperand(0))->isZero() &&
             cast<ConstantSDNode>(Node->getOperand(1))->isZero());
  }

  bool hasIndex() {
    return !(isa<ConstantSDNode>(Node->getOperand(2)) &&
             isa<ConstantSDNode>(Node->getOperand(3)) &&
             cast<ConstantSDNode>(Node->getOperand(2))->isZero() &&
             cast<ConstantSDNode>(Node->getOperand(3))->isZero());
  }

  bool hasScaleImm() {
    return isa<ConstantSDNode>(Node->getOperand(4)) &&
           isa<ConstantSDNode>(Node->getOperand(5));
  }

  bool hasDispImm() {
    return isa<ConstantSDNode>(Node->getOperand(6)) &&
           isa<ConstantSDNode>(Node->getOperand(7));
  }

  uint64_t getScaleImm() {
    assert(hasScaleImm() && "Scale must be immediate.");
    SDValue ScaleHi, ScaleLo;
    std::tie(ScaleHi, ScaleLo) = getScale();
    return Make_64(cast<ConstantSDNode>(ScaleHi)->getZExtValue(),
                   cast<ConstantSDNode>(ScaleLo)->getZExtValue());
  }

  uint64_t getDispImm() {
    assert(hasDispImm() && "Displacement must be immediate.");
    SDValue DispHi, DispLo;
    std::tie(DispHi, DispLo) = getDisp();
    return Make_64(cast<ConstantSDNode>(DispHi)->getZExtValue(),
                   cast<ConstantSDNode>(DispLo)->getZExtValue());
  }
};

struct Next32LeaAddressMode {
  enum { RegBase, FrameIndexBase } BaseType;
  enum { RegScale, ImmediateScale } ScaleType;
  SDValue Base;
  SDValue Index;
  SDValue IndexExt;
  int FrameIndex;
  SDValue ScaleReg;
  int64_t ScaleImm;
  int64_t Disp;

  Next32LeaAddressMode()
      : BaseType(RegBase), ScaleType(ImmediateScale), Base(), Index(),
        IndexExt(), FrameIndex(0), ScaleReg(), ScaleImm(0), Disp(0) {}

  void setScaleReg(SDValue Reg) {
    ScaleType = RegScale;
    ScaleReg = Reg;
  }

  void setScaleImm(int64_t Imm) {
    ScaleType = ImmediateScale;
    ScaleImm = Imm;
  }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  LLVM_DUMP_METHOD void dump(SelectionDAG &DAG) {
    dbgs() << "Base ";
    if (Base.getNode())
      Base.getNode()->dump(&DAG);
    else
      dbgs() << "nul\n";
    if (BaseType == FrameIndexBase)
      dbgs() << " Base.FrameIndex " << FrameIndex << '\n';
    dbgs() << "Index ";
    if (Index.getNode())
      Index.getNode()->dump(&DAG);
    else
      dbgs() << "nul\n";
    if (ScaleType == RegScale) {
      dbgs() << "ScaleReg ";
      if (ScaleReg.getNode())
        ScaleReg.getNode()->dump(&DAG);
      else
        dbgs() << "nul\n";
    } else
      dbgs() << "ScaleImm " << ScaleImm << '\n';
    dbgs() << " Disp " << Disp << '\n';
  }
#endif
};

struct Next32SHLSimplifyPattern {
  SDValue Part;
  SDValue Shift;
  SDValue Add;

  Next32SHLSimplifyPattern() : Part(), Shift(), Add() {}

  int GetShiftAmt() {
    if (Shift.getNode())
      return cast<ConstantSDNode>(Shift.getOperand(1))->getSExtValue();
    return 0;
  }
};

Next32TargetLowering::Next32TargetLowering(const TargetMachine &TM,
                                           const Next32Subtarget &STI)
    : TargetLowering(TM), Subtarget(STI) {
  // FIXME: It's important we do accurate alias-analysis inside basic blocks.
  GatherAllAliasesMaxDepth = 128;
  GatherAllAliasesMaxTFOperands = 128;

  addRegisterClass(MVT::i32, &Next32::GPR32RegClass);

  computeRegisterProperties(STI.getRegisterInfo());

  // The input for conditions is always FLAGS, so I don't think we
  // have special treatment/preference for either of the two boolean
  // disciplines. Therefore, choose 0-or-1 for simplicity. Note that
  // we don't have vector registers, so, not specifying anything for
  // setBooleanVectorContents.
  setBooleanContents(ZeroOrOneBooleanContent);

  setOperationAction(ISD::ADD, MVT::i32, Custom);
  setOperationAction(ISD::ADD, MVT::i64, Expand);
  setOperationAction(ISD::SUB, MVT::i32, Custom);
  setOperationAction(ISD::SUB, MVT::i64, Expand);

  // We're currently not providing custom lowering for SADDO/SSUBO
  // due to lack of interest, reconsider this if we see they're
  // ubiquitous enough and we can lower better than default.
  setOperationAction(ISD::UADDO, MVT::i32, Custom);
  setOperationAction(ISD::USUBO, MVT::i32, Custom);
  setOperationAction(ISD::UADDO_CARRY, MVT::i32, Custom);
  setOperationAction(ISD::USUBO_CARRY, MVT::i32, Custom);

  setMaxAtomicSizeInBitsSupported(128);

  // This effectively covers all loads/stores because the pointer operand is i64
  setOperationAction(ISD::LOAD, MVT::i64, Custom);
  setOperationAction(ISD::LOAD, MVT::i128, Custom);
  setOperationAction(ISD::STORE, MVT::i64, Custom);
  setOperationAction(ISD::STORE, MVT::i128, Custom);
  setOperationAction(ISD::ATOMIC_LOAD, MVT::i64, Custom);
  setOperationAction(ISD::ATOMIC_STORE, MVT::i64, Custom);
  setOperationAction(ISD::ATOMIC_CMP_SWAP, MVT::i64, Custom);
  setOperationAction(ISD::ATOMIC_CMP_SWAP_WITH_SUCCESS, MVT::i64, Custom);
  setOperationAction(ISD::ATOMIC_SWAP, MVT::i64, Custom);
  setOperationAction(ISD::ATOMIC_LOAD_ADD, MVT::i64, Custom);
  setOperationAction(ISD::ATOMIC_LOAD_SUB, MVT::i64, Custom);
  setOperationAction(ISD::ATOMIC_LOAD_AND, MVT::i64, Custom);
  setOperationAction(ISD::ATOMIC_LOAD_OR, MVT::i64, Custom);
  setOperationAction(ISD::ATOMIC_LOAD_XOR, MVT::i64, Custom);
  setOperationAction(ISD::ATOMIC_LOAD_NAND, MVT::i64, Custom);
  setOperationAction(ISD::ATOMIC_LOAD_MIN, MVT::i64, Custom);
  setOperationAction(ISD::ATOMIC_LOAD_MAX, MVT::i64, Custom);
  setOperationAction(ISD::ATOMIC_LOAD_UMIN, MVT::i64, Custom);
  setOperationAction(ISD::ATOMIC_LOAD_UMAX, MVT::i64, Custom);

  setOperationAction(ISD::BR_CC, MVT::i32, Custom);
  setOperationAction(ISD::BR_JT, MVT::Other, Expand);
  setOperationAction(ISD::BRIND, MVT::Other, Expand);
  setOperationAction(ISD::SDIV, MVT::i32, LibCall);
  setOperationAction(ISD::UDIV, MVT::i32, LibCall);
  setOperationAction(ISD::SDIV, MVT::i64, LibCall);
  setOperationAction(ISD::UDIV, MVT::i64, LibCall);
  setOperationAction(ISD::SETCC, MVT::i32, Expand);
  setOperationAction(ISD::SHL_PARTS, MVT::i32, Expand);
  setOperationAction(ISD::SRL_PARTS, MVT::i32, Expand);
  setOperationAction(ISD::SRA_PARTS, MVT::i32, Expand);
  setOperationAction(ISD::SELECT_CC, MVT::i32, Custom);
  setOperationAction(ISD::SELECT, MVT::i32, Expand);
  setOperationAction(ISD::BRCOND, MVT::Other, Custom);
  setOperationAction(ISD::TargetGlobalAddress, MVT::i64, Custom);
  setOperationAction(ISD::GlobalAddress, MVT::i64, Custom);
  setOperationAction(ISD::GlobalAddress, MVT::i32, Custom);
  setOperationAction(ISD::FrameIndex, MVT::i64, Custom);
  setOperationAction(ISD::TargetFrameIndex, MVT::i64, Custom);
  setOperationAction(ISD::BSWAP, MVT::i32, Expand);
  setOperationAction(ISD::CTPOP, MVT::i32, Expand);
  setOperationAction(ISD::UNDEF, MVT::i32, Custom);
  setOperationAction(ISD::ADDRSPACECAST, MVT::i64, Custom);
  setOperationAction(ISD::UREM, MVT::i32, LibCall);
  setOperationAction(ISD::SREM, MVT::i32, LibCall);
  setOperationAction(ISD::UREM, MVT::i64, LibCall);
  setOperationAction(ISD::SREM, MVT::i64, LibCall);
  // For DivRem in next32, setting action to custom for
  // ValueType also disable DivRemPairsPass for that
  // value type, as specified in TargetTransformInfo.
  setOperationAction(ISD::UDIVREM, MVT::i32, Custom);
  setOperationAction(ISD::UDIVREM, MVT::i64, Custom);
  setOperationAction(ISD::SDIVREM, MVT::i32, Custom);
  setOperationAction(ISD::SDIVREM, MVT::i64, Custom);
  setOperationAction(ISD::DYNAMIC_STACKALLOC, MVT::i64, Custom);
  setOperationAction(ISD::STACKSAVE, MVT::i64, Custom);
  setOperationAction(ISD::STACKRESTORE, MVT::i64, Custom);
  setOperationAction(ISD::VASTART, MVT::i64, Custom);
  // We need these three to be custom despite the fact that the normal expand
  // logic suits us well because they have an illegal pointer operand, and the
  // legalizer doesn't know how to deal with this. If we specified Expand, it
  // would fail on operand legalization before expanding the operation itself.
  setOperationAction(ISD::VAEND, MVT::i64, Custom);
  setOperationAction(ISD::VAARG, MVT::i64, Custom);
  setOperationAction(ISD::VACOPY, MVT::i64, Custom);

  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i1, Expand);

  setOperationAction(ISD::ROTL, MVT::i32, Expand);
  setOperationAction(ISD::ROTR, MVT::i32, Expand);

  // We have HW for these, so we want to customize some cases where it would
  // otherwise Expand during Type Legalization (e.g. constant shift amount).
  setOperationAction(ISD::SHL, MVT::i64, Custom);
  setOperationAction(ISD::SRL, MVT::i64, Custom);
  setOperationAction(ISD::SRA, MVT::i64, Custom);

  // We can't use default expansion because pointers are not legal for us
  setOperationAction(ISD::PREFETCH, MVT::i64, Custom);

  // Expand MULH to MUL_LOHI which we natively support in ISA
  setOperationAction(ISD::MULHU, MVT::i32, Expand);
  setOperationAction(ISD::MULHS, MVT::i32, Expand);

  setOperationAction(ISD::MUL, MVT::i64, Expand);
  setOperationAction(ISD::MUL, MVT::i128, Expand);
  // Force MUL i128 expansion by telling legalizer we don't have this libcall
  setLibcallName(RTLIB::MUL_I128, nullptr);

  // Setting the libcall names here will trigger DAG Combiner to generate
  // ISD::[U/S]DIVREM instructions.
  setLibcallName(RTLIB::SDIVREM_I32, "__next32_sdivrem32");
  setLibcallName(RTLIB::UDIVREM_I32, "__next32_udivrem32");
  setLibcallName(RTLIB::SDIVREM_I64, "__next32_sdivrem64");
  setLibcallName(RTLIB::UDIVREM_I64, "__next32_udivrem64");

  // Express native support for signed/unsigned min/max, they're Expand by
  // default.
  setOperationAction(ISD::SMIN, MVT::i32, Legal);
  setOperationAction(ISD::SMAX, MVT::i32, Legal);
  setOperationAction(ISD::UMIN, MVT::i32, Legal);
  setOperationAction(ISD::UMAX, MVT::i32, Legal);

  if (Subtarget.hasVectorInst()) {
    for (MVT VT : MVT::fixedlen_vector_valuetypes()) {
      // Narrowing and extending vector loads/stores aren't handled
      // directly.
      for (MVT InnerVT : MVT::fixedlen_vector_valuetypes()) {
        setTruncStoreAction(VT, InnerVT, Expand);
        setLoadExtAction(ISD::SEXTLOAD, VT, InnerVT, Expand);
        setLoadExtAction(ISD::ZEXTLOAD, VT, InnerVT, Expand);
        setLoadExtAction(ISD::EXTLOAD, VT, InnerVT, Expand);
      }

      if (Next32Helpers::IsValidVectorTy(VT)) {
        setOperationAction(ISD::LOAD, VT, Custom);
        setOperationAction(ISD::STORE, VT, Custom);
        setOperationAction(ISD::MLOAD, VT, Custom);
        setOperationAction(ISD::MSTORE, VT, Custom);
      }
    }

    // Custom lower Intrinsic::get_active_lane_mask.
    setOperationAction(ISD::INTRINSIC_WO_CHAIN, MVT::v2i1, Custom);
    setOperationAction(ISD::INTRINSIC_WO_CHAIN, MVT::v4i1, Custom);
    setOperationAction(ISD::INTRINSIC_WO_CHAIN, MVT::v8i1, Custom);
    setOperationAction(ISD::INTRINSIC_WO_CHAIN, MVT::v16i1, Custom);
  }

  // Jumps are expensive, compared to logic.
  setJumpIsExpensive();

  if (Subtarget.hasLEA()) {
    setTargetDAGCombine(ISD::FrameIndex);
    setTargetDAGCombine(ISD::SHL);
    setTargetDAGCombine(ISD::MUL);
  }

  if (Subtarget.hasAtomicFAddFSub()) {
    setTargetDAGCombine(ISD::ATOMIC_LOAD_FADD);
    setTargetDAGCombine(ISD::ATOMIC_LOAD_FSUB);
  }

  // Hack it via the DAG Combiner because there's no MVT::i256,i512.
  setTargetDAGCombine({ISD::LOAD, ISD::STORE});

  setTargetDAGCombine(ISD::GlobalTLSAddress);
  setTargetDAGCombine(ISD::ADD);
  setTargetDAGCombine(ISD::OR);
  setTargetDAGCombine(ISD::BR);
  setTargetDAGCombine(ISD::EXTRACT_VECTOR_ELT);
  setTargetDAGCombine(ISD::FADD);
}

bool Next32TargetLowering::isOffsetFoldingLegal(
    const GlobalAddressSDNode * /*GA*/) const {
  return false;
}

bool Next32TargetLowering::forceExpandNode(SDNode *N) const {
  if (LSBaseSDNode *LdStNode = dyn_cast<LSBaseSDNode>(N)) {
    EVT VT = LdStNode->getMemoryVT();
    if (!VT.isRound() && VT.getFixedSizeInBits() > 32)
      return true;
  }

  return false;
}

bool Next32TargetLowering::forceCustomLowering(SDNode *N) const {
  if (LSBaseSDNode *LdStNode = dyn_cast<LSBaseSDNode>(N)) {
    unsigned BitWidth = LdStNode->getMemoryVT().getFixedSizeInBits();
    if (BitWidth <= Subtarget.maxLoadStoreSizeBits() &&
        (BitWidth == 256 || BitWidth == 512))
      return true;
  }

  return false;
}

bool Next32TargetLowering::isFMAFasterThanFMulAndFAdd(const MachineFunction &MF,
                                                      EVT VT) const {
  VT = VT.getScalarType();

  if (!VT.isSimple())
    return false;

  switch (VT.getSimpleVT().SimpleTy) {
  case MVT::f16:
  case MVT::f32:
  case MVT::f64:
    return !Subtarget.isGen1();
  default:
    break;
  }

  return false;
}

TargetLowering::ShiftLegalizationStrategy
Next32TargetLowering::preferredShiftLegalizationStrategy(
    SelectionDAG &DAG, SDNode *N, unsigned ExpansionFactor) const {
  return ShiftLegalizationStrategy::ExpandToParts;
}

bool Next32TargetLowering::shouldConvertOrToAdd() const {
  return Subtarget.hasLEA();
}

bool Next32TargetLowering::shouldExpandGetActiveLaneMask(EVT ResVT,
                                                         EVT OpVT) const {
  if (!Subtarget.hasVectorInst())
    return true;

  // We can only support legal predicate result types.
  if (ResVT != MVT::v2i1 && ResVT != MVT::v4i1 && ResVT != MVT::v8i1 &&
      ResVT != MVT::v16i1)
    return true;

  // We only support i32 or i64 scalar inputs.
  if (OpVT != MVT::i32 && OpVT != MVT::i64)
    return true;

  return false;
}

TargetLowering::AtomicExpansionKind
Next32TargetLowering::shouldExpandAtomicRMWInIR(AtomicRMWInst *RMW) const {
  switch (RMW->getOperation()) {
  case AtomicRMWInst::FAdd:
  case AtomicRMWInst::FSub:
    return Subtarget.hasAtomicFAddFSub() ? AtomicExpansionKind::None
                                         : AtomicExpansionKind::CmpXChg;

  case AtomicRMWInst::FMax:
  case AtomicRMWInst::FMin:
  case AtomicRMWInst::UIncWrap:
  case AtomicRMWInst::UDecWrap:
    return AtomicExpansionKind::CmpXChg;

  default:
    return AtomicExpansionKind::None;
  }
}

void Next32TargetLowering::LowerOperationWrapper(
    SDNode *N, SmallVectorImpl<SDValue> &Results, SelectionDAG &DAG) const {
  SDValue Res = LowerOperation(SDValue(N, 0), DAG);

  if (!Res)
    return;

  // If the original node has one result, take the return value from
  // LowerOperation as is. It might not be result number 0.
  if (N->getNumValues() == 1) {
    Results.push_back(Res);
    return;
  }

  if (Res.getResNo() != 0)
    report_fatal_error("LowerOperation returned bad ResNo!");

  if (N->getNumValues() != Res->getNumValues())
    report_fatal_error("Lowering returned the wrong number of results!");

  // Places new result values base on N result number.
  for (unsigned I = 0, E = N->getNumValues(); I != E; ++I) {
    if (Res->getValueType(I) != N->getValueType(I))
      report_fatal_error("Result type mismatch in LowerOperationWrapper!");
    Results.push_back(Res.getValue(I));
  }
}

std::pair<unsigned, const TargetRegisterClass *>
Next32TargetLowering::getRegForInlineAsmConstraint(const TargetRegisterInfo *,
                                                   StringRef /*Constraint*/,
                                                   MVT) const {
  return std::make_pair(0U, &Next32::GPR32RegClass);
}

SDValue Next32TargetLowering::LowerOperation(SDValue Op,
                                             SelectionDAG &DAG) const {
  switch (Op.getOpcode()) {
  case ISD::ADD:
    return LowerADD(Op, DAG);
  case ISD::SUB:
    return LowerSUB(Op, DAG);
  case ISD::UADDO:
  case ISD::USUBO:
    return LowerUADDSUBO(Op, DAG);
  case ISD::UADDO_CARRY:
  case ISD::USUBO_CARRY:
    return LowerUADDSUBO_CARRY(Op, DAG);
  case ISD::SHL:
  case ISD::SRL:
  case ISD::SRA:
    return LowerSHIFT(Op, DAG);
  case ISD::BR_CC:
    return LowerBR_CC(Op, DAG);
  case ISD::SETCC:
    return LowerSETCC(Op, DAG);
  case ISD::SELECT_CC:
    return LowerSELECT_CC(Op, DAG);
  case ISD::BRCOND:
    return LowerBRCOND(Op, DAG);
  case ISD::LOAD:
    return LowerLOAD(Op, DAG);
  case ISD::STORE:
    return LowerSTORE(Op, DAG);
  case ISD::MLOAD:
    return LowerMLOAD(Op, DAG);
  case ISD::MSTORE:
    return LowerMSTORE(Op, DAG);
  case ISD::ATOMIC_LOAD:
    return LowerATOMIC_LOAD(Op, DAG);
  case ISD::ATOMIC_STORE:
    return LowerATOMIC_STORE(Op, DAG);
  case ISD::ATOMIC_CMP_SWAP:
  case ISD::ATOMIC_CMP_SWAP_WITH_SUCCESS:
    return LowerATOMIC_CMP_SWAP(Op, DAG);
  case ISD::ATOMIC_SWAP:
  case ISD::ATOMIC_LOAD_ADD:
  case ISD::ATOMIC_LOAD_SUB:
  case ISD::ATOMIC_LOAD_AND:
  case ISD::ATOMIC_LOAD_OR:
  case ISD::ATOMIC_LOAD_XOR:
  case ISD::ATOMIC_LOAD_NAND:
  case ISD::ATOMIC_LOAD_MIN:
  case ISD::ATOMIC_LOAD_MAX:
  case ISD::ATOMIC_LOAD_UMIN:
  case ISD::ATOMIC_LOAD_UMAX:
    return LowerATOMIC_LOAD_OP(Op, DAG);
  case ISD::UNDEF:
    return DAG.getConstant(Next32Constants::ImplicitDefValue, SDLoc(Op),
                           MVT::i32);
  case ISD::TargetGlobalAddress:
    return LowerTargetGlobalAddress(Op, DAG);
  case ISD::GlobalAddress:
    return LowerGlobalAddress(Op, DAG);
  case ISD::ADDRSPACECAST:
    return LowerADDRSPACECAST(Op, DAG);
  case ISD::UDIVREM:
  case ISD::SDIVREM:
    return LowerDIVREM(Op, DAG);
  case ISD::DYNAMIC_STACKALLOC:
    return LowerDYNAMIC_STACKALLOC(Op, DAG);
  case ISD::STACKSAVE:
    return LowerSTACKSAVE(Op, DAG);
  case ISD::STACKRESTORE:
    return LowerSTACKRESTORE(Op, DAG);
  case ISD::VASTART:
    return LowerVASTART(Op, DAG);
  case ISD::VAEND:
    return LowerVAEND(Op, DAG);
  case ISD::VACOPY:
    return LowerVACOPY(Op, DAG);
  case ISD::VAARG:
    return LowerVAARG(Op, DAG);
  case ISD::PREFETCH:
    return LowerPREFETCH(Op, DAG);
  case ISD::INTRINSIC_WO_CHAIN:
    return LowerINTRINSIC_WO_CHAIN(Op, DAG);
  }
  return Op;
}

/// Replace a node with an illegal result type with a new node built out of
/// custom code.
void Next32TargetLowering::ReplaceNodeResults(SDNode *N,
                                              SmallVectorImpl<SDValue> &Results,
                                              SelectionDAG &DAG) const {
  switch (N->getOpcode()) {
  default:
    llvm_unreachable("Do not know how to custom type legalize this operation!");
  case ISD::MUL:
  case ISD::SHL:
  case ISD::SRL:
  case ISD::SRA:
  case ISD::LOAD:
  case ISD::MLOAD:
  case ISD::ATOMIC_LOAD:
  case ISD::ATOMIC_CMP_SWAP:
  case ISD::ATOMIC_CMP_SWAP_WITH_SUCCESS:
  case ISD::ATOMIC_SWAP:
  case ISD::ATOMIC_LOAD_ADD:
  case ISD::ATOMIC_LOAD_SUB:
  case ISD::ATOMIC_LOAD_AND:
  case ISD::ATOMIC_LOAD_OR:
  case ISD::ATOMIC_LOAD_XOR:
  case ISD::ATOMIC_LOAD_NAND:
  case ISD::ATOMIC_LOAD_MIN:
  case ISD::ATOMIC_LOAD_MAX:
  case ISD::ATOMIC_LOAD_UMIN:
  case ISD::ATOMIC_LOAD_UMAX:
  case ISD::DYNAMIC_STACKALLOC:
  case ISD::STACKSAVE:
  case ISD::UDIVREM:
  case ISD::SDIVREM:
  case ISD::INTRINSIC_WO_CHAIN:
    LowerOperationWrapper(N, Results, DAG);
    return;
  case ISD::TargetGlobalAddress: {
    SDValue Result = LowerTargetGlobalAddress(SDValue(N, 0), DAG);
    Results.push_back(Result);
    return;
  }
  case ISD::GlobalAddress: {
    SDValue Result = LowerGlobalAddress(SDValue(N, 0), DAG);
    Results.push_back(Result);
    return;
  }
  case ISD::TargetFrameIndex: {
    const int FI = cast<FrameIndexSDNode>(N)->getIndex();
    Results.push_back(DAG.getTargetFrameIndex(FI, MVT::i32));
    return;
  }
  case ISD::FrameIndex: {
    SDValue Result = LowerFrameIndex(SDValue(N, 0), DAG, 0, N);
    Results.push_back(Result);
    return;
  }
  case ISD::ADDRSPACECAST: {
    SDValue Result = LowerADDRSPACECAST(SDValue(N, 0), DAG);
    Results.push_back(Result);
    return;
  }
  case ISD::VASTART: {
    SDValue Result = LowerVASTART(SDValue(N, 0), DAG);
    Results.push_back(Result);
    return;
  }
  case ISD::VAEND: {
    SDValue Result = LowerVAEND(SDValue(N, 0), DAG);
    Results.push_back(Result);
    return;
  }
  case ISD::VACOPY: {
    SDValue Result = LowerVACOPY(SDValue(N, 0), DAG);
    Results.push_back(Result);
    return;
  }
  case ISD::VAARG: {
    SDValue Result = LowerVAARG(SDValue(N, 0), DAG);
    Results.push_back(Result.getValue(0));
    Results.push_back(Result.getValue(1));
    return;
  }
  }
}

static void BreakBasePtr(SDValue BasePtr, SmallVectorImpl<SDValue> &Ops,
                         SDLoc &DL, SelectionDAG &DAG) {
  SDValue BasePtrHighIndex = DAG.getConstant(1, DL, MVT::i64);
  SDValue BasePtrLowIndex = DAG.getConstant(0, DL, MVT::i64);
  SDValue BasePtrHigh = DAG.getNode(ISD::EXTRACT_ELEMENT, DL, MVT::i32, BasePtr,
                                    BasePtrHighIndex);
  SDValue BasePtrLow =
      DAG.getNode(ISD::EXTRACT_ELEMENT, DL, MVT::i32, BasePtr, BasePtrLowIndex);
  Ops.push_back(BasePtrHigh);
  Ops.push_back(BasePtrLow);
}

SDValue Next32TargetLowering::PerformDAGCombine(SDNode *N,
                                                DAGCombinerInfo &DCI) const {
  switch (N->getOpcode()) {
  case ISD::MUL:
  case ISD::SHL:
  case ISD::FrameIndex:
    return PerformLeaCandidateCombine(SDValue(N, 0), DCI.DAG);
  case ISD::EXTRACT_VECTOR_ELT:
    return PerformExtractVectorEltWithFMulCombine(N, DCI.DAG);
  case ISD::GlobalTLSAddress:
    return PerformGlobalTLSAddressCombine(N, DCI.DAG);
  case ISD::ADD:
    return PerformADDCombine(N, DCI.DAG);
  case ISD::OR:
    return PerformADDORCombine(N, DCI.DAG);
  case ISD::BR:
    return PerformBRCombine(N, DCI);
  case ISD::LOAD:
  case ISD::STORE:
    return PerformLSCombine(N, DCI.DAG);
  case ISD::ATOMIC_LOAD_FADD:
  case ISD::ATOMIC_LOAD_FSUB:
    return PerformAtomicFADDorFSUBCombine(SDValue(N, 0), DCI.DAG);
  case ISD::FADD:
    return PerformFADDFMAFMULCombine(N, DCI);
  case Next32ISD::ADCFLAGS:
  case Next32ISD::SBBFLAGS:
    return PerformBoolCarryPropagationCombine(N, DCI.DAG);
  case Next32ISD::PSEUDO_LEA:
    return PerformPseudoLeaCombine(N, DCI.DAG);
  }
  return {};
}

/// Combines EXTRACT_VECTOR_ELT and FMUL operations to optimize floating-point
/// multiplications involving vector elements. If extracting a scalar FP value
/// from a vector element is feasible, extract each operand first and then
/// perform FMUL as a scalar operation.
///
/// \param N The node representing the EXTRACT_VECTOR_ELT operation.
/// \param DAG The SelectionDAG to which the nodes belong.
///
/// \returns A new node representing the combined operation, or an empty
///         SDValue if the transformation is not applicable.
SDValue Next32TargetLowering::PerformExtractVectorEltWithFMulCombine(
    SDNode *N, SelectionDAG &DAG) const {
  SDLoc DL(N);

  assert(N->getOpcode() == ISD::EXTRACT_VECTOR_ELT && "Expected extract");
  SDValue Vec = N->getOperand(0);
  SDValue Index = N->getOperand(1);
  EVT VT = N->getValueType(0);
  EVT VecVT = Vec.getValueType();

  // Restrict the combination to cases where the vector has a single use,
  // we are extracting the first element, and the scalar types match.
  // This prevents duplicating vector computations, ensures simplicity,
  // and avoids type mismatches.
  if (!Vec.hasOneUse() || !isNullConstant(Index) || VecVT.getScalarType() != VT)
    return SDValue();

  if (!VT.isFloatingPoint() || Subtarget.isGen1())
    return SDValue();

  switch (Vec->getOpcode()) {
  case ISD::FMUL: {
    SmallVector<SDValue, 4> Ops;
    for (SDValue Op : Vec->ops())
      Ops.push_back(DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, VT, Op, Index));

    SDValue ScalarNode = DAG.getNode(Vec->getOpcode(), DL, VT, Ops);
    ScalarNode->setFlags(Vec->getFlags());
    return ScalarNode;
  }
  default:
    return SDValue();
  }
  llvm_unreachable("All opcodes should return within switch");
}

/// Combine the following patterns to optimize FADD, FMA and FMUL operations:
///
/// 1. `FADD(A, FMA(B, C, D))` where `D = FMUL(E, F)`
///    is transformed to:
///    `FMA(B, C, FMA(E, F, A))`
///
/// 2. `FADD(A, FMA(B, C, D))` where `D = EXTRACT_VECTOR_ELT(FMUL(E, F), Index)`
///    is transformed to:
///    `FMA(B, C, FMA(E, F, A))`
///
/// \param N The node representing the FADD operation.
/// \param DCI The DAG combiner information.
///
/// \return A new node representing the combined operation, or an empty
///         SDValue if the transformation is not applicable.
SDValue
Next32TargetLowering::PerformFADDFMAFMULCombine(SDNode *N,
                                                DAGCombinerInfo &DCI) const {
  assert(N->getOpcode() == ISD::FADD && "Expected FADD");
  if (!N->getFlags().hasAllowContract() ||
      !N->getFlags().hasAllowReassociation())
    return SDValue();

  SDValue FaddOp1 = N->getOperand(0);
  SDValue FaddOp2 = N->getOperand(1);

  if (FaddOp1->getOpcode() != ISD::FMA && FaddOp2->getOpcode() != ISD::FMA)
    return SDValue();

  SDValue FmaOp = FaddOp1->getOpcode() == ISD::FMA ? FaddOp1 : FaddOp2;
  SDValue OtherFaddOp = FaddOp1->getOpcode() == ISD::FMA ? FaddOp2 : FaddOp1;
  SDValue B = FmaOp.getOperand(0);
  SDValue C = FmaOp.getOperand(1);
  SDValue D = FmaOp.getOperand(2);

  SDLoc DL(N);
  EVT VT = N->getValueType(0);
  if (D->getOpcode() == ISD::FMUL) {
    SDValue E = D.getOperand(0);
    SDValue F = D.getOperand(1);
    // Types of operands must match for valid transformation
    if (E.getValueType() != VT || F.getValueType() != VT)
      return SDValue();

    // Create the inner FMA: FMA(E, F, A)
    SDValue InnerFMA =
        DCI.DAG.getNode(ISD::FMA, DL, N->getValueType(0), E, F, OtherFaddOp);
    InnerFMA->setFlags(N->getFlags());
    // Create the outer FMA: FMA(B, C, InnerFMA)
    SDValue OuterFMA =
        DCI.DAG.getNode(ISD::FMA, DL, N->getValueType(0), B, C, InnerFMA);
    OuterFMA->setFlags(N->getFlags());
    // Replace the original FADD node with the new FMA node
    return OuterFMA;
  }

  if (D.getOpcode() == ISD::EXTRACT_VECTOR_ELT) {
    SDValue ScalarFMUL =
        PerformExtractVectorEltWithFMulCombine(D.getNode(), DCI.DAG);
    if (!ScalarFMUL.getNode())
      return SDValue();
    assert(ScalarFMUL->getOpcode() == ISD::FMUL && "Expected scalar FMUL");
    SDValue ScalarFMULOp0 = ScalarFMUL.getOperand(0);
    SDValue ScalarFMULOp1 = ScalarFMUL.getOperand(1);

    // Create the inner FMA: FMA(ScalarFMULOp0, ScalarFMULOp1, OtherFaddOp)
    SDValue InnerFMA = DCI.DAG.getNode(ISD::FMA, DL, VT, ScalarFMULOp0,
                                       ScalarFMULOp1, OtherFaddOp);
    InnerFMA->setFlags(N->getFlags());
    // Create the outer FMA: FMA(B, C, InnerFMA)
    SDValue OuterFMA = DCI.DAG.getNode(ISD::FMA, DL, VT, B, C, InnerFMA);
    OuterFMA->setFlags(N->getFlags());
    // Replace the original FADD node with the new FMA node
    return OuterFMA;
  }

  return SDValue();
}

SDValue Next32TargetLowering::PerformADDCombine(SDNode *N,
                                                SelectionDAG &DAG) const {
  if (SDValue Res = PerformADDORCombine(N, DAG))
    return Res;

  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);

  if (N0->getOpcode() == ISD::FrameIndex && N1->getOpcode() == ISD::Constant) {
    uint64_t Offset = cast<ConstantSDNode>(N1.getNode())->getZExtValue();
    SDLoc DL(N);
    return LowerFrameIndex(N0, DAG, Offset, DL);
  }
  return {};
}

SDValue Next32TargetLowering::PerformADDORCombine(SDNode *N,
                                                  SelectionDAG &DAG) const {
  SDValue Op(N, 0);
  if (SDValue Res = PerformSHLSimplify(Op, DAG))
    return Res;

  return PerformLeaCandidateCombine(Op, DAG);
}

// Try to match following cases, so we can use 64-bit SHL.
//  1. zext/aext (x)
//  2. add/or (zext/aext(x), imm)
//  3. shl (zext/aext(x), imm)
//  4. add/or (shl (zext/aext(x), imm), imm)
bool Next32TargetLowering::MatchSHLPart(SDValue Op, SelectionDAG &DAG,
                                        Next32SHLSimplifyPattern &SP) const {
  if (Op.getOpcode() == ISD::ADD) {
    if (!isa<ConstantSDNode>(Op.getOperand(1)))
      return false;
    SP.Add = Op;
    Op = Op.getOperand(0);
  }

  if (Op.getOpcode() == ISD::OR) {
    if (!DAG.haveNoCommonBitsSet(Op.getOperand(0), Op.getOperand(1)) ||
        !isa<ConstantSDNode>(Op.getOperand(1)) || SP.Add.getNode())
      return false;
    SP.Add = Op;
    Op = Op.getOperand(0);
  }

  if (Op.getOpcode() == ISD::SHL) {
    if (!isa<ConstantSDNode>(Op.getOperand(1)))
      return false;
    SP.Shift = Op;
    Op = Op.getOperand(0);
  }

  if (Op.getOpcode() == ISD::ZERO_EXTEND || Op.getOpcode() == ISD::ANY_EXTEND) {
    SP.Part = Op;
    return true;
  }

  return false;
}

// Since our HW supports 64-bit SHL, optimize cases where i64 is built from two
// i32 and SHL is done on each part.
SDValue Next32TargetLowering::PerformSHLSimplify(SDValue Op,
                                                 SelectionDAG &DAG) const {
  assert((Op.getOpcode() == ISD::ADD || Op.getOpcode() == ISD::OR) &&
         "Unexpected opcode for SHL simplify.");

  if (Op.getValueType() != MVT::i64)
    return SDValue();

  SDValue Add;

  // In some cases, DAG combine sinks add/or immediate node.
  if (isa<ConstantSDNode>(Op.getOperand(1))) {
    Add = Op;
    Op = Op.getOperand(0);

    // Don't do SHL simplify if the next node is not ADD or OR.
    if (Op.getOpcode() != ISD::ADD && Op.getOpcode() != ISD::OR)
      return SDValue();
  }

  if (Op.getOpcode() == ISD::OR &&
      !DAG.haveNoCommonBitsSet(Op.getOperand(0), Op.getOperand(1)))
    return SDValue();

  Next32SHLSimplifyPattern SP0, SP1;
  if (!MatchSHLPart(Op.getOperand(0), DAG, SP0) ||
      !MatchSHLPart(Op.getOperand(1), DAG, SP1))
    return SDValue();

  // Set SP0 to be high part.
  if (SP0.GetShiftAmt() < 32)
    std::swap(SP0, SP1);

  // Check if shifts are matching.
  if (SP1.GetShiftAmt() != SP0.GetShiftAmt() - 32)
    return SDValue();

  // At this point, we should only have 0 or 1 add with immediate.
  for (SDValue V : {SP0.Add, SP1.Add}) {
    if (!V.getNode())
      continue;

    if (Add.getNode())
      return SDValue();

    Add = V;
  }

  auto GetPart = [&](SDValue Part) {
    SDValue Op0 = Part.getOperand(0);
    if (Op0.getValueType() == MVT::i32)
      return Op0;

    // Do the extension to i32, so we can build i64 pair.
    return DAG.getNode(Part.getOpcode(), SDLoc(Part), MVT::i32, Op0);
  };

  SDValue PartHi = GetPart(SP0.Part);
  SDValue PartLo = GetPart(SP1.Part);
  SDValue Res =
      DAG.getNode(ISD::BUILD_PAIR, SDLoc(PartLo), MVT::i64, PartLo, PartHi);
  SDValue Shift = SP1.Shift;

  // Create shift if needed.
  if (Shift.getNode())
    Res =
        DAG.getNode(ISD::SHL, SDLoc(Shift), MVT::i64, Res, Shift.getOperand(1));

  // Create add if needed.
  if (Add.getNode())
    Res = DAG.getNode(ISD::ADD, SDLoc(Add), MVT::i64, Res, Add.getOperand(1));

  return Res;
}

// Helper for MatchLeaRecursively. Add the specified node to the
// specified addressing mode without any further recursion.
bool Next32TargetLowering::MatchLeaBase(SDValue N,
                                        Next32LeaAddressMode &LAM) const {
  // Is the base register already occupied?
  if (LAM.BaseType != Next32LeaAddressMode::RegBase || LAM.Base.getNode()) {
    // If so, check to see if the scale index register is set.
    if (!LAM.Index.getNode()) {
      LAM.Index = N;
      LAM.setScaleImm(1);
      return true;
    }

    // Otherwise, we cannot select it.
    return false;
  }

  // Default, generate it as a register.
  LAM.BaseType = Next32LeaAddressMode::RegBase;
  LAM.Base = N;
  return true;
}

bool Next32TargetLowering::MatchAdd(SDValue &N, SelectionDAG &DAG,
                                    Next32LeaAddressMode &LAM,
                                    unsigned Depth) const {
  auto IsHighPart = [](SDValue Op) {
    switch (Op.getOpcode()) {
    default:
      break;
    case ISD::AND:
      if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(Op.getOperand(1)))
        return (CN->getZExtValue() & UINT64_C(0xFFFFFFFF)) == 0;
      break;
    case ISD::SHL:
      if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(Op.getOperand(1)))
        return CN->getZExtValue() >= 32;
      break;
    }
    return false;
  };

  auto IsLowPart = [](SDValue Op) {
    switch (Op.getOpcode()) {
    default:
      break;
    case ISD::ZERO_EXTEND:
    case ISD::ANY_EXTEND:
      return true;
    case ISD::AND:
      if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(Op.getOperand(1)))
        return (CN->getZExtValue() & UINT64_C(0xFFFFFFFF)) ==
               CN->getZExtValue();
      break;
    case ISD::SRL:
      if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(Op.getOperand(1)))
        return CN->getZExtValue() >= 32;
      break;
    }
    return false;
  };

  // Don't match operands to LEA if they are used to create i64 from two i32
  // parts.
  if ((IsHighPart(N.getOperand(0)) && IsLowPart(N.getOperand(1))) ||
      (IsHighPart(N.getOperand(1)) && IsLowPart(N.getOperand(0))))
    return false;

  Next32LeaAddressMode Backup = LAM;
  if (MatchLeaRecursively(N.getOperand(0), DAG, LAM, Depth + 1) &&
      MatchLeaRecursively(N.getOperand(1), DAG, LAM, Depth + 1))
    return true;
  LAM = Backup;

  // Try again after commuting the operands.
  if (MatchLeaRecursively(N.getOperand(1), DAG, LAM, Depth + 1) &&
      MatchLeaRecursively(N.getOperand(0), DAG, LAM, Depth + 1))
    return true;
  LAM = Backup;

  // If we couldn't fold both operands into the address at the same time,
  // see if we can just put each operand into a register and fold at least
  // the add.
  if (LAM.BaseType == Next32LeaAddressMode::RegBase && !LAM.Base.getNode() &&
      !LAM.Index.getNode()) {
    LAM.Base = N.getOperand(0);
    LAM.Index = N.getOperand(1);
    LAM.setScaleImm(1);
    return true;
  }

  return false;
}

/// This function is used to fold the "add like" OR node with a shift amount to
/// LEA. The fold transformation works as follows:
/// Check the shift operand (`shl`), with constant, which provides the scale.
///
/// 1. Match the outer OR operation with another operand (`or x, zext`).
/// 2. Match the zero extension of the result (`zext`).
/// 3. Match the inner OR operation (`or|add y, constant`).
///
/// If folding is possible, it returns true, calculates the shifted constant,
/// and updates the index operand accordingly.
///
/// \param ShVal The value representing the OR node.
/// \param DAG The selection DAG used for the transformation.
/// \param ShiftAmt The amount by which the value is shifted.
/// \param [out] ShiftedConstValue Output parameter that will hold the shifted
/// constant value if the folding is possible.
/// \param [out] IndexOperand Output parameter that will hold the new index
/// operand if the folding is possible.
///
/// \returns true if the transformation is possible and the shift can be folded.
static bool canFoldShlOrZextOrWithConstant(SDValue ShVal, SelectionDAG &DAG,
                                           unsigned ShiftAmt,
                                           uint64_t &ShiftedConstValue,
                                           SDValue &IndexOperand) {
  // Ensure ShVal is an OR node
  if (ShVal.getOpcode() != ISD::OR)
    return false;

  // Extract operands of the OR node
  SDValue OrOperand0 = ShVal.getOperand(0);
  SDValue OrOperand1 = ShVal.getOperand(1);

  // Check if both operands of OR node have no common bits, OR is add like, can
  // be treated as an ADD node
  if (!DAG.haveNoCommonBitsSet(OrOperand0, OrOperand1))
    return false;

  auto CheckAndFold = [&](SDValue OrOperand, int OrOperandIndex) -> bool {
    if (OrOperand.getOpcode() == ISD::ZERO_EXTEND && OrOperand.hasOneUse() &&
        DAG.isBaseWithConstantOffset(OrOperand.getOperand(0)) &&
        OrOperand.getOperand(0).hasOneUse()) {

      SDValue InnerOrNode = OrOperand.getOperand(0);
      SDValue InnerOrVal = InnerOrNode.getOperand(0);
      SDValue ConstantOp = InnerOrNode.getOperand(1);

      // Check if adding the constant to the lower 32 bits doesn't carry to the
      // high part
      if (isa<ConstantSDNode>(ConstantOp)) {
        uint64_t ConstValue = cast<ConstantSDNode>(ConstantOp)->getZExtValue();
        if (ConstValue < (1ULL << 32) &&
            DAG.haveNoCommonBitsSet(InnerOrVal, ConstantOp)) {
          ShiftedConstValue =
              ConstValue << ShiftAmt; // Shift the constant by the shift amount

          SDValue ZeroExtendedInnerOrVal =
              DAG.getNode(ISD::ZERO_EXTEND, SDLoc(ShVal),
                          OrOperand.getValueType(), InnerOrVal);
          if (OrOperandIndex) {
            IndexOperand =
                DAG.getNode(ISD::OR, SDLoc(ShVal), ShVal.getValueType(),
                            OrOperand0, ZeroExtendedInnerOrVal);
          } else {
            IndexOperand =
                DAG.getNode(ISD::OR, SDLoc(ShVal), ShVal.getValueType(),
                            ZeroExtendedInnerOrVal, OrOperand1);
          }
          return true;
        }
      }
    }

    return false;
  };

  // Check if folding is possible for OrOperand0 or OrOperand1
  if (CheckAndFold(OrOperand0, 0) || CheckAndFold(OrOperand1, 1))
    return true;

  return false;
}

static bool canFoldScaledValueToDisp(SDValue Op, SDValue Ext,
                                     SelectionDAG &DAG) {
  if (!DAG.isBaseWithConstantOffset(Op))
    return false;

  if (Op.getOpcode() != ISD::ADD || !Ext.getNode())
    return true;

  assert((Ext.getOpcode() == ISD::ZERO_EXTEND ||
          Ext.getOpcode() == ISD::SIGN_EXTEND) &&
         "Unexpected extension opcode.");

  bool Sext = Ext.getOpcode() == ISD::SIGN_EXTEND;
  bool NSW = Op.getNode()->getFlags().hasNoSignedWrap();
  bool NUW = Op.getNode()->getFlags().hasNoUnsignedWrap();

  // We can promote a sign/zero extension ahead of an 'add' only if we have
  // 'add nsw' feeding into the 'sext' or 'add nuw' feeding into the 'zext'.
  return (Sext && NSW) || (!Sext && NUW);
}

// Return whether we should match Mul node to LEA. We don't want to do 64-bit
// mul in cases where we can do 32-bit mul when low part of operand is 0, or
// when we can use umul or smul instructions.
bool Next32TargetLowering::ShouldMatchMulToLea(SDValue N) const {
  // if low part of operand is 0, we can do 32-bit mul
  auto IsMULCandidate = [](SDValue Op) -> bool {
    switch (Op.getOpcode()) {
    default:
      break;
    case ISD::BUILD_PAIR:
      if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(Op.getOperand(0)))
        return CN->isZero();
      break;
    case ISD::AND:
      if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(Op.getOperand(1)))
        return CN->getAPIntValue().getLoBits(32).isMinValue();
      break;
    case ISD::SHL:
      if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(Op.getOperand(1)))
        return CN->getZExtValue() >= 32;
      break;
    }
    return false;
  };

  auto IsUMULCandidate = [](SDValue Op) -> bool {
    switch (Op.getOpcode()) {
    default:
      break;
    case ISD::Constant: {
      APInt HiBits = cast<ConstantSDNode>(Op)->getAPIntValue().getHiBits(32);
      return HiBits.isMinValue();
    }
    case ISD::ZERO_EXTEND:
    case ISD::ANY_EXTEND:
      return true;
    case ISD::AND:
      if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(Op.getOperand(1)))
        return CN->getAPIntValue().getHiBits(32).isMinValue();
      break;
    case ISD::SRL:
      if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(Op.getOperand(1)))
        return CN->getZExtValue() >= 32;
      break;
    }
    return false;
  };

  auto IsSMULCandidate = [](SDValue Op) -> bool {
    switch (Op.getOpcode()) {
    default:
      break;
    case ISD::Constant: {
      APInt HiBits = cast<ConstantSDNode>(Op)->getAPIntValue().getHiBits(32);
      return HiBits.isMinValue() || HiBits.isAllOnes();
    }
    case ISD::SIGN_EXTEND:
      return true;
    case ISD::SRA:
      if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(Op.getOperand(1)))
        return CN->getZExtValue() >= 32;
      break;
    }
    return false;
  };

  if (IsMULCandidate(N.getOperand(0)) || IsMULCandidate(N.getOperand(1)))
    return false;

  if (IsUMULCandidate(N.getOperand(0)) && IsUMULCandidate(N.getOperand(1)))
    return false;

  if (IsSMULCandidate(N.getOperand(0)) && IsSMULCandidate(N.getOperand(1)))
    return false;

  return true;
}

bool Next32TargetLowering::MatchLeaRecursively(SDValue N, SelectionDAG &DAG,
                                               Next32LeaAddressMode &LAM,
                                               unsigned Depth) const {
  LLVM_DEBUG({
    dbgs() << "MatchLea: ";
    LAM.dump(DAG);
  });
  // Limit recursion.
  if (Depth > 5)
    MatchLeaBase(N, LAM);

  switch (N.getOpcode()) {
  default:
    break;
  case ISD::Constant:
    LAM.Disp += cast<ConstantSDNode>(N)->getSExtValue();
    return true;
  case ISD::FrameIndex:
    if (LAM.BaseType == Next32LeaAddressMode::RegBase && !LAM.Base.getNode()) {
      LAM.BaseType = Next32LeaAddressMode::FrameIndexBase;
      LAM.FrameIndex = cast<FrameIndexSDNode>(N)->getIndex();
      return true;
    }
    break;
  case ISD::MUL:
    if (LAM.Index.getNode() || !ShouldMatchMulToLea(N))
      break;

    LAM.Index = N.getOperand(0);
    if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(N.getOperand(1)))
      LAM.setScaleImm(CN->getSExtValue());
    else
      LAM.setScaleReg(N.getOperand(1));
    return true;
  case ISD::SHL:
    if (LAM.Index.getNode())
      break;
    if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(N.getOperand(1))) {
      unsigned ShiftAmt = CN->getZExtValue();
      SDValue IndexExt;
      SDValue ShVal = N.getOperand(0);
      LAM.setScaleImm(1LL << ShiftAmt);

      // Look through the extension, as we might fold scaled value to the disp
      // field.
      if (ShVal.getOpcode() == ISD::ZERO_EXTEND ||
          ShVal.getOpcode() == ISD::SIGN_EXTEND) {
        IndexExt = ShVal;
        ShVal = ShVal.getOperand(0);
      }

      uint64_t ShiftedConstValue;
      SDValue IndexOperand;
      if (canFoldShlOrZextOrWithConstant(ShVal, DAG, ShiftAmt,
                                         ShiftedConstValue, IndexOperand)) {
        LAM.Disp += ShiftedConstValue;
        LAM.Index = IndexOperand;
        return true;
      }

      // If the scaled value is an add of something and a constant,
      // we can fold the constant into the disp field here.
      if (canFoldScaledValueToDisp(ShVal, IndexExt, DAG)) {
        ConstantSDNode *AddVal = cast<ConstantSDNode>(ShVal.getOperand(1));
        LAM.Disp += (AddVal->getSExtValue() << ShiftAmt);
        LAM.Index = ShVal.getOperand(0);
        LAM.IndexExt = IndexExt;
        return true;
      }

      LAM.Index = N.getOperand(0);
      return true;
    }
    break;
  case ISD::OR:
    // We want to look through a transformations that turn 'add' into 'or',
    // so we can treat this 'or' exactly like an 'add'.
    if (!DAG.haveNoCommonBitsSet(N.getOperand(0), N.getOperand(1)))
      break;
    LLVM_FALLTHROUGH;
  case ISD::ADD:
    if (MatchAdd(N, DAG, LAM, Depth))
      return true;
    break;
  }

  return MatchLeaBase(N, LAM);
}

// Return whether we should create LEA based on the complexity. This is needed
// to avoid cases where creating LEA could break compilation. Currently, it
// is only happening for SHL and OR where we can't match them but we are
// adding them as Base register. Also avoid generating LEA when the lower 32
// bits of Disp are zero, and Index and Scale are not present. Generating LEA
// should be avoided when the Scale Immediate is non-zero but has its lower 32
// bits set to zero.
bool Next32TargetLowering::ShouldCreateLea(Next32LeaAddressMode &LAM) const {
  unsigned Complexity = 0;
  if (LAM.BaseType == Next32LeaAddressMode::RegBase && LAM.Base.getNode())
    Complexity = 1;
  else if (LAM.BaseType == Next32LeaAddressMode::FrameIndexBase)
    Complexity = 4;

  if (LAM.Index.getNode())
    Complexity++;

  if ((LAM.ScaleImm > 1) || LAM.ScaleReg.getNode())
    Complexity++;

  if (LAM.Disp)
    Complexity++;

  if (Complexity < 2)
    return false;

  if (LAM.BaseType == Next32LeaAddressMode::RegBase && LAM.Base.getNode() &&
      LAM.Disp && !LAM.Index.getNode() && !LAM.ScaleImm && !LAM.ScaleReg)
    return (LAM.Disp & UINT64_C(0xFFFFFFFF)) != 0;

  if (LAM.ScaleType == Next32LeaAddressMode::ImmediateScale && LAM.ScaleImm > 0)
    return (LAM.ScaleImm & UINT64_C(0xFFFFFFFF)) != 0;

  return true;
}

SDValue
Next32TargetLowering::PerformLeaCandidateCombine(SDValue Op,
                                                 SelectionDAG &DAG) const {
  switch (Op.getOpcode()) {
  case ISD::FrameIndex:
  case ISD::SHL:
  case ISD::ADD:
  case ISD::OR:
  case ISD::MUL:
    // Currently, we only support these ops for Lea matching.
    break;
  default:
    llvm_unreachable("Unexpected opcode for matching LEA.");
  }

  // Don't create LEA if target doesn't support it or if it is not 64-bit
  // operation.
  if (!Subtarget.hasLEA() || Op.getValueType() != MVT::i64)
    return SDValue();

  // If only lower 32-bits are used, computing higher part is not needed and
  // LLVM will optimize it. Don't create LEA in that case.
  auto IsOnlyLowPartUsed = [](SDNode *Use) {
    switch (Use->getOpcode()) {
    default:
      break;
    case ISD::TRUNCATE:
    case ISD::SIGN_EXTEND_INREG:
      return true;
    case ISD::AND:
      // Check if only lower 32-bits are masked.
      if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(Use->getOperand(1)))
        return (CN->getZExtValue() & UINT64_C(0xFFFFFFFF)) ==
               CN->getZExtValue();
      break;
    case ISD::SHL:
      // Check if lower 32-bits will be shifted to the upper 32-bits.
      if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(Use->getOperand(1)))
        return CN->getZExtValue() >= 32;
      break;
    }
    return false;
  };
  if (llvm::all_of(Op->uses(), IsOnlyLowPartUsed))
    return SDValue();

  Next32LeaAddressMode LAM;
  if (!MatchLeaRecursively(Op, DAG, LAM, 0))
    return SDValue();

  if (!ShouldCreateLea(LAM))
    return SDValue();

  SDLoc DL(Op);
  SmallVector<SDValue, 8> LeaOperands;

  // Add base operand.
  if (LAM.BaseType == Next32LeaAddressMode::FrameIndexBase) {
    MachineFunction &MF = DAG.getMachineFunction();
    unsigned int SPHi = MF.addLiveIn(Next32::SP_HIGH, &Next32::GPR32RegClass);
    unsigned int SPLo = MF.addLiveIn(Next32::SP_LOW, &Next32::GPR32RegClass);
    LeaOperands.push_back(DAG.getRegister(SPHi, MVT::i32));
    LeaOperands.push_back(DAG.getRegister(SPLo, MVT::i32));
  } else if (LAM.Base.getNode()) {
    BreakBasePtr(LAM.Base, LeaOperands, DL, DAG);
  } else {
    LeaOperands.push_back(DAG.getConstant(0, DL, MVT::i32));
    LeaOperands.push_back(DAG.getConstant(0, DL, MVT::i32));
  }

  // Add index operand.
  if (LAM.Index.getNode()) {
    SDValue Index = LAM.Index;
    SDValue IndexExt = LAM.IndexExt;

    // Extend Index node if needed.
    if (IndexExt.getNode())
      Index = DAG.getNode(IndexExt.getOpcode(), SDLoc(Index),
                          IndexExt.getValueType(), Index);

    BreakBasePtr(Index, LeaOperands, DL, DAG);
  } else {
    LeaOperands.push_back(DAG.getConstant(0, DL, MVT::i32));
    LeaOperands.push_back(DAG.getConstant(0, DL, MVT::i32));
  }

  // Add Scale operand.
  if (LAM.ScaleReg.getNode()) {
    BreakBasePtr(LAM.ScaleReg, LeaOperands, DL, DAG);
  } else {
    LeaOperands.push_back(
        DAG.getConstant((LAM.ScaleImm >> 32) & 0xFFFFFFFF, DL, MVT::i32));
    LeaOperands.push_back(
        DAG.getConstant(LAM.ScaleImm & 0xFFFFFFFF, DL, MVT::i32));
  }

  // Add Disp operand.
  if (LAM.BaseType == Next32LeaAddressMode::FrameIndexBase) {
    SDValue FI = DAG.getTargetFrameIndex(LAM.FrameIndex, MVT::i32);
    SDValue Disp = DAG.getTargetConstant(LAM.Disp, DL, MVT::i64);
    SDValue FrameIndex =
        DAG.getNode(Next32ISD::FRAME_OFFSET_WRAPPER, DL,
                    DAG.getVTList(MVT::i32, MVT::i32), FI, Disp);
    LeaOperands.push_back(SDValue(FrameIndex.getNode(), 0));
    LeaOperands.push_back(SDValue(FrameIndex.getNode(), 1));
  } else {
    LeaOperands.push_back(
        DAG.getConstant((LAM.Disp >> 32) & 0xFFFFFFFF, DL, MVT::i32));
    LeaOperands.push_back(DAG.getConstant(LAM.Disp & 0xFFFFFFFF, DL, MVT::i32));
  }

  SDValue Lea = DAG.getNode(Next32ISD::PSEUDO_LEA, DL,
                            DAG.getVTList(MVT::i32, MVT::i32), LeaOperands);
  return DAG.getNode(ISD::BUILD_PAIR, DL, MVT::i64, Lea.getValue(1),
                     Lea.getValue(0));
}

// Perform fusing of LEA scale to mul.
SDValue Next32TargetLowering::FuseLEAScaleToMul(SDNode *N,
                                                SelectionDAG &DAG) const {
  Next32Lea Lea = Next32Lea(N);

  if (!Lea.hasScaleImm())
    return SDValue();

  uint64_t Scale = Lea.getScaleImm();

  // Don't combine LEA that doesn't do scaling.
  if (!Lea.hasIndex() || (Scale < 2))
    return SDValue();

  SDValue IndexHi, IndexLo;
  std::tie(IndexHi, IndexLo) = Lea.getIndex();

  // Currently, we are only fusing scale value to umul_lohi with imm.
  SDNode *Umul = IndexHi.getNode();
  if (Umul->getOpcode() != ISD::UMUL_LOHI || IndexLo != SDValue(Umul, 0) ||
      IndexHi != SDValue(Umul, 1) || !isa<ConstantSDNode>(Umul->getOperand(1)))
    return SDValue();

  ConstantSDNode *MulC = cast<ConstantSDNode>(Umul->getOperand(1));
  uint64_t NewMulImm = MulC->getZExtValue() * Scale;

  // Bail out if new immediate is too large.
  if (!isUIntN(MulC->getValueSizeInBits(0), NewMulImm))
    return SDValue();

  auto IsSameScale = [&](SDNode *Use) {
    if (Use->getOpcode() != Next32ISD::PSEUDO_LEA)
      return false;

    Next32Lea UseLea = Next32Lea(Use);
    if (!UseLea.hasScaleImm())
      return false;

    SDValue UseLeaIndexHi, UseLeaIndexLo;
    std::tie(UseLeaIndexHi, UseLeaIndexLo) = UseLea.getIndex();
    return UseLeaIndexHi == SDValue(Umul, 1) &&
           UseLeaIndexLo == SDValue(Umul, 0) && UseLea.getScaleImm() == Scale;
  };

  // Only fuse LEA scale to mul iff all users of mul are LEA nodes with the same
  // scale.
  if (!llvm::all_of(Umul->uses(), IsSameScale))
    return SDValue();

  SDValue NewMul = DAG.getNode(
      ISD::UMUL_LOHI, SDLoc(Umul), Umul->getVTList(), Umul->getOperand(0),
      DAG.getConstant(NewMulImm, SDLoc(MulC), MulC->getValueType(0)));

  if (!Lea.hasBase() && Lea.hasDispImm() && (Lea.getDispImm() == 0))
    return DAG.getMergeValues({NewMul.getValue(1), NewMul.getValue(0)},
                              SDLoc(NewMul));

  SDValue NewOperands[] = {N->getOperand(0),
                           N->getOperand(1),
                           NewMul.getValue(1),
                           NewMul.getValue(0),
                           DAG.getConstant(0, SDLoc(N), MVT::i32),
                           DAG.getConstant(1, SDLoc(N), MVT::i32),
                           N->getOperand(6),
                           N->getOperand(7)};
  return DAG.getNode(Next32ISD::PSEUDO_LEA, SDLoc(N), N->getVTList(),
                     NewOperands);
}

// Perform fusing of LEA to LEA.
SDValue Next32TargetLowering::FuseLEAToLEA(SDNode *N, SelectionDAG &DAG) const {
  Next32Lea Lea = Next32Lea(N);

  // To fuse LEAs, Scale and Disp must be immediate.
  // TODO: Add test case when Disp is not immediate, Lea Address Mode is
  // FrameIndexBase.
  if (!Lea.hasScaleImm() || !Lea.hasDispImm())
    return SDValue();

  SDValue LeaIndexHi, LeaIndexLo;
  std::tie(LeaIndexHi, LeaIndexLo) = Lea.getIndex();
  if (LeaIndexHi.getNode()->getOpcode() != Next32ISD::PSEUDO_LEA)
    return SDValue();

  // Check if the Lea Index is the result of the previous one.
  if (SDValue(LeaIndexHi.getNode(), 0) != LeaIndexHi ||
      SDValue(LeaIndexHi.getNode(), 1) != LeaIndexLo)
    return SDValue();

  Next32Lea PredLea = Next32Lea(LeaIndexHi.getNode());

  // To fuse LEAs, Pred Lea Scale and Disp must be immediate.
  // TODO: Add test case when Disp is not immediate, Lea Address Mode is
  // FrameIndexBase.
  if (!PredLea.hasScaleImm() || !PredLea.hasDispImm())
    return SDValue();

  // If Pred Lea has both base and index don't fuse it.
  if (PredLea.hasBase() && PredLea.hasIndex())
    return SDValue();

  assert((PredLea.hasBase() || PredLea.hasIndex()) &&
         "Pseudo lea must have base or index");

  SDValue NewLeaIndexHi, NewLeaIndexLo;
  SDValue NewLeaScaleHi, NewLeaScaleLo;
  uint64_t LeaScale = Lea.getScaleImm();

  // Pred Lea has Base op, no Index op.
  if (PredLea.hasBase()) {
    // New IndexHi, IndexLo becomes Pred Lea BaseHi, BaseLo.
    std::tie(NewLeaIndexHi, NewLeaIndexLo) = PredLea.getBase();

    // New Lea ScaleHi, New Lea ScaleLo becomes Lea Scale.
    std::tie(NewLeaScaleHi, NewLeaScaleLo) = Lea.getScale();
  } else {
    // New IndexHi, IndexLo becomes Pred Lea IndexHi, IndexLo.
    std::tie(NewLeaIndexHi, NewLeaIndexLo) = PredLea.getIndex();

    // New Lea ScaleHi, New Lea ScaleLo.
    uint64_t PredLeaScale = PredLea.getScaleImm();
    uint64_t NewLeaScale = LeaScale * PredLeaScale;
    NewLeaScaleHi =
        DAG.getConstant((NewLeaScale >> 32) & 0xFFFFFFFF, SDLoc(N), MVT::i32);
    NewLeaScaleLo =
        DAG.getConstant((NewLeaScale & 0xFFFFFFFF), SDLoc(N), MVT::i32);
  }

  // New Lea BaseHi, New Lea BaseLo.
  SDValue NewLeaBaseHi, NewLeaBaseLo;
  std::tie(NewLeaBaseHi, NewLeaBaseLo) = Lea.getBase();

  // New Lea DispHi, New Lea DispLo.
  uint64_t LeaDisp = Lea.getDispImm();
  uint64_t PredLeaDisp = PredLea.getDispImm();
  uint64_t NewLeaDispImm = LeaDisp + PredLeaDisp * LeaScale;
  SDValue NewLeaDispHi =
      DAG.getConstant((NewLeaDispImm >> 32) & 0xFFFFFFFF, SDLoc(N), MVT::i32);
  SDValue NewLeaDispLo =
      DAG.getConstant((NewLeaDispImm & 0xFFFFFFFF), SDLoc(N), MVT::i32);

  SDValue LeaOperands[] = {NewLeaBaseHi,  NewLeaBaseLo,  NewLeaIndexHi,
                           NewLeaIndexLo, NewLeaScaleHi, NewLeaScaleLo,
                           NewLeaDispHi,  NewLeaDispLo};

  return DAG.getNode(Next32ISD::PSEUDO_LEA, SDLoc(N), N->getVTList(),
                     LeaOperands);
}

// Perform fusing of LEA scale to mul or LEA to LEA.
SDValue Next32TargetLowering::PerformPseudoLeaCombine(SDNode *N,
                                                      SelectionDAG &DAG) const {
  if (SDValue Res = FuseLEAScaleToMul(N, DAG))
    return Res;

  return FuseLEAToLEA(N, DAG);
}

SDValue
Next32TargetLowering::PerformGlobalTLSAddressCombine(SDNode *N,
                                                     SelectionDAG &DAG) const {
  SDLoc dl(N);
  TargetLowering::ArgListTy Args;

  TargetLowering::CallLoweringInfo CLI(DAG);
  CLI.setDebugLoc(dl)
      .setChain(DAG.getEntryNode())
      .setCallee(CallingConv::C, Type::getInt64Ty(*DAG.getContext()),
                 DAG.getExternalSymbol(Next32Helpers::GetLibcallFunctionName(
                                           Next32Constants::TLS_BASE),
                                       getPointerTy(DAG.getDataLayout(), 1)),
                 std::move(Args));

  std::pair<SDValue, SDValue> CallResult = LowerCallTo(CLI);
  const GlobalValue *GV = cast<GlobalAddressSDNode>(N)->getGlobal();
  SDValue Low =
      DAG.getTargetGlobalAddress(GV, dl, MVT::i32, 0, Next32II::MO_MEM_64LO);
  Low = DAG.getNode(Next32ISD::WRAPPER, dl, MVT::i32, Low);
  Low = DAG.getNode(ISD::ZERO_EXTEND, dl, MVT::i64, Low);
  return DAG.getNode(ISD::ADD, dl, MVT::i64, CallResult.first, Low);
}

//(br (brcond true mbb1) mbb2) -> (br mbb1)
SDValue Next32TargetLowering::PerformBRCombine(SDNode *N,
                                               DAGCombinerInfo &DCI) const {
  SDValue Chain = N->getOperand(0);
  if (Chain.getOpcode() != ISD::BRCOND)
    return {};

  SDValue ChainN1 = Chain.getOperand(1);
  if (ChainN1.getOpcode() != ISD::Constant)
    return {};
  if (ChainN1.getSimpleValueType().SimpleTy != MVT::i32)
    return {};

  SDValue Target = Chain.getOperand(2);
  if (cast<ConstantSDNode>(ChainN1)->isZero())
    Target = N->getOperand(1);

  return DCI.DAG.getNode(ISD::BR, SDLoc(N), MVT::Other, Chain.getOperand(0),
                         Target);
}

// This function is in cahoots with ConvertCarryFlagToBooleanCarry and
// ConvertBooleanCarryToCarryFlag - it detects the conversion pattern
// they create in the DAG to collapse it and use the produced carry
// directly when applicable.
SDValue Next32TargetLowering::PerformBoolCarryPropagationCombine(
    SDNode *N, SelectionDAG &DAG) const {
  assert((N->getOpcode() == Next32ISD::ADCFLAGS ||
          N->getOpcode() == Next32ISD::SBBFLAGS) &&
         "Unexpected node opcode");

  // Start checking whether this is indeed ADDFLAGS(ADCFLAGS(0, 0, C), -1).
  SDValue BoolCarry = N->getOperand(2);
  if (BoolCarry->getOpcode() != Next32ISD::ADDFLAGS ||
      BoolCarry.getResNo() != 1)
    return {};

  // Check for the rest of the pattern.
  SDValue LHS = BoolCarry->getOperand(0);
  SDValue RHS = BoolCarry->getOperand(1);
  if (!(LHS->getOpcode() == Next32ISD::ADCFLAGS && LHS.getResNo() == 0 &&
        isNullConstant(LHS->getOperand(0)) &&
        isNullConstant(LHS->getOperand(1)) && isAllOnesConstant(RHS)))
    return {};

  // Collapse this construct by returning a new node that wires directly into
  // the output flags of the computation from which the original carry was
  // generated.
  SDValue OriginalFlagsCarry = LHS->getOperand(2).getValue(1);
  assert((OriginalFlagsCarry->getOpcode() == Next32ISD::ADDFLAGS ||
          OriginalFlagsCarry->getOpcode() == Next32ISD::SUBFLAGS ||
          OriginalFlagsCarry->getOpcode() == Next32ISD::ADCFLAGS ||
          OriginalFlagsCarry->getOpcode() == Next32ISD::SBBFLAGS) &&
         OriginalFlagsCarry.getResNo() == 1 && "Unexpected carry in opcode");
  SDValue NewOperands[] = {N->getOperand(0), N->getOperand(1),
                           OriginalFlagsCarry};
  return DAG.getNode(N->getOpcode(), SDLoc(N), N->getVTList(), NewOperands);
}

// Append operands that are the same for the memory nodes.
static void AppendMemOps(MemSDNode *Mem, SDLoc &DL, SelectionDAG &DAG,
                         SmallVectorImpl<SDValue> &Ops, unsigned Size) {
  EVT MemVT = Mem->getMemoryVT();

  unsigned int Count = MemVT.isVector()
                           ? Next32Helpers::CountToLog2VecElemFieldValue(
                                 MemVT.getVectorNumElements())
                           : 0;
  unsigned int Align =
      Next32Helpers::BytesToLog2AlignValue(Mem->getAlign().value());
  unsigned int SizeField = Next32Helpers::BitsToSizeFieldValue(Size);
  unsigned int AddrSpace =
      Next32Helpers::GetInstAddressSpace(Mem->getAddressSpace());
  auto MemOp = Next32Helpers::MemNodeTypeToMemOps(Mem);

  const uint64_t MemParams = ((uint64_t)MemOp << 32) | ((Count & 0xFF) << 24) |
                             ((AddrSpace & 0xFF) << 16) |
                             ((Align & 0xFF) << 8) | (SizeField & 0xFF);
  // Append following operands for memory node:
  //  1. Chain
  //  2. Params (mem opcode, count, alignment, size and address space)
  //  3. Address high and low
  //  4. TID register
  Ops.push_back(Mem->getChain());
  Ops.push_back(DAG.getTargetConstant(MemParams, DL, MVT::i64));
  BreakBasePtr(Mem->getBasePtr(), Ops, DL, DAG);
  Ops.push_back(DAG.getRegister(Next32::TID, MVT::i32));
}

// Ops output is in big "word-endian" order
static void BreakOperand(SelectionDAG &DAG, SDValue Op, unsigned int OpCnt,
                         SmallVectorImpl<SDValue> &Ops) {
  if (OpCnt == 1) {
    Ops.push_back(Op);
    return;
  }

  SDLoc DL(Op);

  const unsigned SizeBits = OpCnt * 32;
  auto VTComb = EVT::getEVT(Type::getIntNTy(*DAG.getContext(), SizeBits));
  auto VTPart = VTComb.getHalfSizedIntegerVT(*DAG.getContext());

  assert(Op.getValueType() == VTComb && "Unexpected operand type");

  SDValue One = DAG.getConstant(1, DL, MVT::i32);
  SDValue Hi = DAG.getNode(ISD::EXTRACT_ELEMENT, DL, VTPart, Op, One);
  BreakOperand(DAG, Hi, OpCnt / 2, Ops);

  SDValue Zero = DAG.getConstant(0, DL, MVT::i32);
  SDValue Lo = DAG.getNode(ISD::EXTRACT_ELEMENT, DL, VTPart, Op, Zero);
  BreakOperand(DAG, Lo, OpCnt / 2, Ops);
}

// Ops input is in big "word-endian" order
static SDValue BuildMultiValueResult(SelectionDAG &DAG, ArrayRef<SDValue> Ops) {
  assert(isPowerOf2_64(Ops.size()) && "Invalid Ops size");

  if (Ops.size() == 1)
    return Ops[0];

  SDLoc DL(Ops[0]);
  const unsigned HalfSize = Ops.size() / 2;
  SDValue Hi = BuildMultiValueResult(DAG, Ops.slice(0, HalfSize));
  SDValue Lo = BuildMultiValueResult(DAG, Ops.slice(HalfSize, HalfSize));

  const unsigned SizeBits = Ops.size() * 32;
  auto ResVT = EVT::getEVT(Type::getIntNTy(*DAG.getContext(), SizeBits));

  return DAG.getNode(ISD::BUILD_PAIR, DL, ResVT, Lo, Hi);
}

static SDValue BuildGMemOpResult(SelectionDAG &DAG, SDValue Op,
                                 unsigned int OpCnt) {
  assert(OpCnt != 0 && isPowerOf2_32(OpCnt) && "Unexpected OpCnt");
  SmallVector<SDValue, 4> Ops;

  for (unsigned int i = 0; i < OpCnt; ++i)
    Ops.push_back(Op.getValue(i));

  return BuildMultiValueResult(DAG, Ops);
}

static unsigned int MemDataCount(unsigned int Bits) { return (Bits + 31) / 32; }

static SDVTList GMemOpVTList(SelectionDAG &DAG, unsigned int OpCnt) {
  SmallVector<EVT, 5> ValueTypes;

  for (unsigned int i = 0; i < OpCnt; i++)
    ValueTypes.push_back(MVT::i32);

  ValueTypes.push_back(MVT::Other);

  return DAG.getVTList(ValueTypes);
}

SDValue Next32TargetLowering::PerformLSCombine(SDNode *N,
                                               SelectionDAG &DAG) const {
  auto *LS = cast<LSBaseSDNode>(N);
  const EVT MemVT = LS->getMemoryVT();

  // Defer until legalizer, since vector types should have MVTs.
  // If the type isn't round, let the legalizer expand it before lowering.
  if (MemVT.isVector() || !MemVT.isRound())
    return {};

  /// We don't want to lose out on DAGCombiner chain improvements by lowering
  /// all loads and stores too early -- we're only here as a hack to be able
  /// to catch the i256 and i512 loads and stores because they have extended
  /// types and we can't register custom lowering against them.
  /// TODO: Investigate whether we can get chain improvements for i256 and i512
  /// as well, and remove this hacky method. Maybe DAGCombinerInfo can help?
  const unsigned BitWidth = LS->getMemoryVT().getFixedSizeInBits();
  if (BitWidth > Subtarget.maxLoadStoreSizeBits() ||
      BitWidth <= EVT(MVT::LAST_INTEGER_VALUETYPE).getSizeInBits())
    return {};

  if (N->getOpcode() == ISD::LOAD)
    return LowerLOAD(SDValue(N, 0), DAG);

  return LowerSTORE(SDValue(N, 0), DAG);
}

// Since there is no support for atomic FADD and FSUB in Legalizer in
// SoftenFloatResult function, generate here Next32 specific node to handle
// these instructions. This function is almost the same as LowerATOMIC_LOAD_OP
// function, with additional extensions and casts.
SDValue
Next32TargetLowering::PerformAtomicFADDorFSUBCombine(SDValue Op,
                                                     SelectionDAG &DAG) const {
  SDLoc DL(Op);
  AtomicSDNode *FaOp = cast<AtomicSDNode>(Op);
  auto Size = FaOp->getMemoryVT().getSizeInBits();
  if (!isPowerOf2_32(Size) || Size > 64)
    report_fatal_error("Unexpected atomic fadd or fsub memory size");

  const unsigned int OpCnt = MemDataCount(Size);

  SmallVector<SDValue, 8> GMemOperands;
  // Chain, Mem Params, Addr Hi, Addr Lo, TID
  AppendMemOps(FaOp, DL, DAG, GMemOperands, Size);

  // Cast to integer types since they are only legal for us.
  EVT ValueVT = EVT::getIntegerVT(*DAG.getContext(), Size);
  SDValue Value = DAG.getNode(ISD::BITCAST, DL, ValueVT, FaOp->getVal());

  // Handle sizes that are less than 32bit (e.g. fp16).
  if (Value.getValueSizeInBits() < 32)
    Value = DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i32, Value);

  BreakOperand(DAG, Value, OpCnt, GMemOperands);

  unsigned Opc;
  switch (OpCnt) {
  case 1:
    Opc = Next32ISD::G_MEM_FAOP_S;
    break;
  case 2:
    Opc = Next32ISD::G_MEM_FAOP_D;
    break;
  default:
    report_fatal_error("Unexpected memory size in atomic fadd or fsub");
  }

  SDValue GMemFaOp =
      DAG.getNode(Opc, DL, GMemOpVTList(DAG, OpCnt), GMemOperands);
  SDValue Result = BuildGMemOpResult(DAG, GMemFaOp, OpCnt);

  // Do trunction if value is smaller than 32bit (e.g. fp16).
  if (Result.getValueSizeInBits() > ValueVT.getSizeInBits())
    Result = DAG.getNode(ISD::TRUNCATE, DL, ValueVT, Result);

  // Cast back to the resulting floating point type.
  Result = DAG.getNode(ISD::BITCAST, DL, FaOp->getMemoryVT(), Result);
  SDValue MergeOps[] = {Result, GMemFaOp.getValue(OpCnt)};
  return DAG.getMergeValues(MergeOps, DL);
}

// Append operands that are same for vector load and store.
static void AppendVectorLoadStoreOps(MemSDNode *Mem, SDLoc &DL,
                                     SelectionDAG &DAG,
                                     SmallVectorImpl<SDValue> &Ops) {
  EVT MemVT = Mem->getMemoryVT();
  unsigned int Align =
      Next32Helpers::BytesToLog2AlignValue(Mem->getAlign().value());
  unsigned int Type = Next32Helpers::BitsToSizeFieldValue(
      MemVT.getVectorElementType().getSizeInBits());
  unsigned int Count =
      Next32Helpers::CountToLog2VecElemFieldValue(MemVT.getVectorNumElements());
  unsigned int AddrSpace =
      Next32Helpers::GetInstAddressSpace(Mem->getAddressSpace());
  unsigned int VMemParams = ((AddrSpace & 0xFF) << 24) |
                            ((Align & 0xFF) << 16) | ((Type & 0xFF) << 8) |
                            (Count & 0xFF);

  // Append following operands for vector node:
  //  1. Chain
  //  2. Params (alignment, vector type, vector count and address space)
  //  3. Address high and low
  //  4. TID register
  Ops.push_back(Mem->getChain());
  Ops.push_back(DAG.getTargetConstant(VMemParams, DL, MVT::i32));
  BreakBasePtr(Mem->getBasePtr(), Ops, DL, DAG);
  Ops.push_back(DAG.getRegister(Next32::TID, MVT::i32));
}

SDValue Next32TargetLowering::LowerMLOAD(SDValue Op, SelectionDAG &DAG) const {
  SDLoc DL(Op);
  MaskedLoadSDNode *Load = cast<MaskedLoadSDNode>(Op);
  SDValue Mask = Load->getMask();
  MVT MaskVT = Mask.getSimpleValueType();
  unsigned MaskBits = MaskVT.getVectorNumElements();

  assert(MaskVT.getVectorElementType() == MVT::i1 &&
         "Incorrect vector mask type");
  assert((MaskBits == 2 || MaskBits == 4 || MaskBits == 8 || MaskBits == 16) &&
         "Incorrect number of bits in vector mask");

  // Cast mask to scalar type, and extend it to the legal type.
  EVT NewMaskVT = EVT::getIntegerVT(*DAG.getContext(), MaskBits);
  SDValue IntMask =
      DAG.getAnyExtOrTrunc(DAG.getBitcast(NewMaskVT, Mask), DL, MVT::i32);
  SDValue ResVec = LowerLOADVector(Load, IntMask, DAG);

  SDValue PassThru = Load->getPassThru();
  if (!PassThru->isUndef()) {
    SDValue Select =
        DAG.getSelect(DL, Load->getValueType(0), Mask, ResVec, PassThru);
    ResVec = DAG.getMergeValues({Select, ResVec.getValue(1)}, DL);
  }
  return ResVec;
}

SDValue Next32TargetLowering::LowerMSTORE(SDValue Op, SelectionDAG &DAG) const {
  SDLoc DL(Op);
  MaskedStoreSDNode *Store = cast<MaskedStoreSDNode>(Op);
  SDValue Mask = Store->getMask();
  MVT MaskVT = Mask.getSimpleValueType();
  unsigned MaskBits = MaskVT.getVectorNumElements();

  assert(MaskVT.getVectorElementType() == MVT::i1 &&
         "Incorrect vector mask type");
  assert((MaskBits == 2 || MaskBits == 4 || MaskBits == 8 || MaskBits == 16) &&
         "Incorrect number of bits in vector mask");

  // Cast mask to scalar type, and extend it to the legal type.
  EVT NewMaskVT = EVT::getIntegerVT(*DAG.getContext(), MaskBits);
  Mask = DAG.getNode(ISD::BITCAST, DL, NewMaskVT, Mask);
  Mask = DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i32, Mask);
  return LowerSTOREVector(Store, Mask, Store->getValue(), DAG);
}

SDValue Next32TargetLowering::LowerLOADVector(MemSDNode *Load, SDValue Mask,
                                              SelectionDAG &DAG) const {
  SDLoc DL(Load);
  SmallVector<SDValue, 6> GVMemOperands;
  // Add vector load operands.
  AppendVectorLoadStoreOps(Load, DL, DAG, GVMemOperands);
  GVMemOperands.push_back(Mask);

  const EVT ResVT = Load->getValueType(0);
  const unsigned BitWidth = ResVT.getFixedSizeInBits();
  const unsigned OpCnt = MemDataCount(BitWidth);
  unsigned int Opc;
  switch (OpCnt) {
  case 1:
    Opc = Next32ISD::G_VMEM_READ_1;
    break;
  case 2:
    Opc = Next32ISD::G_VMEM_READ_2;
    break;
  case 4:
    Opc = Next32ISD::G_VMEM_READ_4;
    break;
  case 8:
    Opc = Next32ISD::G_VMEM_READ_8;
    break;
  case 16:
    Opc = Next32ISD::G_VMEM_READ_16;
    break;
  default:
    report_fatal_error("Unexpected vector load size");
  }

  const SDValue GVMemRead =
      DAG.getNode(Opc, DL, GMemOpVTList(DAG, OpCnt), GVMemOperands);

  const SDValue RegIntResult = BuildGMemOpResult(DAG, GVMemRead, OpCnt);
  const EVT IntVT = EVT::getIntegerVT(*DAG.getContext(), BitWidth);
  // Truncate from rounded-up amount of registers to an exactly-sized integer
  // (e.g. for v2i8 this would be i16), then bitcast to that vector.
  const SDValue IntResult = DAG.getAnyExtOrTrunc(RegIntResult, DL, IntVT);
  const SDValue Result = DAG.getBitcast(ResVT, IntResult);
  const SDValue Chain = GVMemRead.getValue(OpCnt);

  return DAG.getMergeValues({Result, Chain}, DL);
}

SDValue Next32TargetLowering::LowerSTOREVector(MemSDNode *Store, SDValue Mask,
                                               SDValue Value,
                                               SelectionDAG &DAG) const {
  SDLoc DL(Store);
  SmallVector<SDValue, 16> GVMemOperands;
  // Add vector store operands.
  AppendVectorLoadStoreOps(Store, DL, DAG, GVMemOperands);
  GVMemOperands.push_back(Mask);

  const unsigned BitWidth = Value.getValueType().getFixedSizeInBits();
  const EVT IntVT = EVT::getIntegerVT(*DAG.getContext(), BitWidth);
  const SDValue IntValue = DAG.getBitcast(IntVT, Value);
  // Extend to at least a register if necessary.
  const SDValue RegIntValue =
      BitWidth >= 32 ? IntValue : DAG.getAnyExtOrTrunc(IntValue, DL, MVT::i32);
  BreakOperand(DAG, RegIntValue, MemDataCount(BitWidth), GVMemOperands);

  SDVTList VTs = DAG.getVTList(MVT::i32, MVT::Other);
  SDValue GMem = DAG.getNode(Next32ISD::G_VMEM_WRITE, DL, VTs, GVMemOperands);
  return GMem.getValue(1);
}

// Legalizer managed to expand non-round types larger than 32 bits,
// here we expand remaining non-round stores into round stores or zero extending
// them to byte-sized stores.
// This method is a copy with some adaptations of
// SelectionDAGLegalize::LegalizeStoreOps.
void Next32TargetLowering::expandOrPromoteNonRoundSTORE(
    SDValue Op, SelectionDAG &DAG, SmallVectorImpl<SDValue> &Ops) const {

  StoreSDNode *ST = cast<StoreSDNode>(Op.getNode());
  SDValue Chain = ST->getChain();
  SDValue Ptr = ST->getBasePtr();
  SDLoc dl(ST);

  SDValue Value = ST->getValue();
  auto &DL = DAG.getDataLayout();
  EVT StVT = ST->getMemoryVT();
  if (StVT.isRound()) {
    Ops.push_back(Op);
    return;
  }
  TypeSize StWidth = StVT.getSizeInBits();
  TypeSize StSize = StVT.getStoreSizeInBits();

  MachineMemOperand::Flags MMOFlags = ST->getMemOperand()->getFlags();
  AAMDNodes AAInfo = ST->getAAInfo();

  if (StWidth != StSize) {
    // Promote to a byte-sized store with upper bits zero if not
    // storing an integral number of bytes.  For example, promote
    // TRUNCSTORE:i1 X -> TRUNCSTORE:i8 (and X, 1)
    EVT NVT = EVT::getIntegerVT(*DAG.getContext(), StSize.getFixedValue());
    Value = DAG.getZeroExtendInReg(Value, dl, StVT);
    SDValue Result =
        DAG.getTruncStore(ST->getChain(), dl, Value, Ptr, ST->getPointerInfo(),
                          NVT, ST->getOriginalAlign(), MMOFlags, AAInfo);

    // TRUNCSTORE:i17 X -> TRUNCSTORE:i24 (and X, 0x1FFFF)
    // We want to further expand the byte-sized non-round load
    // into round loads.
    if (NVT.isRound()) {
      Ops.push_back(Result);
      return;
    } else
      StWidth = NVT.getSizeInBits();
  }
  // Expand non-round store into round stores.
  unsigned StWidthBits = StWidth.getFixedValue();
  unsigned LogStWidth = Log2_32(StWidthBits);
  unsigned RoundWidth = 1 << LogStWidth;
  unsigned ExtraWidth = StWidthBits - RoundWidth;
  EVT RoundVT = EVT::getIntegerVT(*DAG.getContext(), RoundWidth);
  EVT ExtraVT = EVT::getIntegerVT(*DAG.getContext(), ExtraWidth);
  SDValue Lo, Hi;
  unsigned IncrementSize;

  // Store the bottom RoundWidth bits.
  Lo = DAG.getTruncStore(Chain, dl, Value, Ptr, ST->getPointerInfo(), RoundVT,
                         ST->getOriginalAlign(), MMOFlags, AAInfo);

  // Store the remaining ExtraWidth bits.
  IncrementSize = RoundWidth / 8;
  Ptr = DAG.getMemBasePlusOffset(Ptr, TypeSize::getFixed(IncrementSize), dl);
  Hi = DAG.getNode(ISD::SRL, dl, Value.getValueType(), Value,
                   DAG.getConstant(RoundWidth, dl,
                                   getShiftAmountTy(Value.getValueType(), DL)));
  Hi = DAG.getTruncStore(Chain, dl, Hi, Ptr,
                         ST->getPointerInfo().getWithOffset(IncrementSize),
                         ExtraVT, ST->getOriginalAlign(), MMOFlags, AAInfo);

  DAG.getNode(ISD::TokenFactor, dl, MVT::Other, Lo, Hi);

  Ops.push_back(Lo);
  Ops.push_back(Hi);
}

// Legalizer managed to expand non-round types larger than 32 bits,
// here we expand remaining non-round loads into round loads or promoting them
// into byte-sized loads.
// This method is a copy with some adaptations of
// SelectionDAGLegalize::LegalizeLoadOps.
void Next32TargetLowering::expandOrPromoteNonRoundLOAD(
    SDValue Op, SelectionDAG &DAG, SmallVectorImpl<SDValue> &Ops) const {

  LoadSDNode *LD = cast<LoadSDNode>(Op.getNode());
  SDValue Chain = LD->getChain(); // The chain.
  SDValue Ptr = LD->getBasePtr(); // The base pointer.
  SDValue Value;                  // The value returned by the load op.
  SDLoc dl(LD);

  EVT SrcVT = LD->getMemoryVT();
  if (SrcVT.isRound()) {
    Ops.push_back(Op);
    return;
  }
  EVT DstVT = LD->getValueType(0);
  TypeSize SrcWidth = SrcVT.getSizeInBits();

  ISD::LoadExtType ExtType = LD->getExtensionType();
  MachineMemOperand::Flags MMOFlags = LD->getMemOperand()->getFlags();
  AAMDNodes AAInfo = LD->getAAInfo();

  if (SrcWidth != SrcVT.getStoreSizeInBits()) {
    // Promote to a byte-sized load if not loading an integral number of
    // bytes.  For example, promote EXTLOAD:i20 -> EXTLOAD:i24.
    unsigned NewWidth = SrcVT.getStoreSizeInBits();
    EVT NVT = EVT::getIntegerVT(*DAG.getContext(), NewWidth);

    // The extra bits are guaranteed to be zero, since we stored them that
    // way.  A zext load from NVT thus automatically gives zext from SrcVT.
    ISD::LoadExtType NewExtType =
        ExtType == ISD::ZEXTLOAD ? ISD::ZEXTLOAD : ISD::EXTLOAD;

    SDValue Result =
        DAG.getExtLoad(NewExtType, dl, DstVT, Chain, Ptr, LD->getPointerInfo(),
                       NVT, LD->getOriginalAlign(), MMOFlags, AAInfo);

    if (ExtType == ISD::SEXTLOAD)
      // Having the top bits zero doesn't help when sign extending.
      DAG.getNode(ISD::SIGN_EXTEND_INREG, dl, Result.getValueType(), Result,
                  DAG.getValueType(SrcVT));
    else if (ExtType == ISD::ZEXTLOAD || NVT == Result.getValueType())
      // All the top bits are guaranteed to be zero - inform the optimizers.
      DAG.getNode(ISD::AssertZext, dl, Result.getValueType(), Result,
                  DAG.getValueType(SrcVT));

    // If we promoted the non-round load into a non-round byte-sized load,
    // we want to further expand it into round loads.
    if (NVT.isRound()) {
      Ops.push_back(Result);
      return;
    } else {
      ExtType = NewExtType;
      SrcWidth = NVT.getSizeInBits();
    }
  }
  // Expand non-round load into round loads.
  unsigned SrcWidthBits = SrcWidth.getFixedValue();
  unsigned LogSrcWidth = Log2_32(SrcWidthBits);
  unsigned RoundWidth = 1 << LogSrcWidth;
  unsigned ExtraWidth = SrcWidthBits - RoundWidth;
  EVT RoundVT = EVT::getIntegerVT(*DAG.getContext(), RoundWidth);
  EVT ExtraVT = EVT::getIntegerVT(*DAG.getContext(), ExtraWidth);
  SDValue Lo, Hi;
  unsigned IncrementSize;
  auto &DL = DAG.getDataLayout();

  // Load the bottom RoundWidth bits.
  Lo =
      DAG.getExtLoad(ISD::ZEXTLOAD, dl, DstVT, Chain, Ptr, LD->getPointerInfo(),
                     RoundVT, LD->getOriginalAlign(), MMOFlags, AAInfo);

  // Load the remaining ExtraWidth bits.
  IncrementSize = RoundWidth / 8;
  Ptr = DAG.getMemBasePlusOffset(Ptr, TypeSize::getFixed(IncrementSize), dl);
  Hi = DAG.getExtLoad(ExtType, dl, DstVT, Chain, Ptr,
                      LD->getPointerInfo().getWithOffset(IncrementSize),
                      ExtraVT, LD->getOriginalAlign(), MMOFlags, AAInfo);

  // Build a factor node to remember that this load is independent of
  // the other one.
  DAG.getNode(ISD::TokenFactor, dl, MVT::Other, Lo.getValue(1), Hi.getValue(1));

  // Move the top bits to the right place.
  SDValue ShlHi = DAG.getNode(
      ISD::SHL, dl, Hi.getValueType(), Hi,
      DAG.getConstant(RoundWidth, dl, getShiftAmountTy(Hi.getValueType(), DL)));

  // Join the hi and lo parts.
  Value = DAG.getNode(ISD::OR, dl, DstVT, Lo, ShlHi);

  // We want to fill Ops with loads and Hi is ISD::SHL, so we pass its
  // first operand which is a load.
  Ops.push_back(Lo);
  Ops.push_back(Hi);
}

SDValue Next32TargetLowering::LowerLOAD(SDValue Op, SelectionDAG &DAG) const {
  SDLoc DL(Op);
  SmallVector<SDValue, 2> Ops;
  SmallVector<SDValue, 2> ResOps;
  SDValue Result, Chain;

  // Ops can be either the original load, load promoted to a byte-sized type
  // or Lo and Hi from expanding the load into two round loads.
  expandOrPromoteNonRoundLOAD(Op, DAG, Ops);
  unsigned NumLoads = Ops.size();

  for (SDValue LoadOp : Ops) {
    LoadSDNode *Load = cast<LoadSDNode>(LoadOp);
    const EVT MemVT = Load->getMemoryVT();

    const EVT IntMemVT =
        EVT::getIntegerVT(*DAG.getContext(), MemVT.getFixedSizeInBits());
    const unsigned BitWidth = IntMemVT.getFixedSizeInBits();
    if (BitWidth > Subtarget.maxLoadStoreSizeBits())
      report_fatal_error("Unexpected LOAD memory size");

    // Note: for sub-i32 loads this will still yield 1
    const unsigned int OpCnt = MemDataCount(BitWidth);

    if (Load->isIndexed())
      report_fatal_error("Unexpected indexed load");
    if (MemVT.isVector() && Load->getExtensionType() != ISD::NON_EXTLOAD)
      report_fatal_error("Unexpected vector load with extension");

    SmallVector<SDValue, 5> GMemOperands;
    // Chain, Mem Params, Addr Hi, Addr Lo, TID
    AppendMemOps(Load, DL, DAG, GMemOperands, BitWidth);

    unsigned int Opc;
    switch (OpCnt) {
    case 1:
      Opc = Next32ISD::G_MEM_READ_1;
      break;
    case 2:
      Opc = Next32ISD::G_MEM_READ_2;
      break;
    case 4:
      Opc = Next32ISD::G_MEM_READ_4;
      break;
    case 8:
      Opc = Next32ISD::G_MEM_READ_8;
      break;
    case 16:
      Opc = Next32ISD::G_MEM_READ_16;
      break;
    default:
      report_fatal_error("Unexpected load size");
    }

    SDValue GMemRead =
        DAG.getNode(Opc, DL, GMemOpVTList(DAG, OpCnt), GMemOperands);
    Result = BuildGMemOpResult(DAG, GMemRead, OpCnt);
    Chain = GMemRead.getValue(OpCnt);

    // If we expanded a non-round load, we will have a build_pair later on,
    // so we must use half the Value Type.
    const EVT ResVT = Load->getValueType(0);
    const EVT ResIntVT = EVT::getIntegerVT(
        *DAG.getContext(), ResVT.getFixedSizeInBits() / NumLoads);

    // If we have a sub-word load, truncate it to RoundMemVT and let the code
    // below extend it as requested.
    if (Result.getValueSizeInBits() > BitWidth)
      Result = DAG.getZExtOrTrunc(Result, DL, IntMemVT);

    switch (Load->getExtensionType()) {
    case ISD::NON_EXTLOAD:
    case ISD::EXTLOAD:
      Result = DAG.getAnyExtOrTrunc(Result, DL, ResIntVT);
      break;
    case ISD::SEXTLOAD:
      Result = DAG.getSExtOrTrunc(Result, DL, ResIntVT);
      break;
    case ISD::ZEXTLOAD:
      Result = DAG.getZExtOrTrunc(Result, DL, ResIntVT);
      break;
    }

    ResOps.push_back(Result);
  }

  // If NumLoads is 2, it means that expandOrPromoteNonRoundLOAD expanded the
  // load into two round loads (i24 -> i16, i8), we lowered them and put them
  // in ResOps, the only thing left is to build a pair in order for the
  // resulting node value type to match the starting node value type.
  if (NumLoads == 2)
    Result = DAG.getNode(ISD::BUILD_PAIR, DL, Op.getValueType(), ResOps[0],
                         ResOps[1]);

  // We do bitcast shenanigans here because we're running in the context of the
  // DAG combiner prior to softening of floating point types, to be able to
  // catch i256 and i512 loads.
  Result = DAG.getBitcast(Op.getValueType(), Result);
  SDValue MergeOps[] = {Result, Chain};
  return DAG.getMergeValues(MergeOps, DL);
}

SDValue Next32TargetLowering::LowerSTORE(SDValue Op, SelectionDAG &DAG) const {
  SDLoc DL(Op);
  SmallVector<SDValue, 2> Ops;
  SmallVector<SDValue, 2> ResChains;

  // Ops can be either the original store, store promoted to a byte-sized type
  // or Lo and Hi from expanding the store into two round stores.
  expandOrPromoteNonRoundSTORE(Op, DAG, Ops);

  for (SDValue StoreOp : Ops) {
    StoreSDNode *Store = cast<StoreSDNode>(StoreOp);
    SDValue Value = Store->getValue();
    const EVT NMemVT = Store->getMemoryVT();

    if (Store->isIndexed())
      report_fatal_error("Unexpected indexed store");
    if (Store->isTruncatingStore() && NMemVT.isFloatingPoint())
      report_fatal_error("Unexpected FP truncating store");

    const EVT IntMemVT =
        EVT::getIntegerVT(*DAG.getContext(), NMemVT.getFixedSizeInBits());
    const unsigned BitWidth = IntMemVT.getFixedSizeInBits();
    if (BitWidth > Subtarget.maxLoadStoreSizeBits())
      report_fatal_error("Unexpected STORE memory size");

    const EVT ValIntVT = EVT::getIntegerVT(
        *DAG.getContext(), Value.getValueType().getFixedSizeInBits());
    Value = DAG.getBitcast(ValIntVT, Value);
    EVT ExtVT = IntMemVT.getSizeInBits() >= 32 ? IntMemVT : MVT::i32;
    Value = DAG.getAnyExtOrTrunc(Value, DL, ExtVT);

    SmallVector<SDValue, 5> GMemOperands;
    // Chain, Mem Params, Addr Hi, Addr Lo, TID
    AppendMemOps(Store, DL, DAG, GMemOperands, BitWidth);

    BreakOperand(DAG, Value, MemDataCount(BitWidth), GMemOperands);

    SDVTList VTs = DAG.getVTList(MVT::i32, MVT::Other);
    SDValue GMem = DAG.getNode(Next32ISD::G_MEM_WRITE, DL, VTs, GMemOperands);
    ResChains.push_back(GMem.getValue(1));
  }

  return DAG.getTokenFactor(DL, ResChains);
}

SDValue Next32TargetLowering::LowerATOMIC_LOAD(SDValue Op,
                                               SelectionDAG &DAG) const {
  SDLoc DL(Op);
  AtomicSDNode *Load = cast<AtomicSDNode>(Op);
  unsigned Size = Load->getMemoryVT().getSizeInBits();
  if (!isPowerOf2_32(Size) || Size > 64)
    report_fatal_error("Unexpected ATOMIC_LOAD memory size");

  const unsigned int OpCnt = MemDataCount(Size);

  if (OpCnt > 2)
    report_fatal_error("Unexpected atomic load size");

  SmallVector<SDValue, 5> GMemOperands;
  // Chain, Mem Params, Addr Hi, Addr Lo, TID
  AppendMemOps(Load, DL, DAG, GMemOperands, Size);

  unsigned int Opc;
  switch (OpCnt) {
  case 1:
    Opc = Next32ISD::G_MEM_READ_1;
    break;
  case 2:
    Opc = Next32ISD::G_MEM_READ_2;
    break;
  default:
    report_fatal_error("Unexpected load size");
  }

  SDValue GMemRead =
      DAG.getNode(Opc, DL, GMemOpVTList(DAG, OpCnt), GMemOperands);
  SDValue Result = BuildGMemOpResult(DAG, GMemRead, OpCnt);
  SDValue Chain = GMemRead.getValue(OpCnt);

  assert(Result.getValueType() == Load->getValueType(0) &&
         "Incorrect load result type");
  SDValue MergeOps[] = {Result, Chain};
  return DAG.getMergeValues(MergeOps, DL);
}

SDValue Next32TargetLowering::LowerATOMIC_STORE(SDValue Op,
                                                SelectionDAG &DAG) const {
  SDLoc DL(Op);
  AtomicSDNode *Store = cast<AtomicSDNode>(Op);
  unsigned Size = Store->getMemoryVT().getSizeInBits();
  if (!isPowerOf2_32(Size) || Size > 64)
    report_fatal_error("Unexpected ATOMIC_STORE memory size");

  const unsigned int OpCnt = MemDataCount(Size);

  if (OpCnt > 2)
    report_fatal_error("Unexpected store size");

  SDValue StoreVal = Store->getVal();
  // There's no isTruncatingStore() analogue for atomic stores, so make sure
  // that "truncating" atomic stores get a legal i32 operand by any-extending it
  // to i32
  if (Store->getMemoryVT().getSizeInBits() < 32) {
    StoreVal = DAG.getAnyExtOrTrunc(StoreVal, DL, MVT::i32);
  }

  SmallVector<SDValue, 6> GMemOperands;
  // Chain, Mem Params, Addr Hi, Addr Lo, TID
  AppendMemOps(Store, DL, DAG, GMemOperands, Size);

  BreakOperand(DAG, StoreVal, OpCnt, GMemOperands);

  SDVTList VTs = DAG.getVTList(MVT::i32, MVT::Other);
  SDValue GMem = DAG.getNode(Next32ISD::G_MEM_WRITE, DL, VTs, GMemOperands);
  return GMem.getValue(1);
}

SDValue Next32TargetLowering::LowerATOMIC_CMP_SWAP(SDValue Op,
                                                   SelectionDAG &DAG) const {
  SDLoc DL(Op);
  AtomicSDNode *Cas = cast<AtomicSDNode>(Op);
  auto Size = Cas->getMemoryVT().getSizeInBits();
  if (!isPowerOf2_32(Size) || Size > 64)
    report_fatal_error("Unexpected ATOMIC_CMP_SWAP memory size");

  const unsigned int OpCnt = MemDataCount(Size);

  assert(Cas && Cas->isCompareAndSwap() && "Unexpected opcode");

  if (OpCnt > 2)
    report_fatal_error("Unexpected CAS size");

  SmallVector<SDValue, 7> GMemOperands;
  // Chain, Mem Params, Addr Hi, Addr Lo, TID
  AppendMemOps(Cas, DL, DAG, GMemOperands, Size);
  BreakOperand(DAG, Op.getOperand(2), OpCnt, GMemOperands);
  BreakOperand(DAG, Op.getOperand(3), OpCnt, GMemOperands);

  unsigned Opc;
  switch (OpCnt) {
  case 1:
    Opc = Next32ISD::G_MEM_CAS_S;
    break;
  case 2:
    Opc = Next32ISD::G_MEM_CAS_D;
    break;
  default:
    report_fatal_error("Unexpected memory size in CAS operation");
  }

  SDValue GMemCas =
      DAG.getNode(Opc, DL, GMemOpVTList(DAG, OpCnt), GMemOperands);
  SDValue Result = BuildGMemOpResult(DAG, GMemCas, OpCnt);
  SDValue Chain = GMemCas.getValue(OpCnt);

  SmallVector<SDValue, 3> MergeOps;
  MergeOps.push_back(Result);
  if (Cas->getOpcode() == ISD::ATOMIC_CMP_SWAP_WITH_SUCCESS) {
    SDValue Success = DAG.getSetCC(DL, Op->getValueType(1), Result,
                                   Op.getOperand(2), ISD::SETEQ);
    MergeOps.push_back(Success);
  }
  MergeOps.push_back(Chain);

  return DAG.getMergeValues(MergeOps, DL);
}

SDValue Next32TargetLowering::LowerATOMIC_LOAD_OP(SDValue Op,
                                                  SelectionDAG &DAG) const {
  SDLoc DL(Op);
  AtomicSDNode *FaOp = cast<AtomicSDNode>(Op);
  auto Size = FaOp->getMemoryVT().getSizeInBits();
  if (!isPowerOf2_32(Size) || Size > 64)
    report_fatal_error("Unexpected ATOMIC_LOAD_OP memory size");

  const unsigned int OpCnt = MemDataCount(Size);

  SmallVector<SDValue, 8> GMemOperands;
  // Chain, Mem Params, Addr Hi, Addr Lo, TID
  AppendMemOps(FaOp, DL, DAG, GMemOperands, Size);
  BreakOperand(DAG, FaOp->getVal(), OpCnt, GMemOperands);

  unsigned Opc;
  switch (OpCnt) {
  case 1:
    Opc = Next32ISD::G_MEM_FAOP_S;
    break;
  case 2:
    Opc = Next32ISD::G_MEM_FAOP_D;
    break;
  default:
    report_fatal_error("Unexpected memory size in atomic op");
  }

  SDValue GMemFaOp =
      DAG.getNode(Opc, DL, GMemOpVTList(DAG, OpCnt), GMemOperands);
  SDValue Result = BuildGMemOpResult(DAG, GMemFaOp, OpCnt);
  SDValue Chain = GMemFaOp.getValue(OpCnt);

  assert(Result.getValueType() == FaOp->getValueType(0) &&
         "Incorrect load result type");
  SDValue MergeOps[] = {Result, Chain};
  return DAG.getMergeValues(MergeOps, DL);
}

static SDValue LowerGlobalAddress32(const SDLoc &DL, const GlobalValue *GV,
                                    SelectionDAG &DAG) {
  if (GV->getValueType()->isFunctionTy())
    return DAG.getNode(
        Next32ISD::WRAPPER, DL, MVT::i32,
        DAG.getTargetGlobalAddress(GV, DL, MVT::i32, 0, Next32II::MO_FUNCTION));

  llvm_unreachable("Unsupported 32-bit global address type");
}

SDValue Next32TargetLowering::LowerVASTART(SDValue Op,
                                           SelectionDAG &DAG) const {
  SDValue Chain = Op.getOperand(0);
  SDValue Pointer = Op.getOperand(1);
  const Value *SourceValue = cast<SrcValueSDNode>(Op.getOperand(2))->getValue();
  SDLoc DL(Op);

  MachineFunction &MF = DAG.getMachineFunction();
  unsigned int high = MF.addLiveIn(Next32::VA_LOW, &Next32::GPR32RegClass);
  unsigned int low = MF.addLiveIn(Next32::VA_HIGH, &Next32::GPR32RegClass);
  SDValue FirstVariadicAddress =
      DAG.getNode(ISD::BUILD_PAIR, DL, MVT::i64, DAG.getRegister(low, MVT::i32),
                  DAG.getRegister(high, MVT::i32));
  return DAG.getStore(Chain, DL, FirstVariadicAddress, Pointer,
                      MachinePointerInfo(SourceValue));
}

SDValue Next32TargetLowering::LowerVAEND(SDValue Op, SelectionDAG &DAG) const {
  SDValue Chain = Op.getOperand(0);
  return Chain;
}

SDValue Next32TargetLowering::LowerVACOPY(SDValue Op, SelectionDAG &DAG) const {
  return DAG.expandVACopy(Op.getNode());
}

SDValue Next32TargetLowering::LowerVAARG(SDValue Op, SelectionDAG &DAG) const {
  return DAG.expandVAArg(Op.getNode());
}

SDValue Next32TargetLowering::LowerPREFETCH(SDValue Op,
                                            SelectionDAG &DAG) const {
  // TODO: Add support for Next32 PREFETCH instruction for GEN2.
  if (!Subtarget.hasPrefetch())
    return Op->getOperand(0);

  SDLoc DL(Op);
  SDValue Chain = Op.getOperand(0);
  SDValue Addr = Op.getOperand(1);

  SmallVector<SDValue, 4> PrefetchOperands = {Chain};
  BreakBasePtr(Addr, PrefetchOperands, DL, DAG);
  PrefetchOperands.push_back(DAG.getRegister(Next32::TID, MVT::i32));
  return DAG.getNode(Next32ISD::PREFETCH, DL, MVT::Other, PrefetchOperands);
}

SDValue Next32TargetLowering::LowerINTRINSIC_WO_CHAIN(SDValue Op,
                                                      SelectionDAG &DAG) const {
  unsigned IntNo = cast<ConstantSDNode>(Op.getOperand(0))->getZExtValue();
  if (IntNo != Intrinsic::get_active_lane_mask)
    return SDValue();

  SDLoc DL(Op);
  SDValue Index = Op.getOperand(1);
  SDValue TripCount = Op.getOperand(2);
  EVT ResVT = Op.getValueType();
  EVT OpVT = Index.getValueType();
  unsigned MaskBits = ResVT.getVectorNumElements();

  // Compute the predication mask to be used for masked loads and stores.
  // Take the minimum of the (trip count - index) difference and vector size and
  // create the corresponding all-zeros mask.
  SDValue Sub = DAG.getNode(ISD::SUB, DL, OpVT, TripCount, Index);
  SDValue Min = DAG.getNode(ISD::UMIN, DL, OpVT, Sub,
                            DAG.getConstant(MaskBits, DL, OpVT));
  SDValue MinTrunc = DAG.getNode(ISD::TRUNCATE, DL, MVT::i32, Min);
  SDValue Shift = DAG.getNode(ISD::SHL, DL, MVT::i32,
                              DAG.getConstant(-1, DL, MVT::i32), MinTrunc);
  SDValue Not = DAG.getNOT(DL, Shift, MVT::i32);

  // Truncate to the right size and cast it to the resulting type.
  EVT ResTruncVT = EVT::getIntegerVT(*DAG.getContext(), MaskBits);
  SDValue ResTrunc = DAG.getNode(ISD::TRUNCATE, DL, ResTruncVT, Not);
  return DAG.getNode(ISD::BITCAST, DL, ResVT, ResTrunc);
}

static SDValue LowerGlobalAddress64(const SDLoc &DL, const GlobalValue *GV,
                                    SelectionDAG &DAG) {
  unsigned int HighType, LowType;
  if (isa<Function>(GV->getAliaseeObject())) {
    HighType = Next32II::MO_FUNC_64HI;
    LowType = Next32II::MO_FUNC_64LO;
  } else {
    HighType = Next32II::MO_MEM_64HI;
    LowType = Next32II::MO_MEM_64LO;
  }
  SDValue HighGV = DAG.getTargetGlobalAddress(GV, DL, MVT::i32, 0, HighType);
  SDValue LowGV = DAG.getTargetGlobalAddress(GV, DL, MVT::i32, 0, LowType);
  SDValue High = DAG.getNode(Next32ISD::WRAPPER, DL, MVT::i32, HighGV);
  SDValue Low = DAG.getNode(Next32ISD::WRAPPER, DL, MVT::i32, LowGV);
  return DAG.getNode(ISD::BUILD_PAIR, DL, MVT::i64, Low, High);
}

SDValue
Next32TargetLowering::LowerTargetGlobalAddress(SDValue Op,
                                               SelectionDAG &DAG) const {
  SDLoc DL(Op);
  const GlobalValue *GV = cast<GlobalAddressSDNode>(Op)->getGlobal();
  SDValue GV32 =
      DAG.getTargetGlobalAddress(GV, DL, MVT::i32, 0, Next32II::MO_BB);

  return GV32;
}

SDValue Next32TargetLowering::LowerGlobalAddress(SDValue Op,
                                                 SelectionDAG &DAG) const {
  SDLoc DL(Op);
  const GlobalValue *GV = cast<GlobalAddressSDNode>(Op)->getGlobal();

  if (Op.getSimpleValueType() == MVT::i64)
    return LowerGlobalAddress64(DL, GV, DAG);

  if (Op.getSimpleValueType() == MVT::i32)
    return LowerGlobalAddress32(DL, GV, DAG);

  llvm_unreachable("Global address type must be either i32 or i64");
}

static RTLIB::Libcall getDivRemLibcall(const SDNode *N,
                                       MVT::SimpleValueType SVT) {
  assert((N->getOpcode() == ISD::SDIVREM || N->getOpcode() == ISD::UDIVREM) &&
         "Unhandled Opcode in getDivRemLibcall");
  bool isSigned = N->getOpcode() == ISD::SDIVREM;
  RTLIB::Libcall LC;
  switch (SVT) {
  default:
    report_fatal_error("Unexpected request for libcall!");
  case MVT::i32:
    LC = isSigned ? RTLIB::SDIVREM_I32 : RTLIB::UDIVREM_I32;
    break;
  case MVT::i64:
    LC = isSigned ? RTLIB::SDIVREM_I64 : RTLIB::UDIVREM_I64;
    break;
  }
  return LC;
}

SDValue Next32TargetLowering::LowerDIVREM(SDValue Op, SelectionDAG &DAG) const {
  unsigned Opcode = Op->getOpcode();
  assert((Opcode == ISD::SDIVREM || Opcode == ISD::UDIVREM) &&
         "Invalid opcode for Div/Rem lowering");

  EVT VT = Op->getValueType(0);
  if (VT != MVT::i64 && VT != MVT::i32)
    report_fatal_error("Unsupported divrem value type");

  SDLoc DL(Op);
  RTLIB::Libcall LC = getDivRemLibcall(Op.getNode(), VT.getSimpleVT().SimpleTy);
  TargetLowering::MakeLibCallOptions CallOptions;
  CallOptions.setSExt(Opcode == ISD::SDIVREM);
  EVT ExtendedVT = EVT::getIntegerVT(*DAG.getContext(), 2 * VT.getSizeInBits());
  SDValue Ops[] = {Op.getOperand(0), Op.getOperand(1)};

  std::pair<SDValue, SDValue> LibcallRes =
      makeLibCall(DAG, LC, ExtendedVT, Ops, CallOptions, DL);
  // __next32_[s/u]divrem[32/64] return the divider and reminder as one double
  // sized value. The original ISD::SDIVREM created 2 different values, as such
  // we need to split the libcall result.
  SDValue DivIndex = DAG.getConstant(1, DL, MVT::i64);
  SDValue RemIndex = DAG.getConstant(0, DL, MVT::i64);

  // Result parts as the same value type as the arguments.
  SDValue Div =
      DAG.getNode(ISD::EXTRACT_ELEMENT, DL, VT, LibcallRes.first, DivIndex);
  SDValue Rem =
      DAG.getNode(ISD::EXTRACT_ELEMENT, DL, VT, LibcallRes.first, RemIndex);

  return DAG.getMergeValues({Div, Rem}, DL);
}

SDValue Next32TargetLowering::LowerFrameIndex(SDValue Op, SelectionDAG &DAG,
                                              uint64_t Offset, SDLoc DL) const {
  const int FI = cast<FrameIndexSDNode>(Op)->getIndex();
  MachineFunction &MF = DAG.getMachineFunction();

  SDValue LowFI = DAG.getTargetFrameIndex(FI, MVT::i32);
  SDValue Off = DAG.getTargetConstant(Offset, DL, MVT::i64);
  SDVTList FIVTList = DAG.getVTList(MVT::i32, MVT::i32);
  SDValue FrameIndex =
      DAG.getNode(Next32ISD::FRAME_OFFSET_WRAPPER, DL, FIVTList, LowFI, Off);

  SDVTList LowVTList = DAG.getVTList(MVT::i32, MVT::i32);
  unsigned int SPLowReg = MF.addLiveIn(Next32::SP_LOW, &Next32::GPR32RegClass);
  SDValue LowFrameIndex = SDValue(FrameIndex.getNode(), 1);
  // Inputs:  i32 SPLOWReg,      i32 FrameIndex
  // Outputs: i32 AddLowResult,  i32 BoolCarryOut
  SDValue N32LowFI =
      DAG.getNode(ISD::UADDO, DL, LowVTList,
                  DAG.getRegister(SPLowReg, MVT::i32), LowFrameIndex);

  SDValue HighFrameIndex = SDValue(FrameIndex.getNode(), 0);

  unsigned int SPHIGHReg =
      MF.addLiveIn(Next32::SP_HIGH, &Next32::GPR32RegClass);
  SDVTList HighVTList = DAG.getVTList(MVT::i32, MVT::i32);
  // Inputs:  i32 SPHIGHReg,     i32 HighFrameIndex,    i32 BoolCarryIn
  // (N32LowFI BoolCarryOut) Outputs: i32 AddHighResult, i32 BoolCarryOut
  SDValue N32HighFI = DAG.getNode(
      ISD::UADDO_CARRY, DL, HighVTList, DAG.getRegister(SPHIGHReg, MVT::i32),
      HighFrameIndex, SDValue(N32LowFI.getNode(), 1));
  return DAG.getNode(ISD::BUILD_PAIR, DL, MVT::i64, N32LowFI, N32HighFI);
}

SDValue Next32TargetLowering::LowerADDRSPACECAST(SDValue Op,
                                                 SelectionDAG &DAG) const {
  SDValue SrcValue = Op.getOperand(0);
  MVT DstTy = Op.getSimpleValueType();
  return DAG.getZExtOrTrunc(SrcValue, SDLoc(Op), DstTy);
}

SDValue Next32TargetLowering::LowerADD(SDValue Op, SelectionDAG &DAG) const {
  if (!isOneConstant(Op.getOperand(1))) {
    return Op;
  }
  return ReplaceUnaryOpcode(Next32ISD::INC, Op, DAG);
}

SDValue Next32TargetLowering::LowerSUB(SDValue Op, SelectionDAG &DAG) const {
  if (!isOneConstant(Op.getOperand(1))) {
    return Op;
  }
  return ReplaceUnaryOpcode(Next32ISD::DEC, Op, DAG);
}

static SDValue ConvertCarryFlagToBooleanCarry(SDValue Value, EVT VT,
                                              SelectionDAG &DAG) {
  SDLoc DL(Value);
  SDValue Zero = DAG.getConstant(0, DL, MVT::i32);
  SDVTList VTs = DAG.getVTList(VT, MVT::i32);
  // Use first output of ADCFLAGS(0, 0, FLAGS) to convert the carry flag from
  // the input computation's flags to a boolean value
  return DAG.getNode(Next32ISD::ADCFLAGS, DL, VTs, Zero, Zero, Value)
      .getValue(0);
}

static SDValue ConvertBooleanCarryToCarryFlag(SDValue BoolCarry,
                                              SelectionDAG &DAG) {
  SDLoc DL(BoolCarry);
  EVT CarryVT = BoolCarry.getValueType();
  SDValue NegOne = DAG.getAllOnesConstant(DL, CarryVT);
  SDVTList VTs = DAG.getVTList(CarryVT, MVT::i32);
  // Use second output of ADDFLAGS(BoolCarry, -1) as a computation whose FLAGS
  // will have a carry value matching the input boolean
  return DAG.getNode(Next32ISD::ADDFLAGS, DL, VTs, BoolCarry, NegOne)
      .getValue(1);
}

SDValue Next32TargetLowering::LowerUADDSUBO(SDValue Op,
                                            SelectionDAG &DAG) const {
  unsigned Opcode;
  switch (Op.getOpcode()) {
  case ISD::UADDO:
    Opcode = Next32ISD::ADDFLAGS;
    break;
  case ISD::USUBO:
    Opcode = Next32ISD::SUBFLAGS;
    break;
  default:
    llvm_unreachable("Node has unexpected Opcode");
  }

  SDLoc DL(Op);
  SDNode *N = Op.getNode();

  SDValue LHS = Op.getOperand(0);
  SDValue RHS = Op.getOperand(1);
  SDVTList VTs = DAG.getVTList(N->getValueType(0), MVT::i32);

  SDValue Value = DAG.getNode(Opcode, DL, VTs, LHS, RHS);
  SDValue FlagsCarry = Value.getValue(1);
  SDValue BoolCarry =
      ConvertCarryFlagToBooleanCarry(FlagsCarry, N->getValueType(1), DAG);

  SDValue MergeOps[] = {Value, BoolCarry};
  return DAG.getMergeValues(MergeOps, DL);
}

SDValue Next32TargetLowering::LowerUADDSUBO_CARRY(SDValue Op,
                                                  SelectionDAG &DAG) const {
  unsigned Opcode;
  switch (Op.getOpcode()) {
  case ISD::UADDO_CARRY:
    Opcode = Next32ISD::ADCFLAGS;
    break;
  case ISD::USUBO_CARRY:
    Opcode = Next32ISD::SBBFLAGS;
    break;
  default:
    llvm_unreachable("Node has unexpected Opcode");
  }

  SDLoc DL(Op);
  SDNode *N = Op.getNode();

  SDValue LHS = Op.getOperand(0);
  SDValue RHS = Op.getOperand(1);
  SDValue BoolCarryIn = Op.getOperand(2);
  SDValue FlagCarryIn = ConvertBooleanCarryToCarryFlag(BoolCarryIn, DAG);
  SDVTList VTs = DAG.getVTList(N->getValueType(0), MVT::i32);

  SDValue ValueOut = DAG.getNode(Opcode, DL, VTs, LHS, RHS, FlagCarryIn);
  SDValue FlagsOut = ValueOut.getValue(1);
  SDValue BoolCarryOut =
      ConvertCarryFlagToBooleanCarry(FlagsOut, N->getValueType(1), DAG);

  // We need to return a node whose value list matches the value list of the one
  // we're lowering
  SDValue MergeOps[] = {ValueOut, BoolCarryOut};
  return DAG.getMergeValues(MergeOps, DL);
}

// Check whether a we want to force the emission of a 64-bit shift libcall
// ourselves, or pass control back to the legalizer. In the cases where we
// pass control back to the legalizer, it is because we are satisfied with its
// expansion behavior (which may *also* resort to generating a libcall).
// We want to customize DAGTypeLegalizer::ExpandIntRes_Shift's behavior for
// the cases it expands to expensive code (even if we were to declare
// ISD::{SHL,SRL,SRA}_PARTS as Legal): namely, the cases where it knows the
// shift amount to be less than half the result type's bit width.
bool Next32TargetLowering::ShouldForceShiftLibCall(SDValue Op,
                                                   SelectionDAG &DAG) const {
  // We can also get here for operand legalization.
  // Check that the result type itself merits further inspection.
  if (Op.getValueType() != MVT::i64)
    return false;

  SDValue Amt = Op.getOperand(1);
  EVT ShTy = Amt.getValueType();
  EVT NVT = getTypeToTransformTo(*DAG.getContext(), Op.getValueType());
  unsigned ShBits = ShTy.getSizeInBits();
  unsigned NVTBits = NVT.getSizeInBits();

  if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(Amt))
    return CN->getAPIntValue().ult(NVTBits);

  APInt HighBitMask = APInt::getHighBitsSet(ShBits, ShBits - Log2_32(NVTBits));
  KnownBits Known = DAG.computeKnownBits(Amt);

  // All high bits are known to be zero.
  return HighBitMask.isSubsetOf(Known.Zero);
}

/// Emit a 64-bit shift libcall.
SDValue Next32TargetLowering::LowerShiftLibCall(SDValue Op,
                                                SelectionDAG &DAG) const {
  if (!ShouldForceShiftLibCall(Op, DAG))
    return SDValue();

  EVT VT = Op.getValueType();
  RTLIB::Libcall LC = RTLIB::UNKNOWN_LIBCALL;
  bool isSigned;
  assert(VT == MVT::i64 && "Unexpected type!");
  switch (Op.getOpcode()) {
  case ISD::SHL:
    isSigned = false;
    LC = RTLIB::SHL_I64;
    break;
  case ISD::SRL:
    isSigned = false;
    LC = RTLIB::SRL_I64;
    break;
  case ISD::SRA:
    isSigned = true;
    LC = RTLIB::SRA_I64;
    break;
  default:
    llvm_unreachable("Unknown shift!");
  }

  assert(LC != RTLIB::UNKNOWN_LIBCALL && "Unknown libcall!");

  // We may be before operand legalization, legalize the shift amount
  SDValue Amt = Op.getOperand(1);
  EVT ShiftTy = getShiftAmountTy(VT, DAG.getDataLayout());
  if (Amt.getValueType() != ShiftTy)
    Amt = DAG.getZExtOrTrunc(Amt, SDLoc(Amt), ShiftTy);

  SDValue Ops[] = {
      Op.getOperand(0),
      Amt,
  };
  TargetLowering::MakeLibCallOptions CallOptions;
  CallOptions.setSExt(isSigned);
  return makeLibCall(DAG, LC, VT, Ops, CallOptions, SDLoc(Op)).first;
}

/// If the shift amount is a constant larger than 32, there's no need to perform
/// this as a 64-bit shift. Let the legalizer decide what to do.
bool Next32TargetLowering::ShouldLowerLongShift(SDValue Op,
                                                SelectionDAG &DAG) const {
  // We can also get here for operand legalization.
  // Check that the result type itself merits further inspection.
  if (Op.getValueType() != MVT::i64)
    return false;

  SDValue Amt = Op.getOperand(1);
  EVT NVT = getTypeToTransformTo(*DAG.getContext(), Op.getValueType());
  unsigned NVTBits = NVT.getSizeInBits();

  if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(Amt))
    return CN->getAPIntValue().ult(NVTBits);

  return true;
}

/// Create a Next32-specific 64-bit shift.
SDValue Next32TargetLowering::LowerLongShift(SDValue Op,
                                             SelectionDAG &DAG) const {
  if (!ShouldLowerLongShift(Op, DAG))
    return SDValue();

  EVT NVT = getTypeToTransformTo(*DAG.getContext(), Op.getValueType());
  assert(NVT == MVT::i32 && "Unexpected type.");

  SDLoc DL(Op);
  SDValue Zero = DAG.getConstant(0, DL, MVT::i32);
  SDValue Lo =
      DAG.getNode(ISD::EXTRACT_ELEMENT, DL, NVT, Op.getOperand(0), Zero);
  SDValue One = DAG.getConstant(1, DL, MVT::i32);
  SDValue Hi =
      DAG.getNode(ISD::EXTRACT_ELEMENT, DL, NVT, Op.getOperand(0), One);

  SDValue Amt = DAG.getZExtOrTrunc(Op->getOperand(1), DL, NVT);

  unsigned Opc;
  switch (Op->getOpcode()) {
  case ISD::SHL:
    Opc = Next32ISD::SHL64;
    break;
  case ISD::SRL:
    Opc = Next32ISD::SHR64;
    break;
  case ISD::SRA:
    Opc = Next32ISD::SHRI64;
    break;
  default:
    llvm_unreachable("Unexpected opcode.");
  }

  SDVTList NodeTys = DAG.getVTList({NVT, NVT});
  SDValue Shift64 = DAG.getNode(Opc, DL, NodeTys, Lo, Hi, Amt);

  return DAG.getNode(ISD::BUILD_PAIR, DL, Op.getValueType(),
                     Shift64.getValue(0), Shift64.getValue(1));
}

SDValue Next32TargetLowering::LowerSHIFT(SDValue Op, SelectionDAG &DAG) const {
  return Subtarget.hasLongShift() ? LowerLongShift(Op, DAG)
                                  : LowerShiftLibCall(Op, DAG);
}

static SDValue ConvertISDCCToNext32CC(SDValue ISDCCNode, SelectionDAG &DAG) {
  assert(ISDCCNode->getOpcode() == ISD::CONDCODE && "Invalid node type");
  ISD::CondCode ISDCC = cast<CondCodeSDNode>(ISDCCNode)->get();
  Next32Constants::CondCode Next32CC = Next32Helpers::ISDCCToNext32CC(ISDCC);
  return DAG.getTargetConstant(Next32CC, SDLoc(ISDCCNode), MVT::i32);
}

// Note 1: The structure of the SUBFLAGS pseudo-instruction, derived from the
// SUB instruction, has one in/out register (dest), one pure in register (src),
// and one pure out register (flags). Because we're using SUBFLAGS for both
// USUBO (which needs both outputs) and condition comparisons (which need
// only the flags output), in the latter case we get an extraneous register
// and DUP instruction instead of just applying FLAGS directly to dest.
// A potential solution would be a COMPARE pseudo-instruction which doesn't have
// a value output, but this is currently not worth the effort.
// Note 2: Special casing comparisons against 0, 1 and -1 without spending an
// actual immediate value could be handy, but would require us to either add
// NOPFLAGS/INCFLAGS/DECFLAGS pseudo-instructions or enable the SUBFLAGS/COMPARE
// pseudo-instruction to take an immediate as well. The case of 0 is currently
// especially problematic as the instruction set has no provision to encode a
// NOP, and simply taking the FLAGS of an arbitrary previous operation is valid
// only for the conditions E and NE.
static SDValue CreateNext32CompareFlags(SDValue LHS, SDValue RHS,
                                        SelectionDAG &DAG) {
  SDLoc DL(LHS);
  EVT LHSVT = LHS.getValueType();
  assert(LHSVT == RHS.getValueType() && "LHS-RHS value type mismatch");
  SDVTList VTs = DAG.getVTList(LHSVT, MVT::i32);
  return DAG.getNode(Next32ISD::SUBFLAGS, DL, VTs, LHS, RHS).getValue(1);
}

SDValue Next32TargetLowering::LowerBR_CC(SDValue Op, SelectionDAG &DAG) const {
  SDLoc DL(Op);
  SDValue Chain = Op.getOperand(0);
  SDValue Condition = ConvertISDCCToNext32CC(Op.getOperand(1), DAG);
  SDValue LHS = Op.getOperand(2);
  SDValue RHS = Op.getOperand(3);
  SDValue Dest = Op.getOperand(4);
  SDValue CmpFlags = CreateNext32CompareFlags(LHS, RHS, DAG);
  return DAG.getNode(Next32ISD::BR_CC, DL, MVT::Other, Chain, Condition,
                     CmpFlags, Dest);
}

SDValue Next32TargetLowering::LowerBRCOND(SDValue Op, SelectionDAG &DAG) const {
  SDLoc DL(Op);
  SDValue Chain = Op.getOperand(0);
  SDValue CondReg = Op.getOperand(1);
  SDValue Dest = Op.getOperand(2);
  SDValue Condition = DAG.getTargetConstant(Next32Constants::NE, DL, MVT::i32);
  SDValue Zero = DAG.getConstant(0, DL, MVT::i32);
  SDValue CmpFlags = CreateNext32CompareFlags(CondReg, Zero, DAG);
  return DAG.getNode(Next32ISD::BR_CC, DL, MVT::Other, Chain, Condition,
                     CmpFlags, Dest);
}

SDValue Next32TargetLowering::LowerSETCC(SDValue Op, SelectionDAG &DAG) const {
  SDLoc DL(Op);
  SDValue LHS = Op.getOperand(0);
  SDValue RHS = Op.getOperand(1);
  SDValue Condition = ConvertISDCCToNext32CC(Op.getOperand(2), DAG);
  SDValue CmpFlags = CreateNext32CompareFlags(LHS, RHS, DAG);
  SDValue Zero = DAG.getConstant(0, DL, MVT::i32);
  return DAG.getNode(Next32ISD::INCc, DL, MVT::i32, Zero, Condition, CmpFlags);
}

SDValue Next32TargetLowering::LowerSELECT_CC(SDValue Op,
                                             SelectionDAG &DAG) const {
  SDLoc DL(Op);
  SDValue LHS = Op.getOperand(0);
  SDValue RHS = Op.getOperand(1);
  SDValue TrueRet = Op.getOperand(2);
  SDValue FalseRet = Op.getOperand(3);
  SDValue Condition = ConvertISDCCToNext32CC(Op.getOperand(4), DAG);
  SDValue CmpFlags = CreateNext32CompareFlags(LHS, RHS, DAG);

  if (auto *FalseConstant = dyn_cast<ConstantSDNode>(FalseRet)) {
    if (auto *TrueConstant = dyn_cast<ConstantSDNode>(TrueRet)) {
      uint64_t Val =
          TrueConstant->getLimitedValue() - FalseConstant->getLimitedValue();
      if (Val == 1)
        return DAG.getNode(Next32ISD::INCc, DL, MVT::i32, FalseRet, Condition,
                           CmpFlags);
      return DAG.getNode(Next32ISD::ADDc, DL, MVT::i32, FalseRet,
                         DAG.getConstant(Val, DL, MVT::i32), Condition,
                         CmpFlags);
    }
  }
  return DAG.getNode(Next32ISD::SELECTc, DL, MVT::i32, FalseRet, TrueRet,
                     Condition, CmpFlags);
}

SDValue Next32TargetLowering::LowerDYNAMIC_STACKALLOC(SDValue Op,
                                                      SelectionDAG &DAG) const {
  SDLoc DL(Op);
  SDValue Chain = Op.getOperand(0);
  // Dynamic stack alloc of more then 4GB is undefined behaviour
  SDValue Size = DAG.getNode(ISD::TRUNCATE, DL, MVT::i32, Op.getOperand(1));
  // Alignment is guaranteed to be bigger than the stack alignment (if required)
  // or 0
  const unsigned Alignment =
      cast<ConstantSDNode>(Op.getOperand(2))->getZExtValue();
  assert((Alignment == 0) || isPowerOf2_32(Alignment));

  if (Alignment)
    Size = DAG.getNode(ISD::ADD, DL, MVT::i32, Size,
                       DAG.getConstant(Alignment, DL, MVT::i32));

  SDValue TID = DAG.getRegister(Next32::TID, MVT::i32);
  SDVTList ResultTys = DAG.getVTList(MVT::i32, MVT::i32, MVT::i32, MVT::Other);
  SDValue Alloca =
      DAG.getNode(Next32ISD::ALLOCA, DL, ResultTys, Chain, Size, TID);

  SDValue AllocaHigh = Alloca.getValue(0);
  SDValue AllocaLow = Alloca.getValue(1);
  if (Alignment) {
    assert(DAG.getSubtarget().getFrameLowering()->getStackGrowthDirection() ==
           TargetFrameLowering::StackGrowsUp);
    SDVTList VTList = DAG.getVTList(MVT::i32, MVT::i32);
    // Inputs:  i32 AllocaLow,     i32 (Alignemnt -1)
    // Outputs: i32 AddLowResult,  i32 BoolCarryOut
    AllocaLow = DAG.getNode(ISD::UADDO, DL, VTList, AllocaLow,
                            DAG.getConstant(Alignment - 1, DL, MVT::i32));

    // Inputs:  i32 AllocaHigh,    i32 ConstZero (alignment < 4GB),  i32
    // BoolCarryIn (AllocaLow BoolCarryOut) Outputs: i32 AddHighResult, i32
    // BoolCarryOut
    AllocaHigh = DAG.getNode(ISD::UADDO_CARRY, DL, VTList, AllocaHigh,
                             DAG.getConstant(0, DL, MVT::i32),
                             SDValue(AllocaLow.getNode(), 1));

    AllocaLow = DAG.getNode(ISD::AND, DL, MVT::i32, AllocaLow,
                            DAG.getConstant(~(Alignment - 1), DL, MVT::i32));
  }

  SDValue Ptr = DAG.getNode(ISD::BUILD_PAIR, DL, Op.getValueType(), AllocaLow,
                            AllocaHigh);

  SDValue OutChain = Alloca.getValue(3);
  SDValue MergeOps[] = {Ptr, OutChain};
  return DAG.getMergeValues(MergeOps, DL);
}

SDValue Next32TargetLowering::LowerSTACKSAVE(SDValue Op,
                                             SelectionDAG &DAG) const {
  SDValue Chain = Op.getOperand(0);
  SDLoc DL(Op);

  SDValue TID = DAG.getRegister(Next32::TID, MVT::i32);
  SDValue Zero = DAG.getConstant(0, DL, MVT::i32);
  SDVTList ResultTys = DAG.getVTList(MVT::i32, MVT::i32, MVT::i32, MVT::Other);
  SDValue SetFrame =
      DAG.getNode(Next32ISD::SET_FRAME, DL, ResultTys, Chain, Zero, TID);

  SDValue Ptr = DAG.getNode(ISD::BUILD_PAIR, DL, Op.getValueType(),
                            SetFrame.getValue(1), SetFrame.getValue(0));
  SDValue OutChain = SetFrame.getValue(3);
  SDValue MergeOps[] = {Ptr, OutChain};
  return DAG.getMergeValues(MergeOps, DL);
}

SDValue Next32TargetLowering::LowerSTACKRESTORE(SDValue Op,
                                                SelectionDAG &DAG) const {
  SDValue Chain = Op.getOperand(0);
  SDValue RestorePointer = Op.getOperand(1);
  SDLoc DL(Op);

  SDValue RestorePointerHighIndex = DAG.getConstant(1, DL, MVT::i64);
  SDValue RestorePointerLowIndex = DAG.getConstant(0, DL, MVT::i64);
  SDValue RestorePointerHigh =
      DAG.getNode(ISD::EXTRACT_ELEMENT, DL, MVT::i32, RestorePointer,
                  RestorePointerHighIndex);
  SDValue RestorePointerLow =
      DAG.getNode(ISD::EXTRACT_ELEMENT, DL, MVT::i32, RestorePointer,
                  RestorePointerLowIndex);
  SDValue TID = DAG.getRegister(Next32::TID, MVT::i32);

  SDVTList VTs = DAG.getVTList(MVT::i32, MVT::Other);
  SDValue ResetFrame = DAG.getNode(Next32ISD::RESET_FRAME, DL, VTs, Chain,
                                   RestorePointerHigh, RestorePointerLow, TID);
  return ResetFrame.getValue(1);
}

SDValue Next32TargetLowering::ReplaceOpcode(unsigned Opcode, SDValue Op,
                                            SelectionDAG &DAG,
                                            ArrayRef<SDValue> Ops,
                                            SDVTList VTs) const {
  if (VTs.NumVTs == 0) {
    VTs = Op.getNode()->getVTList();
  }
  return DAG.getNode(Opcode, Op, VTs, Ops);
}

SDValue Next32TargetLowering::ReplaceUnaryOpcode(unsigned Opcode, SDValue Op,
                                                 SelectionDAG &DAG,
                                                 SDVTList VTs) const {
  SDValue Ops[] = {Op->getOperand(0)};
  return ReplaceOpcode(Opcode, Op, DAG, Ops, VTs);
}

// Calling Convention Implementation
#include "Next32GenCallingConv.inc"

SDValue Next32TargetLowering::LowerFormalArguments(
    SDValue Chain, CallingConv::ID CallConv, bool IsVarArg,
    const SmallVectorImpl<ISD::InputArg> &Ins, const SDLoc &DL,
    SelectionDAG &DAG, SmallVectorImpl<SDValue> &InVals) const {
  switch (CallConv) {
  default:
    report_fatal_error("Unsupported calling convention");
  case CallingConv::C:
  case CallingConv::Fast:
    break;
  }
  MachineFunction &MF = DAG.getMachineFunction();
  MachineRegisterInfo &RegInfo = MF.getRegInfo();
  // Assign locations to all of the incoming arguments.
  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CallConv, IsVarArg, MF, ArgLocs, *DAG.getContext());
  CCInfo.AnalyzeFormalArguments(Ins, CC_Next32);
  SmallVector<SDValue> FeedOps{Chain};

  for (unsigned I = 0, E = ArgLocs.size(); I != E; ++I) {
    CCValAssign &VA = ArgLocs[I];
    if (VA.isRegLoc()) {
      unsigned VReg = RegInfo.createVirtualRegister(&Next32::GPR32RegClass);
      RegInfo.addLiveIn(VA.getLocReg(), VReg);
      EVT RegVT = VA.getLocVT();
      SDValue Reg = DAG.getRegister(VReg, RegVT);
      SDValue ArgSize =
          DAG.getTargetConstant(Next32Helpers::BitsToSizeFieldValue(
                                    Ins[VA.getValNo()].ArgVT.getSizeInBits()),
                                DL, MVT::i32);
      SDValue FeedPhysIdx = DAG.getTargetConstant(VA.getLocReg(), DL, MVT::i32);
      FeedOps.push_back(FeedPhysIdx);
      FeedOps.push_back(ArgSize);
      InVals.push_back(Reg);
    } else {
      report_fatal_error("Ran out of registers for formal arguments");
    }
  }
  // Generate placeholder for argument FEEDERS, so we can later generate FEEDERS
  // with actual physical registers in order to respect ABI requirements. We are
  // doing this in order to avoid problem when register allocator is under high
  // register pressure and can't match corensponding physical registers (used
  // for arguments) with FEEDERS.
  return DAG.getNode(Next32ISD::FEEDER_ARGS, DL, MVT::Other, FeedOps);
}

void Next32TargetLowering::AnalyzeVarArgsCallOperands(
    CCState &CCInfo, const SmallVectorImpl<ISD::OutputArg> &Outs,
    CCAssignFn FixedFn, CCAssignFn VarArgsFn) {
  unsigned NumOps = Outs.size();
  for (unsigned i = 0; i != NumOps; ++i) {
    MVT ArgVT = Outs[i].VT;
    ISD::ArgFlagsTy ArgFlags = Outs[i].Flags;
    auto Fn = Outs[i].IsFixed ? FixedFn : VarArgsFn;
    if (Fn(i, ArgVT, ArgVT, CCValAssign::Full, ArgFlags, CCInfo))
      llvm_unreachable("Call operand has unhandled type");
  }
}

static Next32ISD::NodeType GetCallOpcode(MachineFunction &MF, SDValue Callee,
                                         CallingConv::ID CallConv) {

  switch (CallConv) {
  default:
    report_fatal_error("Unsupported calling convention");
  case CallingConv::Fast:
  case CallingConv::C:
    if (Callee.getOpcode() == ISD::GlobalAddress ||
        Callee.getOpcode() == ISD::TargetGlobalAddress ||
        Callee.getOpcode() == ISD::ExternalSymbol ||
        Callee.getOpcode() == ISD::TargetExternalSymbol)
      return Next32ISD::CALL;
    else
      return Next32ISD::CALLPTR_WRAPPER;
    break;
  }
}

SDValue
Next32TargetLowering::LowerCall(TargetLowering::CallLoweringInfo &CLI,
                                SmallVectorImpl<SDValue> &InVals) const {
  SelectionDAG &DAG = CLI.DAG;
  auto &Outs = CLI.Outs;
  auto &OutVals = CLI.OutVals;
  auto &Ins = CLI.Ins;
  SDValue Chain = CLI.Chain;
  SDValue Callee = CLI.Callee;
  bool &IsTailCall = CLI.IsTailCall;
  CallingConv::ID CallConv = CLI.CallConv;
  bool IsVarArg = CLI.IsVarArg;
  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo &MFI = MF.getFrameInfo();
  // Next32 target does not support tail call optimization.
  IsTailCall = false;
  unsigned int Opcode = GetCallOpcode(MF, Callee, CallConv);

  // Analyze operands of the call, assigning locations to each operand.
  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CallConv, IsVarArg, MF, ArgLocs, *DAG.getContext());

  // This is a hack, this https://reviews.llvm.org/D42374 will allow us to
  // remove this hack once merged.
  if (CLI.IsVarArg)
    AnalyzeVarArgsCallOperands(CCInfo, Outs, CC_Next32, CC_Next32VarArg);
  else
    CCInfo.AnalyzeCallOperands(Outs, CC_Next32);

  auto PtrVT = MVT::i32;

  SmallVector<std::pair<SDValue, SDValue>, 8> RegsToPass;
  SmallVector<SDValue, 8> MemOpChains;
  SDValue FirstVarArgFI;
  SDValue ByValLow;

  // Walk arg assignments
  for (unsigned i = 0, e = static_cast<unsigned>(ArgLocs.size()); i != e; ++i) {
    ISD::ArgFlagsTy Flags = Outs[i].Flags;
    CCValAssign &VA = ArgLocs[i];
    SDValue Arg = OutVals[i];

    // Promote the value if needed.
    switch (VA.getLocInfo()) {
    default:
      llvm_unreachable("Unknown loc info");
    case CCValAssign::Full:
      break;
    case CCValAssign::SExt:
      Arg = DAG.getNode(ISD::SIGN_EXTEND, CLI.DL, VA.getLocVT(), Arg);
      break;
    case CCValAssign::ZExt:
      Arg = DAG.getNode(ISD::ZERO_EXTEND, CLI.DL, VA.getLocVT(), Arg);
      break;
    case CCValAssign::AExt:
      Arg = DAG.getNode(ISD::ANY_EXTEND, CLI.DL, VA.getLocVT(), Arg);
      break;
    }

    // Push arguments into RegsToPass vector
    if (VA.isRegLoc()) {
      if (!Flags.isByVal()) {
        SDValue ArgSize = DAG.getTargetConstant(
            Next32Helpers::BitsToSizeFieldValue(
                Outs[VA.getValNo()].ArgVT.getSizeInBits()),
            CLI.DL, MVT::i32);
        RegsToPass.push_back(std::make_pair(Arg, ArgSize));

      } else if (!ByValLow) {
        assert(Flags.isSplit());

        // Low part of byval argument. Capture it and use it later
        ByValLow = Arg;

      } else {
        assert(Flags.isSplitEnd());

        // High part of byval argument.
        SDValue ByValHigh = Arg;

        // Create a stack object to hold the copy of the argument
        const Align ByValAlign = Flags.getNonZeroByValAlign();
        int FI = MFI.CreateStackObject(Flags.getByValSize(), ByValAlign, false);
        SDValue ByValFI =
            DAG.getFrameIndex(FI, getPointerTy(DAG.getDataLayout()));

        // Copy the argument into the new stack object
        SDValue WholeValue =
            DAG.getNode(ISD::BUILD_PAIR, CLI.DL, MVT::i64, ByValLow, ByValHigh);
        MemOpChains.push_back(DAG.getMemcpy(
            Chain, CLI.DL, ByValFI, WholeValue,
            DAG.getIntPtrConstant(Flags.getByValSize(), CLI.DL), ByValAlign,
            /*isVolatile*/ false, /*AlwaysInline=*/false,
            /*CI=*/nullptr, std::nullopt, MachinePointerInfo(), MachinePointerInfo()));

        // Pass a pointer to the copied argument in the stack
        SDValue ByValPtrLoIndex = DAG.getConstant(0, CLI.DL, MVT::i32);
        SDValue ByValPtrHiIndex = DAG.getConstant(1, CLI.DL, MVT::i32);
        SDValue ByValPtrLo = DAG.getNode(ISD::EXTRACT_ELEMENT, CLI.DL, MVT::i32,
                                         ByValFI, ByValPtrLoIndex);
        SDValue ByValPtrHi = DAG.getNode(ISD::EXTRACT_ELEMENT, CLI.DL, MVT::i32,
                                         ByValFI, ByValPtrHiIndex);
        SDValue ArgSize = DAG.getTargetConstant(
            llvm::Next32Constants::InstructionSize::InstructionSize64, CLI.DL,
            MVT::i32);
        RegsToPass.push_back(std::make_pair(ByValPtrLo, ArgSize));
        RegsToPass.push_back(std::make_pair(ByValPtrHi, ArgSize));

        // Done handling the byval argument, reset the state
        ByValLow = SDValue();
      }
    } else if (VA.isMemLoc()) {
      if (Outs[i].IsFixed)
        report_fatal_error(
            "Stack argument passing is only supported on variadic parameters");

      // Second parameter to CreateFixedObject assumes the stack direction.
      assert(DAG.getSubtarget().getFrameLowering()->getStackGrowthDirection() ==
             TargetFrameLowering::StackGrowsUp);

      MachineFrameInfo &MFI = MF.getFrameInfo();
      int FI = MFI.CreateFixedObject(Arg.getValueSizeInBits() / 8,
                                     VA.getLocMemOffset(), false);
      SDValue FIN = DAG.getFrameIndex(FI, getPointerTy(DAG.getDataLayout()));
      if (!FirstVarArgFI)
        FirstVarArgFI = FIN;
      MemOpChains.push_back(DAG.getStore(Chain, CLI.DL, Arg, FIN,
                                         MachinePointerInfo(), MaybeAlign(),
                                         MachineMemOperand::MOVolatile));
    } else
      llvm_unreachable("call arg pass bug");
  }
  if (!MemOpChains.empty())
    Chain = DAG.getNode(ISD::TokenFactor, CLI.DL, MVT::Other, MemOpChains);
  if (CLI.IsVarArg) {
    SDValue VaLow;
    SDValue VaHigh;
    if (FirstVarArgFI) {
      SDValue VaLowIndex = DAG.getConstant(0, CLI.DL, MVT::i32);
      SDValue VaHighIndex = DAG.getConstant(1, CLI.DL, MVT::i32);
      VaLow = DAG.getNode(ISD::EXTRACT_ELEMENT, CLI.DL, MVT::i32, FirstVarArgFI,
                          VaLowIndex);
      VaHigh = DAG.getNode(ISD::EXTRACT_ELEMENT, CLI.DL, MVT::i32,
                           FirstVarArgFI, VaHighIndex);
    } else {
      VaLow = DAG.getConstant(0, CLI.DL, MVT::i32);
      VaHigh = VaLow;
    }
    SDValue ArgSize = DAG.getTargetConstant(
        llvm::Next32Constants::InstructionSize::InstructionSize64, CLI.DL,
        MVT::i32);
    RegsToPass.push_back(std::make_pair(VaHigh, ArgSize));
    RegsToPass.push_back(std::make_pair(VaLow, ArgSize));
  }

  // If the callee is a GlobalAddress node (quite common, every direct call is)
  // turn it into a TargetGlobalAddress node so that legalize doesn't hack it.
  // Likewise ExternalSymbol -> TargetExternalSymbol.
  Next32Constants::RRIAttribute FunctionAttributes =
      Next32Constants::RRIAttribute::Default;
  if (auto *G = dyn_cast<GlobalAddressSDNode>(Callee)) {
    FunctionAttributes =
        Next32Helpers::GetFunctionAttribute(G->getGlobal()->getName(), this);
    Callee = DAG.getTargetGlobalAddress(G->getGlobal(), CLI.DL, PtrVT,
                                        G->getOffset(), 0);
  } else if (auto *E = dyn_cast<ExternalSymbolSDNode>(Callee)) {
    FunctionAttributes =
        Next32Helpers::GetFunctionAttribute(E->getSymbol(), this);
    Callee = DAG.getTargetExternalSymbol(E->getSymbol(), PtrVT, 0);
  }

  // Returns a chain & a flag for retval copy to use.
  SDVTList NodeTys = DAG.getVTList(MVT::Other, MVT::Glue);
  SmallVector<SDValue, 8> Ops;
  Ops.push_back(Chain);
  Ops.push_back(DAG.getTargetConstant(Ins.size(), CLI.DL, MVT::i32));
  if (Opcode != Next32ISD::CALLPTR_WRAPPER) {
    Ops.push_back(Callee);
  } else {
    SDValue LowIndex = DAG.getConstant(1, CLI.DL, MVT::i32);
    SDValue CalleeLow =
        DAG.getNode(ISD::EXTRACT_ELEMENT, CLI.DL, MVT::i32, Callee, LowIndex);
    Ops.push_back(CalleeLow);
    SDValue HighIndex = DAG.getConstant(0, CLI.DL, MVT::i32);
    SDValue CalleeHigh =
        DAG.getNode(ISD::EXTRACT_ELEMENT, CLI.DL, MVT::i32, Callee, HighIndex);
    Ops.push_back(CalleeHigh);
  }
  // Add argument registers to the end of the list so that they are
  // known live into the call.
  for (auto &Reg : RegsToPass) {
    Ops.push_back(Reg.first);
    Ops.push_back(Reg.second);
  }

  SmallVector<SDValue, 16> RetSiteSyms;

  // Return indication
  MCSymbol *Sym = MF.getContext().createTempSymbol();
  if (auto *ELFSym = dyn_cast<MCSymbolELF>(Sym))
    ELFSym->setType(ELF::STT_COMMON);
  SDValue RetSiteSym = DAG.getMCSymbol(Sym, MVT::Other);
  RetSiteSyms.push_back(RetSiteSym);
  Ops.push_back(RetSiteSym);

  Ops.push_back(DAG.getTargetConstant(FunctionAttributes, CLI.DL, MVT::i32));
  Chain = DAG.getNode(Opcode, CLI.DL, NodeTys, Ops);
  SDValue InFlag = Chain.getValue(1);
  // Handle result values, copying them out of physregs into vregs that we
  // return.
  return LowerCallResult(Chain, InFlag, CallConv, IsVarArg, Ins, CLI.DL, DAG,
                         InVals, RetSiteSyms);
}

SDValue
Next32TargetLowering::LowerReturn(SDValue Chain, CallingConv::ID CallConv,
                                  bool /*isVarArg*/,
                                  const SmallVectorImpl<ISD::OutputArg> &Outs,
                                  const SmallVectorImpl<SDValue> &OutVals,
                                  const SDLoc &DL, SelectionDAG &DAG) const {
  MachineFunction &MF = DAG.getMachineFunction();
  SmallVector<SDValue, 8> RetOps(1, Chain);
  unsigned int RetFidReg = 0;
  SDValue RetFid;

  RetFidReg = MF.addLiveIn(Next32::RET_FID, &Next32::RESERVEDRegClass);
  RetFid = DAG.getRegister(RetFidReg, MVT::i32);
  RetOps.push_back(RetFid);
  SDValue ArgReg = DAG.getRegister(Next32::TID, MVT::i32);
  RetOps.push_back(ArgReg);
  RetOps.push_back(DAG.getTargetConstant(
      llvm::Next32Constants::InstructionSize::InstructionSize32, DL, MVT::i32));
  unsigned int index = 0;
  for (auto &OutVal : OutVals) {
    RetFidReg = MF.addLiveIn(Next32::RET_FID, &Next32::RESERVEDRegClass);
    RetFid = DAG.getRegister(RetFidReg, MVT::i32);
    RetOps.push_back(RetFid);
    if (OutVal.getOpcode() == ISD::UNDEF)
      RetOps.push_back(DAG.getConstant(0, DL, MVT::i32));
    else
      RetOps.push_back(OutVal);
    RetOps.push_back(DAG.getTargetConstant(
        Next32Helpers::BitsToSizeFieldValue(Outs[index].ArgVT.getSizeInBits()),
        DL, MVT::i32));
    ++index;
  }
  return DAG.getNode(Next32ISD::RET_FLAG, DL, MVT::Other, RetOps);
}

SDValue Next32TargetLowering::LowerCallResult(
    SDValue Chain, SDValue InFlag, CallingConv::ID CallConv, bool IsVarArg,
    const SmallVectorImpl<ISD::InputArg> &Ins, const SDLoc &DL,
    SelectionDAG &DAG, SmallVectorImpl<SDValue> &InVals,
    const SmallVectorImpl<SDValue> &RetSiteSyms) const {

  MachineFunction &MF = DAG.getMachineFunction();
  // Assign locations to each value returned by this call.
  SmallVector<CCValAssign, 16> RVLocs;
  CCState CCInfo(CallConv, IsVarArg, MF, RVLocs, *DAG.getContext());

  CCInfo.AnalyzeCallResult(Ins, CC_Next32);
  unsigned int SymIndex = 0;
  SDVTList NodeTys = DAG.getVTList(MVT::Other, MVT::Glue);

  Chain = DAG.getNode(Next32ISD::SYM, DL, NodeTys, Chain, RetSiteSyms[SymIndex],
                      InFlag)
              .getValue(0);
  InFlag = Chain.getValue(1);
  SDValue ArgSize = DAG.getTargetConstant(
      llvm::Next32Constants::InstructionSize::InstructionSize32, DL, MVT::i32);
  Chain = DAG.getNode(Next32ISD::CALL_TERMINATOR_TID, DL, NodeTys, Chain,
                      ArgSize, InFlag)
              .getValue(0);
  InFlag = Chain.getValue(1);

  if (RVLocs.size() != 0) {
    SmallVector<EVT, 4> ValueTypes;
    SmallVector<SDValue, 6> CTOps = {Chain};
    for (size_t i = 0; i < RVLocs.size(); i++) {
      auto In = Ins[i];
      unsigned int SizeVal =
          llvm::Next32Constants::InstructionSize::InstructionSize32;
      if ((!In.Flags.isZExt() && !In.Flags.isSExt()) ||
          In.ArgVT.getSizeInBits() > 32)
        SizeVal = Next32Helpers::BitsToSizeFieldValue(In.ArgVT.getSizeInBits());
      // Collect the sizes of original input types in order to generate correct
      // suffixes for FEEDER instructions.
      CTOps.push_back(DAG.getTargetConstant(SizeVal, DL, MVT::i32));
      // Collect all input type sizes in order to generate proper return value.
      ValueTypes.push_back(MVT::i32);
    }

    CTOps.push_back(InFlag);
    ValueTypes.push_back(MVT::Other);
    ValueTypes.push_back(MVT::Glue);
    NodeTys = DAG.getVTList(ValueTypes);
    Chain = DAG.getNode(Next32ISD::CALL_TERMINATOR, DL, NodeTys, CTOps)
                .getValue(RVLocs.size());
    for (size_t i = 0; i < RVLocs.size(); i++) {
      InVals.push_back(Chain.getValue(i));
    }
  }
  return Chain;
}

const char *Next32TargetLowering::getTargetNodeName(unsigned Opcode) const {
  switch ((Next32ISD::NodeType)Opcode) {
  case Next32ISD::ADDc:
    return "Next32ISD::ADDc";
  case Next32ISD::XORc:
    return "Next32ISD::XORc";
  case Next32ISD::ORc:
    return "Next32ISD::ORc";
  case Next32ISD::ANDc:
    return "Next32ISD::ANDc";
  case Next32ISD::SUBc:
    return "Next32ISD::SUBc";
  case Next32ISD::ADDFLAGS:
    return "Next32ISD::ADDFLAGS";
  case Next32ISD::SUBFLAGS:
    return "Next32ISD::SUBFLAGS";
  case Next32ISD::ADCFLAGS:
    return "Next32ISD::ADCFLAGS";
  case Next32ISD::SBBFLAGS:
    return "Next32ISD::SBBFLAGS";
  case Next32ISD::SHLc:
    return "Next32ISD::SHLc";
  case Next32ISD::SHRc:
    return "Next32ISD::SHRc";
  case Next32ISD::SHRIc:
    return "Next32ISD::SHRIc";
  case Next32ISD::SHL64:
    return "Next32ISD::SHL64";
  case Next32ISD::SHR64:
    return "Next32ISD::SHR64";
  case Next32ISD::SHRI64:
    return "Next32ISD::SHRI64";
  case Next32ISD::CTLZc:
    return "Next32ISD::CTLZc";
  case Next32ISD::CTTZc:
    return "Next32ISD::CTTZc";
  case Next32ISD::SELECT:
    return "Next32ISD::SELECT";
  case Next32ISD::SELECTc:
    return "Next32ISD::SELECTc";
  case Next32ISD::CHAIN:
    return "Next32ISD::CHAIN";
  case Next32ISD::CHAINc:
    return "Next32ISD::CHAINc";
  case Next32ISD::CHAINP:
    return "Next32ISD::CHAINP";
  case Next32ISD::CHAINPc:
    return "Next32ISD::CHAINPc";
  case Next32ISD::WRITER:
    return "Next32ISD::WRITER";
  case Next32ISD::NOTc:
    return "Next32ISD::NOTc";
  case Next32ISD::NEGc:
    return "Next32ISD::NEGc";
  case Next32ISD::INC:
    return "Next32ISD::INC";
  case Next32ISD::INCc:
    return "Next32ISD::INCc";
  case Next32ISD::DEC:
    return "Next32ISD::DEC";
  case Next32ISD::DECc:
    return "Next32ISD::DECc";
  case Next32ISD::RET_FLAG:
    return "Next32ISD::RET_FLAG";
  case Next32ISD::FEEDER_ARGS:
    return "Next32ISD::FEEDER_ARGS";
  case Next32ISD::CALL:
    return "Next32ISD::CALL";
  case Next32ISD::CALLPTR:
    return "Next32ISD::CALLPTR";
  case Next32ISD::CALLPTR_WRAPPER:
    return "Next32ISD::CALLPTR_WRAPPER";
  case Next32ISD::CALL_TERMINATOR_TID:
    return "Next32ISD::CALL_TERMINATOR_TID";
  case Next32ISD::CALL_TERMINATOR:
    return "Next32ISD::CALL_TERMINATOR";
  case Next32ISD::BR_CC:
    return "Next32ISD::BR_CC";
  case Next32ISD::DUP:
    return "Next32ISD::DUP";
  case Next32ISD::SYM:
    return "Next32ISD::SYM";
  case Next32ISD::SET_FRAME:
    return "Next32ISD::SET_FRAME";
  case Next32ISD::RESET_FRAME:
    return "Next32ISD::RESET_FRAME";
  case Next32ISD::ALLOCA:
    return "Next32ISD::ALLOCA";
  case Next32ISD::WRAPPER:
    return "Next32ISD::WRAPPER";
  case Next32ISD::FRAME_OFFSET_WRAPPER:
    return "Next32ISD::FRAME_OFFSET_WRAPPER";
  case Next32ISD::G_VMEM_WRITE:
    return "Next32ISD::G_VMEM_WRITE";
  case Next32ISD::G_VMEM_READ_1:
    return "Next32ISD::G_VMEM_READ_1";
  case Next32ISD::G_VMEM_READ_2:
    return "Next32ISD::G_VMEM_READ_2";
  case Next32ISD::G_VMEM_READ_4:
    return "Next32ISD::G_VMEM_READ_4";
  case Next32ISD::G_VMEM_READ_8:
    return "Next32ISD::G_VMEM_READ_8";
  case Next32ISD::G_VMEM_READ_16:
    return "Next32ISD::G_VMEM_READ_16";
  case Next32ISD::G_MEM_WRITE:
    return "Next32ISD::G_MEM_WRITE";
  case Next32ISD::G_MEM_READ_1:
    return "Next32ISD::G_MEM_READ_1";
  case Next32ISD::G_MEM_READ_2:
    return "Next32ISD::G_MEM_READ_2";
  case Next32ISD::G_MEM_READ_4:
    return "Next32ISD::G_MEM_READ_4";
  case Next32ISD::G_MEM_READ_8:
    return "Next32ISD::G_MEM_READ_8";
  case Next32ISD::G_MEM_READ_16:
    return "Next32ISD::G_MEM_READ_16";
  case Next32ISD::G_MEM_FAOP_S:
    return "Next32ISD::G_MEM_FAOP_S";
  case Next32ISD::G_MEM_FAOP_D:
    return "Next32ISD::G_MEM_FAOP_D";
  case Next32ISD::G_MEM_CAS_S:
    return "Next32ISD::G_MEM_CAS_S";
  case Next32ISD::G_MEM_CAS_D:
    return "Next32ISD::G_MEM_CAS_D";
  case Next32ISD::PREFETCH:
    return "Next32ISD::PREFETCH";
  case Next32ISD::TOKEN_ID_F:
    return "Next32ISD::TOKEN_ID_F";
  case Next32ISD::BARRIER:
    return "Next32ISD::BARRIER";
  case Next32ISD::SET_TID:
    return "Next32ISD::SET_TID";
  case Next32ISD::PSEUDO_LEA:
    return "Next32ISD::PSEUDO_LEA";
  default:
    break;
  }
  return nullptr;
}

// Recombine CALL_TERMINATOR_PARAMS and CALL_TERMINATOR_RESULTS into a single
// MachineInstr with variadic operands and results.
static MachineBasicBlock *
lowerCallTerminatorResults(MachineInstr &CTResults, DebugLoc DL,
                           MachineBasicBlock *BB, const TargetInstrInfo &TII) {
  MachineInstr &CTParams = *CTResults.getPrevNode();
  assert(CTParams.getOpcode() == Next32::CALL_TERMINATOR_PARAMS);

  MachineFunction &MF = *BB->getParent();
  const MCInstrDesc &MCID = TII.get(Next32::CALL_TERMINATOR);
  MachineInstrBuilder MIB(MF, MF.CreateMachineInstr(MCID, DL));

  for (auto Def : CTResults.defs())
    MIB.add(Def);
  for (auto Use : CTParams.uses())
    MIB.add(Use);

  BB->insert(CTResults.getIterator(), MIB);
  CTParams.eraseFromParent();
  CTResults.eraseFromParent();

  return BB;
}

MachineBasicBlock *
Next32TargetLowering::EmitInstrWithCustomInserter(MachineInstr &MI,
                                                  MachineBasicBlock *BB) const {
  const TargetInstrInfo &TII = *Subtarget.getInstrInfo();
  DebugLoc DL = MI.getDebugLoc();

  switch (MI.getOpcode()) {
  default:
    return BB;
  case Next32::CALL_TERMINATOR_RESULTS:
    return lowerCallTerminatorResults(MI, DL, BB, TII);
  }
}

MVT Next32TargetLowering::getScalarShiftAmountTy(const DataLayout &DL,
                                                 EVT) const {
  return MVT::i32;
}

EVT Next32TargetLowering::getSetCCResultType(const DataLayout &DL,
                                             LLVMContext &Context,
                                             EVT /*VT*/) const {
  return EVT::getIntegerVT(Context, 32);
}

unsigned Next32TargetLowering::getNumRegistersForCallingConv(
    LLVMContext &Context, CallingConv::ID CC, EVT VT) const {
  unsigned int NumRegs =
      TargetLowering::getNumRegistersForCallingConv(Context, CC, VT);

  // Make sure that we always return number of registers which data size can be
  // encoded in feeders and writers.
  if (VT.isInteger())
    NumRegs = PowerOf2Ceil(NumRegs);
  return NumRegs;
}

Register
Next32TargetLowering::getRegisterByName(const char *RegName, LLT Ty,
                                        const MachineFunction &MF) const {
  unsigned Reg =
      StringSwitch<unsigned>(RegName).Case("TID", Next32::TID).Default(0);

  if (Reg)
    return Reg;
  report_fatal_error(
      Twine("Invalid register name \"" + StringRef(RegName) + "\"."));
}
