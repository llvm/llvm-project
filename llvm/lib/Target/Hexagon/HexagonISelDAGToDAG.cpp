//===-- HexagonISelDAGToDAG.cpp - A dag to dag inst selector for Hexagon --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines an instruction selector for the Hexagon target.
//
//===----------------------------------------------------------------------===//

#include "Hexagon.h"
#include "HexagonISelLowering.h"
#include "HexagonMachineFunctionInfo.h"
#include "HexagonTargetMachine.h"
#include "llvm/CodeGen/FunctionLoweringInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
using namespace llvm;

#define DEBUG_TYPE "hexagon-isel"

static
cl::opt<bool>
EnableAddressRebalancing("isel-rebalance-addr", cl::Hidden, cl::init(true),
  cl::desc("Rebalance address calculation trees to improve "
          "instruction selection"));

// Rebalance only if this allows e.g. combining a GA with an offset or
// factoring out a shift.
static
cl::opt<bool>
RebalanceOnlyForOptimizations("rebalance-only-opt", cl::Hidden, cl::init(false),
  cl::desc("Rebalance address tree only if this allows optimizations"));

static
cl::opt<bool>
RebalanceOnlyImbalancedTrees("rebalance-only-imbal", cl::Hidden,
  cl::init(false), cl::desc("Rebalance address tree only if it is imbalanced"));

//===----------------------------------------------------------------------===//
// Instruction Selector Implementation
//===----------------------------------------------------------------------===//

//===--------------------------------------------------------------------===//
/// HexagonDAGToDAGISel - Hexagon specific code to select Hexagon machine
/// instructions for SelectionDAG operations.
///
namespace {
class HexagonDAGToDAGISel : public SelectionDAGISel {
  const HexagonSubtarget *HST;
  const HexagonInstrInfo *HII;
  const HexagonRegisterInfo *HRI;
public:
  explicit HexagonDAGToDAGISel(HexagonTargetMachine &tm,
                               CodeGenOpt::Level OptLevel)
      : SelectionDAGISel(tm, OptLevel), HST(nullptr), HII(nullptr),
        HRI(nullptr) {}

  bool runOnMachineFunction(MachineFunction &MF) override {
    // Reset the subtarget each time through.
    HST = &MF.getSubtarget<HexagonSubtarget>();
    HII = HST->getInstrInfo();
    HRI = HST->getRegisterInfo();
    SelectionDAGISel::runOnMachineFunction(MF);
    return true;
  }

  void PreprocessISelDAG() override;
  void EmitFunctionEntryCode() override;

  void Select(SDNode *N) override;

  // Complex Pattern Selectors.
  inline bool SelectAddrGA(SDValue &N, SDValue &R);
  inline bool SelectAddrGP(SDValue &N, SDValue &R);
  bool SelectGlobalAddress(SDValue &N, SDValue &R, bool UseGP);
  bool SelectAddrFI(SDValue &N, SDValue &R);

  const char *getPassName() const override {
    return "Hexagon DAG->DAG Pattern Instruction Selection";
  }

  // Generate a machine instruction node corresponding to the circ/brev
  // load intrinsic.
  MachineSDNode *LoadInstrForLoadIntrinsic(SDNode *IntN);
  // Given the circ/brev load intrinsic and the already generated machine
  // instruction, generate the appropriate store (that is a part of the
  // intrinsic's functionality).
  SDNode *StoreInstrForLoadIntrinsic(MachineSDNode *LoadN, SDNode *IntN);

  void SelectFrameIndex(SDNode *N);
  /// SelectInlineAsmMemoryOperand - Implement addressing mode selection for
  /// inline asm expressions.
  bool SelectInlineAsmMemoryOperand(const SDValue &Op,
                                    unsigned ConstraintID,
                                    std::vector<SDValue> &OutOps) override;
  bool tryLoadOfLoadIntrinsic(LoadSDNode *N);
  void SelectLoad(SDNode *N);
  void SelectIndexedLoad(LoadSDNode *LD, const SDLoc &dl);
  void SelectIndexedStore(StoreSDNode *ST, const SDLoc &dl);
  void SelectStore(SDNode *N);
  void SelectSHL(SDNode *N);
  void SelectMul(SDNode *N);
  void SelectZeroExtend(SDNode *N);
  void SelectIntrinsicWChain(SDNode *N);
  void SelectIntrinsicWOChain(SDNode *N);
  void SelectConstant(SDNode *N);
  void SelectConstantFP(SDNode *N);
  void SelectAdd(SDNode *N);
  void SelectBitcast(SDNode *N);
  void SelectBitOp(SDNode *N);

  // XformMskToBitPosU5Imm - Returns the bit position which
  // the single bit 32 bit mask represents.
  // Used in Clr and Set bit immediate memops.
  SDValue XformMskToBitPosU5Imm(uint32_t Imm, const SDLoc &DL) {
    unsigned BitPos = Log2_32(Imm);
    assert(BitPos < 32 && "Constant out of range for 32 BitPos Memops");
    return CurDAG->getTargetConstant(BitPos, DL, MVT::i32);
  }

  // XformMskToBitPosU4Imm - Returns the bit position which the single-bit
  // 16 bit mask represents. Used in Clr and Set bit immediate memops.
  SDValue XformMskToBitPosU4Imm(uint16_t Imm, const SDLoc &DL) {
    return XformMskToBitPosU5Imm(Imm, DL);
  }

  // XformMskToBitPosU3Imm - Returns the bit position which the single-bit
  // 8 bit mask represents. Used in Clr and Set bit immediate memops.
  SDValue XformMskToBitPosU3Imm(uint8_t Imm, const SDLoc &DL) {
    return XformMskToBitPosU5Imm(Imm, DL);
  }

  // Return true if there is exactly one bit set in V, i.e., if V is one of the
  // following integers: 2^0, 2^1, ..., 2^31.
  bool ImmIsSingleBit(uint32_t v) const {
    return isPowerOf2_32(v);
  }

  // XformM5ToU5Imm - Return a target constant with the specified value, of
  // type i32 where the negative literal is transformed into a positive literal
  // for use in -= memops.
  inline SDValue XformM5ToU5Imm(signed Imm, const SDLoc &DL) {
    assert((Imm >= -31 && Imm <= -1) && "Constant out of range for Memops");
    return CurDAG->getTargetConstant(-Imm, DL, MVT::i32);
  }

  // XformU7ToU7M1Imm - Return a target constant decremented by 1, in range
  // [1..128], used in cmpb.gtu instructions.
  inline SDValue XformU7ToU7M1Imm(signed Imm, const SDLoc &DL) {
    assert((Imm >= 1 && Imm <= 128) && "Constant out of range for cmpb op");
    return CurDAG->getTargetConstant(Imm - 1, DL, MVT::i8);
  }

  // XformS8ToS8M1Imm - Return a target constant decremented by 1.
  inline SDValue XformSToSM1Imm(signed Imm, const SDLoc &DL) {
    return CurDAG->getTargetConstant(Imm - 1, DL, MVT::i32);
  }

  // XformU8ToU8M1Imm - Return a target constant decremented by 1.
  inline SDValue XformUToUM1Imm(unsigned Imm, const SDLoc &DL) {
    assert((Imm >= 1) && "Cannot decrement unsigned int less than 1");
    return CurDAG->getTargetConstant(Imm - 1, DL, MVT::i32);
  }

  // XformSToSM2Imm - Return a target constant decremented by 2.
  inline SDValue XformSToSM2Imm(unsigned Imm, const SDLoc &DL) {
    return CurDAG->getTargetConstant(Imm - 2, DL, MVT::i32);
  }

  // XformSToSM3Imm - Return a target constant decremented by 3.
  inline SDValue XformSToSM3Imm(unsigned Imm, const SDLoc &DL) {
    return CurDAG->getTargetConstant(Imm - 3, DL, MVT::i32);
  }

  // Include the pieces autogenerated from the target description.
  #include "HexagonGenDAGISel.inc"

private:
  bool isValueExtension(const SDValue &Val, unsigned FromBits, SDValue &Src);
  bool orIsAdd(const SDNode *N) const;
  bool isAlignedMemNode(const MemSDNode *N) const;

  SmallDenseMap<SDNode *,int> RootWeights;
  SmallDenseMap<SDNode *,int> RootHeights;
  SmallDenseMap<const Value *,int> GAUsesInFunction;
  int getWeight(SDNode *N);
  int getHeight(SDNode *N);
  SDValue getMultiplierForSHL(SDNode *N);
  SDValue factorOutPowerOf2(SDValue V, unsigned Power);
  unsigned getUsesInFunction(const Value *V);
  SDValue balanceSubTree(SDNode *N, bool Factorize = false);
  void rebalanceAddressTrees();
}; // end HexagonDAGToDAGISel
}  // end anonymous namespace


/// createHexagonISelDag - This pass converts a legalized DAG into a
/// Hexagon-specific DAG, ready for instruction scheduling.
///
namespace llvm {
FunctionPass *createHexagonISelDag(HexagonTargetMachine &TM,
                                   CodeGenOpt::Level OptLevel) {
  return new HexagonDAGToDAGISel(TM, OptLevel);
}
}

// Intrinsics that return a a predicate.
static bool doesIntrinsicReturnPredicate(unsigned ID) {
  switch (ID) {
    default:
      return false;
    case Intrinsic::hexagon_C2_cmpeq:
    case Intrinsic::hexagon_C2_cmpgt:
    case Intrinsic::hexagon_C2_cmpgtu:
    case Intrinsic::hexagon_C2_cmpgtup:
    case Intrinsic::hexagon_C2_cmpgtp:
    case Intrinsic::hexagon_C2_cmpeqp:
    case Intrinsic::hexagon_C2_bitsset:
    case Intrinsic::hexagon_C2_bitsclr:
    case Intrinsic::hexagon_C2_cmpeqi:
    case Intrinsic::hexagon_C2_cmpgti:
    case Intrinsic::hexagon_C2_cmpgtui:
    case Intrinsic::hexagon_C2_cmpgei:
    case Intrinsic::hexagon_C2_cmpgeui:
    case Intrinsic::hexagon_C2_cmplt:
    case Intrinsic::hexagon_C2_cmpltu:
    case Intrinsic::hexagon_C2_bitsclri:
    case Intrinsic::hexagon_C2_and:
    case Intrinsic::hexagon_C2_or:
    case Intrinsic::hexagon_C2_xor:
    case Intrinsic::hexagon_C2_andn:
    case Intrinsic::hexagon_C2_not:
    case Intrinsic::hexagon_C2_orn:
    case Intrinsic::hexagon_C2_pxfer_map:
    case Intrinsic::hexagon_C2_any8:
    case Intrinsic::hexagon_C2_all8:
    case Intrinsic::hexagon_A2_vcmpbeq:
    case Intrinsic::hexagon_A2_vcmpbgtu:
    case Intrinsic::hexagon_A2_vcmpheq:
    case Intrinsic::hexagon_A2_vcmphgt:
    case Intrinsic::hexagon_A2_vcmphgtu:
    case Intrinsic::hexagon_A2_vcmpweq:
    case Intrinsic::hexagon_A2_vcmpwgt:
    case Intrinsic::hexagon_A2_vcmpwgtu:
    case Intrinsic::hexagon_C2_tfrrp:
    case Intrinsic::hexagon_S2_tstbit_i:
    case Intrinsic::hexagon_S2_tstbit_r:
      return true;
  }
}

void HexagonDAGToDAGISel::SelectIndexedLoad(LoadSDNode *LD, const SDLoc &dl) {
  SDValue Chain = LD->getChain();
  SDValue Base = LD->getBasePtr();
  SDValue Offset = LD->getOffset();
  int32_t Inc = cast<ConstantSDNode>(Offset.getNode())->getSExtValue();
  EVT LoadedVT = LD->getMemoryVT();
  unsigned Opcode = 0;

  // Check for zero extended loads. Treat any-extend loads as zero extended
  // loads.
  ISD::LoadExtType ExtType = LD->getExtensionType();
  bool IsZeroExt = (ExtType == ISD::ZEXTLOAD || ExtType == ISD::EXTLOAD);
  bool IsValidInc = HII->isValidAutoIncImm(LoadedVT, Inc);

  assert(LoadedVT.isSimple());
  switch (LoadedVT.getSimpleVT().SimpleTy) {
  case MVT::i8:
    if (IsZeroExt)
      Opcode = IsValidInc ? Hexagon::L2_loadrub_pi : Hexagon::L2_loadrub_io;
    else
      Opcode = IsValidInc ? Hexagon::L2_loadrb_pi : Hexagon::L2_loadrb_io;
    break;
  case MVT::i16:
    if (IsZeroExt)
      Opcode = IsValidInc ? Hexagon::L2_loadruh_pi : Hexagon::L2_loadruh_io;
    else
      Opcode = IsValidInc ? Hexagon::L2_loadrh_pi : Hexagon::L2_loadrh_io;
    break;
  case MVT::i32:
    Opcode = IsValidInc ? Hexagon::L2_loadri_pi : Hexagon::L2_loadri_io;
    break;
  case MVT::i64:
    Opcode = IsValidInc ? Hexagon::L2_loadrd_pi : Hexagon::L2_loadrd_io;
    break;
  // 64B
  case MVT::v64i8:
  case MVT::v32i16:
  case MVT::v16i32:
  case MVT::v8i64:
    if (isAlignedMemNode(LD))
      Opcode = IsValidInc ? Hexagon::V6_vL32b_pi : Hexagon::V6_vL32b_ai;
    else
      Opcode = IsValidInc ? Hexagon::V6_vL32Ub_pi : Hexagon::V6_vL32Ub_ai;
    break;
  // 128B
  case MVT::v128i8:
  case MVT::v64i16:
  case MVT::v32i32:
  case MVT::v16i64:
    if (isAlignedMemNode(LD))
      Opcode = IsValidInc ? Hexagon::V6_vL32b_pi_128B
                          : Hexagon::V6_vL32b_ai_128B;
    else
      Opcode = IsValidInc ? Hexagon::V6_vL32Ub_pi_128B
                          : Hexagon::V6_vL32Ub_ai_128B;
    break;
  default:
    llvm_unreachable("Unexpected memory type in indexed load");
  }

  SDValue IncV = CurDAG->getTargetConstant(Inc, dl, MVT::i32);
  MachineSDNode::mmo_iterator MemOp = MF->allocateMemRefsArray(1);
  MemOp[0] = LD->getMemOperand();

  auto getExt64 = [this,ExtType] (MachineSDNode *N, const SDLoc &dl)
        -> MachineSDNode* {
    if (ExtType == ISD::ZEXTLOAD || ExtType == ISD::EXTLOAD) {
      SDValue Zero = CurDAG->getTargetConstant(0, dl, MVT::i32);
      return CurDAG->getMachineNode(Hexagon::A4_combineir, dl, MVT::i64,
                                    Zero, SDValue(N, 0));
    }
    if (ExtType == ISD::SEXTLOAD)
      return CurDAG->getMachineNode(Hexagon::A2_sxtw, dl, MVT::i64,
                                    SDValue(N, 0));
    return N;
  };

  //                  Loaded value   Next address   Chain
  SDValue From[3] = { SDValue(LD,0), SDValue(LD,1), SDValue(LD,2) };
  SDValue To[3];

  EVT ValueVT = LD->getValueType(0);
  if (ValueVT == MVT::i64 && ExtType != ISD::NON_EXTLOAD) {
    // A load extending to i64 will actually produce i32, which will then
    // need to be extended to i64.
    assert(LoadedVT.getSizeInBits() <= 32);
    ValueVT = MVT::i32;
  }

  if (IsValidInc) {
    MachineSDNode *L = CurDAG->getMachineNode(Opcode, dl, ValueVT,
                                              MVT::i32, MVT::Other, Base,
                                              IncV, Chain);
    L->setMemRefs(MemOp, MemOp+1);
    To[1] = SDValue(L, 1); // Next address.
    To[2] = SDValue(L, 2); // Chain.
    // Handle special case for extension to i64.
    if (LD->getValueType(0) == MVT::i64)
      L = getExt64(L, dl);
    To[0] = SDValue(L, 0); // Loaded (extended) value.
  } else {
    SDValue Zero = CurDAG->getTargetConstant(0, dl, MVT::i32);
    MachineSDNode *L = CurDAG->getMachineNode(Opcode, dl, ValueVT, MVT::Other,
                                              Base, Zero, Chain);
    L->setMemRefs(MemOp, MemOp+1);
    To[2] = SDValue(L, 1); // Chain.
    MachineSDNode *A = CurDAG->getMachineNode(Hexagon::A2_addi, dl, MVT::i32,
                                              Base, IncV);
    To[1] = SDValue(A, 0); // Next address.
    // Handle special case for extension to i64.
    if (LD->getValueType(0) == MVT::i64)
      L = getExt64(L, dl);
    To[0] = SDValue(L, 0); // Loaded (extended) value.
  }
  ReplaceUses(From, To, 3);
  CurDAG->RemoveDeadNode(LD);
}


MachineSDNode *HexagonDAGToDAGISel::LoadInstrForLoadIntrinsic(SDNode *IntN) {
  if (IntN->getOpcode() != ISD::INTRINSIC_W_CHAIN)
    return nullptr;

  SDLoc dl(IntN);
  unsigned IntNo = cast<ConstantSDNode>(IntN->getOperand(1))->getZExtValue();

  static std::map<unsigned,unsigned> LoadPciMap = {
    { Intrinsic::hexagon_circ_ldb,  Hexagon::L2_loadrb_pci  },
    { Intrinsic::hexagon_circ_ldub, Hexagon::L2_loadrub_pci },
    { Intrinsic::hexagon_circ_ldh,  Hexagon::L2_loadrh_pci  },
    { Intrinsic::hexagon_circ_lduh, Hexagon::L2_loadruh_pci },
    { Intrinsic::hexagon_circ_ldw,  Hexagon::L2_loadri_pci  },
    { Intrinsic::hexagon_circ_ldd,  Hexagon::L2_loadrd_pci  },
  };
  auto FLC = LoadPciMap.find(IntNo);
  if (FLC != LoadPciMap.end()) {
    SDNode *Mod = CurDAG->getMachineNode(Hexagon::A2_tfrrcr, dl, MVT::i32,
          IntN->getOperand(4));
    EVT ValTy = (IntNo == Intrinsic::hexagon_circ_ldd) ? MVT::i64 : MVT::i32;
    EVT RTys[] = { ValTy, MVT::i32, MVT::Other };
    // Operands: { Base, Increment, Modifier, Chain }
    auto Inc = cast<ConstantSDNode>(IntN->getOperand(5));
    SDValue I = CurDAG->getTargetConstant(Inc->getSExtValue(), dl, MVT::i32);
    MachineSDNode *Res = CurDAG->getMachineNode(FLC->second, dl, RTys,
          { IntN->getOperand(2), I, SDValue(Mod,0), IntN->getOperand(0) });
    return Res;
  }

  static std::map<unsigned,unsigned> LoadPbrMap = {
    { Intrinsic::hexagon_brev_ldb,  Hexagon::L2_loadrb_pbr  },
    { Intrinsic::hexagon_brev_ldub, Hexagon::L2_loadrub_pbr },
    { Intrinsic::hexagon_brev_ldh,  Hexagon::L2_loadrh_pbr  },
    { Intrinsic::hexagon_brev_lduh, Hexagon::L2_loadruh_pbr },
    { Intrinsic::hexagon_brev_ldw,  Hexagon::L2_loadri_pbr  },
    { Intrinsic::hexagon_brev_ldd,  Hexagon::L2_loadrd_pbr  },
  };
  auto FLB = LoadPbrMap.find(IntNo);
  if (FLB != LoadPbrMap.end()) {
    SDNode *Mod = CurDAG->getMachineNode(Hexagon::A2_tfrrcr, dl, MVT::i32,
            IntN->getOperand(4));
    EVT ValTy = (IntNo == Intrinsic::hexagon_brev_ldd) ? MVT::i64 : MVT::i32;
    EVT RTys[] = { ValTy, MVT::i32, MVT::Other };
    // Operands: { Base, Modifier, Chain }
    MachineSDNode *Res = CurDAG->getMachineNode(FLB->second, dl, RTys,
          { IntN->getOperand(2), SDValue(Mod,0), IntN->getOperand(0) });
    return Res;
  }

  return nullptr;
}

SDNode *HexagonDAGToDAGISel::StoreInstrForLoadIntrinsic(MachineSDNode *LoadN,
      SDNode *IntN) {
  // The "LoadN" is just a machine load instruction. The intrinsic also
  // involves storing it. Generate an appropriate store to the location
  // given in the intrinsic's operand(3).
  uint64_t F = HII->get(LoadN->getMachineOpcode()).TSFlags;
  unsigned SizeBits = (F >> HexagonII::MemAccessSizePos) &
                      HexagonII::MemAccesSizeMask;
  unsigned Size = 1U << (SizeBits-1);

  SDLoc dl(IntN);
  MachinePointerInfo PI;
  SDValue TS;
  SDValue Loc = IntN->getOperand(3);

  if (Size >= 4)
    TS = CurDAG->getStore(SDValue(LoadN, 2), dl, SDValue(LoadN, 0), Loc, PI,
                          Size);
  else
    TS = CurDAG->getTruncStore(SDValue(LoadN, 2), dl, SDValue(LoadN, 0), Loc,
                               PI, MVT::getIntegerVT(Size * 8), Size);

  SDNode *StoreN;
  {
    HandleSDNode Handle(TS);
    SelectStore(TS.getNode());
    StoreN = Handle.getValue().getNode();
  }

  // Load's results are { Loaded value, Updated pointer, Chain }
  ReplaceUses(SDValue(IntN, 0), SDValue(LoadN, 1));
  ReplaceUses(SDValue(IntN, 1), SDValue(StoreN, 0));
  return StoreN;
}

bool HexagonDAGToDAGISel::tryLoadOfLoadIntrinsic(LoadSDNode *N) {
  // The intrinsics for load circ/brev perform two operations:
  // 1. Load a value V from the specified location, using the addressing
  //    mode corresponding to the intrinsic.
  // 2. Store V into a specified location. This location is typically a
  //    local, temporary object.
  // In many cases, the program using these intrinsics will immediately
  // load V again from the local object. In those cases, when certain
  // conditions are met, the last load can be removed.
  // This function identifies and optimizes this pattern. If the pattern
  // cannot be optimized, it returns nullptr, which will cause the load
  // to be selected separately from the intrinsic (which will be handled
  // in SelectIntrinsicWChain).

  SDValue Ch = N->getOperand(0);
  SDValue Loc = N->getOperand(1);

  // Assume that the load and the intrinsic are connected directly with a
  // chain:
  //   t1: i32,ch = int.load ..., ..., ..., Loc, ...    // <-- C
  //   t2: i32,ch = load t1:1, Loc, ...
  SDNode *C = Ch.getNode();

  if (C->getOpcode() != ISD::INTRINSIC_W_CHAIN)
    return false;

  // The second load can only be eliminated if its extension type matches
  // that of the load instruction corresponding to the intrinsic. The user
  // can provide an address of an unsigned variable to store the result of
  // a sign-extending intrinsic into (or the other way around).
  ISD::LoadExtType IntExt;
  switch (cast<ConstantSDNode>(C->getOperand(1))->getZExtValue()) {
    case Intrinsic::hexagon_brev_ldub:
    case Intrinsic::hexagon_brev_lduh:
    case Intrinsic::hexagon_circ_ldub:
    case Intrinsic::hexagon_circ_lduh:
      IntExt = ISD::ZEXTLOAD;
      break;
    case Intrinsic::hexagon_brev_ldw:
    case Intrinsic::hexagon_brev_ldd:
    case Intrinsic::hexagon_circ_ldw:
    case Intrinsic::hexagon_circ_ldd:
      IntExt = ISD::NON_EXTLOAD;
      break;
    default:
      IntExt = ISD::SEXTLOAD;
      break;
  }
  if (N->getExtensionType() != IntExt)
    return false;

  // Make sure the target location for the loaded value in the load intrinsic
  // is the location from which LD (or N) is loading.
  if (C->getNumOperands() < 4 || Loc.getNode() != C->getOperand(3).getNode())
    return false;

  if (MachineSDNode *L = LoadInstrForLoadIntrinsic(C)) {
    SDNode *S = StoreInstrForLoadIntrinsic(L, C);
    SDValue F[] = { SDValue(N,0), SDValue(N,1), SDValue(C,0), SDValue(C,1) };
    SDValue T[] = { SDValue(L,0), SDValue(S,0), SDValue(L,1), SDValue(S,0) };
    ReplaceUses(F, T, array_lengthof(T));
    // This transformation will leave the intrinsic dead. If it remains in
    // the DAG, the selection code will see it again, but without the load,
    // and it will generate a store that is normally required for it.
    CurDAG->RemoveDeadNode(C);
    return true;
  }

  return false;
}

void HexagonDAGToDAGISel::SelectLoad(SDNode *N) {
  SDLoc dl(N);
  LoadSDNode *LD = cast<LoadSDNode>(N);
  ISD::MemIndexedMode AM = LD->getAddressingMode();

  // Handle indexed loads.
  if (AM != ISD::UNINDEXED) {
    SelectIndexedLoad(LD, dl);
    return;
  }

  // Handle patterns using circ/brev load intrinsics.
  if (tryLoadOfLoadIntrinsic(LD))
    return;

  SelectCode(LD);
}

void HexagonDAGToDAGISel::SelectIndexedStore(StoreSDNode *ST, const SDLoc &dl) {
  SDValue Chain = ST->getChain();
  SDValue Base = ST->getBasePtr();
  SDValue Offset = ST->getOffset();
  SDValue Value = ST->getValue();
  // Get the constant value.
  int32_t Inc = cast<ConstantSDNode>(Offset.getNode())->getSExtValue();
  EVT StoredVT = ST->getMemoryVT();
  EVT ValueVT = Value.getValueType();

  bool IsValidInc = HII->isValidAutoIncImm(StoredVT, Inc);
  unsigned Opcode = 0;

  assert(StoredVT.isSimple());
  switch (StoredVT.getSimpleVT().SimpleTy) {
  case MVT::i8:
    Opcode = IsValidInc ? Hexagon::S2_storerb_pi : Hexagon::S2_storerb_io;
    break;
  case MVT::i16:
    Opcode = IsValidInc ? Hexagon::S2_storerh_pi : Hexagon::S2_storerh_io;
    break;
  case MVT::i32:
    Opcode = IsValidInc ? Hexagon::S2_storeri_pi : Hexagon::S2_storeri_io;
    break;
  case MVT::i64:
    Opcode = IsValidInc ? Hexagon::S2_storerd_pi : Hexagon::S2_storerd_io;
    break;
  // 64B
  case MVT::v64i8:
  case MVT::v32i16:
  case MVT::v16i32:
  case MVT::v8i64:
    if (isAlignedMemNode(ST))
      Opcode = IsValidInc ? Hexagon::V6_vS32b_pi : Hexagon::V6_vS32b_ai;
    else
      Opcode = IsValidInc ? Hexagon::V6_vS32Ub_pi : Hexagon::V6_vS32Ub_ai;
    break;
  // 128B
  case MVT::v128i8:
  case MVT::v64i16:
  case MVT::v32i32:
  case MVT::v16i64:
    if (isAlignedMemNode(ST))
      Opcode = IsValidInc ? Hexagon::V6_vS32b_pi_128B
                          : Hexagon::V6_vS32b_ai_128B;
    else
      Opcode = IsValidInc ? Hexagon::V6_vS32Ub_pi_128B
                          : Hexagon::V6_vS32Ub_ai_128B;
    break;
  default:
    llvm_unreachable("Unexpected memory type in indexed store");
  }

  if (ST->isTruncatingStore() && ValueVT.getSizeInBits() == 64) {
    assert(StoredVT.getSizeInBits() < 64 && "Not a truncating store");
    Value = CurDAG->getTargetExtractSubreg(Hexagon::subreg_loreg,
                                           dl, MVT::i32, Value);
  }

  SDValue IncV = CurDAG->getTargetConstant(Inc, dl, MVT::i32);
  MachineSDNode::mmo_iterator MemOp = MF->allocateMemRefsArray(1);
  MemOp[0] = ST->getMemOperand();

  //                  Next address   Chain
  SDValue From[2] = { SDValue(ST,0), SDValue(ST,1) };
  SDValue To[2];

  if (IsValidInc) {
    // Build post increment store.
    SDValue Ops[] = { Base, IncV, Value, Chain };
    MachineSDNode *S = CurDAG->getMachineNode(Opcode, dl, MVT::i32, MVT::Other,
                                              Ops);
    S->setMemRefs(MemOp, MemOp + 1);
    To[0] = SDValue(S, 0);
    To[1] = SDValue(S, 1);
  } else {
    SDValue Zero = CurDAG->getTargetConstant(0, dl, MVT::i32);
    SDValue Ops[] = { Base, Zero, Value, Chain };
    MachineSDNode *S = CurDAG->getMachineNode(Opcode, dl, MVT::Other, Ops);
    S->setMemRefs(MemOp, MemOp + 1);
    To[1] = SDValue(S, 0);
    MachineSDNode *A = CurDAG->getMachineNode(Hexagon::A2_addi, dl, MVT::i32,
                                              Base, IncV);
    To[0] = SDValue(A, 0);
  }

  ReplaceUses(From, To, 2);
  CurDAG->RemoveDeadNode(ST);
}

void HexagonDAGToDAGISel::SelectStore(SDNode *N) {
  SDLoc dl(N);
  StoreSDNode *ST = cast<StoreSDNode>(N);
  ISD::MemIndexedMode AM = ST->getAddressingMode();

  // Handle indexed stores.
  if (AM != ISD::UNINDEXED) {
    SelectIndexedStore(ST, dl);
    return;
  }

  SelectCode(ST);
}

void HexagonDAGToDAGISel::SelectMul(SDNode *N) {
  SDLoc dl(N);

  // %conv.i = sext i32 %tmp1 to i64
  // %conv2.i = sext i32 %add to i64
  // %mul.i = mul nsw i64 %conv2.i, %conv.i
  //
  //   --- match with the following ---
  //
  // %mul.i = mpy (%tmp1, %add)
  //

  if (N->getValueType(0) == MVT::i64) {
    // Shifting a i64 signed multiply.
    SDValue MulOp0 = N->getOperand(0);
    SDValue MulOp1 = N->getOperand(1);

    SDValue OP0;
    SDValue OP1;

    // Handle sign_extend and sextload.
    if (MulOp0.getOpcode() == ISD::SIGN_EXTEND) {
      SDValue Sext0 = MulOp0.getOperand(0);
      if (Sext0.getNode()->getValueType(0) != MVT::i32) {
        SelectCode(N);
        return;
      }
      OP0 = Sext0;
    } else if (MulOp0.getOpcode() == ISD::LOAD) {
      LoadSDNode *LD = cast<LoadSDNode>(MulOp0.getNode());
      if (LD->getMemoryVT() != MVT::i32 ||
          LD->getExtensionType() != ISD::SEXTLOAD ||
          LD->getAddressingMode() != ISD::UNINDEXED) {
        SelectCode(N);
        return;
      }
      SDValue Chain = LD->getChain();
      SDValue TargetConst0 = CurDAG->getTargetConstant(0, dl, MVT::i32);
      OP0 = SDValue(CurDAG->getMachineNode(Hexagon::L2_loadri_io, dl, MVT::i32,
                                            MVT::Other,
                                            LD->getBasePtr(), TargetConst0,
                                            Chain), 0);
    } else {
      SelectCode(N);
      return;
    }

    // Same goes for the second operand.
    if (MulOp1.getOpcode() == ISD::SIGN_EXTEND) {
      SDValue Sext1 = MulOp1.getOperand(0);
      if (Sext1.getNode()->getValueType(0) != MVT::i32) {
        SelectCode(N);
        return;
      }
      OP1 = Sext1;
    } else if (MulOp1.getOpcode() == ISD::LOAD) {
      LoadSDNode *LD = cast<LoadSDNode>(MulOp1.getNode());
      if (LD->getMemoryVT() != MVT::i32 ||
          LD->getExtensionType() != ISD::SEXTLOAD ||
          LD->getAddressingMode() != ISD::UNINDEXED) {
        SelectCode(N);
        return;
      }
      SDValue Chain = LD->getChain();
      SDValue TargetConst0 = CurDAG->getTargetConstant(0, dl, MVT::i32);
      OP1 = SDValue(CurDAG->getMachineNode(Hexagon::L2_loadri_io, dl, MVT::i32,
                                            MVT::Other,
                                            LD->getBasePtr(), TargetConst0,
                                            Chain), 0);
    } else {
      SelectCode(N);
      return;
    }

    // Generate a mpy instruction.
    SDNode *Result = CurDAG->getMachineNode(Hexagon::M2_dpmpyss_s0, dl,
                                            MVT::i64, OP0, OP1);
    ReplaceNode(N, Result);
    return;
  }

  SelectCode(N);
}

void HexagonDAGToDAGISel::SelectSHL(SDNode *N) {
  SDLoc dl(N);
  SDValue Shl_0 = N->getOperand(0);
  SDValue Shl_1 = N->getOperand(1);

  auto Default = [this,N] () -> void { SelectCode(N); };

  if (N->getValueType(0) != MVT::i32 || Shl_1.getOpcode() != ISD::Constant)
    return Default();

  // RHS is const.
  int32_t ShlConst = cast<ConstantSDNode>(Shl_1)->getSExtValue();

  if (Shl_0.getOpcode() == ISD::MUL) {
    SDValue Mul_0 = Shl_0.getOperand(0); // Val
    SDValue Mul_1 = Shl_0.getOperand(1); // Const
    // RHS of mul is const.
    if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(Mul_1)) {
      int32_t ValConst = C->getSExtValue() << ShlConst;
      if (isInt<9>(ValConst)) {
        SDValue Val = CurDAG->getTargetConstant(ValConst, dl, MVT::i32);
        SDNode *Result = CurDAG->getMachineNode(Hexagon::M2_mpysmi, dl,
                                                MVT::i32, Mul_0, Val);
        ReplaceNode(N, Result);
        return;
      }
    }
    return Default();
  }

  if (Shl_0.getOpcode() == ISD::SUB) {
    SDValue Sub_0 = Shl_0.getOperand(0); // Const 0
    SDValue Sub_1 = Shl_0.getOperand(1); // Val
    if (ConstantSDNode *C1 = dyn_cast<ConstantSDNode>(Sub_0)) {
      if (C1->getSExtValue() != 0 || Sub_1.getOpcode() != ISD::SHL)
        return Default();
      SDValue Shl2_0 = Sub_1.getOperand(0); // Val
      SDValue Shl2_1 = Sub_1.getOperand(1); // Const
      if (ConstantSDNode *C2 = dyn_cast<ConstantSDNode>(Shl2_1)) {
        int32_t ValConst = 1 << (ShlConst + C2->getSExtValue());
        if (isInt<9>(-ValConst)) {
          SDValue Val = CurDAG->getTargetConstant(-ValConst, dl, MVT::i32);
          SDNode *Result = CurDAG->getMachineNode(Hexagon::M2_mpysmi, dl,
                                                  MVT::i32, Shl2_0, Val);
          ReplaceNode(N, Result);
          return;
        }
      }
    }
  }

  return Default();
}


//
// If there is an zero_extend followed an intrinsic in DAG (this means - the
// result of the intrinsic is predicate); convert the zero_extend to
// transfer instruction.
//
// Zero extend -> transfer is lowered here. Otherwise, zero_extend will be
// converted into a MUX as predicate registers defined as 1 bit in the
// compiler. Architecture defines them as 8-bit registers.
// We want to preserve all the lower 8-bits and, not just 1 LSB bit.
//
void HexagonDAGToDAGISel::SelectZeroExtend(SDNode *N) {
  SDLoc dl(N);

  SDValue Op0 = N->getOperand(0);
  EVT OpVT = Op0.getValueType();
  unsigned OpBW = OpVT.getSizeInBits();

  // Special handling for zero-extending a vector of booleans.
  if (OpVT.isVector() && OpVT.getVectorElementType() == MVT::i1 && OpBW <= 64) {
    SDNode *Mask = CurDAG->getMachineNode(Hexagon::C2_mask, dl, MVT::i64, Op0);
    unsigned NE = OpVT.getVectorNumElements();
    EVT ExVT = N->getValueType(0);
    unsigned ES = ExVT.getVectorElementType().getSizeInBits();
    uint64_t MV = 0, Bit = 1;
    for (unsigned i = 0; i < NE; ++i) {
      MV |= Bit;
      Bit <<= ES;
    }
    SDValue Ones = CurDAG->getTargetConstant(MV, dl, MVT::i64);
    SDNode *OnesReg = CurDAG->getMachineNode(Hexagon::CONST64, dl,
                                             MVT::i64, Ones);
    if (ExVT.getSizeInBits() == 32) {
      SDNode *And = CurDAG->getMachineNode(Hexagon::A2_andp, dl, MVT::i64,
                                           SDValue(Mask,0), SDValue(OnesReg,0));
      SDValue SubR = CurDAG->getTargetConstant(Hexagon::subreg_loreg, dl,
                                               MVT::i32);
      ReplaceNode(N, CurDAG->getMachineNode(Hexagon::EXTRACT_SUBREG, dl, ExVT,
                                            SDValue(And, 0), SubR));
      return;
    }
    ReplaceNode(N,
                CurDAG->getMachineNode(Hexagon::A2_andp, dl, ExVT,
                                       SDValue(Mask, 0), SDValue(OnesReg, 0)));
    return;
  }

  SDNode *Int = N->getOperand(0).getNode();
  if ((Int->getOpcode() == ISD::INTRINSIC_WO_CHAIN)) {
    unsigned ID = cast<ConstantSDNode>(Int->getOperand(0))->getZExtValue();
    if (doesIntrinsicReturnPredicate(ID)) {
      // Now we need to differentiate target data types.
      if (N->getValueType(0) == MVT::i64) {
        // Convert the zero_extend to Rs = Pd followed by A2_combinew(0,Rs).
        SDValue TargetConst0 = CurDAG->getTargetConstant(0, dl, MVT::i32);
        SDNode *Result_1 = CurDAG->getMachineNode(Hexagon::C2_tfrpr, dl,
                                                  MVT::i32, SDValue(Int, 0));
        SDNode *Result_2 = CurDAG->getMachineNode(Hexagon::A2_tfrsi, dl,
                                                  MVT::i32, TargetConst0);
        SDNode *Result_3 = CurDAG->getMachineNode(Hexagon::A2_combinew, dl,
                                                  MVT::i64, MVT::Other,
                                                  SDValue(Result_2, 0),
                                                  SDValue(Result_1, 0));
        ReplaceNode(N, Result_3);
        return;
      }
      if (N->getValueType(0) == MVT::i32) {
        // Convert the zero_extend to Rs = Pd
        SDNode* RsPd = CurDAG->getMachineNode(Hexagon::C2_tfrpr, dl,
                                              MVT::i32, SDValue(Int, 0));
        ReplaceNode(N, RsPd);
        return;
      }
      llvm_unreachable("Unexpected value type");
    }
  }
  SelectCode(N);
}


//
// Handling intrinsics for circular load and bitreverse load.
//
void HexagonDAGToDAGISel::SelectIntrinsicWChain(SDNode *N) {
  if (MachineSDNode *L = LoadInstrForLoadIntrinsic(N)) {
    StoreInstrForLoadIntrinsic(L, N);
    CurDAG->RemoveDeadNode(N);
    return;
  }
  SelectCode(N);
}

void HexagonDAGToDAGISel::SelectIntrinsicWOChain(SDNode *N) {
  unsigned IID = cast<ConstantSDNode>(N->getOperand(0))->getZExtValue();
  unsigned Bits;
  switch (IID) {
  case Intrinsic::hexagon_S2_vsplatrb:
    Bits = 8;
    break;
  case Intrinsic::hexagon_S2_vsplatrh:
    Bits = 16;
    break;
  default:
    SelectCode(N);
    return;
  }

  SDValue V = N->getOperand(1);
  SDValue U;
  if (isValueExtension(V, Bits, U)) {
    SDValue R = CurDAG->getNode(N->getOpcode(), SDLoc(N), N->getValueType(0),
                                N->getOperand(0), U);
    ReplaceNode(N, R.getNode());
    SelectCode(R.getNode());
    return;
  }
  SelectCode(N);
}

//
// Map floating point constant values.
//
void HexagonDAGToDAGISel::SelectConstantFP(SDNode *N) {
  SDLoc dl(N);
  ConstantFPSDNode *CN = dyn_cast<ConstantFPSDNode>(N);
  APInt A = CN->getValueAPF().bitcastToAPInt();
  if (N->getValueType(0) == MVT::f32) {
    SDValue V = CurDAG->getTargetConstant(A.getZExtValue(), dl, MVT::i32);
    ReplaceNode(N, CurDAG->getMachineNode(Hexagon::A2_tfrsi, dl, MVT::f32, V));
    return;
  }
  if (N->getValueType(0) == MVT::f64) {
    SDValue V = CurDAG->getTargetConstant(A.getZExtValue(), dl, MVT::i64);
    ReplaceNode(N, CurDAG->getMachineNode(Hexagon::CONST64, dl, MVT::f64, V));
    return;
  }

  SelectCode(N);
}

//
// Map boolean values.
//
void HexagonDAGToDAGISel::SelectConstant(SDNode *N) {
  if (N->getValueType(0) == MVT::i1) {
    assert(!(cast<ConstantSDNode>(N)->getZExtValue() >> 1));
    unsigned Opc = (cast<ConstantSDNode>(N)->getSExtValue() != 0)
                      ? Hexagon::PS_true
                      : Hexagon::PS_false;
    ReplaceNode(N, CurDAG->getMachineNode(Opc, SDLoc(N), MVT::i1));
    return;
  }

  SelectCode(N);
}


//
// Map add followed by a asr -> asr +=.
//
void HexagonDAGToDAGISel::SelectAdd(SDNode *N) {
  SDLoc dl(N);
  if (N->getValueType(0) != MVT::i32) {
    SelectCode(N);
    return;
  }
  // Identify nodes of the form: add(asr(...)).
  SDNode* Src1 = N->getOperand(0).getNode();
  if (Src1->getOpcode() != ISD::SRA || !Src1->hasOneUse() ||
      Src1->getValueType(0) != MVT::i32) {
    SelectCode(N);
    return;
  }

  // Build Rd = Rd' + asr(Rs, Rt). The machine constraints will ensure that
  // Rd and Rd' are assigned to the same register
  SDNode* Result = CurDAG->getMachineNode(Hexagon::S2_asr_r_r_acc, dl, MVT::i32,
                                          N->getOperand(1),
                                          Src1->getOperand(0),
                                          Src1->getOperand(1));
  ReplaceNode(N, Result);
}

//
// Map the following, where possible.
// AND/FABS -> clrbit
// OR -> setbit
// XOR/FNEG ->toggle_bit.
//
void HexagonDAGToDAGISel::SelectBitOp(SDNode *N) {
  SDLoc dl(N);
  EVT ValueVT = N->getValueType(0);

  // We handle only 32 and 64-bit bit ops.
  if (!(ValueVT == MVT::i32 || ValueVT == MVT::i64 ||
        ValueVT == MVT::f32 || ValueVT == MVT::f64)) {
    SelectCode(N);
    return;
  }

  // We handly only fabs and fneg for V5.
  unsigned Opc = N->getOpcode();
  if ((Opc == ISD::FABS || Opc == ISD::FNEG) && !HST->hasV5TOps()) {
    SelectCode(N);
    return;
  }

  int64_t Val = 0;
  if (Opc != ISD::FABS && Opc != ISD::FNEG) {
    if (N->getOperand(1).getOpcode() == ISD::Constant)
      Val = cast<ConstantSDNode>((N)->getOperand(1))->getSExtValue();
    else {
      SelectCode(N);
      return;
    }
  }

  if (Opc == ISD::AND) {
    // Check if this is a bit-clearing AND, if not select code the usual way.
    if ((ValueVT == MVT::i32 && isPowerOf2_32(~Val)) ||
        (ValueVT == MVT::i64 && isPowerOf2_64(~Val)))
      Val = ~Val;
    else {
      SelectCode(N);
      return;
    }
  }

  // If OR or AND is being fed by shl, srl and, sra don't do this change,
  // because Hexagon provide |= &= on shl, srl, and sra.
  // Traverse the DAG to see if there is shl, srl and sra.
  if (Opc == ISD::OR || Opc == ISD::AND) {
    switch (N->getOperand(0)->getOpcode()) {
      default:
        break;
      case ISD::SRA:
      case ISD::SRL:
      case ISD::SHL:
        SelectCode(N);
        return;
    }
  }

  // Make sure it's power of 2.
  unsigned BitPos = 0;
  if (Opc != ISD::FABS && Opc != ISD::FNEG) {
    if ((ValueVT == MVT::i32 && !isPowerOf2_32(Val)) ||
        (ValueVT == MVT::i64 && !isPowerOf2_64(Val))) {
      SelectCode(N);
      return;
    }

    // Get the bit position.
    BitPos = countTrailingZeros(uint64_t(Val));
  } else {
    // For fabs and fneg, it's always the 31st bit.
    BitPos = 31;
  }

  unsigned BitOpc = 0;
  // Set the right opcode for bitwise operations.
  switch (Opc) {
    default:
      llvm_unreachable("Only bit-wise/abs/neg operations are allowed.");
    case ISD::AND:
    case ISD::FABS:
      BitOpc = Hexagon::S2_clrbit_i;
      break;
    case ISD::OR:
      BitOpc = Hexagon::S2_setbit_i;
      break;
    case ISD::XOR:
    case ISD::FNEG:
      BitOpc = Hexagon::S2_togglebit_i;
      break;
  }

  SDNode *Result;
  // Get the right SDVal for the opcode.
  SDValue SDVal = CurDAG->getTargetConstant(BitPos, dl, MVT::i32);

  if (ValueVT == MVT::i32 || ValueVT == MVT::f32) {
    Result = CurDAG->getMachineNode(BitOpc, dl, ValueVT,
                                    N->getOperand(0), SDVal);
  } else {
    // 64-bit gymnastic to use REG_SEQUENCE. But it's worth it.
    EVT SubValueVT;
    if (ValueVT == MVT::i64)
      SubValueVT = MVT::i32;
    else
      SubValueVT = MVT::f32;

    SDNode *Reg = N->getOperand(0).getNode();
    SDValue RegClass = CurDAG->getTargetConstant(Hexagon::DoubleRegsRegClassID,
                                                 dl, MVT::i64);

    SDValue SubregHiIdx = CurDAG->getTargetConstant(Hexagon::subreg_hireg, dl,
                                                    MVT::i32);
    SDValue SubregLoIdx = CurDAG->getTargetConstant(Hexagon::subreg_loreg, dl,
                                                    MVT::i32);

    SDValue SubregHI = CurDAG->getTargetExtractSubreg(Hexagon::subreg_hireg, dl,
                                                    MVT::i32, SDValue(Reg, 0));

    SDValue SubregLO = CurDAG->getTargetExtractSubreg(Hexagon::subreg_loreg, dl,
                                                    MVT::i32, SDValue(Reg, 0));

    // Clear/set/toggle hi or lo registers depending on the bit position.
    if (SubValueVT != MVT::f32 && BitPos < 32) {
      SDNode *Result0 = CurDAG->getMachineNode(BitOpc, dl, SubValueVT,
                                               SubregLO, SDVal);
      const SDValue Ops[] = { RegClass, SubregHI, SubregHiIdx,
                              SDValue(Result0, 0), SubregLoIdx };
      Result = CurDAG->getMachineNode(TargetOpcode::REG_SEQUENCE,
                                      dl, ValueVT, Ops);
    } else {
      if (Opc != ISD::FABS && Opc != ISD::FNEG)
        SDVal = CurDAG->getTargetConstant(BitPos-32, dl, MVT::i32);
      SDNode *Result0 = CurDAG->getMachineNode(BitOpc, dl, SubValueVT,
                                               SubregHI, SDVal);
      const SDValue Ops[] = { RegClass, SDValue(Result0, 0), SubregHiIdx,
                              SubregLO, SubregLoIdx };
      Result = CurDAG->getMachineNode(TargetOpcode::REG_SEQUENCE,
                                      dl, ValueVT, Ops);
    }
  }

  ReplaceNode(N, Result);
}


void HexagonDAGToDAGISel::SelectFrameIndex(SDNode *N) {
  MachineFrameInfo &MFI = MF->getFrameInfo();
  const HexagonFrameLowering *HFI = HST->getFrameLowering();
  int FX = cast<FrameIndexSDNode>(N)->getIndex();
  unsigned StkA = HFI->getStackAlignment();
  unsigned MaxA = MFI.getMaxAlignment();
  SDValue FI = CurDAG->getTargetFrameIndex(FX, MVT::i32);
  SDLoc DL(N);
  SDValue Zero = CurDAG->getTargetConstant(0, DL, MVT::i32);
  SDNode *R = nullptr;

  // Use PS_fi when:
  // - the object is fixed, or
  // - there are no objects with higher-than-default alignment, or
  // - there are no dynamically allocated objects.
  // Otherwise, use PS_fia.
  if (FX < 0 || MaxA <= StkA || !MFI.hasVarSizedObjects()) {
    R = CurDAG->getMachineNode(Hexagon::PS_fi, DL, MVT::i32, FI, Zero);
  } else {
    auto &HMFI = *MF->getInfo<HexagonMachineFunctionInfo>();
    unsigned AR = HMFI.getStackAlignBaseVReg();
    SDValue CH = CurDAG->getEntryNode();
    SDValue Ops[] = { CurDAG->getCopyFromReg(CH, DL, AR, MVT::i32), FI, Zero };
    R = CurDAG->getMachineNode(Hexagon::PS_fia, DL, MVT::i32, Ops);
  }

  ReplaceNode(N, R);
}


void HexagonDAGToDAGISel::SelectBitcast(SDNode *N) {
  EVT SVT = N->getOperand(0).getValueType();
  EVT DVT = N->getValueType(0);
  if (!SVT.isVector() || !DVT.isVector() ||
      SVT.getVectorElementType() == MVT::i1 ||
      DVT.getVectorElementType() == MVT::i1 ||
      SVT.getSizeInBits() != DVT.getSizeInBits()) {
    SelectCode(N);
    return;
  }

  CurDAG->ReplaceAllUsesOfValueWith(SDValue(N,0), N->getOperand(0));
  CurDAG->RemoveDeadNode(N);
}


void HexagonDAGToDAGISel::Select(SDNode *N) {
  if (N->isMachineOpcode()) {
    N->setNodeId(-1);
    return;   // Already selected.
  }

  switch (N->getOpcode()) {
  case ISD::Constant:
    SelectConstant(N);
    return;

  case ISD::ConstantFP:
    SelectConstantFP(N);
    return;

  case ISD::FrameIndex:
    SelectFrameIndex(N);
    return;

  case ISD::ADD:
    SelectAdd(N);
    return;

  case ISD::BITCAST:
    SelectBitcast(N);
    return;

  case ISD::SHL:
    SelectSHL(N);
    return;

  case ISD::LOAD:
    SelectLoad(N);
    return;

  case ISD::STORE:
    SelectStore(N);
    return;

  case ISD::MUL:
    SelectMul(N);
    return;

  case ISD::AND:
  case ISD::OR:
  case ISD::XOR:
  case ISD::FABS:
  case ISD::FNEG:
    SelectBitOp(N);
    return;

  case ISD::ZERO_EXTEND:
    SelectZeroExtend(N);
    return;

  case ISD::INTRINSIC_W_CHAIN:
    SelectIntrinsicWChain(N);
    return;

  case ISD::INTRINSIC_WO_CHAIN:
    SelectIntrinsicWOChain(N);
    return;
  }

  SelectCode(N);
}

bool HexagonDAGToDAGISel::
SelectInlineAsmMemoryOperand(const SDValue &Op, unsigned ConstraintID,
                             std::vector<SDValue> &OutOps) {
  SDValue Inp = Op, Res;

  switch (ConstraintID) {
  default:
    return true;
  case InlineAsm::Constraint_i:
  case InlineAsm::Constraint_o: // Offsetable.
  case InlineAsm::Constraint_v: // Not offsetable.
  case InlineAsm::Constraint_m: // Memory.
    if (SelectAddrFI(Inp, Res))
      OutOps.push_back(Res);
    else
      OutOps.push_back(Inp);
    break;
  }

  OutOps.push_back(CurDAG->getTargetConstant(0, SDLoc(Op), MVT::i32));
  return false;
}


void HexagonDAGToDAGISel::PreprocessISelDAG() {
  SelectionDAG &DAG = *CurDAG;
  std::vector<SDNode*> Nodes;
  for (SDNode &Node : DAG.allnodes())
    Nodes.push_back(&Node);

  // Simplify: (or (select c x 0) z)  ->  (select c (or x z) z)
  //           (or (select c 0 y) z)  ->  (select c z (or y z))
  // This may not be the right thing for all targets, so do it here.
  for (auto I : Nodes) {
    if (I->getOpcode() != ISD::OR)
      continue;

    auto IsZero = [] (const SDValue &V) -> bool {
      if (ConstantSDNode *SC = dyn_cast<ConstantSDNode>(V.getNode()))
        return SC->isNullValue();
      return false;
    };
    auto IsSelect0 = [IsZero] (const SDValue &Op) -> bool {
      if (Op.getOpcode() != ISD::SELECT)
        return false;
      return IsZero(Op.getOperand(1)) || IsZero(Op.getOperand(2));
    };

    SDValue N0 = I->getOperand(0), N1 = I->getOperand(1);
    EVT VT = I->getValueType(0);
    bool SelN0 = IsSelect0(N0);
    SDValue SOp = SelN0 ? N0 : N1;
    SDValue VOp = SelN0 ? N1 : N0;

    if (SOp.getOpcode() == ISD::SELECT && SOp.getNode()->hasOneUse()) {
      SDValue SC = SOp.getOperand(0);
      SDValue SX = SOp.getOperand(1);
      SDValue SY = SOp.getOperand(2);
      SDLoc DLS = SOp;
      if (IsZero(SY)) {
        SDValue NewOr = DAG.getNode(ISD::OR, DLS, VT, SX, VOp);
        SDValue NewSel = DAG.getNode(ISD::SELECT, DLS, VT, SC, NewOr, VOp);
        DAG.ReplaceAllUsesWith(I, NewSel.getNode());
      } else if (IsZero(SX)) {
        SDValue NewOr = DAG.getNode(ISD::OR, DLS, VT, SY, VOp);
        SDValue NewSel = DAG.getNode(ISD::SELECT, DLS, VT, SC, VOp, NewOr);
        DAG.ReplaceAllUsesWith(I, NewSel.getNode());
      }
    }
  }

  // Transform: (store ch addr (add x (add (shl y c) e)))
  //        to: (store ch addr (add x (shl (add y d) c))),
  // where e = (shl d c) for some integer d.
  // The purpose of this is to enable generation of loads/stores with
  // shifted addressing mode, i.e. mem(x+y<<#c). For that, the shift
  // value c must be 0, 1 or 2.
  for (auto I : Nodes) {
    if (I->getOpcode() != ISD::STORE)
      continue;

    // I matched: (store ch addr Off)
    SDValue Off = I->getOperand(2);
    // Off needs to match: (add x (add (shl y c) (shl d c))))
    if (Off.getOpcode() != ISD::ADD)
      continue;
    // Off matched: (add x T0)
    SDValue T0 = Off.getOperand(1);
    // T0 needs to match: (add T1 T2):
    if (T0.getOpcode() != ISD::ADD)
      continue;
    // T0 matched: (add T1 T2)
    SDValue T1 = T0.getOperand(0);
    SDValue T2 = T0.getOperand(1);
    // T1 needs to match: (shl y c)
    if (T1.getOpcode() != ISD::SHL)
      continue;
    SDValue C = T1.getOperand(1);
    ConstantSDNode *CN = dyn_cast<ConstantSDNode>(C.getNode());
    if (CN == nullptr)
      continue;
    unsigned CV = CN->getZExtValue();
    if (CV > 2)
      continue;
    // T2 needs to match e, where e = (shl d c) for some d.
    ConstantSDNode *EN = dyn_cast<ConstantSDNode>(T2.getNode());
    if (EN == nullptr)
      continue;
    unsigned EV = EN->getZExtValue();
    if (EV % (1 << CV) != 0)
      continue;
    unsigned DV = EV / (1 << CV);

    // Replace T0 with: (shl (add y d) c)
    SDLoc DL = SDLoc(I);
    EVT VT = T0.getValueType();
    SDValue D = DAG.getConstant(DV, DL, VT);
    // NewAdd = (add y d)
    SDValue NewAdd = DAG.getNode(ISD::ADD, DL, VT, T1.getOperand(0), D);
    // NewShl = (shl NewAdd c)
    SDValue NewShl = DAG.getNode(ISD::SHL, DL, VT, NewAdd, C);
    ReplaceNode(T0.getNode(), NewShl.getNode());
  }

  if (EnableAddressRebalancing) {
    rebalanceAddressTrees();

    DEBUG(
      dbgs() << "************* SelectionDAG after preprocessing: ***********\n";
      CurDAG->dump();
      dbgs() << "************* End SelectionDAG after preprocessing ********\n";
    );
  }
}

void HexagonDAGToDAGISel::EmitFunctionEntryCode() {
  auto &HST = static_cast<const HexagonSubtarget&>(MF->getSubtarget());
  auto &HFI = *HST.getFrameLowering();
  if (!HFI.needsAligna(*MF))
    return;

  MachineFrameInfo &MFI = MF->getFrameInfo();
  MachineBasicBlock *EntryBB = &MF->front();
  unsigned AR = FuncInfo->CreateReg(MVT::i32);
  unsigned MaxA = MFI.getMaxAlignment();
  BuildMI(EntryBB, DebugLoc(), HII->get(Hexagon::PS_aligna), AR)
      .addImm(MaxA);
  MF->getInfo<HexagonMachineFunctionInfo>()->setStackAlignBaseVReg(AR);
}

// Match a frame index that can be used in an addressing mode.
bool HexagonDAGToDAGISel::SelectAddrFI(SDValue& N, SDValue &R) {
  if (N.getOpcode() != ISD::FrameIndex)
    return false;
  auto &HFI = *HST->getFrameLowering();
  MachineFrameInfo &MFI = MF->getFrameInfo();
  int FX = cast<FrameIndexSDNode>(N)->getIndex();
  if (!MFI.isFixedObjectIndex(FX) && HFI.needsAligna(*MF))
    return false;
  R = CurDAG->getTargetFrameIndex(FX, MVT::i32);
  return true;
}

inline bool HexagonDAGToDAGISel::SelectAddrGA(SDValue &N, SDValue &R) {
  return SelectGlobalAddress(N, R, false);
}

inline bool HexagonDAGToDAGISel::SelectAddrGP(SDValue &N, SDValue &R) {
  return SelectGlobalAddress(N, R, true);
}

bool HexagonDAGToDAGISel::SelectGlobalAddress(SDValue &N, SDValue &R,
                                              bool UseGP) {
  switch (N.getOpcode()) {
  case ISD::ADD: {
    SDValue N0 = N.getOperand(0);
    SDValue N1 = N.getOperand(1);
    unsigned GAOpc = N0.getOpcode();
    if (UseGP && GAOpc != HexagonISD::CONST32_GP)
      return false;
    if (!UseGP && GAOpc != HexagonISD::CONST32)
      return false;
    if (ConstantSDNode *Const = dyn_cast<ConstantSDNode>(N1)) {
      SDValue Addr = N0.getOperand(0);
      if (GlobalAddressSDNode *GA = dyn_cast<GlobalAddressSDNode>(Addr)) {
        if (GA->getOpcode() == ISD::TargetGlobalAddress) {
          uint64_t NewOff = GA->getOffset() + (uint64_t)Const->getSExtValue();
          R = CurDAG->getTargetGlobalAddress(GA->getGlobal(), SDLoc(Const),
                                             N.getValueType(), NewOff);
          return true;
        }
      }
    }
    break;
  }
  case HexagonISD::CONST32:
    // The operand(0) of CONST32 is TargetGlobalAddress, which is what we
    // want in the instruction.
    if (!UseGP)
      R = N.getOperand(0);
    return !UseGP;
  case HexagonISD::CONST32_GP:
    if (UseGP)
      R = N.getOperand(0);
    return UseGP;
  default:
    return false;
  }

  return false;
}

bool HexagonDAGToDAGISel::isValueExtension(const SDValue &Val,
      unsigned FromBits, SDValue &Src) {
  unsigned Opc = Val.getOpcode();
  switch (Opc) {
  case ISD::SIGN_EXTEND:
  case ISD::ZERO_EXTEND:
  case ISD::ANY_EXTEND: {
    SDValue const &Op0 = Val.getOperand(0);
    EVT T = Op0.getValueType();
    if (T.isInteger() && T.getSizeInBits() == FromBits) {
      Src = Op0;
      return true;
    }
    break;
  }
  case ISD::SIGN_EXTEND_INREG:
  case ISD::AssertSext:
  case ISD::AssertZext:
    if (Val.getOperand(0).getValueType().isInteger()) {
      VTSDNode *T = cast<VTSDNode>(Val.getOperand(1));
      if (T->getVT().getSizeInBits() == FromBits) {
        Src = Val.getOperand(0);
        return true;
      }
    }
    break;
  case ISD::AND: {
    // Check if this is an AND with "FromBits" of lower bits set to 1.
    uint64_t FromMask = (1 << FromBits) - 1;
    if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(Val.getOperand(0))) {
      if (C->getZExtValue() == FromMask) {
        Src = Val.getOperand(1);
        return true;
      }
    }
    if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(Val.getOperand(1))) {
      if (C->getZExtValue() == FromMask) {
        Src = Val.getOperand(0);
        return true;
      }
    }
    break;
  }
  case ISD::OR:
  case ISD::XOR: {
    // OR/XOR with the lower "FromBits" bits set to 0.
    uint64_t FromMask = (1 << FromBits) - 1;
    if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(Val.getOperand(0))) {
      if ((C->getZExtValue() & FromMask) == 0) {
        Src = Val.getOperand(1);
        return true;
      }
    }
    if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(Val.getOperand(1))) {
      if ((C->getZExtValue() & FromMask) == 0) {
        Src = Val.getOperand(0);
        return true;
      }
    }
  }
  default:
    break;
  }
  return false;
}


bool HexagonDAGToDAGISel::orIsAdd(const SDNode *N) const {
  assert(N->getOpcode() == ISD::OR);
  auto *C = dyn_cast<ConstantSDNode>(N->getOperand(1));
  assert(C);

  // Detect when "or" is used to add an offset to a stack object.
  if (auto *FN = dyn_cast<FrameIndexSDNode>(N->getOperand(0))) {
    MachineFrameInfo &MFI = MF->getFrameInfo();
    unsigned A = MFI.getObjectAlignment(FN->getIndex());
    assert(isPowerOf2_32(A));
    int32_t Off = C->getSExtValue();
    // If the alleged offset fits in the zero bits guaranteed by
    // the alignment, then this or is really an add.
    return (Off >= 0) && (((A-1) & Off) == unsigned(Off));
  }
  return false;
}

bool HexagonDAGToDAGISel::isAlignedMemNode(const MemSDNode *N) const {
  return N->getAlignment() >= N->getMemoryVT().getStoreSize();
}

////////////////////////////////////////////////////////////////////////////////
// Rebalancing of address calculation trees

static bool isOpcodeHandled(const SDNode *N) {
  switch (N->getOpcode()) {
    case ISD::ADD:
    case ISD::MUL:
      return true;
    case ISD::SHL:
      // We only handle constant shifts because these can be easily flattened
      // into multiplications by 2^Op1.
      return isa<ConstantSDNode>(N->getOperand(1).getNode());
    default:
      return false;
  }
}

/// \brief Return the weight of an SDNode
int HexagonDAGToDAGISel::getWeight(SDNode *N) {
  if (!isOpcodeHandled(N))
    return 1;
  assert(RootWeights.count(N) && "Cannot get weight of unseen root!");
  assert(RootWeights[N] != -1 && "Cannot get weight of unvisited root!");
  assert(RootWeights[N] != -2 && "Cannot get weight of RAWU'd root!");
  return RootWeights[N];
}

int HexagonDAGToDAGISel::getHeight(SDNode *N) {
  if (!isOpcodeHandled(N))
    return 0;
  assert(RootWeights.count(N) && RootWeights[N] >= 0 &&
      "Cannot query height of unvisited/RAUW'd node!");
  return RootHeights[N];
}

namespace {
struct WeightedLeaf {
  SDValue Value;
  int Weight;
  int InsertionOrder;

  WeightedLeaf() : Value(SDValue()) { }

  WeightedLeaf(SDValue Value, int Weight, int InsertionOrder) :
    Value(Value), Weight(Weight), InsertionOrder(InsertionOrder) {
    assert(Weight >= 0 && "Weight must be >= 0");
  }

  static bool Compare(const WeightedLeaf &A, const WeightedLeaf &B) {
    assert(A.Value.getNode() && B.Value.getNode());
    return A.Weight == B.Weight ?
            (A.InsertionOrder > B.InsertionOrder) :
            (A.Weight > B.Weight);
  }
};

/// A specialized priority queue for WeigthedLeaves. It automatically folds
/// constants and allows removal of non-top elements while maintaining the
/// priority order.
class LeafPrioQueue {
  SmallVector<WeightedLeaf, 8> Q;
  bool HaveConst;
  WeightedLeaf ConstElt;
  unsigned Opcode;

public:
  bool empty() {
    return (!HaveConst && Q.empty());
  }

  size_t size() {
    return Q.size() + HaveConst;
  }

  bool hasConst() {
    return HaveConst;
  }

  const WeightedLeaf &top() {
    if (HaveConst)
      return ConstElt;
    return Q.front();
  }

  WeightedLeaf pop() {
    if (HaveConst) {
      HaveConst = false;
      return ConstElt;
    }
    std::pop_heap(Q.begin(), Q.end(), WeightedLeaf::Compare);
    return Q.pop_back_val();
  }

  void push(WeightedLeaf L, bool SeparateConst=true) {
    if (!HaveConst && SeparateConst && isa<ConstantSDNode>(L.Value)) {
      if (Opcode == ISD::MUL &&
          cast<ConstantSDNode>(L.Value)->getSExtValue() == 1)
        return;
      if (Opcode == ISD::ADD &&
          cast<ConstantSDNode>(L.Value)->getSExtValue() == 0)
        return;

      HaveConst = true;
      ConstElt = L;
    } else {
      Q.push_back(L);
      std::push_heap(Q.begin(), Q.end(), WeightedLeaf::Compare);
    }
  }

  /// Push L to the bottom of the queue regardless of its weight. If L is
  /// constant, it will not be folded with other constants in the queue.
  void pushToBottom(WeightedLeaf L) {
    L.Weight = 1000;
    push(L, false);
  }

  /// Search for a SHL(x, [<=MaxAmount]) subtree in the queue, return the one of
  /// lowest weight and remove it from the queue.
  WeightedLeaf findSHL(uint64_t MaxAmount);

  WeightedLeaf findMULbyConst();

  LeafPrioQueue(unsigned Opcode) :
    HaveConst(false), Opcode(Opcode) { }
};
} // end anonymous namespace

WeightedLeaf LeafPrioQueue::findSHL(uint64_t MaxAmount) {
  int ResultPos;
  WeightedLeaf Result;

  for (int Pos = 0, End = Q.size(); Pos != End; ++Pos) {
    const WeightedLeaf &L = Q[Pos];
    const SDValue &Val = L.Value;
    if (Val.getOpcode() != ISD::SHL ||
        !isa<ConstantSDNode>(Val.getOperand(1)) ||
        Val.getConstantOperandVal(1) > MaxAmount)
      continue;
    if (!Result.Value.getNode() || Result.Weight > L.Weight ||
        (Result.Weight == L.Weight && Result.InsertionOrder > L.InsertionOrder))
    {
      Result = L;
      ResultPos = Pos;
    }
  }

  if (Result.Value.getNode()) {
    Q.erase(&Q[ResultPos]);
    std::make_heap(Q.begin(), Q.end(), WeightedLeaf::Compare);
  }

  return Result;
}

WeightedLeaf LeafPrioQueue::findMULbyConst() {
  int ResultPos;
  WeightedLeaf Result;

  for (int Pos = 0, End = Q.size(); Pos != End; ++Pos) {
    const WeightedLeaf &L = Q[Pos];
    const SDValue &Val = L.Value;
    if (Val.getOpcode() != ISD::MUL ||
        !isa<ConstantSDNode>(Val.getOperand(1)) ||
        Val.getConstantOperandVal(1) > 127)
      continue;
    if (!Result.Value.getNode() || Result.Weight > L.Weight ||
        (Result.Weight == L.Weight && Result.InsertionOrder > L.InsertionOrder))
    {
      Result = L;
      ResultPos = Pos;
    }
  }

  if (Result.Value.getNode()) {
    Q.erase(&Q[ResultPos]);
    std::make_heap(Q.begin(), Q.end(), WeightedLeaf::Compare);
  }

  return Result;
}

SDValue HexagonDAGToDAGISel::getMultiplierForSHL(SDNode *N) {
  uint64_t MulFactor = 1ull << N->getConstantOperandVal(1);
  return CurDAG->getConstant(MulFactor, SDLoc(N),
                             N->getOperand(1).getValueType());
}

/// @returns the value x for which 2^x is a factor of Val
static unsigned getPowerOf2Factor(SDValue Val) {
  if (Val.getOpcode() == ISD::MUL) {
    unsigned MaxFactor = 0;
    for (int i = 0; i < 2; ++i) {
      ConstantSDNode *C = dyn_cast<ConstantSDNode>(Val.getOperand(i));
      if (!C)
        continue;
      const APInt &CInt = C->getAPIntValue();
      if (CInt.getBoolValue())
        MaxFactor = CInt.countTrailingZeros();
    }
    return MaxFactor;
  }
  if (Val.getOpcode() == ISD::SHL) {
    if (!isa<ConstantSDNode>(Val.getOperand(1).getNode()))
      return 0;
    return (unsigned) Val.getConstantOperandVal(1);
  }

  return 0;
}

/// @returns true if V>>Amount will eliminate V's operation on its child
static bool willShiftRightEliminate(SDValue V, unsigned Amount) {
  if (V.getOpcode() == ISD::MUL) {
    SDValue Ops[] = { V.getOperand(0), V.getOperand(1) };
    for (int i = 0; i < 2; ++i)
      if (isa<ConstantSDNode>(Ops[i].getNode()) &&
          V.getConstantOperandVal(i) % (1ULL << Amount) == 0) {
        uint64_t NewConst = V.getConstantOperandVal(i) >> Amount;
        return (NewConst == 1);
      }
  } else if (V.getOpcode() == ISD::SHL) {
    return (Amount == V.getConstantOperandVal(1));
  }

  return false;
}

SDValue HexagonDAGToDAGISel::factorOutPowerOf2(SDValue V, unsigned Power) {
  SDValue Ops[] = { V.getOperand(0), V.getOperand(1) };
  if (V.getOpcode() == ISD::MUL) {
    for (int i=0; i < 2; ++i) {
      if (isa<ConstantSDNode>(Ops[i].getNode()) &&
          V.getConstantOperandVal(i) % ((uint64_t)1 << Power) == 0) {
        uint64_t NewConst = V.getConstantOperandVal(i) >> Power;
        if (NewConst == 1)
          return Ops[!i];
        Ops[i] = CurDAG->getConstant(NewConst,
                                     SDLoc(V), V.getValueType());
        break;
      }
    }
  } else if (V.getOpcode() == ISD::SHL) {
    uint64_t ShiftAmount = V.getConstantOperandVal(1);
    if (ShiftAmount == Power)
      return Ops[0];
    Ops[1] = CurDAG->getConstant(ShiftAmount - Power,
                                 SDLoc(V), V.getValueType());
  }

  return CurDAG->getNode(V.getOpcode(), SDLoc(V), V.getValueType(), Ops);
}

static bool isTargetConstant(const SDValue &V) {
  return V.getOpcode() == HexagonISD::CONST32 ||
         V.getOpcode() == HexagonISD::CONST32_GP;
}

unsigned HexagonDAGToDAGISel::getUsesInFunction(const Value *V) {
  if (GAUsesInFunction.count(V))
    return GAUsesInFunction[V];

  unsigned Result = 0;
  const Function *CurF = CurDAG->getMachineFunction().getFunction();
  for (const User *U : V->users()) {
    if (isa<Instruction>(U) &&
        cast<Instruction>(U)->getParent()->getParent() == CurF)
      ++Result;
  }

  GAUsesInFunction[V] = Result;

  return Result;
}

/// Note - After calling this, N may be dead. It may have been replaced by a
/// new node, so always use the returned value in place of N.
///
/// @returns The SDValue taking the place of N (which could be N if it is
/// unchanged)
SDValue HexagonDAGToDAGISel::balanceSubTree(SDNode *N, bool TopLevel) {
  assert(RootWeights.count(N) && "Cannot balance non-root node.");
  assert(RootWeights[N] != -2 && "This node was RAUW'd!");
  assert(!TopLevel || N->getOpcode() == ISD::ADD);

  // Return early if this node was already visited
  if (RootWeights[N] != -1)
    return SDValue(N, 0);

  assert(isOpcodeHandled(N));

  SDValue Op0 = N->getOperand(0);
  SDValue Op1 = N->getOperand(1);

  // Return early if the operands will remain unchanged or are all roots
  if ((!isOpcodeHandled(Op0.getNode()) || RootWeights.count(Op0.getNode())) &&
      (!isOpcodeHandled(Op1.getNode()) || RootWeights.count(Op1.getNode()))) {
    SDNode *Op0N = Op0.getNode();
    int Weight;
    if (isOpcodeHandled(Op0N) && RootWeights[Op0N] == -1) {
      Weight = getWeight(balanceSubTree(Op0N).getNode());
      // Weight = calculateWeight(Op0N);
    } else
      Weight = getWeight(Op0N);

    SDNode *Op1N = N->getOperand(1).getNode(); // Op1 may have been RAUWd
    if (isOpcodeHandled(Op1N) && RootWeights[Op1N] == -1) {
      Weight += getWeight(balanceSubTree(Op1N).getNode());
      // Weight += calculateWeight(Op1N);
    } else
      Weight += getWeight(Op1N);

    RootWeights[N] = Weight;
    RootHeights[N] = std::max(getHeight(N->getOperand(0).getNode()),
                              getHeight(N->getOperand(1).getNode())) + 1;

    DEBUG(dbgs() << "--> No need to balance root (Weight=" << Weight
                 << " Height=" << RootHeights[N] << "): ");
    DEBUG(N->dump());

    return SDValue(N, 0);
  }

  DEBUG(dbgs() << "** Balancing root node: ");
  DEBUG(N->dump());

  unsigned NOpcode = N->getOpcode();

  LeafPrioQueue Leaves(NOpcode);
  SmallVector<SDValue, 4> Worklist;
  Worklist.push_back(SDValue(N, 0));

  // SHL nodes will be converted to MUL nodes
  if (NOpcode == ISD::SHL)
    NOpcode = ISD::MUL;

  bool CanFactorize = false;
  WeightedLeaf Mul1, Mul2;
  unsigned MaxPowerOf2 = 0;
  WeightedLeaf GA;

  // Do not try to factor out a shift if there is already a shift at the tip of
  // the tree.
  bool HaveTopLevelShift = false;
  if (TopLevel &&
      ((isOpcodeHandled(Op0.getNode()) && Op0.getOpcode() == ISD::SHL &&
                        Op0.getConstantOperandVal(1) < 4) ||
       (isOpcodeHandled(Op1.getNode()) && Op1.getOpcode() == ISD::SHL &&
                        Op1.getConstantOperandVal(1) < 4)))
    HaveTopLevelShift = true;

  // Flatten the subtree into an ordered list of leaves; at the same time
  // determine whether the tree is already balanced.
  int InsertionOrder = 0;
  SmallDenseMap<SDValue, int> NodeHeights;
  bool Imbalanced = false;
  int CurrentWeight = 0;
  while (!Worklist.empty()) {
    SDValue Child = Worklist.pop_back_val();

    if (Child.getNode() != N && RootWeights.count(Child.getNode())) {
      // CASE 1: Child is a root note

      int Weight = RootWeights[Child.getNode()];
      if (Weight == -1) {
        Child = balanceSubTree(Child.getNode());
        // calculateWeight(Child.getNode());
        Weight = getWeight(Child.getNode());
      } else if (Weight == -2) {
        // Whoops, this node was RAUWd by one of the balanceSubTree calls we
        // made. Our worklist isn't up to date anymore.
        // Restart the whole process.
        DEBUG(dbgs() << "--> Subtree was RAUWd. Restarting...\n");
        return balanceSubTree(N, TopLevel);
      }

      NodeHeights[Child] = 1;
      CurrentWeight += Weight;

      unsigned PowerOf2;
      if (TopLevel && !CanFactorize && !HaveTopLevelShift &&
          (Child.getOpcode() == ISD::MUL || Child.getOpcode() == ISD::SHL) &&
          Child.hasOneUse() && (PowerOf2 = getPowerOf2Factor(Child))) {
        // Try to identify two factorizable MUL/SHL children greedily. Leave
        // them out of the priority queue for now so we can deal with them
        // after.
        if (!Mul1.Value.getNode()) {
          Mul1 = WeightedLeaf(Child, Weight, InsertionOrder++);
          MaxPowerOf2 = PowerOf2;
        } else {
          Mul2 = WeightedLeaf(Child, Weight, InsertionOrder++);
          MaxPowerOf2 = std::min(MaxPowerOf2, PowerOf2);

          // Our addressing modes can only shift by a maximum of 3
          if (MaxPowerOf2 > 3)
            MaxPowerOf2 = 3;

          CanFactorize = true;
        }
      } else
        Leaves.push(WeightedLeaf(Child, Weight, InsertionOrder++));
    } else if (!isOpcodeHandled(Child.getNode())) {
      // CASE 2: Child is an unhandled kind of node (e.g. constant)
      int Weight = getWeight(Child.getNode());

      NodeHeights[Child] = getHeight(Child.getNode());
      CurrentWeight += Weight;

      if (isTargetConstant(Child) && !GA.Value.getNode())
        GA = WeightedLeaf(Child, Weight, InsertionOrder++);
      else
        Leaves.push(WeightedLeaf(Child, Weight, InsertionOrder++));
    } else {
      // CASE 3: Child is a subtree of same opcode
      // Visit children first, then flatten.
      unsigned ChildOpcode = Child.getOpcode();
      assert(ChildOpcode == NOpcode ||
             (NOpcode == ISD::MUL && ChildOpcode == ISD::SHL));

      // Convert SHL to MUL
      SDValue Op1;
      if (ChildOpcode == ISD::SHL)
        Op1 = getMultiplierForSHL(Child.getNode());
      else
        Op1 = Child->getOperand(1);

      if (!NodeHeights.count(Op1) || !NodeHeights.count(Child->getOperand(0))) {
        assert(!NodeHeights.count(Child) && "Parent visited before children?");
        // Visit children first, then re-visit this node
        Worklist.push_back(Child);
        Worklist.push_back(Op1);
        Worklist.push_back(Child->getOperand(0));
      } else {
        // Back at this node after visiting the children
        if (std::abs(NodeHeights[Op1] - NodeHeights[Child->getOperand(0)]) > 1)
          Imbalanced = true;

        NodeHeights[Child] = std::max(NodeHeights[Op1],
                                      NodeHeights[Child->getOperand(0)]) + 1;
      }
    }
  }

  DEBUG(dbgs() << "--> Current height=" << NodeHeights[SDValue(N, 0)]
               << " weight=" << CurrentWeight << " imbalanced="
               << Imbalanced << "\n");

  // Transform MUL(x, C * 2^Y) + SHL(z, Y) -> SHL(ADD(MUL(x, C), z), Y)
  //  This factors out a shift in order to match memw(a<<Y+b).
  if (CanFactorize && (willShiftRightEliminate(Mul1.Value, MaxPowerOf2) ||
                       willShiftRightEliminate(Mul2.Value, MaxPowerOf2))) {
    DEBUG(dbgs() << "--> Found common factor for two MUL children!\n");
    int Weight = Mul1.Weight + Mul2.Weight;
    int Height = std::max(NodeHeights[Mul1.Value], NodeHeights[Mul2.Value]) + 1;
    SDValue Mul1Factored = factorOutPowerOf2(Mul1.Value, MaxPowerOf2);
    SDValue Mul2Factored = factorOutPowerOf2(Mul2.Value, MaxPowerOf2);
    SDValue Sum = CurDAG->getNode(ISD::ADD, SDLoc(N), Mul1.Value.getValueType(),
                                  Mul1Factored, Mul2Factored);
    SDValue Const = CurDAG->getConstant(MaxPowerOf2, SDLoc(N),
                                        Mul1.Value.getValueType());
    SDValue New = CurDAG->getNode(ISD::SHL, SDLoc(N), Mul1.Value.getValueType(),
                                  Sum, Const);
    NodeHeights[New] = Height;
    Leaves.push(WeightedLeaf(New, Weight, Mul1.InsertionOrder));
  } else if (Mul1.Value.getNode()) {
    // We failed to factorize two MULs, so now the Muls are left outside the
    // queue... add them back.
    Leaves.push(Mul1);
    if (Mul2.Value.getNode())
      Leaves.push(Mul2);
    CanFactorize = false;
  }

  // Combine GA + Constant -> GA+Offset, but only if GA is not used elsewhere
  // and the root node itself is not used more than twice. This reduces the
  // amount of additional constant extenders introduced by this optimization.
  bool CombinedGA = false;
  if (NOpcode == ISD::ADD && GA.Value.getNode() && Leaves.hasConst() &&
      GA.Value.hasOneUse() && N->use_size() < 3) {
    GlobalAddressSDNode *GANode =
      cast<GlobalAddressSDNode>(GA.Value.getOperand(0));
    ConstantSDNode *Offset = cast<ConstantSDNode>(Leaves.top().Value);

    if (getUsesInFunction(GANode->getGlobal()) == 1 && Offset->hasOneUse() &&
        getTargetLowering()->isOffsetFoldingLegal(GANode)) {
      DEBUG(dbgs() << "--> Combining GA and offset (" << Offset->getSExtValue()
          << "): ");
      DEBUG(GANode->dump());

      SDValue NewTGA =
        CurDAG->getTargetGlobalAddress(GANode->getGlobal(), SDLoc(GA.Value),
            GANode->getValueType(0),
            GANode->getOffset() + (uint64_t)Offset->getSExtValue());
      GA.Value = CurDAG->getNode(GA.Value.getOpcode(), SDLoc(GA.Value),
          GA.Value.getValueType(), NewTGA);
      GA.Weight += Leaves.top().Weight;

      NodeHeights[GA.Value] = getHeight(GA.Value.getNode());
      CombinedGA = true;

      Leaves.pop(); // Remove the offset constant from the queue
    }
  }

  if ((RebalanceOnlyForOptimizations && !CanFactorize && !CombinedGA) ||
      (RebalanceOnlyImbalancedTrees && !Imbalanced)) {
    RootWeights[N] = CurrentWeight;
    RootHeights[N] = NodeHeights[SDValue(N, 0)];

    return SDValue(N, 0);
  }

  // Combine GA + SHL(x, C<=31) so we will match Rx=add(#u8,asl(Rx,#U5))
  if (NOpcode == ISD::ADD && GA.Value.getNode()) {
    WeightedLeaf SHL = Leaves.findSHL(31);
    if (SHL.Value.getNode()) {
      int Height = std::max(NodeHeights[GA.Value], NodeHeights[SHL.Value]) + 1;
      GA.Value = CurDAG->getNode(ISD::ADD, SDLoc(GA.Value),
                                 GA.Value.getValueType(),
                                 GA.Value, SHL.Value);
      GA.Weight = SHL.Weight; // Specifically ignore the GA weight here
      NodeHeights[GA.Value] = Height;
    }
  }

  if (GA.Value.getNode())
    Leaves.push(GA);

  // If this is the top level and we haven't factored out a shift, we should try
  // to move a constant to the bottom to match addressing modes like memw(rX+C)
  if (TopLevel && !CanFactorize && Leaves.hasConst()) {
    DEBUG(dbgs() << "--> Pushing constant to tip of tree.");
    Leaves.pushToBottom(Leaves.pop());
  }

  const DataLayout &DL = CurDAG->getDataLayout();
  const TargetLowering &TLI = *getTargetLowering();

  // Rebuild the tree using Huffman's algorithm
  while (Leaves.size() > 1) {
    WeightedLeaf L0 = Leaves.pop();

    // See whether we can grab a MUL to form an add(Rx,mpyi(Ry,#u6)),
    // otherwise just get the next leaf
    WeightedLeaf L1 = Leaves.findMULbyConst();
    if (!L1.Value.getNode())
      L1 = Leaves.pop();

    assert(L0.Weight <= L1.Weight && "Priority queue is broken!");

    SDValue V0 = L0.Value;
    int V0Weight = L0.Weight;
    SDValue V1 = L1.Value;
    int V1Weight = L1.Weight;

    // Make sure that none of these nodes have been RAUW'd
    if ((RootWeights.count(V0.getNode()) && RootWeights[V0.getNode()] == -2) ||
        (RootWeights.count(V1.getNode()) && RootWeights[V1.getNode()] == -2)) {
      DEBUG(dbgs() << "--> Subtree was RAUWd. Restarting...\n");
      return balanceSubTree(N, TopLevel);
    }

    ConstantSDNode *V0C = dyn_cast<ConstantSDNode>(V0);
    ConstantSDNode *V1C = dyn_cast<ConstantSDNode>(V1);
    EVT VT = N->getValueType(0);
    SDValue NewNode;

    if (V0C && !V1C) {
      std::swap(V0, V1);
      std::swap(V0C, V1C);
    }

    // Calculate height of this node
    assert(NodeHeights.count(V0) && NodeHeights.count(V1) &&
           "Children must have been visited before re-combining them!");
    int Height = std::max(NodeHeights[V0], NodeHeights[V1]) + 1;

    // Rebuild this node (and restore SHL from MUL if needed)
    if (V1C && NOpcode == ISD::MUL && V1C->getAPIntValue().isPowerOf2())
      NewNode = CurDAG->getNode(
          ISD::SHL, SDLoc(V0), VT, V0,
          CurDAG->getConstant(
              V1C->getAPIntValue().logBase2(), SDLoc(N),
              TLI.getScalarShiftAmountTy(DL, V0.getValueType())));
    else
      NewNode = CurDAG->getNode(NOpcode, SDLoc(N), VT, V0, V1);

    NodeHeights[NewNode] = Height;

    int Weight = V0Weight + V1Weight;
    Leaves.push(WeightedLeaf(NewNode, Weight, L0.InsertionOrder));

    DEBUG(dbgs() << "--> Built new node (Weight=" << Weight << ",Height="
                 << Height << "):\n");
    DEBUG(NewNode.dump());
  }

  assert(Leaves.size() == 1);
  SDValue NewRoot = Leaves.top().Value;

  assert(NodeHeights.count(NewRoot));
  int Height = NodeHeights[NewRoot];

  // Restore SHL if we earlier converted it to a MUL
  if (NewRoot.getOpcode() == ISD::MUL) {
    ConstantSDNode *V1C = dyn_cast<ConstantSDNode>(NewRoot.getOperand(1));
    if (V1C && V1C->getAPIntValue().isPowerOf2()) {
      EVT VT = NewRoot.getValueType();
      SDValue V0 = NewRoot.getOperand(0);
      NewRoot = CurDAG->getNode(
          ISD::SHL, SDLoc(NewRoot), VT, V0,
          CurDAG->getConstant(
              V1C->getAPIntValue().logBase2(), SDLoc(NewRoot),
              TLI.getScalarShiftAmountTy(DL, V0.getValueType())));
    }
  }

  if (N != NewRoot.getNode()) {
    DEBUG(dbgs() << "--> Root is now: ");
    DEBUG(NewRoot.dump());

    // Replace all uses of old root by new root
    CurDAG->ReplaceAllUsesWith(N, NewRoot.getNode());
    // Mark that we have RAUW'd N
    RootWeights[N] = -2;
  } else {
    DEBUG(dbgs() << "--> Root unchanged.\n");
  }

  RootWeights[NewRoot.getNode()] = Leaves.top().Weight;
  RootHeights[NewRoot.getNode()] = Height;

  return NewRoot;
}

void HexagonDAGToDAGISel::rebalanceAddressTrees() {
  for (auto I = CurDAG->allnodes_begin(), E = CurDAG->allnodes_end(); I != E;) {
    SDNode *N = &*I++;
    if (N->getOpcode() != ISD::LOAD && N->getOpcode() != ISD::STORE)
      continue;

    SDValue BasePtr = cast<MemSDNode>(N)->getBasePtr();
    if (BasePtr.getOpcode() != ISD::ADD)
      continue;

    // We've already processed this node
    if (RootWeights.count(BasePtr.getNode()))
      continue;

    DEBUG(dbgs() << "** Rebalancing address calculation in node: ");
    DEBUG(N->dump());

    // FindRoots
    SmallVector<SDNode *, 4> Worklist;

    Worklist.push_back(BasePtr.getOperand(0).getNode());
    Worklist.push_back(BasePtr.getOperand(1).getNode());

    while (!Worklist.empty()) {
      SDNode *N = Worklist.pop_back_val();
      unsigned Opcode = N->getOpcode();

      if (!isOpcodeHandled(N))
        continue;

      Worklist.push_back(N->getOperand(0).getNode());
      Worklist.push_back(N->getOperand(1).getNode());

      // Not a root if it has only one use and same opcode as its parent
      if (N->hasOneUse() && Opcode == N->use_begin()->getOpcode())
        continue;

      // This root node has already been processed
      if (RootWeights.count(N))
        continue;

      RootWeights[N] = -1;
    }

    // Balance node itself
    RootWeights[BasePtr.getNode()] = -1;
    SDValue NewBasePtr = balanceSubTree(BasePtr.getNode(), /*TopLevel=*/ true);

    if (N->getOpcode() == ISD::LOAD)
      N = CurDAG->UpdateNodeOperands(N, N->getOperand(0),
            NewBasePtr, N->getOperand(2));
    else
      N = CurDAG->UpdateNodeOperands(N, N->getOperand(0), N->getOperand(1),
            NewBasePtr, N->getOperand(3));

    DEBUG(dbgs() << "--> Final node: ");
    DEBUG(N->dump());
  }

  CurDAG->RemoveDeadNodes();
  GAUsesInFunction.clear();
  RootHeights.clear();
  RootWeights.clear();
}

