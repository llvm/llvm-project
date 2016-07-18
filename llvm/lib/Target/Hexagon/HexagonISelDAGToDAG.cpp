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
cl::opt<unsigned>
MaxNumOfUsesForConstExtenders("ga-max-num-uses-for-constant-extenders",
  cl::Hidden, cl::init(2),
  cl::desc("Maximum number of uses of a global address such that we still us a"
           "constant extended instruction"));

//===----------------------------------------------------------------------===//
// Instruction Selector Implementation
//===----------------------------------------------------------------------===//

//===--------------------------------------------------------------------===//
/// HexagonDAGToDAGISel - Hexagon specific code to select Hexagon machine
/// instructions for SelectionDAG operations.
///
namespace {
class HexagonDAGToDAGISel : public SelectionDAGISel {
  const HexagonTargetMachine &HTM;
  const HexagonSubtarget *HST;
  const HexagonInstrInfo *HII;
  const HexagonRegisterInfo *HRI;
public:
  explicit HexagonDAGToDAGISel(HexagonTargetMachine &tm,
                               CodeGenOpt::Level OptLevel)
      : SelectionDAGISel(tm, OptLevel), HTM(tm), HST(nullptr), HII(nullptr),
        HRI(nullptr) {}

  bool runOnMachineFunction(MachineFunction &MF) override {
    // Reset the subtarget each time through.
    HST = &MF.getSubtarget<HexagonSubtarget>();
    HII = HST->getInstrInfo();
    HRI = HST->getRegisterInfo();
    SelectionDAGISel::runOnMachineFunction(MF);
    return true;
  }

  virtual void PreprocessISelDAG() override;
  virtual void EmitFunctionEntryCode() override;

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
  void SelectBaseOffsetLoad(LoadSDNode *LD, SDLoc dl);
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
    int32_t bitPos;
    bitPos = Log2_32(Imm);
    assert(bitPos >= 0 && bitPos < 32 &&
           "Constant out of range for 32 BitPos Memops");
    return CurDAG->getTargetConstant(bitPos, DL, MVT::i32);
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

  //
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
    SDNode *Result = CurDAG->getMachineNode(Hexagon::M2_dpmpyss_s0, dl, MVT::i64,
                                            OP0, OP1);
    ReplaceNode(N, Result);
    return;
  }

  SelectCode(N);
}

void HexagonDAGToDAGISel::SelectSHL(SDNode *N) {
  SDLoc dl(N);
  if (N->getValueType(0) == MVT::i32) {
    SDValue Shl_0 = N->getOperand(0);
    SDValue Shl_1 = N->getOperand(1);
    // RHS is const.
    if (Shl_1.getOpcode() == ISD::Constant) {
      if (Shl_0.getOpcode() == ISD::MUL) {
        SDValue Mul_0 = Shl_0.getOperand(0); // Val
        SDValue Mul_1 = Shl_0.getOperand(1); // Const
        // RHS of mul is const.
        if (Mul_1.getOpcode() == ISD::Constant) {
          int32_t ShlConst =
            cast<ConstantSDNode>(Shl_1.getNode())->getSExtValue();
          int32_t MulConst =
            cast<ConstantSDNode>(Mul_1.getNode())->getSExtValue();
          int32_t ValConst = MulConst << ShlConst;
          SDValue Val = CurDAG->getTargetConstant(ValConst, dl,
                                                  MVT::i32);
          if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(Val.getNode()))
            if (isInt<9>(CN->getSExtValue())) {
              SDNode* Result =
                CurDAG->getMachineNode(Hexagon::M2_mpysmi, dl,
                                       MVT::i32, Mul_0, Val);
              ReplaceNode(N, Result);
              return;
            }

        }
      } else if (Shl_0.getOpcode() == ISD::SUB) {
        SDValue Sub_0 = Shl_0.getOperand(0); // Const 0
        SDValue Sub_1 = Shl_0.getOperand(1); // Val
        if (Sub_0.getOpcode() == ISD::Constant) {
          int32_t SubConst =
            cast<ConstantSDNode>(Sub_0.getNode())->getSExtValue();
          if (SubConst == 0) {
            if (Sub_1.getOpcode() == ISD::SHL) {
              SDValue Shl2_0 = Sub_1.getOperand(0); // Val
              SDValue Shl2_1 = Sub_1.getOperand(1); // Const
              if (Shl2_1.getOpcode() == ISD::Constant) {
                int32_t ShlConst =
                  cast<ConstantSDNode>(Shl_1.getNode())->getSExtValue();
                int32_t Shl2Const =
                  cast<ConstantSDNode>(Shl2_1.getNode())->getSExtValue();
                int32_t ValConst = 1 << (ShlConst+Shl2Const);
                SDValue Val = CurDAG->getTargetConstant(-ValConst, dl,
                                                        MVT::i32);
                if (ConstantSDNode *CN =
                    dyn_cast<ConstantSDNode>(Val.getNode()))
                  if (isInt<9>(CN->getSExtValue())) {
                    SDNode* Result =
                      CurDAG->getMachineNode(Hexagon::M2_mpysmi, dl, MVT::i32,
                                             Shl2_0, Val);
                    ReplaceNode(N, Result);
                    return;
                  }
              }
            }
          }
        }
      }
    }
  }
  SelectCode(N);
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
    SDNode *OnesReg = CurDAG->getMachineNode(Hexagon::CONST64_Int_Real, dl,
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

  SDNode *IsIntrinsic = N->getOperand(0).getNode();
  if ((IsIntrinsic->getOpcode() == ISD::INTRINSIC_WO_CHAIN)) {
    unsigned ID =
      cast<ConstantSDNode>(IsIntrinsic->getOperand(0))->getZExtValue();
    if (doesIntrinsicReturnPredicate(ID)) {
      // Now we need to differentiate target data types.
      if (N->getValueType(0) == MVT::i64) {
        // Convert the zero_extend to Rs = Pd followed by A2_combinew(0,Rs).
        SDValue TargetConst0 = CurDAG->getTargetConstant(0, dl, MVT::i32);
        SDNode *Result_1 = CurDAG->getMachineNode(Hexagon::C2_tfrpr, dl,
                                                  MVT::i32,
                                                  SDValue(IsIntrinsic, 0));
        SDNode *Result_2 = CurDAG->getMachineNode(Hexagon::A2_tfrsi, dl,
                                                  MVT::i32,
                                                  TargetConst0);
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
                                              MVT::i32,
                                              SDValue(IsIntrinsic, 0));
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
  const APFloat &APF = CN->getValueAPF();
  if (N->getValueType(0) == MVT::f32) {
    ReplaceNode(
        N, CurDAG->getMachineNode(Hexagon::TFRI_f, dl, MVT::f32,
                                  CurDAG->getTargetConstantFP(
                                      APF.convertToFloat(), dl, MVT::f32)));
    return;
  }
  else if (N->getValueType(0) == MVT::f64) {
    ReplaceNode(
        N, CurDAG->getMachineNode(Hexagon::CONST64_Float_Real, dl, MVT::f64,
                                  CurDAG->getTargetConstantFP(
                                      APF.convertToDouble(), dl, MVT::f64)));
    return;
  }

  SelectCode(N);
}

//
// Map predicate true (encoded as -1 in LLVM) to a XOR.
//
void HexagonDAGToDAGISel::SelectConstant(SDNode *N) {
  SDLoc dl(N);
  if (N->getValueType(0) == MVT::i1) {
    SDNode* Result = 0;
    int32_t Val = cast<ConstantSDNode>(N)->getSExtValue();
    if (Val == -1) {
      Result = CurDAG->getMachineNode(Hexagon::TFR_PdTrue, dl, MVT::i1);
    } else if (Val == 0) {
      Result = CurDAG->getMachineNode(Hexagon::TFR_PdFalse, dl, MVT::i1);
    }
    if (Result) {
      ReplaceNode(N, Result);
      return;
    }
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
  if (Src1->getOpcode() != ISD::SRA || !Src1->hasOneUse()
      || Src1->getValueType(0) != MVT::i32) {
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
  MachineFrameInfo *MFI = MF->getFrameInfo();
  const HexagonFrameLowering *HFI = HST->getFrameLowering();
  int FX = cast<FrameIndexSDNode>(N)->getIndex();
  unsigned StkA = HFI->getStackAlignment();
  unsigned MaxA = MFI->getMaxAlignment();
  SDValue FI = CurDAG->getTargetFrameIndex(FX, MVT::i32);
  SDLoc DL(N);
  SDValue Zero = CurDAG->getTargetConstant(0, DL, MVT::i32);
  SDNode *R = 0;

  // Use TFR_FI when:
  // - the object is fixed, or
  // - there are no objects with higher-than-default alignment, or
  // - there are no dynamically allocated objects.
  // Otherwise, use TFR_FIA.
  if (FX < 0 || MaxA <= StkA || !MFI->hasVarSizedObjects()) {
    R = CurDAG->getMachineNode(Hexagon::TFR_FI, DL, MVT::i32, FI, Zero);
  } else {
    auto &HMFI = *MF->getInfo<HexagonMachineFunctionInfo>();
    unsigned AR = HMFI.getStackAlignBaseVReg();
    SDValue CH = CurDAG->getEntryNode();
    SDValue Ops[] = { CurDAG->getCopyFromReg(CH, DL, AR, MVT::i32), FI, Zero };
    R = CurDAG->getMachineNode(Hexagon::TFR_FIA, DL, MVT::i32, Ops);
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
}

void HexagonDAGToDAGISel::EmitFunctionEntryCode() {
  auto &HST = static_cast<const HexagonSubtarget&>(MF->getSubtarget());
  auto &HFI = *HST.getFrameLowering();
  if (!HFI.needsAligna(*MF))
    return;

  MachineFrameInfo *MFI = MF->getFrameInfo();
  MachineBasicBlock *EntryBB = &MF->front();
  unsigned AR = FuncInfo->CreateReg(MVT::i32);
  unsigned MaxA = MFI->getMaxAlignment();
  BuildMI(EntryBB, DebugLoc(), HII->get(Hexagon::ALIGNA), AR)
      .addImm(MaxA);
  MF->getInfo<HexagonMachineFunctionInfo>()->setStackAlignBaseVReg(AR);
}

// Match a frame index that can be used in an addressing mode.
bool HexagonDAGToDAGISel::SelectAddrFI(SDValue& N, SDValue &R) {
  if (N.getOpcode() != ISD::FrameIndex)
    return false;
  auto &HFI = *HST->getFrameLowering();
  MachineFrameInfo *MFI = MF->getFrameInfo();
  int FX = cast<FrameIndexSDNode>(N)->getIndex();
  if (!MFI->isFixedObjectIndex(FX) && HFI.needsAligna(*MF))
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
    MachineFrameInfo *MFI = MF->getFrameInfo();
    unsigned A = MFI->getObjectAlignment(FN->getIndex());
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
