//===-- SystemZISelDAGToDAG.cpp - A dag to dag inst selector for SystemZ --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines an instruction selector for the SystemZ target.
//
//===----------------------------------------------------------------------===//

#include "SystemZTargetMachine.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {
// Used to build addressing modes.
struct SystemZAddressingMode {
  // The shape of the address.
  enum AddrForm {
    // base+displacement
    FormBD,

    // base+displacement+index for load and store operands
    FormBDXNormal,

    // base+displacement+index for load address operands
    FormBDXLA,

    // base+displacement+index+ADJDYNALLOC
    FormBDXDynAlloc
  };
  AddrForm Form;

  // The type of displacement.  The enum names here correspond directly
  // to the definitions in SystemZOperand.td.  We could split them into
  // flags -- single/pair, 128-bit, etc. -- but it hardly seems worth it.
  enum DispRange {
    Disp12Only,
    Disp12Pair,
    Disp20Only,
    Disp20Only128,
    Disp20Pair
  };
  DispRange DR;

  // The parts of the address.  The address is equivalent to:
  //
  //     Base + Disp + Index + (IncludesDynAlloc ? ADJDYNALLOC : 0)
  SDValue Base;
  int64_t Disp;
  SDValue Index;
  bool IncludesDynAlloc;

  SystemZAddressingMode(AddrForm form, DispRange dr)
    : Form(form), DR(dr), Base(), Disp(0), Index(),
      IncludesDynAlloc(false) {}

  // True if the address can have an index register.
  bool hasIndexField() { return Form != FormBD; }

  // True if the address can (and must) include ADJDYNALLOC.
  bool isDynAlloc() { return Form == FormBDXDynAlloc; }

  void dump() {
    errs() << "SystemZAddressingMode " << this << '\n';

    errs() << " Base ";
    if (Base.getNode() != 0)
      Base.getNode()->dump();
    else
      errs() << "null\n";

    if (hasIndexField()) {
      errs() << " Index ";
      if (Index.getNode() != 0)
        Index.getNode()->dump();
      else
        errs() << "null\n";
    }

    errs() << " Disp " << Disp;
    if (IncludesDynAlloc)
      errs() << " + ADJDYNALLOC";
    errs() << '\n';
  }
};

// Return a mask with Count low bits set.
static uint64_t allOnes(unsigned int Count) {
  return Count == 0 ? 0 : (uint64_t(1) << (Count - 1) << 1) - 1;
}

// Represents operands 2 to 5 of the ROTATE AND ... SELECTED BITS operation
// given by Opcode.  The operands are: Input (R2), Start (I3), End (I4) and
// Rotate (I5).  The combined operand value is effectively:
//
//   (or (rotl Input, Rotate), ~Mask)
//
// for RNSBG and:
//
//   (and (rotl Input, Rotate), Mask)
//
// otherwise.  The output value has BitSize bits, although Input may be
// narrower (in which case the upper bits are don't care).
struct RxSBGOperands {
  RxSBGOperands(unsigned Op, SDValue N)
    : Opcode(Op), BitSize(N.getValueType().getSizeInBits()),
      Mask(allOnes(BitSize)), Input(N), Start(64 - BitSize), End(63),
      Rotate(0) {}

  unsigned Opcode;
  unsigned BitSize;
  uint64_t Mask;
  SDValue Input;
  unsigned Start;
  unsigned End;
  unsigned Rotate;
};

class SystemZDAGToDAGISel : public SelectionDAGISel {
  const SystemZTargetLowering &Lowering;
  const SystemZSubtarget &Subtarget;

  // Used by SystemZOperands.td to create integer constants.
  inline SDValue getImm(const SDNode *Node, uint64_t Imm) const {
    return CurDAG->getTargetConstant(Imm, Node->getValueType(0));
  }

  const SystemZTargetMachine &getTargetMachine() const {
    return static_cast<const SystemZTargetMachine &>(TM);
  }

  const SystemZInstrInfo *getInstrInfo() const {
    return getTargetMachine().getInstrInfo();
  }

  // Try to fold more of the base or index of AM into AM, where IsBase
  // selects between the base and index.
  bool expandAddress(SystemZAddressingMode &AM, bool IsBase) const;

  // Try to describe N in AM, returning true on success.
  bool selectAddress(SDValue N, SystemZAddressingMode &AM) const;

  // Extract individual target operands from matched address AM.
  void getAddressOperands(const SystemZAddressingMode &AM, EVT VT,
                          SDValue &Base, SDValue &Disp) const;
  void getAddressOperands(const SystemZAddressingMode &AM, EVT VT,
                          SDValue &Base, SDValue &Disp, SDValue &Index) const;

  // Try to match Addr as a FormBD address with displacement type DR.
  // Return true on success, storing the base and displacement in
  // Base and Disp respectively.
  bool selectBDAddr(SystemZAddressingMode::DispRange DR, SDValue Addr,
                    SDValue &Base, SDValue &Disp) const;

  // Try to match Addr as a FormBDX address with displacement type DR.
  // Return true on success and if the result had no index.  Store the
  // base and displacement in Base and Disp respectively.
  bool selectMVIAddr(SystemZAddressingMode::DispRange DR, SDValue Addr,
                     SDValue &Base, SDValue &Disp) const;

  // Try to match Addr as a FormBDX* address of form Form with
  // displacement type DR.  Return true on success, storing the base,
  // displacement and index in Base, Disp and Index respectively.
  bool selectBDXAddr(SystemZAddressingMode::AddrForm Form,
                     SystemZAddressingMode::DispRange DR, SDValue Addr,
                     SDValue &Base, SDValue &Disp, SDValue &Index) const;

  // PC-relative address matching routines used by SystemZOperands.td.
  bool selectPCRelAddress(SDValue Addr, SDValue &Target) const {
    if (SystemZISD::isPCREL(Addr.getOpcode())) {
      Target = Addr.getOperand(0);
      return true;
    }
    return false;
  }

  // BD matching routines used by SystemZOperands.td.
  bool selectBDAddr12Only(SDValue Addr, SDValue &Base, SDValue &Disp) const {
    return selectBDAddr(SystemZAddressingMode::Disp12Only, Addr, Base, Disp);
  }
  bool selectBDAddr12Pair(SDValue Addr, SDValue &Base, SDValue &Disp) const {
    return selectBDAddr(SystemZAddressingMode::Disp12Pair, Addr, Base, Disp);
  }
  bool selectBDAddr20Only(SDValue Addr, SDValue &Base, SDValue &Disp) const {
    return selectBDAddr(SystemZAddressingMode::Disp20Only, Addr, Base, Disp);
  }
  bool selectBDAddr20Pair(SDValue Addr, SDValue &Base, SDValue &Disp) const {
    return selectBDAddr(SystemZAddressingMode::Disp20Pair, Addr, Base, Disp);
  }

  // MVI matching routines used by SystemZOperands.td.
  bool selectMVIAddr12Pair(SDValue Addr, SDValue &Base, SDValue &Disp) const {
    return selectMVIAddr(SystemZAddressingMode::Disp12Pair, Addr, Base, Disp);
  }
  bool selectMVIAddr20Pair(SDValue Addr, SDValue &Base, SDValue &Disp) const {
    return selectMVIAddr(SystemZAddressingMode::Disp20Pair, Addr, Base, Disp);
  }

  // BDX matching routines used by SystemZOperands.td.
  bool selectBDXAddr12Only(SDValue Addr, SDValue &Base, SDValue &Disp,
                           SDValue &Index) const {
    return selectBDXAddr(SystemZAddressingMode::FormBDXNormal,
                         SystemZAddressingMode::Disp12Only,
                         Addr, Base, Disp, Index);
  }
  bool selectBDXAddr12Pair(SDValue Addr, SDValue &Base, SDValue &Disp,
                           SDValue &Index) const {
    return selectBDXAddr(SystemZAddressingMode::FormBDXNormal,
                         SystemZAddressingMode::Disp12Pair,
                         Addr, Base, Disp, Index);
  }
  bool selectDynAlloc12Only(SDValue Addr, SDValue &Base, SDValue &Disp,
                            SDValue &Index) const {
    return selectBDXAddr(SystemZAddressingMode::FormBDXDynAlloc,
                         SystemZAddressingMode::Disp12Only,
                         Addr, Base, Disp, Index);
  }
  bool selectBDXAddr20Only(SDValue Addr, SDValue &Base, SDValue &Disp,
                           SDValue &Index) const {
    return selectBDXAddr(SystemZAddressingMode::FormBDXNormal,
                         SystemZAddressingMode::Disp20Only,
                         Addr, Base, Disp, Index);
  }
  bool selectBDXAddr20Only128(SDValue Addr, SDValue &Base, SDValue &Disp,
                              SDValue &Index) const {
    return selectBDXAddr(SystemZAddressingMode::FormBDXNormal,
                         SystemZAddressingMode::Disp20Only128,
                         Addr, Base, Disp, Index);
  }
  bool selectBDXAddr20Pair(SDValue Addr, SDValue &Base, SDValue &Disp,
                           SDValue &Index) const {
    return selectBDXAddr(SystemZAddressingMode::FormBDXNormal,
                         SystemZAddressingMode::Disp20Pair,
                         Addr, Base, Disp, Index);
  }
  bool selectLAAddr12Pair(SDValue Addr, SDValue &Base, SDValue &Disp,
                          SDValue &Index) const {
    return selectBDXAddr(SystemZAddressingMode::FormBDXLA,
                         SystemZAddressingMode::Disp12Pair,
                         Addr, Base, Disp, Index);
  }
  bool selectLAAddr20Pair(SDValue Addr, SDValue &Base, SDValue &Disp,
                          SDValue &Index) const {
    return selectBDXAddr(SystemZAddressingMode::FormBDXLA,
                         SystemZAddressingMode::Disp20Pair,
                         Addr, Base, Disp, Index);
  }

  // Check whether (or Op (and X InsertMask)) is effectively an insertion
  // of X into bits InsertMask of some Y != Op.  Return true if so and
  // set Op to that Y.
  bool detectOrAndInsertion(SDValue &Op, uint64_t InsertMask) const;

  // Try to update RxSBG so that only the bits of RxSBG.Input in Mask are used.
  // Return true on success.
  bool refineRxSBGMask(RxSBGOperands &RxSBG, uint64_t Mask) const;

  // Try to fold some of RxSBG.Input into other fields of RxSBG.
  // Return true on success.
  bool expandRxSBG(RxSBGOperands &RxSBG) const;

  // Return an undefined value of type VT.
  SDValue getUNDEF(SDLoc DL, EVT VT) const;

  // Convert N to VT, if it isn't already.
  SDValue convertTo(SDLoc DL, EVT VT, SDValue N) const;

  // Try to implement AND or shift node N using RISBG with the zero flag set.
  // Return the selected node on success, otherwise return null.
  SDNode *tryRISBGZero(SDNode *N);

  // Try to use RISBG or Opcode to implement OR or XOR node N.
  // Return the selected node on success, otherwise return null.
  SDNode *tryRxSBG(SDNode *N, unsigned Opcode);

  // If Op0 is null, then Node is a constant that can be loaded using:
  //
  //   (Opcode UpperVal LowerVal)
  //
  // If Op0 is nonnull, then Node can be implemented using:
  //
  //   (Opcode (Opcode Op0 UpperVal) LowerVal)
  SDNode *splitLargeImmediate(unsigned Opcode, SDNode *Node, SDValue Op0,
                              uint64_t UpperVal, uint64_t LowerVal);

  // Return true if Load and Store are loads and stores of the same size
  // and are guaranteed not to overlap.  Such operations can be implemented
  // using block (SS-format) instructions.
  //
  // Partial overlap would lead to incorrect code, since the block operations
  // are logically bytewise, even though they have a fast path for the
  // non-overlapping case.  We also need to avoid full overlap (i.e. two
  // addresses that might be equal at run time) because although that case
  // would be handled correctly, it might be implemented by millicode.
  bool canUseBlockOperation(StoreSDNode *Store, LoadSDNode *Load) const;

  // N is a (store (load Y), X) pattern.  Return true if it can use an MVC
  // from Y to X.
  bool storeLoadCanUseMVC(SDNode *N) const;

  // N is a (store (op (load A[0]), (load A[1])), X) pattern.  Return true
  // if A[1 - I] == X and if N can use a block operation like NC from A[I]
  // to X.
  bool storeLoadCanUseBlockBinary(SDNode *N, unsigned I) const;

public:
  SystemZDAGToDAGISel(SystemZTargetMachine &TM, CodeGenOpt::Level OptLevel)
    : SelectionDAGISel(TM, OptLevel),
      Lowering(*TM.getTargetLowering()),
      Subtarget(*TM.getSubtargetImpl()) { }

  // Override MachineFunctionPass.
  virtual const char *getPassName() const LLVM_OVERRIDE {
    return "SystemZ DAG->DAG Pattern Instruction Selection";
  }

  // Override SelectionDAGISel.
  virtual SDNode *Select(SDNode *Node) LLVM_OVERRIDE;
  virtual bool SelectInlineAsmMemoryOperand(const SDValue &Op,
                                            char ConstraintCode,
                                            std::vector<SDValue> &OutOps)
    LLVM_OVERRIDE;

  // Include the pieces autogenerated from the target description.
  #include "SystemZGenDAGISel.inc"
};
} // end anonymous namespace

FunctionPass *llvm::createSystemZISelDag(SystemZTargetMachine &TM,
                                         CodeGenOpt::Level OptLevel) {
  return new SystemZDAGToDAGISel(TM, OptLevel);
}

// Return true if Val should be selected as a displacement for an address
// with range DR.  Here we're interested in the range of both the instruction
// described by DR and of any pairing instruction.
static bool selectDisp(SystemZAddressingMode::DispRange DR, int64_t Val) {
  switch (DR) {
  case SystemZAddressingMode::Disp12Only:
    return isUInt<12>(Val);

  case SystemZAddressingMode::Disp12Pair:
  case SystemZAddressingMode::Disp20Only:
  case SystemZAddressingMode::Disp20Pair:
    return isInt<20>(Val);

  case SystemZAddressingMode::Disp20Only128:
    return isInt<20>(Val) && isInt<20>(Val + 8);
  }
  llvm_unreachable("Unhandled displacement range");
}

// Change the base or index in AM to Value, where IsBase selects
// between the base and index.
static void changeComponent(SystemZAddressingMode &AM, bool IsBase,
                            SDValue Value) {
  if (IsBase)
    AM.Base = Value;
  else
    AM.Index = Value;
}

// The base or index of AM is equivalent to Value + ADJDYNALLOC,
// where IsBase selects between the base and index.  Try to fold the
// ADJDYNALLOC into AM.
static bool expandAdjDynAlloc(SystemZAddressingMode &AM, bool IsBase,
                              SDValue Value) {
  if (AM.isDynAlloc() && !AM.IncludesDynAlloc) {
    changeComponent(AM, IsBase, Value);
    AM.IncludesDynAlloc = true;
    return true;
  }
  return false;
}

// The base of AM is equivalent to Base + Index.  Try to use Index as
// the index register.
static bool expandIndex(SystemZAddressingMode &AM, SDValue Base,
                        SDValue Index) {
  if (AM.hasIndexField() && !AM.Index.getNode()) {
    AM.Base = Base;
    AM.Index = Index;
    return true;
  }
  return false;
}

// The base or index of AM is equivalent to Op0 + Op1, where IsBase selects
// between the base and index.  Try to fold Op1 into AM's displacement.
static bool expandDisp(SystemZAddressingMode &AM, bool IsBase,
                       SDValue Op0, uint64_t Op1) {
  // First try adjusting the displacement.
  int64_t TestDisp = AM.Disp + Op1;
  if (selectDisp(AM.DR, TestDisp)) {
    changeComponent(AM, IsBase, Op0);
    AM.Disp = TestDisp;
    return true;
  }

  // We could consider forcing the displacement into a register and
  // using it as an index, but it would need to be carefully tuned.
  return false;
}

bool SystemZDAGToDAGISel::expandAddress(SystemZAddressingMode &AM,
                                        bool IsBase) const {
  SDValue N = IsBase ? AM.Base : AM.Index;
  unsigned Opcode = N.getOpcode();
  if (Opcode == ISD::TRUNCATE) {
    N = N.getOperand(0);
    Opcode = N.getOpcode();
  }
  if (Opcode == ISD::ADD || CurDAG->isBaseWithConstantOffset(N)) {
    SDValue Op0 = N.getOperand(0);
    SDValue Op1 = N.getOperand(1);

    unsigned Op0Code = Op0->getOpcode();
    unsigned Op1Code = Op1->getOpcode();

    if (Op0Code == SystemZISD::ADJDYNALLOC)
      return expandAdjDynAlloc(AM, IsBase, Op1);
    if (Op1Code == SystemZISD::ADJDYNALLOC)
      return expandAdjDynAlloc(AM, IsBase, Op0);

    if (Op0Code == ISD::Constant)
      return expandDisp(AM, IsBase, Op1,
                        cast<ConstantSDNode>(Op0)->getSExtValue());
    if (Op1Code == ISD::Constant)
      return expandDisp(AM, IsBase, Op0,
                        cast<ConstantSDNode>(Op1)->getSExtValue());

    if (IsBase && expandIndex(AM, Op0, Op1))
      return true;
  }
  if (Opcode == SystemZISD::PCREL_OFFSET) {
    SDValue Full = N.getOperand(0);
    SDValue Base = N.getOperand(1);
    SDValue Anchor = Base.getOperand(0);
    uint64_t Offset = (cast<GlobalAddressSDNode>(Full)->getOffset() -
                       cast<GlobalAddressSDNode>(Anchor)->getOffset());
    return expandDisp(AM, IsBase, Base, Offset);
  }
  return false;
}

// Return true if an instruction with displacement range DR should be
// used for displacement value Val.  selectDisp(DR, Val) must already hold.
static bool isValidDisp(SystemZAddressingMode::DispRange DR, int64_t Val) {
  assert(selectDisp(DR, Val) && "Invalid displacement");
  switch (DR) {
  case SystemZAddressingMode::Disp12Only:
  case SystemZAddressingMode::Disp20Only:
  case SystemZAddressingMode::Disp20Only128:
    return true;

  case SystemZAddressingMode::Disp12Pair:
    // Use the other instruction if the displacement is too large.
    return isUInt<12>(Val);

  case SystemZAddressingMode::Disp20Pair:
    // Use the other instruction if the displacement is small enough.
    return !isUInt<12>(Val);
  }
  llvm_unreachable("Unhandled displacement range");
}

// Return true if Base + Disp + Index should be performed by LA(Y).
static bool shouldUseLA(SDNode *Base, int64_t Disp, SDNode *Index) {
  // Don't use LA(Y) for constants.
  if (!Base)
    return false;

  // Always use LA(Y) for frame addresses, since we know that the destination
  // register is almost always (perhaps always) going to be different from
  // the frame register.
  if (Base->getOpcode() == ISD::FrameIndex)
    return true;

  if (Disp) {
    // Always use LA(Y) if there is a base, displacement and index.
    if (Index)
      return true;

    // Always use LA if the displacement is small enough.  It should always
    // be no worse than AGHI (and better if it avoids a move).
    if (isUInt<12>(Disp))
      return true;

    // For similar reasons, always use LAY if the constant is too big for AGHI.
    // LAY should be no worse than AGFI.
    if (!isInt<16>(Disp))
      return true;
  } else {
    // Don't use LA for plain registers.
    if (!Index)
      return false;

    // Don't use LA for plain addition if the index operand is only used
    // once.  It should be a natural two-operand addition in that case.
    if (Index->hasOneUse())
      return false;

    // Prefer addition if the second operation is sign-extended, in the
    // hope of using AGF.
    unsigned IndexOpcode = Index->getOpcode();
    if (IndexOpcode == ISD::SIGN_EXTEND ||
        IndexOpcode == ISD::SIGN_EXTEND_INREG)
      return false;
  }

  // Don't use LA for two-operand addition if either operand is only
  // used once.  The addition instructions are better in that case.
  if (Base->hasOneUse())
    return false;

  return true;
}

// Return true if Addr is suitable for AM, updating AM if so.
bool SystemZDAGToDAGISel::selectAddress(SDValue Addr,
                                        SystemZAddressingMode &AM) const {
  // Start out assuming that the address will need to be loaded separately,
  // then try to extend it as much as we can.
  AM.Base = Addr;

  // First try treating the address as a constant.
  if (Addr.getOpcode() == ISD::Constant &&
      expandDisp(AM, true, SDValue(),
                 cast<ConstantSDNode>(Addr)->getSExtValue()))
    ;
  else
    // Otherwise try expanding each component.
    while (expandAddress(AM, true) ||
           (AM.Index.getNode() && expandAddress(AM, false)))
      continue;

  // Reject cases where it isn't profitable to use LA(Y).
  if (AM.Form == SystemZAddressingMode::FormBDXLA &&
      !shouldUseLA(AM.Base.getNode(), AM.Disp, AM.Index.getNode()))
    return false;

  // Reject cases where the other instruction in a pair should be used.
  if (!isValidDisp(AM.DR, AM.Disp))
    return false;

  // Make sure that ADJDYNALLOC is included where necessary.
  if (AM.isDynAlloc() && !AM.IncludesDynAlloc)
    return false;

  DEBUG(AM.dump());
  return true;
}

// Insert a node into the DAG at least before Pos.  This will reposition
// the node as needed, and will assign it a node ID that is <= Pos's ID.
// Note that this does *not* preserve the uniqueness of node IDs!
// The selection DAG must no longer depend on their uniqueness when this
// function is used.
static void insertDAGNode(SelectionDAG *DAG, SDNode *Pos, SDValue N) {
  if (N.getNode()->getNodeId() == -1 ||
      N.getNode()->getNodeId() > Pos->getNodeId()) {
    DAG->RepositionNode(Pos, N.getNode());
    N.getNode()->setNodeId(Pos->getNodeId());
  }
}

void SystemZDAGToDAGISel::getAddressOperands(const SystemZAddressingMode &AM,
                                             EVT VT, SDValue &Base,
                                             SDValue &Disp) const {
  Base = AM.Base;
  if (!Base.getNode())
    // Register 0 means "no base".  This is mostly useful for shifts.
    Base = CurDAG->getRegister(0, VT);
  else if (Base.getOpcode() == ISD::FrameIndex) {
    // Lower a FrameIndex to a TargetFrameIndex.
    int64_t FrameIndex = cast<FrameIndexSDNode>(Base)->getIndex();
    Base = CurDAG->getTargetFrameIndex(FrameIndex, VT);
  } else if (Base.getValueType() != VT) {
    // Truncate values from i64 to i32, for shifts.
    assert(VT == MVT::i32 && Base.getValueType() == MVT::i64 &&
           "Unexpected truncation");
    SDLoc DL(Base);
    SDValue Trunc = CurDAG->getNode(ISD::TRUNCATE, DL, VT, Base);
    insertDAGNode(CurDAG, Base.getNode(), Trunc);
    Base = Trunc;
  }

  // Lower the displacement to a TargetConstant.
  Disp = CurDAG->getTargetConstant(AM.Disp, VT);
}

void SystemZDAGToDAGISel::getAddressOperands(const SystemZAddressingMode &AM,
                                             EVT VT, SDValue &Base,
                                             SDValue &Disp,
                                             SDValue &Index) const {
  getAddressOperands(AM, VT, Base, Disp);

  Index = AM.Index;
  if (!Index.getNode())
    // Register 0 means "no index".
    Index = CurDAG->getRegister(0, VT);
}

bool SystemZDAGToDAGISel::selectBDAddr(SystemZAddressingMode::DispRange DR,
                                       SDValue Addr, SDValue &Base,
                                       SDValue &Disp) const {
  SystemZAddressingMode AM(SystemZAddressingMode::FormBD, DR);
  if (!selectAddress(Addr, AM))
    return false;

  getAddressOperands(AM, Addr.getValueType(), Base, Disp);
  return true;
}

bool SystemZDAGToDAGISel::selectMVIAddr(SystemZAddressingMode::DispRange DR,
                                        SDValue Addr, SDValue &Base,
                                        SDValue &Disp) const {
  SystemZAddressingMode AM(SystemZAddressingMode::FormBDXNormal, DR);
  if (!selectAddress(Addr, AM) || AM.Index.getNode())
    return false;

  getAddressOperands(AM, Addr.getValueType(), Base, Disp);
  return true;
}

bool SystemZDAGToDAGISel::selectBDXAddr(SystemZAddressingMode::AddrForm Form,
                                        SystemZAddressingMode::DispRange DR,
                                        SDValue Addr, SDValue &Base,
                                        SDValue &Disp, SDValue &Index) const {
  SystemZAddressingMode AM(Form, DR);
  if (!selectAddress(Addr, AM))
    return false;

  getAddressOperands(AM, Addr.getValueType(), Base, Disp, Index);
  return true;
}

bool SystemZDAGToDAGISel::detectOrAndInsertion(SDValue &Op,
                                               uint64_t InsertMask) const {
  // We're only interested in cases where the insertion is into some operand
  // of Op, rather than into Op itself.  The only useful case is an AND.
  if (Op.getOpcode() != ISD::AND)
    return false;

  // We need a constant mask.
  ConstantSDNode *MaskNode =
    dyn_cast<ConstantSDNode>(Op.getOperand(1).getNode());
  if (!MaskNode)
    return false;

  // It's not an insertion of Op.getOperand(0) if the two masks overlap.
  uint64_t AndMask = MaskNode->getZExtValue();
  if (InsertMask & AndMask)
    return false;

  // It's only an insertion if all bits are covered or are known to be zero.
  // The inner check covers all cases but is more expensive.
  uint64_t Used = allOnes(Op.getValueType().getSizeInBits());
  if (Used != (AndMask | InsertMask)) {
    APInt KnownZero, KnownOne;
    CurDAG->ComputeMaskedBits(Op.getOperand(0), KnownZero, KnownOne);
    if (Used != (AndMask | InsertMask | KnownZero.getZExtValue()))
      return false;
  }

  Op = Op.getOperand(0);
  return true;
}

bool SystemZDAGToDAGISel::refineRxSBGMask(RxSBGOperands &RxSBG,
                                          uint64_t Mask) const {
  const SystemZInstrInfo *TII = getInstrInfo();
  if (RxSBG.Rotate != 0)
    Mask = (Mask << RxSBG.Rotate) | (Mask >> (64 - RxSBG.Rotate));
  Mask &= RxSBG.Mask;
  if (TII->isRxSBGMask(Mask, RxSBG.BitSize, RxSBG.Start, RxSBG.End)) {
    RxSBG.Mask = Mask;
    return true;
  }
  return false;
}

// Return true if any bits of (RxSBG.Input & Mask) are significant.
static bool maskMatters(RxSBGOperands &RxSBG, uint64_t Mask) {
  // Rotate the mask in the same way as RxSBG.Input is rotated.
  if (RxSBG.Rotate != 0)
    Mask = ((Mask << RxSBG.Rotate) | (Mask >> (64 - RxSBG.Rotate)));
  return (Mask & RxSBG.Mask) != 0;
}

bool SystemZDAGToDAGISel::expandRxSBG(RxSBGOperands &RxSBG) const {
  SDValue N = RxSBG.Input;
  unsigned Opcode = N.getOpcode();
  switch (Opcode) {
  case ISD::AND: {
    if (RxSBG.Opcode == SystemZ::RNSBG)
      return false;

    ConstantSDNode *MaskNode =
      dyn_cast<ConstantSDNode>(N.getOperand(1).getNode());
    if (!MaskNode)
      return false;

    SDValue Input = N.getOperand(0);
    uint64_t Mask = MaskNode->getZExtValue();
    if (!refineRxSBGMask(RxSBG, Mask)) {
      // If some bits of Input are already known zeros, those bits will have
      // been removed from the mask.  See if adding them back in makes the
      // mask suitable.
      APInt KnownZero, KnownOne;
      CurDAG->ComputeMaskedBits(Input, KnownZero, KnownOne);
      Mask |= KnownZero.getZExtValue();
      if (!refineRxSBGMask(RxSBG, Mask))
        return false;
    }
    RxSBG.Input = Input;
    return true;
  }

  case ISD::OR: {
    if (RxSBG.Opcode != SystemZ::RNSBG)
      return false;

    ConstantSDNode *MaskNode =
      dyn_cast<ConstantSDNode>(N.getOperand(1).getNode());
    if (!MaskNode)
      return false;

    SDValue Input = N.getOperand(0);
    uint64_t Mask = ~MaskNode->getZExtValue();
    if (!refineRxSBGMask(RxSBG, Mask)) {
      // If some bits of Input are already known ones, those bits will have
      // been removed from the mask.  See if adding them back in makes the
      // mask suitable.
      APInt KnownZero, KnownOne;
      CurDAG->ComputeMaskedBits(Input, KnownZero, KnownOne);
      Mask &= ~KnownOne.getZExtValue();
      if (!refineRxSBGMask(RxSBG, Mask))
        return false;
    }
    RxSBG.Input = Input;
    return true;
  }

  case ISD::ROTL: {
    // Any 64-bit rotate left can be merged into the RxSBG.
    if (RxSBG.BitSize != 64 || N.getValueType() != MVT::i64)
      return false;
    ConstantSDNode *CountNode
      = dyn_cast<ConstantSDNode>(N.getOperand(1).getNode());
    if (!CountNode)
      return false;

    RxSBG.Rotate = (RxSBG.Rotate + CountNode->getZExtValue()) & 63;
    RxSBG.Input = N.getOperand(0);
    return true;
  }
      
  case ISD::SIGN_EXTEND:
  case ISD::ZERO_EXTEND:
  case ISD::ANY_EXTEND: {
    // Check that the extension bits are don't-care (i.e. are masked out
    // by the final mask).
    unsigned InnerBitSize = N.getOperand(0).getValueType().getSizeInBits();
    if (maskMatters(RxSBG, allOnes(RxSBG.BitSize) - allOnes(InnerBitSize)))
      return false;

    RxSBG.Input = N.getOperand(0);
    return true;
  }

  case ISD::SHL: {
    ConstantSDNode *CountNode =
      dyn_cast<ConstantSDNode>(N.getOperand(1).getNode());
    if (!CountNode)
      return false;

    uint64_t Count = CountNode->getZExtValue();
    unsigned BitSize = N.getValueType().getSizeInBits();
    if (Count < 1 || Count >= BitSize)
      return false;

    if (RxSBG.Opcode == SystemZ::RNSBG) {
      // Treat (shl X, count) as (rotl X, size-count) as long as the bottom
      // count bits from RxSBG.Input are ignored.
      if (maskMatters(RxSBG, allOnes(Count)))
        return false;
    } else {
      // Treat (shl X, count) as (and (rotl X, count), ~0<<count).
      if (!refineRxSBGMask(RxSBG, allOnes(BitSize - Count) << Count))
        return false;
    }

    RxSBG.Rotate = (RxSBG.Rotate + Count) & 63;
    RxSBG.Input = N.getOperand(0);
    return true;
  }

  case ISD::SRL:
  case ISD::SRA: {
    ConstantSDNode *CountNode =
      dyn_cast<ConstantSDNode>(N.getOperand(1).getNode());
    if (!CountNode)
      return false;

    uint64_t Count = CountNode->getZExtValue();
    unsigned BitSize = N.getValueType().getSizeInBits();
    if (Count < 1 || Count >= BitSize)
      return false;

    if (RxSBG.Opcode == SystemZ::RNSBG || Opcode == ISD::SRA) {
      // Treat (srl|sra X, count) as (rotl X, size-count) as long as the top
      // count bits from RxSBG.Input are ignored.
      if (maskMatters(RxSBG, allOnes(Count) << (BitSize - Count)))
        return false;
    } else {
      // Treat (srl X, count), mask) as (and (rotl X, size-count), ~0>>count),
      // which is similar to SLL above.
      if (!refineRxSBGMask(RxSBG, allOnes(BitSize - Count)))
        return false;
    }

    RxSBG.Rotate = (RxSBG.Rotate - Count) & 63;
    RxSBG.Input = N.getOperand(0);
    return true;
  }
  default:
    return false;
  }
}

SDValue SystemZDAGToDAGISel::getUNDEF(SDLoc DL, EVT VT) const {
  SDNode *N = CurDAG->getMachineNode(TargetOpcode::IMPLICIT_DEF, DL, VT);
  return SDValue(N, 0);
}

SDValue SystemZDAGToDAGISel::convertTo(SDLoc DL, EVT VT, SDValue N) const {
  if (N.getValueType() == MVT::i32 && VT == MVT::i64)
    return CurDAG->getTargetInsertSubreg(SystemZ::subreg_l32,
                                         DL, VT, getUNDEF(DL, MVT::i64), N);
  if (N.getValueType() == MVT::i64 && VT == MVT::i32)
    return CurDAG->getTargetExtractSubreg(SystemZ::subreg_l32, DL, VT, N);
  assert(N.getValueType() == VT && "Unexpected value types");
  return N;
}

SDNode *SystemZDAGToDAGISel::tryRISBGZero(SDNode *N) {
  EVT VT = N->getValueType(0);
  RxSBGOperands RISBG(SystemZ::RISBG, SDValue(N, 0));
  unsigned Count = 0;
  while (expandRxSBG(RISBG))
    if (RISBG.Input.getOpcode() != ISD::ANY_EXTEND)
      Count += 1;
  if (Count == 0)
    return 0;
  if (Count == 1) {
    // Prefer to use normal shift instructions over RISBG, since they can handle
    // all cases and are sometimes shorter.
    if (N->getOpcode() != ISD::AND)
      return 0;

    // Prefer register extensions like LLC over RISBG.  Also prefer to start
    // out with normal ANDs if one instruction would be enough.  We can convert
    // these ANDs into an RISBG later if a three-address instruction is useful.
    if (VT == MVT::i32 ||
        RISBG.Mask == 0xff ||
        RISBG.Mask == 0xffff ||
        SystemZ::isImmLF(~RISBG.Mask) ||
        SystemZ::isImmHF(~RISBG.Mask)) {
      // Force the new mask into the DAG, since it may include known-one bits.
      ConstantSDNode *MaskN = cast<ConstantSDNode>(N->getOperand(1).getNode());
      if (MaskN->getZExtValue() != RISBG.Mask) {
        SDValue NewMask = CurDAG->getConstant(RISBG.Mask, VT);
        N = CurDAG->UpdateNodeOperands(N, N->getOperand(0), NewMask);
        return SelectCode(N);
      }
      return 0;
    }
  }  

  unsigned Opcode = SystemZ::RISBG;
  EVT OpcodeVT = MVT::i64;
  if (VT == MVT::i32 && Subtarget.hasHighWord()) {
    Opcode = SystemZ::RISBMux;
    OpcodeVT = MVT::i32;
    RISBG.Start &= 31;
    RISBG.End &= 31;
  }
  SDValue Ops[5] = {
    getUNDEF(SDLoc(N), OpcodeVT),
    convertTo(SDLoc(N), OpcodeVT, RISBG.Input),
    CurDAG->getTargetConstant(RISBG.Start, MVT::i32),
    CurDAG->getTargetConstant(RISBG.End | 128, MVT::i32),
    CurDAG->getTargetConstant(RISBG.Rotate, MVT::i32)
  };
  N = CurDAG->getMachineNode(Opcode, SDLoc(N), OpcodeVT, Ops);
  return convertTo(SDLoc(N), VT, SDValue(N, 0)).getNode();
}

SDNode *SystemZDAGToDAGISel::tryRxSBG(SDNode *N, unsigned Opcode) {
  // Try treating each operand of N as the second operand of the RxSBG
  // and see which goes deepest.
  RxSBGOperands RxSBG[] = {
    RxSBGOperands(Opcode, N->getOperand(0)),
    RxSBGOperands(Opcode, N->getOperand(1))
  };
  unsigned Count[] = { 0, 0 };
  for (unsigned I = 0; I < 2; ++I)
    while (expandRxSBG(RxSBG[I]))
      if (RxSBG[I].Input.getOpcode() != ISD::ANY_EXTEND)
        Count[I] += 1;

  // Do nothing if neither operand is suitable.
  if (Count[0] == 0 && Count[1] == 0)
    return 0;

  // Pick the deepest second operand.
  unsigned I = Count[0] > Count[1] ? 0 : 1;
  SDValue Op0 = N->getOperand(I ^ 1);

  // Prefer IC for character insertions from memory.
  if (Opcode == SystemZ::ROSBG && (RxSBG[I].Mask & 0xff) == 0)
    if (LoadSDNode *Load = dyn_cast<LoadSDNode>(Op0.getNode()))
      if (Load->getMemoryVT() == MVT::i8)
        return 0;

  // See whether we can avoid an AND in the first operand by converting
  // ROSBG to RISBG.
  if (Opcode == SystemZ::ROSBG && detectOrAndInsertion(Op0, RxSBG[I].Mask))
    Opcode = SystemZ::RISBG;
           
  EVT VT = N->getValueType(0);
  SDValue Ops[5] = {
    convertTo(SDLoc(N), MVT::i64, Op0),
    convertTo(SDLoc(N), MVT::i64, RxSBG[I].Input),
    CurDAG->getTargetConstant(RxSBG[I].Start, MVT::i32),
    CurDAG->getTargetConstant(RxSBG[I].End, MVT::i32),
    CurDAG->getTargetConstant(RxSBG[I].Rotate, MVT::i32)
  };
  N = CurDAG->getMachineNode(Opcode, SDLoc(N), MVT::i64, Ops);
  return convertTo(SDLoc(N), VT, SDValue(N, 0)).getNode();
}

SDNode *SystemZDAGToDAGISel::splitLargeImmediate(unsigned Opcode, SDNode *Node,
                                                 SDValue Op0, uint64_t UpperVal,
                                                 uint64_t LowerVal) {
  EVT VT = Node->getValueType(0);
  SDLoc DL(Node);
  SDValue Upper = CurDAG->getConstant(UpperVal, VT);
  if (Op0.getNode())
    Upper = CurDAG->getNode(Opcode, DL, VT, Op0, Upper);
  Upper = SDValue(Select(Upper.getNode()), 0);

  SDValue Lower = CurDAG->getConstant(LowerVal, VT);
  SDValue Or = CurDAG->getNode(Opcode, DL, VT, Upper, Lower);
  return Or.getNode();
}

bool SystemZDAGToDAGISel::canUseBlockOperation(StoreSDNode *Store,
                                               LoadSDNode *Load) const {
  // Check that the two memory operands have the same size.
  if (Load->getMemoryVT() != Store->getMemoryVT())
    return false;

  // Volatility stops an access from being decomposed.
  if (Load->isVolatile() || Store->isVolatile())
    return false;

  // There's no chance of overlap if the load is invariant.
  if (Load->isInvariant())
    return true;

  // Otherwise we need to check whether there's an alias.
  const Value *V1 = Load->getSrcValue();
  const Value *V2 = Store->getSrcValue();
  if (!V1 || !V2)
    return false;

  // Reject equality.
  uint64_t Size = Load->getMemoryVT().getStoreSize();
  int64_t End1 = Load->getSrcValueOffset() + Size;
  int64_t End2 = Store->getSrcValueOffset() + Size;
  if (V1 == V2 && End1 == End2)
    return false;

  return !AA->alias(AliasAnalysis::Location(V1, End1, Load->getTBAAInfo()),
                    AliasAnalysis::Location(V2, End2, Store->getTBAAInfo()));
}

bool SystemZDAGToDAGISel::storeLoadCanUseMVC(SDNode *N) const {
  StoreSDNode *Store = cast<StoreSDNode>(N);
  LoadSDNode *Load = cast<LoadSDNode>(Store->getValue());

  // Prefer not to use MVC if either address can use ... RELATIVE LONG
  // instructions.
  uint64_t Size = Load->getMemoryVT().getStoreSize();
  if (Size > 1 && Size <= 8) {
    // Prefer LHRL, LRL and LGRL.
    if (SystemZISD::isPCREL(Load->getBasePtr().getOpcode()))
      return false;
    // Prefer STHRL, STRL and STGRL.
    if (SystemZISD::isPCREL(Store->getBasePtr().getOpcode()))
      return false;
  }

  return canUseBlockOperation(Store, Load);
}

bool SystemZDAGToDAGISel::storeLoadCanUseBlockBinary(SDNode *N,
                                                     unsigned I) const {
  StoreSDNode *StoreA = cast<StoreSDNode>(N);
  LoadSDNode *LoadA = cast<LoadSDNode>(StoreA->getValue().getOperand(1 - I));
  LoadSDNode *LoadB = cast<LoadSDNode>(StoreA->getValue().getOperand(I));
  return !LoadA->isVolatile() && canUseBlockOperation(StoreA, LoadB);
}

SDNode *SystemZDAGToDAGISel::Select(SDNode *Node) {
  // Dump information about the Node being selected
  DEBUG(errs() << "Selecting: "; Node->dump(CurDAG); errs() << "\n");

  // If we have a custom node, we already have selected!
  if (Node->isMachineOpcode()) {
    DEBUG(errs() << "== "; Node->dump(CurDAG); errs() << "\n");
    Node->setNodeId(-1);
    return 0;
  }

  unsigned Opcode = Node->getOpcode();
  SDNode *ResNode = 0;
  switch (Opcode) {
  case ISD::OR:
    if (Node->getOperand(1).getOpcode() != ISD::Constant)
      ResNode = tryRxSBG(Node, SystemZ::ROSBG);
    goto or_xor;

  case ISD::XOR:
    if (Node->getOperand(1).getOpcode() != ISD::Constant)
      ResNode = tryRxSBG(Node, SystemZ::RXSBG);
    // Fall through.
  or_xor:
    // If this is a 64-bit operation in which both 32-bit halves are nonzero,
    // split the operation into two.
    if (!ResNode && Node->getValueType(0) == MVT::i64)
      if (ConstantSDNode *Op1 = dyn_cast<ConstantSDNode>(Node->getOperand(1))) {
        uint64_t Val = Op1->getZExtValue();
        if (!SystemZ::isImmLF(Val) && !SystemZ::isImmHF(Val))
          Node = splitLargeImmediate(Opcode, Node, Node->getOperand(0),
                                     Val - uint32_t(Val), uint32_t(Val));
      }
    break;

  case ISD::AND:
    if (Node->getOperand(1).getOpcode() != ISD::Constant)
      ResNode = tryRxSBG(Node, SystemZ::RNSBG);
    // Fall through.
  case ISD::ROTL:
  case ISD::SHL:
  case ISD::SRL:
    if (!ResNode)
      ResNode = tryRISBGZero(Node);
    break;

  case ISD::Constant:
    // If this is a 64-bit constant that is out of the range of LLILF,
    // LLIHF and LGFI, split it into two 32-bit pieces.
    if (Node->getValueType(0) == MVT::i64) {
      uint64_t Val = cast<ConstantSDNode>(Node)->getZExtValue();
      if (!SystemZ::isImmLF(Val) && !SystemZ::isImmHF(Val) && !isInt<32>(Val))
        Node = splitLargeImmediate(ISD::OR, Node, SDValue(),
                                   Val - uint32_t(Val), uint32_t(Val));
    }
    break;

  case ISD::ATOMIC_LOAD_SUB:
    // Try to convert subtractions of constants to additions.
    if (ConstantSDNode *Op2 = dyn_cast<ConstantSDNode>(Node->getOperand(2))) {
      uint64_t Value = -Op2->getZExtValue();
      EVT VT = Node->getValueType(0);
      if (VT == MVT::i32 || isInt<32>(Value)) {
        SDValue Ops[] = { Node->getOperand(0), Node->getOperand(1),
                          CurDAG->getConstant(int32_t(Value), VT) };
        Node = CurDAG->MorphNodeTo(Node, ISD::ATOMIC_LOAD_ADD,
                                   Node->getVTList(), Ops, array_lengthof(Ops));
      }
    }
    break;

  case SystemZISD::SELECT_CCMASK: {
    SDValue Op0 = Node->getOperand(0);
    SDValue Op1 = Node->getOperand(1);
    // Prefer to put any load first, so that it can be matched as a
    // conditional load.
    if (Op1.getOpcode() == ISD::LOAD && Op0.getOpcode() != ISD::LOAD) {
      SDValue CCValid = Node->getOperand(2);
      SDValue CCMask = Node->getOperand(3);
      uint64_t ConstCCValid =
        cast<ConstantSDNode>(CCValid.getNode())->getZExtValue();
      uint64_t ConstCCMask =
        cast<ConstantSDNode>(CCMask.getNode())->getZExtValue();
      // Invert the condition.
      CCMask = CurDAG->getConstant(ConstCCValid ^ ConstCCMask,
                                   CCMask.getValueType());
      SDValue Op4 = Node->getOperand(4);
      Node = CurDAG->UpdateNodeOperands(Node, Op1, Op0, CCValid, CCMask, Op4);
    }
    break;
  }
  }

  // Select the default instruction
  if (!ResNode)
    ResNode = SelectCode(Node);

  DEBUG(errs() << "=> ";
        if (ResNode == NULL || ResNode == Node)
          Node->dump(CurDAG);
        else
          ResNode->dump(CurDAG);
        errs() << "\n";
        );
  return ResNode;
}

bool SystemZDAGToDAGISel::
SelectInlineAsmMemoryOperand(const SDValue &Op,
                             char ConstraintCode,
                             std::vector<SDValue> &OutOps) {
  assert(ConstraintCode == 'm' && "Unexpected constraint code");
  // Accept addresses with short displacements, which are compatible
  // with Q, R, S and T.  But keep the index operand for future expansion.
  SDValue Base, Disp, Index;
  if (!selectBDXAddr(SystemZAddressingMode::FormBD,
                     SystemZAddressingMode::Disp12Only,
                     Op, Base, Disp, Index))
    return true;
  OutOps.push_back(Base);
  OutOps.push_back(Disp);
  OutOps.push_back(Index);
  return false;
}
