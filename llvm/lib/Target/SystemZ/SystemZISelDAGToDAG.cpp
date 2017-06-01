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

#define DEBUG_TYPE "systemz-isel"

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
    if (Base.getNode())
      Base.getNode()->dump();
    else
      errs() << "null\n";

    if (hasIndexField()) {
      errs() << " Index ";
      if (Index.getNode())
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
  assert(Count <= 64);
  if (Count > 63)
    return UINT64_MAX;
  return (uint64_t(1) << Count) - 1;
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
// narrower (in which case the upper bits are don't care), or wider (in which
// case the result will be truncated as part of the operation).
struct RxSBGOperands {
  RxSBGOperands(unsigned Op, SDValue N)
    : Opcode(Op), BitSize(N.getValueSizeInBits()),
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
  const SystemZSubtarget *Subtarget;

  // Used by SystemZOperands.td to create integer constants.
  inline SDValue getImm(const SDNode *Node, uint64_t Imm) const {
    return CurDAG->getTargetConstant(Imm, SDLoc(Node), Node->getValueType(0));
  }

  const SystemZTargetMachine &getTargetMachine() const {
    return static_cast<const SystemZTargetMachine &>(TM);
  }

  const SystemZInstrInfo *getInstrInfo() const {
    return Subtarget->getInstrInfo();
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

  // Try to match Addr as an address with a base, 12-bit displacement
  // and index, where the index is element Elem of a vector.
  // Return true on success, storing the base, displacement and vector
  // in Base, Disp and Index respectively.
  bool selectBDVAddr12Only(SDValue Addr, SDValue Elem, SDValue &Base,
                           SDValue &Disp, SDValue &Index) const;

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
  SDValue getUNDEF(const SDLoc &DL, EVT VT) const;

  // Convert N to VT, if it isn't already.
  SDValue convertTo(const SDLoc &DL, EVT VT, SDValue N) const;

  // Try to implement AND or shift node N using RISBG with the zero flag set.
  // Return the selected node on success, otherwise return null.
  bool tryRISBGZero(SDNode *N);

  // Try to use RISBG or Opcode to implement OR or XOR node N.
  // Return the selected node on success, otherwise return null.
  bool tryRxSBG(SDNode *N, unsigned Opcode);

  // If Op0 is null, then Node is a constant that can be loaded using:
  //
  //   (Opcode UpperVal LowerVal)
  //
  // If Op0 is nonnull, then Node can be implemented using:
  //
  //   (Opcode (Opcode Op0 UpperVal) LowerVal)
  void splitLargeImmediate(unsigned Opcode, SDNode *Node, SDValue Op0,
                           uint64_t UpperVal, uint64_t LowerVal);

  // Try to use gather instruction Opcode to implement vector insertion N.
  bool tryGather(SDNode *N, unsigned Opcode);

  // Try to use scatter instruction Opcode to implement store Store.
  bool tryScatter(StoreSDNode *Store, unsigned Opcode);

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
      : SelectionDAGISel(TM, OptLevel) {}

  bool runOnMachineFunction(MachineFunction &MF) override {
    Subtarget = &MF.getSubtarget<SystemZSubtarget>();
    return SelectionDAGISel::runOnMachineFunction(MF);
  }

  // Override MachineFunctionPass.
  StringRef getPassName() const override {
    return "SystemZ DAG->DAG Pattern Instruction Selection";
  }

  // Override SelectionDAGISel.
  void Select(SDNode *Node) override;
  bool SelectInlineAsmMemoryOperand(const SDValue &Op, unsigned ConstraintID,
                                    std::vector<SDValue> &OutOps) override;

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
  // Also see if it's a bare ADJDYNALLOC.
  else if (Addr.getOpcode() == SystemZISD::ADJDYNALLOC &&
           expandAdjDynAlloc(AM, true, SDValue()))
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
    DAG->RepositionNode(Pos->getIterator(), N.getNode());
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
  Disp = CurDAG->getTargetConstant(AM.Disp, SDLoc(Base), VT);
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

bool SystemZDAGToDAGISel::selectBDVAddr12Only(SDValue Addr, SDValue Elem,
                                              SDValue &Base,
                                              SDValue &Disp,
                                              SDValue &Index) const {
  SDValue Regs[2];
  if (selectBDXAddr12Only(Addr, Regs[0], Disp, Regs[1]) &&
      Regs[0].getNode() && Regs[1].getNode()) {
    for (unsigned int I = 0; I < 2; ++I) {
      Base = Regs[I];
      Index = Regs[1 - I];
      // We can't tell here whether the index vector has the right type
      // for the access; the caller needs to do that instead.
      if (Index.getOpcode() == ISD::ZERO_EXTEND)
        Index = Index.getOperand(0);
      if (Index.getOpcode() == ISD::EXTRACT_VECTOR_ELT &&
          Index.getOperand(1) == Elem) {
        Index = Index.getOperand(0);
        return true;
      }
    }
  }
  return false;
}

bool SystemZDAGToDAGISel::detectOrAndInsertion(SDValue &Op,
                                               uint64_t InsertMask) const {
  // We're only interested in cases where the insertion is into some operand
  // of Op, rather than into Op itself.  The only useful case is an AND.
  if (Op.getOpcode() != ISD::AND)
    return false;

  // We need a constant mask.
  auto *MaskNode = dyn_cast<ConstantSDNode>(Op.getOperand(1).getNode());
  if (!MaskNode)
    return false;

  // It's not an insertion of Op.getOperand(0) if the two masks overlap.
  uint64_t AndMask = MaskNode->getZExtValue();
  if (InsertMask & AndMask)
    return false;

  // It's only an insertion if all bits are covered or are known to be zero.
  // The inner check covers all cases but is more expensive.
  uint64_t Used = allOnes(Op.getValueSizeInBits());
  if (Used != (AndMask | InsertMask)) {
    APInt KnownZero, KnownOne;
    CurDAG->computeKnownBits(Op.getOperand(0), KnownZero, KnownOne);
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
  case ISD::TRUNCATE: {
    if (RxSBG.Opcode == SystemZ::RNSBG)
      return false;
    uint64_t BitSize = N.getValueSizeInBits();
    uint64_t Mask = allOnes(BitSize);
    if (!refineRxSBGMask(RxSBG, Mask))
      return false;
    RxSBG.Input = N.getOperand(0);
    return true;
  }
  case ISD::AND: {
    if (RxSBG.Opcode == SystemZ::RNSBG)
      return false;

    auto *MaskNode = dyn_cast<ConstantSDNode>(N.getOperand(1).getNode());
    if (!MaskNode)
      return false;

    SDValue Input = N.getOperand(0);
    uint64_t Mask = MaskNode->getZExtValue();
    if (!refineRxSBGMask(RxSBG, Mask)) {
      // If some bits of Input are already known zeros, those bits will have
      // been removed from the mask.  See if adding them back in makes the
      // mask suitable.
      APInt KnownZero, KnownOne;
      CurDAG->computeKnownBits(Input, KnownZero, KnownOne);
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

    auto *MaskNode = dyn_cast<ConstantSDNode>(N.getOperand(1).getNode());
    if (!MaskNode)
      return false;

    SDValue Input = N.getOperand(0);
    uint64_t Mask = ~MaskNode->getZExtValue();
    if (!refineRxSBGMask(RxSBG, Mask)) {
      // If some bits of Input are already known ones, those bits will have
      // been removed from the mask.  See if adding them back in makes the
      // mask suitable.
      APInt KnownZero, KnownOne;
      CurDAG->computeKnownBits(Input, KnownZero, KnownOne);
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
    auto *CountNode = dyn_cast<ConstantSDNode>(N.getOperand(1).getNode());
    if (!CountNode)
      return false;

    RxSBG.Rotate = (RxSBG.Rotate + CountNode->getZExtValue()) & 63;
    RxSBG.Input = N.getOperand(0);
    return true;
  }

  case ISD::ANY_EXTEND:
    // Bits above the extended operand are don't-care.
    RxSBG.Input = N.getOperand(0);
    return true;

  case ISD::ZERO_EXTEND:
    if (RxSBG.Opcode != SystemZ::RNSBG) {
      // Restrict the mask to the extended operand.
      unsigned InnerBitSize = N.getOperand(0).getValueSizeInBits();
      if (!refineRxSBGMask(RxSBG, allOnes(InnerBitSize)))
        return false;

      RxSBG.Input = N.getOperand(0);
      return true;
    }
    LLVM_FALLTHROUGH;

  case ISD::SIGN_EXTEND: {
    // Check that the extension bits are don't-care (i.e. are masked out
    // by the final mask).
    unsigned InnerBitSize = N.getOperand(0).getValueSizeInBits();
    if (maskMatters(RxSBG, allOnes(RxSBG.BitSize) - allOnes(InnerBitSize)))
      return false;

    RxSBG.Input = N.getOperand(0);
    return true;
  }

  case ISD::SHL: {
    auto *CountNode = dyn_cast<ConstantSDNode>(N.getOperand(1).getNode());
    if (!CountNode)
      return false;

    uint64_t Count = CountNode->getZExtValue();
    unsigned BitSize = N.getValueSizeInBits();
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
    auto *CountNode = dyn_cast<ConstantSDNode>(N.getOperand(1).getNode());
    if (!CountNode)
      return false;

    uint64_t Count = CountNode->getZExtValue();
    unsigned BitSize = N.getValueSizeInBits();
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

SDValue SystemZDAGToDAGISel::getUNDEF(const SDLoc &DL, EVT VT) const {
  SDNode *N = CurDAG->getMachineNode(TargetOpcode::IMPLICIT_DEF, DL, VT);
  return SDValue(N, 0);
}

SDValue SystemZDAGToDAGISel::convertTo(const SDLoc &DL, EVT VT,
                                       SDValue N) const {
  if (N.getValueType() == MVT::i32 && VT == MVT::i64)
    return CurDAG->getTargetInsertSubreg(SystemZ::subreg_l32,
                                         DL, VT, getUNDEF(DL, MVT::i64), N);
  if (N.getValueType() == MVT::i64 && VT == MVT::i32)
    return CurDAG->getTargetExtractSubreg(SystemZ::subreg_l32, DL, VT, N);
  assert(N.getValueType() == VT && "Unexpected value types");
  return N;
}

bool SystemZDAGToDAGISel::tryRISBGZero(SDNode *N) {
  SDLoc DL(N);
  EVT VT = N->getValueType(0);
  if (!VT.isInteger() || VT.getSizeInBits() > 64)
    return false;
  RxSBGOperands RISBG(SystemZ::RISBG, SDValue(N, 0));
  unsigned Count = 0;
  while (expandRxSBG(RISBG))
    // The widening or narrowing is expected to be free.
    // Counting widening or narrowing as a saved operation will result in
    // preferring an R*SBG over a simple shift/logical instruction.
    if (RISBG.Input.getOpcode() != ISD::ANY_EXTEND &&
        RISBG.Input.getOpcode() != ISD::TRUNCATE)
      Count += 1;
  if (Count == 0)
    return false;

  // Prefer to use normal shift instructions over RISBG, since they can handle
  // all cases and are sometimes shorter.
  if (Count == 1 && N->getOpcode() != ISD::AND)
    return false;

  // Prefer register extensions like LLC over RISBG.  Also prefer to start
  // out with normal ANDs if one instruction would be enough.  We can convert
  // these ANDs into an RISBG later if a three-address instruction is useful.
  if (RISBG.Rotate == 0) {
    bool PreferAnd = false;
    // Prefer AND for any 32-bit and-immediate operation.
    if (VT == MVT::i32)
      PreferAnd = true;
    // As well as for any 64-bit operation that can be implemented via LLC(R),
    // LLH(R), LLGT(R), or one of the and-immediate instructions.
    else if (RISBG.Mask == 0xff ||
             RISBG.Mask == 0xffff ||
             RISBG.Mask == 0x7fffffff ||
             SystemZ::isImmLF(~RISBG.Mask) ||
             SystemZ::isImmHF(~RISBG.Mask))
     PreferAnd = true;
    // And likewise for the LLZRGF instruction, which doesn't have a register
    // to register version.
    else if (auto *Load = dyn_cast<LoadSDNode>(RISBG.Input)) {
      if (Load->getMemoryVT() == MVT::i32 &&
          (Load->getExtensionType() == ISD::EXTLOAD ||
           Load->getExtensionType() == ISD::ZEXTLOAD) &&
          RISBG.Mask == 0xffffff00 &&
          Subtarget->hasLoadAndZeroRightmostByte())
      PreferAnd = true;
    }
    if (PreferAnd) {
      // Replace the current node with an AND.  Note that the current node
      // might already be that same AND, in which case it is already CSE'd
      // with it, and we must not call ReplaceNode.
      SDValue In = convertTo(DL, VT, RISBG.Input);
      SDValue Mask = CurDAG->getConstant(RISBG.Mask, DL, VT);
      SDValue New = CurDAG->getNode(ISD::AND, DL, VT, In, Mask);
      if (N != New.getNode()) {
        insertDAGNode(CurDAG, N, Mask);
        insertDAGNode(CurDAG, N, New);
        ReplaceNode(N, New.getNode());
        N = New.getNode();
      }
      // Now, select the machine opcode to implement this operation.
      SelectCode(N);
      return true;
    }
  }

  unsigned Opcode = SystemZ::RISBG;
  // Prefer RISBGN if available, since it does not clobber CC.
  if (Subtarget->hasMiscellaneousExtensions())
    Opcode = SystemZ::RISBGN;
  EVT OpcodeVT = MVT::i64;
  if (VT == MVT::i32 && Subtarget->hasHighWord()) {
    Opcode = SystemZ::RISBMux;
    OpcodeVT = MVT::i32;
    RISBG.Start &= 31;
    RISBG.End &= 31;
  }
  SDValue Ops[5] = {
    getUNDEF(DL, OpcodeVT),
    convertTo(DL, OpcodeVT, RISBG.Input),
    CurDAG->getTargetConstant(RISBG.Start, DL, MVT::i32),
    CurDAG->getTargetConstant(RISBG.End | 128, DL, MVT::i32),
    CurDAG->getTargetConstant(RISBG.Rotate, DL, MVT::i32)
  };
  SDValue New = convertTo(
      DL, VT, SDValue(CurDAG->getMachineNode(Opcode, DL, OpcodeVT, Ops), 0));
  ReplaceUses(N, New.getNode());
  CurDAG->RemoveDeadNode(N);
  return true;
}

bool SystemZDAGToDAGISel::tryRxSBG(SDNode *N, unsigned Opcode) {
  SDLoc DL(N);
  EVT VT = N->getValueType(0);
  if (!VT.isInteger() || VT.getSizeInBits() > 64)
    return false;
  // Try treating each operand of N as the second operand of the RxSBG
  // and see which goes deepest.
  RxSBGOperands RxSBG[] = {
    RxSBGOperands(Opcode, N->getOperand(0)),
    RxSBGOperands(Opcode, N->getOperand(1))
  };
  unsigned Count[] = { 0, 0 };
  for (unsigned I = 0; I < 2; ++I)
    while (expandRxSBG(RxSBG[I]))
      // The widening or narrowing is expected to be free.
      // Counting widening or narrowing as a saved operation will result in
      // preferring an R*SBG over a simple shift/logical instruction.
      if (RxSBG[I].Input.getOpcode() != ISD::ANY_EXTEND &&
          RxSBG[I].Input.getOpcode() != ISD::TRUNCATE)
        Count[I] += 1;

  // Do nothing if neither operand is suitable.
  if (Count[0] == 0 && Count[1] == 0)
    return false;

  // Pick the deepest second operand.
  unsigned I = Count[0] > Count[1] ? 0 : 1;
  SDValue Op0 = N->getOperand(I ^ 1);

  // Prefer IC for character insertions from memory.
  if (Opcode == SystemZ::ROSBG && (RxSBG[I].Mask & 0xff) == 0)
    if (auto *Load = dyn_cast<LoadSDNode>(Op0.getNode()))
      if (Load->getMemoryVT() == MVT::i8)
        return false;

  // See whether we can avoid an AND in the first operand by converting
  // ROSBG to RISBG.
  if (Opcode == SystemZ::ROSBG && detectOrAndInsertion(Op0, RxSBG[I].Mask)) {
    Opcode = SystemZ::RISBG;
    // Prefer RISBGN if available, since it does not clobber CC.
    if (Subtarget->hasMiscellaneousExtensions())
      Opcode = SystemZ::RISBGN;
  }

  SDValue Ops[5] = {
    convertTo(DL, MVT::i64, Op0),
    convertTo(DL, MVT::i64, RxSBG[I].Input),
    CurDAG->getTargetConstant(RxSBG[I].Start, DL, MVT::i32),
    CurDAG->getTargetConstant(RxSBG[I].End, DL, MVT::i32),
    CurDAG->getTargetConstant(RxSBG[I].Rotate, DL, MVT::i32)
  };
  SDValue New = convertTo(
      DL, VT, SDValue(CurDAG->getMachineNode(Opcode, DL, MVT::i64, Ops), 0));
  ReplaceNode(N, New.getNode());
  return true;
}

void SystemZDAGToDAGISel::splitLargeImmediate(unsigned Opcode, SDNode *Node,
                                              SDValue Op0, uint64_t UpperVal,
                                              uint64_t LowerVal) {
  EVT VT = Node->getValueType(0);
  SDLoc DL(Node);
  SDValue Upper = CurDAG->getConstant(UpperVal, DL, VT);
  if (Op0.getNode())
    Upper = CurDAG->getNode(Opcode, DL, VT, Op0, Upper);

  {
    // When we haven't passed in Op0, Upper will be a constant. In order to
    // prevent folding back to the large immediate in `Or = getNode(...)` we run
    // SelectCode first and end up with an opaque machine node. This means that
    // we need to use a handle to keep track of Upper in case it gets CSE'd by
    // SelectCode.
    //
    // Note that in the case where Op0 is passed in we could just call
    // SelectCode(Upper) later, along with the SelectCode(Or), and avoid needing
    // the handle at all, but it's fine to do it here.
    //
    // TODO: This is a pretty hacky way to do this. Can we do something that
    // doesn't require a two paragraph explanation?
    HandleSDNode Handle(Upper);
    SelectCode(Upper.getNode());
    Upper = Handle.getValue();
  }

  SDValue Lower = CurDAG->getConstant(LowerVal, DL, VT);
  SDValue Or = CurDAG->getNode(Opcode, DL, VT, Upper, Lower);

  ReplaceUses(Node, Or.getNode());
  CurDAG->RemoveDeadNode(Node);

  SelectCode(Or.getNode());
}

bool SystemZDAGToDAGISel::tryGather(SDNode *N, unsigned Opcode) {
  SDValue ElemV = N->getOperand(2);
  auto *ElemN = dyn_cast<ConstantSDNode>(ElemV);
  if (!ElemN)
    return false;

  unsigned Elem = ElemN->getZExtValue();
  EVT VT = N->getValueType(0);
  if (Elem >= VT.getVectorNumElements())
    return false;

  auto *Load = dyn_cast<LoadSDNode>(N->getOperand(1));
  if (!Load || !Load->hasOneUse())
    return false;
  if (Load->getMemoryVT().getSizeInBits() !=
      Load->getValueType(0).getSizeInBits())
    return false;

  SDValue Base, Disp, Index;
  if (!selectBDVAddr12Only(Load->getBasePtr(), ElemV, Base, Disp, Index) ||
      Index.getValueType() != VT.changeVectorElementTypeToInteger())
    return false;

  SDLoc DL(Load);
  SDValue Ops[] = {
    N->getOperand(0), Base, Disp, Index,
    CurDAG->getTargetConstant(Elem, DL, MVT::i32), Load->getChain()
  };
  SDNode *Res = CurDAG->getMachineNode(Opcode, DL, VT, MVT::Other, Ops);
  ReplaceUses(SDValue(Load, 1), SDValue(Res, 1));
  ReplaceNode(N, Res);
  return true;
}

bool SystemZDAGToDAGISel::tryScatter(StoreSDNode *Store, unsigned Opcode) {
  SDValue Value = Store->getValue();
  if (Value.getOpcode() != ISD::EXTRACT_VECTOR_ELT)
    return false;
  if (Store->getMemoryVT().getSizeInBits() != Value.getValueSizeInBits())
    return false;

  SDValue ElemV = Value.getOperand(1);
  auto *ElemN = dyn_cast<ConstantSDNode>(ElemV);
  if (!ElemN)
    return false;

  SDValue Vec = Value.getOperand(0);
  EVT VT = Vec.getValueType();
  unsigned Elem = ElemN->getZExtValue();
  if (Elem >= VT.getVectorNumElements())
    return false;

  SDValue Base, Disp, Index;
  if (!selectBDVAddr12Only(Store->getBasePtr(), ElemV, Base, Disp, Index) ||
      Index.getValueType() != VT.changeVectorElementTypeToInteger())
    return false;

  SDLoc DL(Store);
  SDValue Ops[] = {
    Vec, Base, Disp, Index, CurDAG->getTargetConstant(Elem, DL, MVT::i32),
    Store->getChain()
  };
  ReplaceNode(Store, CurDAG->getMachineNode(Opcode, DL, MVT::Other, Ops));
  return true;
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
  if (Load->isInvariant() && Load->isDereferenceable())
    return true;

  // Otherwise we need to check whether there's an alias.
  const Value *V1 = Load->getMemOperand()->getValue();
  const Value *V2 = Store->getMemOperand()->getValue();
  if (!V1 || !V2)
    return false;

  // Reject equality.
  uint64_t Size = Load->getMemoryVT().getStoreSize();
  int64_t End1 = Load->getSrcValueOffset() + Size;
  int64_t End2 = Store->getSrcValueOffset() + Size;
  if (V1 == V2 && End1 == End2)
    return false;

  return !AA->alias(MemoryLocation(V1, End1, Load->getAAInfo()),
                    MemoryLocation(V2, End2, Store->getAAInfo()));
}

bool SystemZDAGToDAGISel::storeLoadCanUseMVC(SDNode *N) const {
  auto *Store = cast<StoreSDNode>(N);
  auto *Load = cast<LoadSDNode>(Store->getValue());

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
  auto *StoreA = cast<StoreSDNode>(N);
  auto *LoadA = cast<LoadSDNode>(StoreA->getValue().getOperand(1 - I));
  auto *LoadB = cast<LoadSDNode>(StoreA->getValue().getOperand(I));
  return !LoadA->isVolatile() && canUseBlockOperation(StoreA, LoadB);
}

void SystemZDAGToDAGISel::Select(SDNode *Node) {
  // Dump information about the Node being selected
  DEBUG(errs() << "Selecting: "; Node->dump(CurDAG); errs() << "\n");

  // If we have a custom node, we already have selected!
  if (Node->isMachineOpcode()) {
    DEBUG(errs() << "== "; Node->dump(CurDAG); errs() << "\n");
    Node->setNodeId(-1);
    return;
  }

  unsigned Opcode = Node->getOpcode();
  switch (Opcode) {
  case ISD::OR:
    if (Node->getOperand(1).getOpcode() != ISD::Constant)
      if (tryRxSBG(Node, SystemZ::ROSBG))
        return;
    goto or_xor;

  case ISD::XOR:
    if (Node->getOperand(1).getOpcode() != ISD::Constant)
      if (tryRxSBG(Node, SystemZ::RXSBG))
        return;
    // Fall through.
  or_xor:
    // If this is a 64-bit operation in which both 32-bit halves are nonzero,
    // split the operation into two.
    if (Node->getValueType(0) == MVT::i64)
      if (auto *Op1 = dyn_cast<ConstantSDNode>(Node->getOperand(1))) {
        uint64_t Val = Op1->getZExtValue();
        if (!SystemZ::isImmLF(Val) && !SystemZ::isImmHF(Val)) {
          splitLargeImmediate(Opcode, Node, Node->getOperand(0),
                              Val - uint32_t(Val), uint32_t(Val));
          return;
        }
      }
    break;

  case ISD::AND:
    if (Node->getOperand(1).getOpcode() != ISD::Constant)
      if (tryRxSBG(Node, SystemZ::RNSBG))
        return;
    LLVM_FALLTHROUGH;
  case ISD::ROTL:
  case ISD::SHL:
  case ISD::SRL:
  case ISD::ZERO_EXTEND:
    if (tryRISBGZero(Node))
      return;
    break;

  case ISD::Constant:
    // If this is a 64-bit constant that is out of the range of LLILF,
    // LLIHF and LGFI, split it into two 32-bit pieces.
    if (Node->getValueType(0) == MVT::i64) {
      uint64_t Val = cast<ConstantSDNode>(Node)->getZExtValue();
      if (!SystemZ::isImmLF(Val) && !SystemZ::isImmHF(Val) && !isInt<32>(Val)) {
        splitLargeImmediate(ISD::OR, Node, SDValue(), Val - uint32_t(Val),
                            uint32_t(Val));
        return;
      }
    }
    break;

  case SystemZISD::SELECT_CCMASK: {
    SDValue Op0 = Node->getOperand(0);
    SDValue Op1 = Node->getOperand(1);
    // Prefer to put any load first, so that it can be matched as a
    // conditional load.  Likewise for constants in range for LOCHI.
    if ((Op1.getOpcode() == ISD::LOAD && Op0.getOpcode() != ISD::LOAD) ||
        (Subtarget->hasLoadStoreOnCond2() &&
         Node->getValueType(0).isInteger() &&
         Op1.getOpcode() == ISD::Constant &&
         isInt<16>(cast<ConstantSDNode>(Op1)->getSExtValue()) &&
         !(Op0.getOpcode() == ISD::Constant &&
           isInt<16>(cast<ConstantSDNode>(Op0)->getSExtValue())))) {
      SDValue CCValid = Node->getOperand(2);
      SDValue CCMask = Node->getOperand(3);
      uint64_t ConstCCValid =
        cast<ConstantSDNode>(CCValid.getNode())->getZExtValue();
      uint64_t ConstCCMask =
        cast<ConstantSDNode>(CCMask.getNode())->getZExtValue();
      // Invert the condition.
      CCMask = CurDAG->getConstant(ConstCCValid ^ ConstCCMask, SDLoc(Node),
                                   CCMask.getValueType());
      SDValue Op4 = Node->getOperand(4);
      Node = CurDAG->UpdateNodeOperands(Node, Op1, Op0, CCValid, CCMask, Op4);
    }
    break;
  }

  case ISD::INSERT_VECTOR_ELT: {
    EVT VT = Node->getValueType(0);
    unsigned ElemBitSize = VT.getScalarSizeInBits();
    if (ElemBitSize == 32) {
      if (tryGather(Node, SystemZ::VGEF))
        return;
    } else if (ElemBitSize == 64) {
      if (tryGather(Node, SystemZ::VGEG))
        return;
    }
    break;
  }

  case ISD::STORE: {
    auto *Store = cast<StoreSDNode>(Node);
    unsigned ElemBitSize = Store->getValue().getValueSizeInBits();
    if (ElemBitSize == 32) {
      if (tryScatter(Store, SystemZ::VSCEF))
        return;
    } else if (ElemBitSize == 64) {
      if (tryScatter(Store, SystemZ::VSCEG))
        return;
    }
    break;
  }
  }

  SelectCode(Node);
}

bool SystemZDAGToDAGISel::
SelectInlineAsmMemoryOperand(const SDValue &Op,
                             unsigned ConstraintID,
                             std::vector<SDValue> &OutOps) {
  SystemZAddressingMode::AddrForm Form;
  SystemZAddressingMode::DispRange DispRange;
  SDValue Base, Disp, Index;

  switch(ConstraintID) {
  default:
    llvm_unreachable("Unexpected asm memory constraint");
  case InlineAsm::Constraint_i:
  case InlineAsm::Constraint_Q:
    // Accept an address with a short displacement, but no index.
    Form = SystemZAddressingMode::FormBD;
    DispRange = SystemZAddressingMode::Disp12Only;
    break;
  case InlineAsm::Constraint_R:
    // Accept an address with a short displacement and an index.
    Form = SystemZAddressingMode::FormBDXNormal;
    DispRange = SystemZAddressingMode::Disp12Only;
    break;
  case InlineAsm::Constraint_S:
    // Accept an address with a long displacement, but no index.
    Form = SystemZAddressingMode::FormBD;
    DispRange = SystemZAddressingMode::Disp20Only;
    break;
  case InlineAsm::Constraint_T:
  case InlineAsm::Constraint_m:
    // Accept an address with a long displacement and an index.
    // m works the same as T, as this is the most general case.
    Form = SystemZAddressingMode::FormBDXNormal;
    DispRange = SystemZAddressingMode::Disp20Only;
    break;
  }

  if (selectBDXAddr(Form, DispRange, Op, Base, Disp, Index)) {
    const TargetRegisterClass *TRC =
      Subtarget->getRegisterInfo()->getPointerRegClass(*MF);
    SDLoc DL(Base);
    SDValue RC = CurDAG->getTargetConstant(TRC->getID(), DL, MVT::i32);

    // Make sure that the base address doesn't go into %r0.
    // If it's a TargetFrameIndex or a fixed register, we shouldn't do anything.
    if (Base.getOpcode() != ISD::TargetFrameIndex &&
        Base.getOpcode() != ISD::Register) {
      Base =
        SDValue(CurDAG->getMachineNode(TargetOpcode::COPY_TO_REGCLASS,
                                       DL, Base.getValueType(),
                                       Base, RC), 0);
    }

    // Make sure that the index register isn't assigned to %r0 either.
    if (Index.getOpcode() != ISD::Register) {
      Index =
        SDValue(CurDAG->getMachineNode(TargetOpcode::COPY_TO_REGCLASS,
                                       DL, Index.getValueType(),
                                       Index, RC), 0);
    }

    OutOps.push_back(Base);
    OutOps.push_back(Disp);
    OutOps.push_back(Index);
    return false;
  }

  return true;
}
