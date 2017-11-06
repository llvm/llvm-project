//===- X86ISelDAGToDAG.cpp - A DAG pattern matching inst selector for X86 -===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a DAG pattern matching instruction selector for X86,
// converting from a legalized dag to a X86 dag.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrBuilder.h"
#include "X86MachineFunctionInfo.h"
#include "X86RegisterInfo.h"
#include "X86Subtarget.h"
#include "X86TargetMachine.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/IR/ConstantRange.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/KnownBits.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include <stdint.h>
using namespace llvm;

#define DEBUG_TYPE "x86-isel"

STATISTIC(NumLoadMoved, "Number of loads moved below TokenFactor");

//===----------------------------------------------------------------------===//
//                      Pattern Matcher Implementation
//===----------------------------------------------------------------------===//

namespace {
  /// This corresponds to X86AddressMode, but uses SDValue's instead of register
  /// numbers for the leaves of the matched tree.
  struct X86ISelAddressMode {
    enum {
      RegBase,
      FrameIndexBase
    } BaseType;

    // This is really a union, discriminated by BaseType!
    SDValue Base_Reg;
    int Base_FrameIndex;

    unsigned Scale;
    SDValue IndexReg;
    int32_t Disp;
    SDValue Segment;
    const GlobalValue *GV;
    const Constant *CP;
    const BlockAddress *BlockAddr;
    const char *ES;
    MCSymbol *MCSym;
    int JT;
    unsigned Align;    // CP alignment.
    unsigned char SymbolFlags;  // X86II::MO_*

    X86ISelAddressMode()
        : BaseType(RegBase), Base_FrameIndex(0), Scale(1), IndexReg(), Disp(0),
          Segment(), GV(nullptr), CP(nullptr), BlockAddr(nullptr), ES(nullptr),
          MCSym(nullptr), JT(-1), Align(0), SymbolFlags(X86II::MO_NO_FLAG) {}

    bool hasSymbolicDisplacement() const {
      return GV != nullptr || CP != nullptr || ES != nullptr ||
             MCSym != nullptr || JT != -1 || BlockAddr != nullptr;
    }

    bool hasBaseOrIndexReg() const {
      return BaseType == FrameIndexBase ||
             IndexReg.getNode() != nullptr || Base_Reg.getNode() != nullptr;
    }

    /// Return true if this addressing mode is already RIP-relative.
    bool isRIPRelative() const {
      if (BaseType != RegBase) return false;
      if (RegisterSDNode *RegNode =
            dyn_cast_or_null<RegisterSDNode>(Base_Reg.getNode()))
        return RegNode->getReg() == X86::RIP;
      return false;
    }

    void setBaseReg(SDValue Reg) {
      BaseType = RegBase;
      Base_Reg = Reg;
    }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
    void dump() {
      dbgs() << "X86ISelAddressMode " << this << '\n';
      dbgs() << "Base_Reg ";
      if (Base_Reg.getNode())
        Base_Reg.getNode()->dump();
      else
        dbgs() << "nul";
      dbgs() << " Base.FrameIndex " << Base_FrameIndex << '\n'
             << " Scale" << Scale << '\n'
             << "IndexReg ";
      if (IndexReg.getNode())
        IndexReg.getNode()->dump();
      else
        dbgs() << "nul";
      dbgs() << " Disp " << Disp << '\n'
             << "GV ";
      if (GV)
        GV->dump();
      else
        dbgs() << "nul";
      dbgs() << " CP ";
      if (CP)
        CP->dump();
      else
        dbgs() << "nul";
      dbgs() << '\n'
             << "ES ";
      if (ES)
        dbgs() << ES;
      else
        dbgs() << "nul";
      dbgs() << " MCSym ";
      if (MCSym)
        dbgs() << MCSym;
      else
        dbgs() << "nul";
      dbgs() << " JT" << JT << " Align" << Align << '\n';
    }
#endif
  };
}

namespace {
  //===--------------------------------------------------------------------===//
  /// ISel - X86-specific code to select X86 machine instructions for
  /// SelectionDAG operations.
  ///
  class X86DAGToDAGISel final : public SelectionDAGISel {
    /// Keep a pointer to the X86Subtarget around so that we can
    /// make the right decision when generating code for different targets.
    const X86Subtarget *Subtarget;

    /// If true, selector should try to optimize for code size instead of
    /// performance.
    bool OptForSize;

    /// If true, selector should try to optimize for minimum code size.
    bool OptForMinSize;

  public:
    explicit X86DAGToDAGISel(X86TargetMachine &tm, CodeGenOpt::Level OptLevel)
        : SelectionDAGISel(tm, OptLevel), OptForSize(false),
          OptForMinSize(false) {}

    StringRef getPassName() const override {
      return "X86 DAG->DAG Instruction Selection";
    }

    bool runOnMachineFunction(MachineFunction &MF) override {
      // Reset the subtarget each time through.
      Subtarget = &MF.getSubtarget<X86Subtarget>();
      SelectionDAGISel::runOnMachineFunction(MF);
      return true;
    }

    void EmitFunctionEntryCode() override;

    bool IsProfitableToFold(SDValue N, SDNode *U, SDNode *Root) const override;

    void PreprocessISelDAG() override;

// Include the pieces autogenerated from the target description.
#include "X86GenDAGISel.inc"

  private:
    void Select(SDNode *N) override;

    bool foldOffsetIntoAddress(uint64_t Offset, X86ISelAddressMode &AM);
    bool matchLoadInAddress(LoadSDNode *N, X86ISelAddressMode &AM);
    bool matchWrapper(SDValue N, X86ISelAddressMode &AM);
    bool matchAddress(SDValue N, X86ISelAddressMode &AM);
    bool matchAdd(SDValue N, X86ISelAddressMode &AM, unsigned Depth);
    bool matchAddressRecursively(SDValue N, X86ISelAddressMode &AM,
                                 unsigned Depth);
    bool matchAddressBase(SDValue N, X86ISelAddressMode &AM);
    bool selectAddr(SDNode *Parent, SDValue N, SDValue &Base,
                    SDValue &Scale, SDValue &Index, SDValue &Disp,
                    SDValue &Segment);
    bool selectVectorAddr(SDNode *Parent, SDValue N, SDValue &Base,
                          SDValue &Scale, SDValue &Index, SDValue &Disp,
                          SDValue &Segment);
    template <class GatherScatterSDNode>
    bool selectAddrOfGatherScatterNode(GatherScatterSDNode *Parent, SDValue N,
                                       SDValue &Base, SDValue &Scale,
                                       SDValue &Index, SDValue &Disp,
                                       SDValue &Segment);
    bool selectMOV64Imm32(SDValue N, SDValue &Imm);
    bool selectLEAAddr(SDValue N, SDValue &Base,
                       SDValue &Scale, SDValue &Index, SDValue &Disp,
                       SDValue &Segment);
    bool selectLEA64_32Addr(SDValue N, SDValue &Base,
                            SDValue &Scale, SDValue &Index, SDValue &Disp,
                            SDValue &Segment);
    bool selectTLSADDRAddr(SDValue N, SDValue &Base,
                           SDValue &Scale, SDValue &Index, SDValue &Disp,
                           SDValue &Segment);
    bool selectScalarSSELoad(SDNode *Root, SDValue N,
                             SDValue &Base, SDValue &Scale,
                             SDValue &Index, SDValue &Disp,
                             SDValue &Segment,
                             SDValue &NodeWithChain);
    bool selectRelocImm(SDValue N, SDValue &Op);

    bool tryFoldLoad(SDNode *P, SDValue N,
                     SDValue &Base, SDValue &Scale,
                     SDValue &Index, SDValue &Disp,
                     SDValue &Segment);

    /// Implement addressing mode selection for inline asm expressions.
    bool SelectInlineAsmMemoryOperand(const SDValue &Op,
                                      unsigned ConstraintID,
                                      std::vector<SDValue> &OutOps) override;

    void emitSpecialCodeForMain();

    inline void getAddressOperands(X86ISelAddressMode &AM, const SDLoc &DL,
                                   SDValue &Base, SDValue &Scale,
                                   SDValue &Index, SDValue &Disp,
                                   SDValue &Segment) {
      Base = (AM.BaseType == X86ISelAddressMode::FrameIndexBase)
                 ? CurDAG->getTargetFrameIndex(
                       AM.Base_FrameIndex,
                       TLI->getPointerTy(CurDAG->getDataLayout()))
                 : AM.Base_Reg;
      Scale = getI8Imm(AM.Scale, DL);
      Index = AM.IndexReg;
      // These are 32-bit even in 64-bit mode since RIP-relative offset
      // is 32-bit.
      if (AM.GV)
        Disp = CurDAG->getTargetGlobalAddress(AM.GV, SDLoc(),
                                              MVT::i32, AM.Disp,
                                              AM.SymbolFlags);
      else if (AM.CP)
        Disp = CurDAG->getTargetConstantPool(AM.CP, MVT::i32,
                                             AM.Align, AM.Disp, AM.SymbolFlags);
      else if (AM.ES) {
        assert(!AM.Disp && "Non-zero displacement is ignored with ES.");
        Disp = CurDAG->getTargetExternalSymbol(AM.ES, MVT::i32, AM.SymbolFlags);
      } else if (AM.MCSym) {
        assert(!AM.Disp && "Non-zero displacement is ignored with MCSym.");
        assert(AM.SymbolFlags == 0 && "oo");
        Disp = CurDAG->getMCSymbol(AM.MCSym, MVT::i32);
      } else if (AM.JT != -1) {
        assert(!AM.Disp && "Non-zero displacement is ignored with JT.");
        Disp = CurDAG->getTargetJumpTable(AM.JT, MVT::i32, AM.SymbolFlags);
      } else if (AM.BlockAddr)
        Disp = CurDAG->getTargetBlockAddress(AM.BlockAddr, MVT::i32, AM.Disp,
                                             AM.SymbolFlags);
      else
        Disp = CurDAG->getTargetConstant(AM.Disp, DL, MVT::i32);

      if (AM.Segment.getNode())
        Segment = AM.Segment;
      else
        Segment = CurDAG->getRegister(0, MVT::i32);
    }

    // Utility function to determine whether we should avoid selecting
    // immediate forms of instructions for better code size or not.
    // At a high level, we'd like to avoid such instructions when
    // we have similar constants used within the same basic block
    // that can be kept in a register.
    //
    bool shouldAvoidImmediateInstFormsForSize(SDNode *N) const {
      uint32_t UseCount = 0;

      // Do not want to hoist if we're not optimizing for size.
      // TODO: We'd like to remove this restriction.
      // See the comment in X86InstrInfo.td for more info.
      if (!OptForSize)
        return false;

      // Walk all the users of the immediate.
      for (SDNode::use_iterator UI = N->use_begin(),
           UE = N->use_end(); (UI != UE) && (UseCount < 2); ++UI) {

        SDNode *User = *UI;

        // This user is already selected. Count it as a legitimate use and
        // move on.
        if (User->isMachineOpcode()) {
          UseCount++;
          continue;
        }

        // We want to count stores of immediates as real uses.
        if (User->getOpcode() == ISD::STORE &&
            User->getOperand(1).getNode() == N) {
          UseCount++;
          continue;
        }

        // We don't currently match users that have > 2 operands (except
        // for stores, which are handled above)
        // Those instruction won't match in ISEL, for now, and would
        // be counted incorrectly.
        // This may change in the future as we add additional instruction
        // types.
        if (User->getNumOperands() != 2)
          continue;

        // Immediates that are used for offsets as part of stack
        // manipulation should be left alone. These are typically
        // used to indicate SP offsets for argument passing and
        // will get pulled into stores/pushes (implicitly).
        if (User->getOpcode() == X86ISD::ADD ||
            User->getOpcode() == ISD::ADD    ||
            User->getOpcode() == X86ISD::SUB ||
            User->getOpcode() == ISD::SUB) {

          // Find the other operand of the add/sub.
          SDValue OtherOp = User->getOperand(0);
          if (OtherOp.getNode() == N)
            OtherOp = User->getOperand(1);

          // Don't count if the other operand is SP.
          RegisterSDNode *RegNode;
          if (OtherOp->getOpcode() == ISD::CopyFromReg &&
              (RegNode = dyn_cast_or_null<RegisterSDNode>(
                 OtherOp->getOperand(1).getNode())))
            if ((RegNode->getReg() == X86::ESP) ||
                (RegNode->getReg() == X86::RSP))
              continue;
        }

        // ... otherwise, count this and move on.
        UseCount++;
      }

      // If we have more than 1 use, then recommend for hoisting.
      return (UseCount > 1);
    }

    /// Return a target constant with the specified value of type i8.
    inline SDValue getI8Imm(unsigned Imm, const SDLoc &DL) {
      return CurDAG->getTargetConstant(Imm, DL, MVT::i8);
    }

    /// Return a target constant with the specified value, of type i32.
    inline SDValue getI32Imm(unsigned Imm, const SDLoc &DL) {
      return CurDAG->getTargetConstant(Imm, DL, MVT::i32);
    }

    SDValue getExtractVEXTRACTImmediate(SDNode *N, unsigned VecWidth,
                                        const SDLoc &DL) {
      assert((VecWidth == 128 || VecWidth == 256) && "Unexpected vector width");
      uint64_t Index = N->getConstantOperandVal(1);
      MVT VecVT = N->getOperand(0).getSimpleValueType();
      return getI8Imm((Index * VecVT.getScalarSizeInBits()) / VecWidth, DL);
    }

    SDValue getInsertVINSERTImmediate(SDNode *N, unsigned VecWidth,
                                      const SDLoc &DL) {
      assert((VecWidth == 128 || VecWidth == 256) && "Unexpected vector width");
      uint64_t Index = N->getConstantOperandVal(2);
      MVT VecVT = N->getSimpleValueType(0);
      return getI8Imm((Index * VecVT.getScalarSizeInBits()) / VecWidth, DL);
    }

    /// Return an SDNode that returns the value of the global base register.
    /// Output instructions required to initialize the global base register,
    /// if necessary.
    SDNode *getGlobalBaseReg();

    /// Return a reference to the TargetMachine, casted to the target-specific
    /// type.
    const X86TargetMachine &getTargetMachine() const {
      return static_cast<const X86TargetMachine &>(TM);
    }

    /// Return a reference to the TargetInstrInfo, casted to the target-specific
    /// type.
    const X86InstrInfo *getInstrInfo() const {
      return Subtarget->getInstrInfo();
    }

    /// \brief Address-mode matching performs shift-of-and to and-of-shift
    /// reassociation in order to expose more scaled addressing
    /// opportunities.
    bool ComplexPatternFuncMutatesDAG() const override {
      return true;
    }

    bool isSExtAbsoluteSymbolRef(unsigned Width, SDNode *N) const;

    /// Returns whether this is a relocatable immediate in the range
    /// [-2^Width .. 2^Width-1].
    template <unsigned Width> bool isSExtRelocImm(SDNode *N) const {
      if (auto *CN = dyn_cast<ConstantSDNode>(N))
        return isInt<Width>(CN->getSExtValue());
      return isSExtAbsoluteSymbolRef(Width, N);
    }

    // Indicates we should prefer to use a non-temporal load for this load.
    bool useNonTemporalLoad(LoadSDNode *N) const {
      if (!N->isNonTemporal())
        return false;

      unsigned StoreSize = N->getMemoryVT().getStoreSize();

      if (N->getAlignment() < StoreSize)
        return false;

      switch (StoreSize) {
      default: llvm_unreachable("Unsupported store size");
      case 16:
        return Subtarget->hasSSE41();
      case 32:
        return Subtarget->hasAVX2();
      case 64:
        return Subtarget->hasAVX512();
      }
    }

    bool foldLoadStoreIntoMemOperand(SDNode *Node);

    bool matchBEXTRFromAnd(SDNode *Node);

    bool isMaskZeroExtended(SDNode *N) const;
  };
}


// Returns true if this masked compare can be implemented legally with this
// type.
static bool isLegalMaskCompare(SDNode *N, const X86Subtarget *Subtarget) {
  unsigned Opcode = N->getOpcode();
  if (Opcode == X86ISD::PCMPEQM || Opcode == X86ISD::PCMPGTM ||
      Opcode == X86ISD::CMPM || Opcode == X86ISD::TESTM ||
      Opcode == X86ISD::TESTNM || Opcode == X86ISD::CMPMU) {
    // We can get 256-bit 8 element types here without VLX being enabled. When
    // this happens we will use 512-bit operations and the mask will not be
    // zero extended.
    EVT OpVT = N->getOperand(0).getValueType();
    if (OpVT == MVT::v8i32 || OpVT == MVT::v8f32)
      return Subtarget->hasVLX();

    return true;
  }

  return false;
}

// Returns true if we can assume the writer of the mask has zero extended it
// for us.
bool X86DAGToDAGISel::isMaskZeroExtended(SDNode *N) const {
  // If this is an AND, check if we have a compare on either side. As long as
  // one side guarantees the mask is zero extended, the AND will preserve those
  // zeros.
  if (N->getOpcode() == ISD::AND)
    return isLegalMaskCompare(N->getOperand(0).getNode(), Subtarget) ||
           isLegalMaskCompare(N->getOperand(1).getNode(), Subtarget);

  return isLegalMaskCompare(N, Subtarget);
}

bool
X86DAGToDAGISel::IsProfitableToFold(SDValue N, SDNode *U, SDNode *Root) const {
  if (OptLevel == CodeGenOpt::None) return false;

  if (!N.hasOneUse())
    return false;

  if (N.getOpcode() != ISD::LOAD)
    return true;

  // If N is a load, do additional profitability checks.
  if (U == Root) {
    switch (U->getOpcode()) {
    default: break;
    case X86ISD::ADD:
    case X86ISD::SUB:
    case X86ISD::AND:
    case X86ISD::XOR:
    case X86ISD::OR:
    case ISD::ADD:
    case ISD::ADDCARRY:
    case ISD::AND:
    case ISD::OR:
    case ISD::XOR: {
      SDValue Op1 = U->getOperand(1);

      // If the other operand is a 8-bit immediate we should fold the immediate
      // instead. This reduces code size.
      // e.g.
      // movl 4(%esp), %eax
      // addl $4, %eax
      // vs.
      // movl $4, %eax
      // addl 4(%esp), %eax
      // The former is 2 bytes shorter. In case where the increment is 1, then
      // the saving can be 4 bytes (by using incl %eax).
      if (ConstantSDNode *Imm = dyn_cast<ConstantSDNode>(Op1))
        if (Imm->getAPIntValue().isSignedIntN(8))
          return false;

      // If the other operand is a TLS address, we should fold it instead.
      // This produces
      // movl    %gs:0, %eax
      // leal    i@NTPOFF(%eax), %eax
      // instead of
      // movl    $i@NTPOFF, %eax
      // addl    %gs:0, %eax
      // if the block also has an access to a second TLS address this will save
      // a load.
      // FIXME: This is probably also true for non-TLS addresses.
      if (Op1.getOpcode() == X86ISD::Wrapper) {
        SDValue Val = Op1.getOperand(0);
        if (Val.getOpcode() == ISD::TargetGlobalTLSAddress)
          return false;
      }
    }
    }
  }

  return true;
}

/// Replace the original chain operand of the call with
/// load's chain operand and move load below the call's chain operand.
static void moveBelowOrigChain(SelectionDAG *CurDAG, SDValue Load,
                               SDValue Call, SDValue OrigChain) {
  SmallVector<SDValue, 8> Ops;
  SDValue Chain = OrigChain.getOperand(0);
  if (Chain.getNode() == Load.getNode())
    Ops.push_back(Load.getOperand(0));
  else {
    assert(Chain.getOpcode() == ISD::TokenFactor &&
           "Unexpected chain operand");
    for (unsigned i = 0, e = Chain.getNumOperands(); i != e; ++i)
      if (Chain.getOperand(i).getNode() == Load.getNode())
        Ops.push_back(Load.getOperand(0));
      else
        Ops.push_back(Chain.getOperand(i));
    SDValue NewChain =
      CurDAG->getNode(ISD::TokenFactor, SDLoc(Load), MVT::Other, Ops);
    Ops.clear();
    Ops.push_back(NewChain);
  }
  Ops.append(OrigChain->op_begin() + 1, OrigChain->op_end());
  CurDAG->UpdateNodeOperands(OrigChain.getNode(), Ops);
  CurDAG->UpdateNodeOperands(Load.getNode(), Call.getOperand(0),
                             Load.getOperand(1), Load.getOperand(2));

  Ops.clear();
  Ops.push_back(SDValue(Load.getNode(), 1));
  Ops.append(Call->op_begin() + 1, Call->op_end());
  CurDAG->UpdateNodeOperands(Call.getNode(), Ops);
}

/// Return true if call address is a load and it can be
/// moved below CALLSEQ_START and the chains leading up to the call.
/// Return the CALLSEQ_START by reference as a second output.
/// In the case of a tail call, there isn't a callseq node between the call
/// chain and the load.
static bool isCalleeLoad(SDValue Callee, SDValue &Chain, bool HasCallSeq) {
  // The transformation is somewhat dangerous if the call's chain was glued to
  // the call. After MoveBelowOrigChain the load is moved between the call and
  // the chain, this can create a cycle if the load is not folded. So it is
  // *really* important that we are sure the load will be folded.
  if (Callee.getNode() == Chain.getNode() || !Callee.hasOneUse())
    return false;
  LoadSDNode *LD = dyn_cast<LoadSDNode>(Callee.getNode());
  if (!LD ||
      LD->isVolatile() ||
      LD->getAddressingMode() != ISD::UNINDEXED ||
      LD->getExtensionType() != ISD::NON_EXTLOAD)
    return false;

  // Now let's find the callseq_start.
  while (HasCallSeq && Chain.getOpcode() != ISD::CALLSEQ_START) {
    if (!Chain.hasOneUse())
      return false;
    Chain = Chain.getOperand(0);
  }

  if (!Chain.getNumOperands())
    return false;
  // Since we are not checking for AA here, conservatively abort if the chain
  // writes to memory. It's not safe to move the callee (a load) across a store.
  if (isa<MemSDNode>(Chain.getNode()) &&
      cast<MemSDNode>(Chain.getNode())->writeMem())
    return false;
  if (Chain.getOperand(0).getNode() == Callee.getNode())
    return true;
  if (Chain.getOperand(0).getOpcode() == ISD::TokenFactor &&
      Callee.getValue(1).isOperandOf(Chain.getOperand(0).getNode()) &&
      Callee.getValue(1).hasOneUse())
    return true;
  return false;
}

void X86DAGToDAGISel::PreprocessISelDAG() {
  // OptFor[Min]Size are used in pattern predicates that isel is matching.
  OptForSize = MF->getFunction()->optForSize();
  OptForMinSize = MF->getFunction()->optForMinSize();
  assert((!OptForMinSize || OptForSize) && "OptForMinSize implies OptForSize");

  for (SelectionDAG::allnodes_iterator I = CurDAG->allnodes_begin(),
       E = CurDAG->allnodes_end(); I != E; ) {
    SDNode *N = &*I++; // Preincrement iterator to avoid invalidation issues.

    if (OptLevel != CodeGenOpt::None &&
        // Only does this when target favors doesn't favor register indirect
        // call.
        ((N->getOpcode() == X86ISD::CALL && !Subtarget->slowTwoMemOps()) ||
         (N->getOpcode() == X86ISD::TC_RETURN &&
          // Only does this if load can be folded into TC_RETURN.
          (Subtarget->is64Bit() ||
           !getTargetMachine().isPositionIndependent())))) {
      /// Also try moving call address load from outside callseq_start to just
      /// before the call to allow it to be folded.
      ///
      ///     [Load chain]
      ///         ^
      ///         |
      ///       [Load]
      ///       ^    ^
      ///       |    |
      ///      /      \--
      ///     /          |
      ///[CALLSEQ_START] |
      ///     ^          |
      ///     |          |
      /// [LOAD/C2Reg]   |
      ///     |          |
      ///      \        /
      ///       \      /
      ///       [CALL]
      bool HasCallSeq = N->getOpcode() == X86ISD::CALL;
      SDValue Chain = N->getOperand(0);
      SDValue Load  = N->getOperand(1);
      if (!isCalleeLoad(Load, Chain, HasCallSeq))
        continue;
      moveBelowOrigChain(CurDAG, Load, SDValue(N, 0), Chain);
      ++NumLoadMoved;
      continue;
    }

    // Lower fpround and fpextend nodes that target the FP stack to be store and
    // load to the stack.  This is a gross hack.  We would like to simply mark
    // these as being illegal, but when we do that, legalize produces these when
    // it expands calls, then expands these in the same legalize pass.  We would
    // like dag combine to be able to hack on these between the call expansion
    // and the node legalization.  As such this pass basically does "really
    // late" legalization of these inline with the X86 isel pass.
    // FIXME: This should only happen when not compiled with -O0.
    if (N->getOpcode() != ISD::FP_ROUND && N->getOpcode() != ISD::FP_EXTEND)
      continue;

    MVT SrcVT = N->getOperand(0).getSimpleValueType();
    MVT DstVT = N->getSimpleValueType(0);

    // If any of the sources are vectors, no fp stack involved.
    if (SrcVT.isVector() || DstVT.isVector())
      continue;

    // If the source and destination are SSE registers, then this is a legal
    // conversion that should not be lowered.
    const X86TargetLowering *X86Lowering =
        static_cast<const X86TargetLowering *>(TLI);
    bool SrcIsSSE = X86Lowering->isScalarFPTypeInSSEReg(SrcVT);
    bool DstIsSSE = X86Lowering->isScalarFPTypeInSSEReg(DstVT);
    if (SrcIsSSE && DstIsSSE)
      continue;

    if (!SrcIsSSE && !DstIsSSE) {
      // If this is an FPStack extension, it is a noop.
      if (N->getOpcode() == ISD::FP_EXTEND)
        continue;
      // If this is a value-preserving FPStack truncation, it is a noop.
      if (N->getConstantOperandVal(1))
        continue;
    }

    // Here we could have an FP stack truncation or an FPStack <-> SSE convert.
    // FPStack has extload and truncstore.  SSE can fold direct loads into other
    // operations.  Based on this, decide what we want to do.
    MVT MemVT;
    if (N->getOpcode() == ISD::FP_ROUND)
      MemVT = DstVT;  // FP_ROUND must use DstVT, we can't do a 'trunc load'.
    else
      MemVT = SrcIsSSE ? SrcVT : DstVT;

    SDValue MemTmp = CurDAG->CreateStackTemporary(MemVT);
    SDLoc dl(N);

    // FIXME: optimize the case where the src/dest is a load or store?
    SDValue Store =
        CurDAG->getTruncStore(CurDAG->getEntryNode(), dl, N->getOperand(0),
                              MemTmp, MachinePointerInfo(), MemVT);
    SDValue Result = CurDAG->getExtLoad(ISD::EXTLOAD, dl, DstVT, Store, MemTmp,
                                        MachinePointerInfo(), MemVT);

    // We're about to replace all uses of the FP_ROUND/FP_EXTEND with the
    // extload we created.  This will cause general havok on the dag because
    // anything below the conversion could be folded into other existing nodes.
    // To avoid invalidating 'I', back it up to the convert node.
    --I;
    CurDAG->ReplaceAllUsesOfValueWith(SDValue(N, 0), Result);

    // Now that we did that, the node is dead.  Increment the iterator to the
    // next node to process, then delete N.
    ++I;
    CurDAG->DeleteNode(N);
  }
}


/// Emit any code that needs to be executed only in the main function.
void X86DAGToDAGISel::emitSpecialCodeForMain() {
  if (Subtarget->isTargetCygMing()) {
    TargetLowering::ArgListTy Args;
    auto &DL = CurDAG->getDataLayout();

    TargetLowering::CallLoweringInfo CLI(*CurDAG);
    CLI.setChain(CurDAG->getRoot())
        .setCallee(CallingConv::C, Type::getVoidTy(*CurDAG->getContext()),
                   CurDAG->getExternalSymbol("__main", TLI->getPointerTy(DL)),
                   std::move(Args));
    const TargetLowering &TLI = CurDAG->getTargetLoweringInfo();
    std::pair<SDValue, SDValue> Result = TLI.LowerCallTo(CLI);
    CurDAG->setRoot(Result.second);
  }
}

void X86DAGToDAGISel::EmitFunctionEntryCode() {
  // If this is main, emit special code for main.
  if (const Function *Fn = MF->getFunction())
    if (Fn->hasExternalLinkage() && Fn->getName() == "main")
      emitSpecialCodeForMain();
}

static bool isDispSafeForFrameIndex(int64_t Val) {
  // On 64-bit platforms, we can run into an issue where a frame index
  // includes a displacement that, when added to the explicit displacement,
  // will overflow the displacement field. Assuming that the frame index
  // displacement fits into a 31-bit integer  (which is only slightly more
  // aggressive than the current fundamental assumption that it fits into
  // a 32-bit integer), a 31-bit disp should always be safe.
  return isInt<31>(Val);
}

bool X86DAGToDAGISel::foldOffsetIntoAddress(uint64_t Offset,
                                            X86ISelAddressMode &AM) {
  // Cannot combine ExternalSymbol displacements with integer offsets.
  if (Offset != 0 && (AM.ES || AM.MCSym))
    return true;
  int64_t Val = AM.Disp + Offset;
  CodeModel::Model M = TM.getCodeModel();
  if (Subtarget->is64Bit()) {
    if (!X86::isOffsetSuitableForCodeModel(Val, M,
                                           AM.hasSymbolicDisplacement()))
      return true;
    // In addition to the checks required for a register base, check that
    // we do not try to use an unsafe Disp with a frame index.
    if (AM.BaseType == X86ISelAddressMode::FrameIndexBase &&
        !isDispSafeForFrameIndex(Val))
      return true;
  }
  AM.Disp = Val;
  return false;

}

bool X86DAGToDAGISel::matchLoadInAddress(LoadSDNode *N, X86ISelAddressMode &AM){
  SDValue Address = N->getOperand(1);

  // load gs:0 -> GS segment register.
  // load fs:0 -> FS segment register.
  //
  // This optimization is valid because the GNU TLS model defines that
  // gs:0 (or fs:0 on X86-64) contains its own address.
  // For more information see http://people.redhat.com/drepper/tls.pdf
  if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(Address))
    if (C->getSExtValue() == 0 && AM.Segment.getNode() == nullptr &&
        (Subtarget->isTargetGlibc() || Subtarget->isTargetAndroid() ||
         Subtarget->isTargetFuchsia()))
      switch (N->getPointerInfo().getAddrSpace()) {
      case 256:
        AM.Segment = CurDAG->getRegister(X86::GS, MVT::i16);
        return false;
      case 257:
        AM.Segment = CurDAG->getRegister(X86::FS, MVT::i16);
        return false;
      // Address space 258 is not handled here, because it is not used to
      // address TLS areas.
      }

  return true;
}

/// Try to match X86ISD::Wrapper and X86ISD::WrapperRIP nodes into an addressing
/// mode. These wrap things that will resolve down into a symbol reference.
/// If no match is possible, this returns true, otherwise it returns false.
bool X86DAGToDAGISel::matchWrapper(SDValue N, X86ISelAddressMode &AM) {
  // If the addressing mode already has a symbol as the displacement, we can
  // never match another symbol.
  if (AM.hasSymbolicDisplacement())
    return true;

  SDValue N0 = N.getOperand(0);
  CodeModel::Model M = TM.getCodeModel();

  // Handle X86-64 rip-relative addresses.  We check this before checking direct
  // folding because RIP is preferable to non-RIP accesses.
  if (Subtarget->is64Bit() && N.getOpcode() == X86ISD::WrapperRIP &&
      // Under X86-64 non-small code model, GV (and friends) are 64-bits, so
      // they cannot be folded into immediate fields.
      // FIXME: This can be improved for kernel and other models?
      (M == CodeModel::Small || M == CodeModel::Kernel)) {
    // Base and index reg must be 0 in order to use %rip as base.
    if (AM.hasBaseOrIndexReg())
      return true;
    if (GlobalAddressSDNode *G = dyn_cast<GlobalAddressSDNode>(N0)) {
      X86ISelAddressMode Backup = AM;
      AM.GV = G->getGlobal();
      AM.SymbolFlags = G->getTargetFlags();
      if (foldOffsetIntoAddress(G->getOffset(), AM)) {
        AM = Backup;
        return true;
      }
    } else if (ConstantPoolSDNode *CP = dyn_cast<ConstantPoolSDNode>(N0)) {
      X86ISelAddressMode Backup = AM;
      AM.CP = CP->getConstVal();
      AM.Align = CP->getAlignment();
      AM.SymbolFlags = CP->getTargetFlags();
      if (foldOffsetIntoAddress(CP->getOffset(), AM)) {
        AM = Backup;
        return true;
      }
    } else if (ExternalSymbolSDNode *S = dyn_cast<ExternalSymbolSDNode>(N0)) {
      AM.ES = S->getSymbol();
      AM.SymbolFlags = S->getTargetFlags();
    } else if (auto *S = dyn_cast<MCSymbolSDNode>(N0)) {
      AM.MCSym = S->getMCSymbol();
    } else if (JumpTableSDNode *J = dyn_cast<JumpTableSDNode>(N0)) {
      AM.JT = J->getIndex();
      AM.SymbolFlags = J->getTargetFlags();
    } else if (BlockAddressSDNode *BA = dyn_cast<BlockAddressSDNode>(N0)) {
      X86ISelAddressMode Backup = AM;
      AM.BlockAddr = BA->getBlockAddress();
      AM.SymbolFlags = BA->getTargetFlags();
      if (foldOffsetIntoAddress(BA->getOffset(), AM)) {
        AM = Backup;
        return true;
      }
    } else
      llvm_unreachable("Unhandled symbol reference node.");

    if (N.getOpcode() == X86ISD::WrapperRIP)
      AM.setBaseReg(CurDAG->getRegister(X86::RIP, MVT::i64));
    return false;
  }

  // Handle the case when globals fit in our immediate field: This is true for
  // X86-32 always and X86-64 when in -mcmodel=small mode.  In 64-bit
  // mode, this only applies to a non-RIP-relative computation.
  if (!Subtarget->is64Bit() ||
      M == CodeModel::Small || M == CodeModel::Kernel) {
    assert(N.getOpcode() != X86ISD::WrapperRIP &&
           "RIP-relative addressing already handled");
    if (GlobalAddressSDNode *G = dyn_cast<GlobalAddressSDNode>(N0)) {
      AM.GV = G->getGlobal();
      AM.Disp += G->getOffset();
      AM.SymbolFlags = G->getTargetFlags();
    } else if (ConstantPoolSDNode *CP = dyn_cast<ConstantPoolSDNode>(N0)) {
      AM.CP = CP->getConstVal();
      AM.Align = CP->getAlignment();
      AM.Disp += CP->getOffset();
      AM.SymbolFlags = CP->getTargetFlags();
    } else if (ExternalSymbolSDNode *S = dyn_cast<ExternalSymbolSDNode>(N0)) {
      AM.ES = S->getSymbol();
      AM.SymbolFlags = S->getTargetFlags();
    } else if (auto *S = dyn_cast<MCSymbolSDNode>(N0)) {
      AM.MCSym = S->getMCSymbol();
    } else if (JumpTableSDNode *J = dyn_cast<JumpTableSDNode>(N0)) {
      AM.JT = J->getIndex();
      AM.SymbolFlags = J->getTargetFlags();
    } else if (BlockAddressSDNode *BA = dyn_cast<BlockAddressSDNode>(N0)) {
      AM.BlockAddr = BA->getBlockAddress();
      AM.Disp += BA->getOffset();
      AM.SymbolFlags = BA->getTargetFlags();
    } else
      llvm_unreachable("Unhandled symbol reference node.");
    return false;
  }

  return true;
}

/// Add the specified node to the specified addressing mode, returning true if
/// it cannot be done. This just pattern matches for the addressing mode.
bool X86DAGToDAGISel::matchAddress(SDValue N, X86ISelAddressMode &AM) {
  if (matchAddressRecursively(N, AM, 0))
    return true;

  // Post-processing: Convert lea(,%reg,2) to lea(%reg,%reg), which has
  // a smaller encoding and avoids a scaled-index.
  if (AM.Scale == 2 &&
      AM.BaseType == X86ISelAddressMode::RegBase &&
      AM.Base_Reg.getNode() == nullptr) {
    AM.Base_Reg = AM.IndexReg;
    AM.Scale = 1;
  }

  // Post-processing: Convert foo to foo(%rip), even in non-PIC mode,
  // because it has a smaller encoding.
  // TODO: Which other code models can use this?
  if (TM.getCodeModel() == CodeModel::Small &&
      Subtarget->is64Bit() &&
      AM.Scale == 1 &&
      AM.BaseType == X86ISelAddressMode::RegBase &&
      AM.Base_Reg.getNode() == nullptr &&
      AM.IndexReg.getNode() == nullptr &&
      AM.SymbolFlags == X86II::MO_NO_FLAG &&
      AM.hasSymbolicDisplacement())
    AM.Base_Reg = CurDAG->getRegister(X86::RIP, MVT::i64);

  return false;
}

bool X86DAGToDAGISel::matchAdd(SDValue N, X86ISelAddressMode &AM,
                               unsigned Depth) {
  // Add an artificial use to this node so that we can keep track of
  // it if it gets CSE'd with a different node.
  HandleSDNode Handle(N);

  X86ISelAddressMode Backup = AM;
  if (!matchAddressRecursively(N.getOperand(0), AM, Depth+1) &&
      !matchAddressRecursively(Handle.getValue().getOperand(1), AM, Depth+1))
    return false;
  AM = Backup;

  // Try again after commuting the operands.
  if (!matchAddressRecursively(Handle.getValue().getOperand(1), AM, Depth+1) &&
      !matchAddressRecursively(Handle.getValue().getOperand(0), AM, Depth+1))
    return false;
  AM = Backup;

  // If we couldn't fold both operands into the address at the same time,
  // see if we can just put each operand into a register and fold at least
  // the add.
  if (AM.BaseType == X86ISelAddressMode::RegBase &&
      !AM.Base_Reg.getNode() &&
      !AM.IndexReg.getNode()) {
    N = Handle.getValue();
    AM.Base_Reg = N.getOperand(0);
    AM.IndexReg = N.getOperand(1);
    AM.Scale = 1;
    return false;
  }
  N = Handle.getValue();
  return true;
}

// Insert a node into the DAG at least before the Pos node's position. This
// will reposition the node as needed, and will assign it a node ID that is <=
// the Pos node's ID. Note that this does *not* preserve the uniqueness of node
// IDs! The selection DAG must no longer depend on their uniqueness when this
// is used.
static void insertDAGNode(SelectionDAG &DAG, SDValue Pos, SDValue N) {
  if (N.getNode()->getNodeId() == -1 ||
      N.getNode()->getNodeId() > Pos.getNode()->getNodeId()) {
    DAG.RepositionNode(Pos.getNode()->getIterator(), N.getNode());
    N.getNode()->setNodeId(Pos.getNode()->getNodeId());
  }
}

// Transform "(X >> (8-C1)) & (0xff << C1)" to "((X >> 8) & 0xff) << C1" if
// safe. This allows us to convert the shift and and into an h-register
// extract and a scaled index. Returns false if the simplification is
// performed.
static bool foldMaskAndShiftToExtract(SelectionDAG &DAG, SDValue N,
                                      uint64_t Mask,
                                      SDValue Shift, SDValue X,
                                      X86ISelAddressMode &AM) {
  if (Shift.getOpcode() != ISD::SRL ||
      !isa<ConstantSDNode>(Shift.getOperand(1)) ||
      !Shift.hasOneUse())
    return true;

  int ScaleLog = 8 - Shift.getConstantOperandVal(1);
  if (ScaleLog <= 0 || ScaleLog >= 4 ||
      Mask != (0xffu << ScaleLog))
    return true;

  MVT VT = N.getSimpleValueType();
  SDLoc DL(N);
  SDValue Eight = DAG.getConstant(8, DL, MVT::i8);
  SDValue NewMask = DAG.getConstant(0xff, DL, VT);
  SDValue Srl = DAG.getNode(ISD::SRL, DL, VT, X, Eight);
  SDValue And = DAG.getNode(ISD::AND, DL, VT, Srl, NewMask);
  SDValue ShlCount = DAG.getConstant(ScaleLog, DL, MVT::i8);
  SDValue Shl = DAG.getNode(ISD::SHL, DL, VT, And, ShlCount);

  // Insert the new nodes into the topological ordering. We must do this in
  // a valid topological ordering as nothing is going to go back and re-sort
  // these nodes. We continually insert before 'N' in sequence as this is
  // essentially a pre-flattened and pre-sorted sequence of nodes. There is no
  // hierarchy left to express.
  insertDAGNode(DAG, N, Eight);
  insertDAGNode(DAG, N, Srl);
  insertDAGNode(DAG, N, NewMask);
  insertDAGNode(DAG, N, And);
  insertDAGNode(DAG, N, ShlCount);
  insertDAGNode(DAG, N, Shl);
  DAG.ReplaceAllUsesWith(N, Shl);
  AM.IndexReg = And;
  AM.Scale = (1 << ScaleLog);
  return false;
}

// Transforms "(X << C1) & C2" to "(X & (C2>>C1)) << C1" if safe and if this
// allows us to fold the shift into this addressing mode. Returns false if the
// transform succeeded.
static bool foldMaskedShiftToScaledMask(SelectionDAG &DAG, SDValue N,
                                        uint64_t Mask,
                                        SDValue Shift, SDValue X,
                                        X86ISelAddressMode &AM) {
  if (Shift.getOpcode() != ISD::SHL ||
      !isa<ConstantSDNode>(Shift.getOperand(1)))
    return true;

  // Not likely to be profitable if either the AND or SHIFT node has more
  // than one use (unless all uses are for address computation). Besides,
  // isel mechanism requires their node ids to be reused.
  if (!N.hasOneUse() || !Shift.hasOneUse())
    return true;

  // Verify that the shift amount is something we can fold.
  unsigned ShiftAmt = Shift.getConstantOperandVal(1);
  if (ShiftAmt != 1 && ShiftAmt != 2 && ShiftAmt != 3)
    return true;

  MVT VT = N.getSimpleValueType();
  SDLoc DL(N);
  SDValue NewMask = DAG.getConstant(Mask >> ShiftAmt, DL, VT);
  SDValue NewAnd = DAG.getNode(ISD::AND, DL, VT, X, NewMask);
  SDValue NewShift = DAG.getNode(ISD::SHL, DL, VT, NewAnd, Shift.getOperand(1));

  // Insert the new nodes into the topological ordering. We must do this in
  // a valid topological ordering as nothing is going to go back and re-sort
  // these nodes. We continually insert before 'N' in sequence as this is
  // essentially a pre-flattened and pre-sorted sequence of nodes. There is no
  // hierarchy left to express.
  insertDAGNode(DAG, N, NewMask);
  insertDAGNode(DAG, N, NewAnd);
  insertDAGNode(DAG, N, NewShift);
  DAG.ReplaceAllUsesWith(N, NewShift);

  AM.Scale = 1 << ShiftAmt;
  AM.IndexReg = NewAnd;
  return false;
}

// Implement some heroics to detect shifts of masked values where the mask can
// be replaced by extending the shift and undoing that in the addressing mode
// scale. Patterns such as (shl (srl x, c1), c2) are canonicalized into (and
// (srl x, SHIFT), MASK) by DAGCombines that don't know the shl can be done in
// the addressing mode. This results in code such as:
//
//   int f(short *y, int *lookup_table) {
//     ...
//     return *y + lookup_table[*y >> 11];
//   }
//
// Turning into:
//   movzwl (%rdi), %eax
//   movl %eax, %ecx
//   shrl $11, %ecx
//   addl (%rsi,%rcx,4), %eax
//
// Instead of:
//   movzwl (%rdi), %eax
//   movl %eax, %ecx
//   shrl $9, %ecx
//   andl $124, %rcx
//   addl (%rsi,%rcx), %eax
//
// Note that this function assumes the mask is provided as a mask *after* the
// value is shifted. The input chain may or may not match that, but computing
// such a mask is trivial.
static bool foldMaskAndShiftToScale(SelectionDAG &DAG, SDValue N,
                                    uint64_t Mask,
                                    SDValue Shift, SDValue X,
                                    X86ISelAddressMode &AM) {
  if (Shift.getOpcode() != ISD::SRL || !Shift.hasOneUse() ||
      !isa<ConstantSDNode>(Shift.getOperand(1)))
    return true;

  unsigned ShiftAmt = Shift.getConstantOperandVal(1);
  unsigned MaskLZ = countLeadingZeros(Mask);
  unsigned MaskTZ = countTrailingZeros(Mask);

  // The amount of shift we're trying to fit into the addressing mode is taken
  // from the trailing zeros of the mask.
  unsigned AMShiftAmt = MaskTZ;

  // There is nothing we can do here unless the mask is removing some bits.
  // Also, the addressing mode can only represent shifts of 1, 2, or 3 bits.
  if (AMShiftAmt <= 0 || AMShiftAmt > 3) return true;

  // We also need to ensure that mask is a continuous run of bits.
  if (countTrailingOnes(Mask >> MaskTZ) + MaskTZ + MaskLZ != 64) return true;

  // Scale the leading zero count down based on the actual size of the value.
  // Also scale it down based on the size of the shift.
  unsigned ScaleDown = (64 - X.getSimpleValueType().getSizeInBits()) + ShiftAmt;
  if (MaskLZ < ScaleDown)
    return true;
  MaskLZ -= ScaleDown;

  // The final check is to ensure that any masked out high bits of X are
  // already known to be zero. Otherwise, the mask has a semantic impact
  // other than masking out a couple of low bits. Unfortunately, because of
  // the mask, zero extensions will be removed from operands in some cases.
  // This code works extra hard to look through extensions because we can
  // replace them with zero extensions cheaply if necessary.
  bool ReplacingAnyExtend = false;
  if (X.getOpcode() == ISD::ANY_EXTEND) {
    unsigned ExtendBits = X.getSimpleValueType().getSizeInBits() -
                          X.getOperand(0).getSimpleValueType().getSizeInBits();
    // Assume that we'll replace the any-extend with a zero-extend, and
    // narrow the search to the extended value.
    X = X.getOperand(0);
    MaskLZ = ExtendBits > MaskLZ ? 0 : MaskLZ - ExtendBits;
    ReplacingAnyExtend = true;
  }
  APInt MaskedHighBits =
    APInt::getHighBitsSet(X.getSimpleValueType().getSizeInBits(), MaskLZ);
  KnownBits Known;
  DAG.computeKnownBits(X, Known);
  if (MaskedHighBits != Known.Zero) return true;

  // We've identified a pattern that can be transformed into a single shift
  // and an addressing mode. Make it so.
  MVT VT = N.getSimpleValueType();
  if (ReplacingAnyExtend) {
    assert(X.getValueType() != VT);
    // We looked through an ANY_EXTEND node, insert a ZERO_EXTEND.
    SDValue NewX = DAG.getNode(ISD::ZERO_EXTEND, SDLoc(X), VT, X);
    insertDAGNode(DAG, N, NewX);
    X = NewX;
  }
  SDLoc DL(N);
  SDValue NewSRLAmt = DAG.getConstant(ShiftAmt + AMShiftAmt, DL, MVT::i8);
  SDValue NewSRL = DAG.getNode(ISD::SRL, DL, VT, X, NewSRLAmt);
  SDValue NewSHLAmt = DAG.getConstant(AMShiftAmt, DL, MVT::i8);
  SDValue NewSHL = DAG.getNode(ISD::SHL, DL, VT, NewSRL, NewSHLAmt);

  // Insert the new nodes into the topological ordering. We must do this in
  // a valid topological ordering as nothing is going to go back and re-sort
  // these nodes. We continually insert before 'N' in sequence as this is
  // essentially a pre-flattened and pre-sorted sequence of nodes. There is no
  // hierarchy left to express.
  insertDAGNode(DAG, N, NewSRLAmt);
  insertDAGNode(DAG, N, NewSRL);
  insertDAGNode(DAG, N, NewSHLAmt);
  insertDAGNode(DAG, N, NewSHL);
  DAG.ReplaceAllUsesWith(N, NewSHL);

  AM.Scale = 1 << AMShiftAmt;
  AM.IndexReg = NewSRL;
  return false;
}

bool X86DAGToDAGISel::matchAddressRecursively(SDValue N, X86ISelAddressMode &AM,
                                              unsigned Depth) {
  SDLoc dl(N);
  DEBUG({
      dbgs() << "MatchAddress: ";
      AM.dump();
    });
  // Limit recursion.
  if (Depth > 5)
    return matchAddressBase(N, AM);

  // If this is already a %rip relative address, we can only merge immediates
  // into it.  Instead of handling this in every case, we handle it here.
  // RIP relative addressing: %rip + 32-bit displacement!
  if (AM.isRIPRelative()) {
    // FIXME: JumpTable and ExternalSymbol address currently don't like
    // displacements.  It isn't very important, but this should be fixed for
    // consistency.
    if (!(AM.ES || AM.MCSym) && AM.JT != -1)
      return true;

    if (ConstantSDNode *Cst = dyn_cast<ConstantSDNode>(N))
      if (!foldOffsetIntoAddress(Cst->getSExtValue(), AM))
        return false;
    return true;
  }

  switch (N.getOpcode()) {
  default: break;
  case ISD::LOCAL_RECOVER: {
    if (!AM.hasSymbolicDisplacement() && AM.Disp == 0)
      if (const auto *ESNode = dyn_cast<MCSymbolSDNode>(N.getOperand(0))) {
        // Use the symbol and don't prefix it.
        AM.MCSym = ESNode->getMCSymbol();
        return false;
      }
    break;
  }
  case ISD::Constant: {
    uint64_t Val = cast<ConstantSDNode>(N)->getSExtValue();
    if (!foldOffsetIntoAddress(Val, AM))
      return false;
    break;
  }

  case X86ISD::Wrapper:
  case X86ISD::WrapperRIP:
    if (!matchWrapper(N, AM))
      return false;
    break;

  case ISD::LOAD:
    if (!matchLoadInAddress(cast<LoadSDNode>(N), AM))
      return false;
    break;

  case ISD::FrameIndex:
    if (AM.BaseType == X86ISelAddressMode::RegBase &&
        AM.Base_Reg.getNode() == nullptr &&
        (!Subtarget->is64Bit() || isDispSafeForFrameIndex(AM.Disp))) {
      AM.BaseType = X86ISelAddressMode::FrameIndexBase;
      AM.Base_FrameIndex = cast<FrameIndexSDNode>(N)->getIndex();
      return false;
    }
    break;

  case ISD::SHL:
    if (AM.IndexReg.getNode() != nullptr || AM.Scale != 1)
      break;

    if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(N.getOperand(1))) {
      unsigned Val = CN->getZExtValue();
      // Note that we handle x<<1 as (,x,2) rather than (x,x) here so
      // that the base operand remains free for further matching. If
      // the base doesn't end up getting used, a post-processing step
      // in MatchAddress turns (,x,2) into (x,x), which is cheaper.
      if (Val == 1 || Val == 2 || Val == 3) {
        AM.Scale = 1 << Val;
        SDValue ShVal = N.getOperand(0);

        // Okay, we know that we have a scale by now.  However, if the scaled
        // value is an add of something and a constant, we can fold the
        // constant into the disp field here.
        if (CurDAG->isBaseWithConstantOffset(ShVal)) {
          AM.IndexReg = ShVal.getOperand(0);
          ConstantSDNode *AddVal = cast<ConstantSDNode>(ShVal.getOperand(1));
          uint64_t Disp = (uint64_t)AddVal->getSExtValue() << Val;
          if (!foldOffsetIntoAddress(Disp, AM))
            return false;
        }

        AM.IndexReg = ShVal;
        return false;
      }
    }
    break;

  case ISD::SRL: {
    // Scale must not be used already.
    if (AM.IndexReg.getNode() != nullptr || AM.Scale != 1) break;

    SDValue And = N.getOperand(0);
    if (And.getOpcode() != ISD::AND) break;
    SDValue X = And.getOperand(0);

    // We only handle up to 64-bit values here as those are what matter for
    // addressing mode optimizations.
    if (X.getSimpleValueType().getSizeInBits() > 64) break;

    // The mask used for the transform is expected to be post-shift, but we
    // found the shift first so just apply the shift to the mask before passing
    // it down.
    if (!isa<ConstantSDNode>(N.getOperand(1)) ||
        !isa<ConstantSDNode>(And.getOperand(1)))
      break;
    uint64_t Mask = And.getConstantOperandVal(1) >> N.getConstantOperandVal(1);

    // Try to fold the mask and shift into the scale, and return false if we
    // succeed.
    if (!foldMaskAndShiftToScale(*CurDAG, N, Mask, N, X, AM))
      return false;
    break;
  }

  case ISD::SMUL_LOHI:
  case ISD::UMUL_LOHI:
    // A mul_lohi where we need the low part can be folded as a plain multiply.
    if (N.getResNo() != 0) break;
    LLVM_FALLTHROUGH;
  case ISD::MUL:
  case X86ISD::MUL_IMM:
    // X*[3,5,9] -> X+X*[2,4,8]
    if (AM.BaseType == X86ISelAddressMode::RegBase &&
        AM.Base_Reg.getNode() == nullptr &&
        AM.IndexReg.getNode() == nullptr) {
      if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(N.getOperand(1)))
        if (CN->getZExtValue() == 3 || CN->getZExtValue() == 5 ||
            CN->getZExtValue() == 9) {
          AM.Scale = unsigned(CN->getZExtValue())-1;

          SDValue MulVal = N.getOperand(0);
          SDValue Reg;

          // Okay, we know that we have a scale by now.  However, if the scaled
          // value is an add of something and a constant, we can fold the
          // constant into the disp field here.
          if (MulVal.getNode()->getOpcode() == ISD::ADD && MulVal.hasOneUse() &&
              isa<ConstantSDNode>(MulVal.getOperand(1))) {
            Reg = MulVal.getOperand(0);
            ConstantSDNode *AddVal =
              cast<ConstantSDNode>(MulVal.getOperand(1));
            uint64_t Disp = AddVal->getSExtValue() * CN->getZExtValue();
            if (foldOffsetIntoAddress(Disp, AM))
              Reg = N.getOperand(0);
          } else {
            Reg = N.getOperand(0);
          }

          AM.IndexReg = AM.Base_Reg = Reg;
          return false;
        }
    }
    break;

  case ISD::SUB: {
    // Given A-B, if A can be completely folded into the address and
    // the index field with the index field unused, use -B as the index.
    // This is a win if a has multiple parts that can be folded into
    // the address. Also, this saves a mov if the base register has
    // other uses, since it avoids a two-address sub instruction, however
    // it costs an additional mov if the index register has other uses.

    // Add an artificial use to this node so that we can keep track of
    // it if it gets CSE'd with a different node.
    HandleSDNode Handle(N);

    // Test if the LHS of the sub can be folded.
    X86ISelAddressMode Backup = AM;
    if (matchAddressRecursively(N.getOperand(0), AM, Depth+1)) {
      AM = Backup;
      break;
    }
    // Test if the index field is free for use.
    if (AM.IndexReg.getNode() || AM.isRIPRelative()) {
      AM = Backup;
      break;
    }

    int Cost = 0;
    SDValue RHS = Handle.getValue().getOperand(1);
    // If the RHS involves a register with multiple uses, this
    // transformation incurs an extra mov, due to the neg instruction
    // clobbering its operand.
    if (!RHS.getNode()->hasOneUse() ||
        RHS.getNode()->getOpcode() == ISD::CopyFromReg ||
        RHS.getNode()->getOpcode() == ISD::TRUNCATE ||
        RHS.getNode()->getOpcode() == ISD::ANY_EXTEND ||
        (RHS.getNode()->getOpcode() == ISD::ZERO_EXTEND &&
         RHS.getOperand(0).getValueType() == MVT::i32))
      ++Cost;
    // If the base is a register with multiple uses, this
    // transformation may save a mov.
    // FIXME: Don't rely on DELETED_NODEs.
    if ((AM.BaseType == X86ISelAddressMode::RegBase && AM.Base_Reg.getNode() &&
         AM.Base_Reg->getOpcode() != ISD::DELETED_NODE &&
         !AM.Base_Reg.getNode()->hasOneUse()) ||
        AM.BaseType == X86ISelAddressMode::FrameIndexBase)
      --Cost;
    // If the folded LHS was interesting, this transformation saves
    // address arithmetic.
    if ((AM.hasSymbolicDisplacement() && !Backup.hasSymbolicDisplacement()) +
        ((AM.Disp != 0) && (Backup.Disp == 0)) +
        (AM.Segment.getNode() && !Backup.Segment.getNode()) >= 2)
      --Cost;
    // If it doesn't look like it may be an overall win, don't do it.
    if (Cost >= 0) {
      AM = Backup;
      break;
    }

    // Ok, the transformation is legal and appears profitable. Go for it.
    SDValue Zero = CurDAG->getConstant(0, dl, N.getValueType());
    SDValue Neg = CurDAG->getNode(ISD::SUB, dl, N.getValueType(), Zero, RHS);
    AM.IndexReg = Neg;
    AM.Scale = 1;

    // Insert the new nodes into the topological ordering.
    insertDAGNode(*CurDAG, Handle.getValue(), Zero);
    insertDAGNode(*CurDAG, Handle.getValue(), Neg);
    return false;
  }

  case ISD::ADD:
    if (!matchAdd(N, AM, Depth))
      return false;
    break;

  case ISD::OR:
    // We want to look through a transform in InstCombine and DAGCombiner that
    // turns 'add' into 'or', so we can treat this 'or' exactly like an 'add'.
    // Example: (or (and x, 1), (shl y, 3)) --> (add (and x, 1), (shl y, 3))
    // An 'lea' can then be used to match the shift (multiply) and add:
    // and $1, %esi
    // lea (%rsi, %rdi, 8), %rax
    if (CurDAG->haveNoCommonBitsSet(N.getOperand(0), N.getOperand(1)) &&
        !matchAdd(N, AM, Depth))
      return false;
    break;

  case ISD::AND: {
    // Perform some heroic transforms on an and of a constant-count shift
    // with a constant to enable use of the scaled offset field.

    // Scale must not be used already.
    if (AM.IndexReg.getNode() != nullptr || AM.Scale != 1) break;

    SDValue Shift = N.getOperand(0);
    if (Shift.getOpcode() != ISD::SRL && Shift.getOpcode() != ISD::SHL) break;
    SDValue X = Shift.getOperand(0);

    // We only handle up to 64-bit values here as those are what matter for
    // addressing mode optimizations.
    if (X.getSimpleValueType().getSizeInBits() > 64) break;

    if (!isa<ConstantSDNode>(N.getOperand(1)))
      break;
    uint64_t Mask = N.getConstantOperandVal(1);

    // Try to fold the mask and shift into an extract and scale.
    if (!foldMaskAndShiftToExtract(*CurDAG, N, Mask, Shift, X, AM))
      return false;

    // Try to fold the mask and shift directly into the scale.
    if (!foldMaskAndShiftToScale(*CurDAG, N, Mask, Shift, X, AM))
      return false;

    // Try to swap the mask and shift to place shifts which can be done as
    // a scale on the outside of the mask.
    if (!foldMaskedShiftToScaledMask(*CurDAG, N, Mask, Shift, X, AM))
      return false;
    break;
  }
  }

  return matchAddressBase(N, AM);
}

/// Helper for MatchAddress. Add the specified node to the
/// specified addressing mode without any further recursion.
bool X86DAGToDAGISel::matchAddressBase(SDValue N, X86ISelAddressMode &AM) {
  // Is the base register already occupied?
  if (AM.BaseType != X86ISelAddressMode::RegBase || AM.Base_Reg.getNode()) {
    // If so, check to see if the scale index register is set.
    if (!AM.IndexReg.getNode()) {
      AM.IndexReg = N;
      AM.Scale = 1;
      return false;
    }

    // Otherwise, we cannot select it.
    return true;
  }

  // Default, generate it as a register.
  AM.BaseType = X86ISelAddressMode::RegBase;
  AM.Base_Reg = N;
  return false;
}

template <class GatherScatterSDNode>
bool X86DAGToDAGISel::selectAddrOfGatherScatterNode(
    GatherScatterSDNode *Mgs, SDValue N, SDValue &Base, SDValue &Scale,
    SDValue &Index, SDValue &Disp, SDValue &Segment) {
  X86ISelAddressMode AM;
  unsigned AddrSpace = Mgs->getPointerInfo().getAddrSpace();
  // AddrSpace 256 -> GS, 257 -> FS, 258 -> SS.
  if (AddrSpace == 256)
    AM.Segment = CurDAG->getRegister(X86::GS, MVT::i16);
  if (AddrSpace == 257)
    AM.Segment = CurDAG->getRegister(X86::FS, MVT::i16);
  if (AddrSpace == 258)
    AM.Segment = CurDAG->getRegister(X86::SS, MVT::i16);

  SDLoc DL(N);
  Base = Mgs->getBasePtr();
  Index = Mgs->getIndex();
  unsigned ScalarSize = Mgs->getValue().getScalarValueSizeInBits();
  Scale = getI8Imm(ScalarSize/8, DL);

  // If Base is 0, the whole address is in index and the Scale is 1
  if (isa<ConstantSDNode>(Base)) {
    assert(cast<ConstantSDNode>(Base)->isNullValue() &&
           "Unexpected base in gather/scatter");
    Scale = getI8Imm(1, DL);
    Base = CurDAG->getRegister(0, MVT::i32);
  }
  if (AM.Segment.getNode())
    Segment = AM.Segment;
  else
    Segment = CurDAG->getRegister(0, MVT::i32);
  Disp = CurDAG->getTargetConstant(0, DL, MVT::i32);
  return true;
}

bool X86DAGToDAGISel::selectVectorAddr(SDNode *Parent, SDValue N, SDValue &Base,
                                       SDValue &Scale, SDValue &Index,
                                       SDValue &Disp, SDValue &Segment) {
  if (auto Mgs = dyn_cast<MaskedGatherScatterSDNode>(Parent))
    return selectAddrOfGatherScatterNode<MaskedGatherScatterSDNode>(
        Mgs, N, Base, Scale, Index, Disp, Segment);
  if (auto X86Gather = dyn_cast<X86MaskedGatherSDNode>(Parent))
    return selectAddrOfGatherScatterNode<X86MaskedGatherSDNode>(
        X86Gather, N, Base, Scale, Index, Disp, Segment);
  return false;
}

/// Returns true if it is able to pattern match an addressing mode.
/// It returns the operands which make up the maximal addressing mode it can
/// match by reference.
///
/// Parent is the parent node of the addr operand that is being matched.  It
/// is always a load, store, atomic node, or null.  It is only null when
/// checking memory operands for inline asm nodes.
bool X86DAGToDAGISel::selectAddr(SDNode *Parent, SDValue N, SDValue &Base,
                                 SDValue &Scale, SDValue &Index,
                                 SDValue &Disp, SDValue &Segment) {
  X86ISelAddressMode AM;

  if (Parent &&
      // This list of opcodes are all the nodes that have an "addr:$ptr" operand
      // that are not a MemSDNode, and thus don't have proper addrspace info.
      Parent->getOpcode() != ISD::INTRINSIC_W_CHAIN && // unaligned loads, fixme
      Parent->getOpcode() != ISD::INTRINSIC_VOID && // nontemporal stores
      Parent->getOpcode() != X86ISD::TLSCALL && // Fixme
      Parent->getOpcode() != X86ISD::EH_SJLJ_SETJMP && // setjmp
      Parent->getOpcode() != X86ISD::EH_SJLJ_LONGJMP) { // longjmp
    unsigned AddrSpace =
      cast<MemSDNode>(Parent)->getPointerInfo().getAddrSpace();
    // AddrSpace 256 -> GS, 257 -> FS, 258 -> SS.
    if (AddrSpace == 256)
      AM.Segment = CurDAG->getRegister(X86::GS, MVT::i16);
    if (AddrSpace == 257)
      AM.Segment = CurDAG->getRegister(X86::FS, MVT::i16);
    if (AddrSpace == 258)
      AM.Segment = CurDAG->getRegister(X86::SS, MVT::i16);
  }

  if (matchAddress(N, AM))
    return false;

  MVT VT = N.getSimpleValueType();
  if (AM.BaseType == X86ISelAddressMode::RegBase) {
    if (!AM.Base_Reg.getNode())
      AM.Base_Reg = CurDAG->getRegister(0, VT);
  }

  if (!AM.IndexReg.getNode())
    AM.IndexReg = CurDAG->getRegister(0, VT);

  getAddressOperands(AM, SDLoc(N), Base, Scale, Index, Disp, Segment);
  return true;
}

// We can only fold a load if all nodes between it and the root node have a
// single use. If there are additional uses, we could end up duplicating the
// load.
static bool hasSingleUsesFromRoot(SDNode *Root, SDNode *N) {
  SDNode *User = *N->use_begin();
  while (User != Root) {
    if (!User->hasOneUse())
      return false;
    User = *User->use_begin();
  }

  return true;
}

/// Match a scalar SSE load. In particular, we want to match a load whose top
/// elements are either undef or zeros. The load flavor is derived from the
/// type of N, which is either v4f32 or v2f64.
///
/// We also return:
///   PatternChainNode: this is the matched node that has a chain input and
///   output.
bool X86DAGToDAGISel::selectScalarSSELoad(SDNode *Root,
                                          SDValue N, SDValue &Base,
                                          SDValue &Scale, SDValue &Index,
                                          SDValue &Disp, SDValue &Segment,
                                          SDValue &PatternNodeWithChain) {
  // We can allow a full vector load here since narrowing a load is ok.
  if (ISD::isNON_EXTLoad(N.getNode())) {
    PatternNodeWithChain = N;
    if (IsProfitableToFold(PatternNodeWithChain, N.getNode(), Root) &&
        IsLegalToFold(PatternNodeWithChain, *N->use_begin(), Root, OptLevel) &&
        hasSingleUsesFromRoot(Root, N.getNode())) {
      LoadSDNode *LD = cast<LoadSDNode>(PatternNodeWithChain);
      return selectAddr(LD, LD->getBasePtr(), Base, Scale, Index, Disp,
                        Segment);
    }
  }

  // We can also match the special zero extended load opcode.
  if (N.getOpcode() == X86ISD::VZEXT_LOAD) {
    PatternNodeWithChain = N;
    if (IsProfitableToFold(PatternNodeWithChain, N.getNode(), Root) &&
        IsLegalToFold(PatternNodeWithChain, *N->use_begin(), Root, OptLevel) &&
        hasSingleUsesFromRoot(Root, N.getNode())) {
      auto *MI = cast<MemIntrinsicSDNode>(PatternNodeWithChain);
      return selectAddr(MI, MI->getBasePtr(), Base, Scale, Index, Disp,
                        Segment);
    }
  }

  // Need to make sure that the SCALAR_TO_VECTOR and load are both only used
  // once. Otherwise the load might get duplicated and the chain output of the
  // duplicate load will not be observed by all dependencies.
  if (N.getOpcode() == ISD::SCALAR_TO_VECTOR && N.getNode()->hasOneUse()) {
    PatternNodeWithChain = N.getOperand(0);
    if (ISD::isNON_EXTLoad(PatternNodeWithChain.getNode()) &&
        IsProfitableToFold(PatternNodeWithChain, N.getNode(), Root) &&
        IsLegalToFold(PatternNodeWithChain, N.getNode(), Root, OptLevel) &&
        hasSingleUsesFromRoot(Root, N.getNode())) {
      LoadSDNode *LD = cast<LoadSDNode>(PatternNodeWithChain);
      return selectAddr(LD, LD->getBasePtr(), Base, Scale, Index, Disp,
                        Segment);
    }
  }

  // Also handle the case where we explicitly require zeros in the top
  // elements.  This is a vector shuffle from the zero vector.
  if (N.getOpcode() == X86ISD::VZEXT_MOVL && N.getNode()->hasOneUse() &&
      // Check to see if the top elements are all zeros (or bitcast of zeros).
      N.getOperand(0).getOpcode() == ISD::SCALAR_TO_VECTOR &&
      N.getOperand(0).getNode()->hasOneUse()) {
    PatternNodeWithChain = N.getOperand(0).getOperand(0);
    if (ISD::isNON_EXTLoad(PatternNodeWithChain.getNode()) &&
        IsProfitableToFold(PatternNodeWithChain, N.getNode(), Root) &&
        IsLegalToFold(PatternNodeWithChain, N.getNode(), Root, OptLevel) &&
        hasSingleUsesFromRoot(Root, N.getNode())) {
      // Okay, this is a zero extending load.  Fold it.
      LoadSDNode *LD = cast<LoadSDNode>(PatternNodeWithChain);
      return selectAddr(LD, LD->getBasePtr(), Base, Scale, Index, Disp,
                        Segment);
    }
  }

  return false;
}


bool X86DAGToDAGISel::selectMOV64Imm32(SDValue N, SDValue &Imm) {
  if (const ConstantSDNode *CN = dyn_cast<ConstantSDNode>(N)) {
    uint64_t ImmVal = CN->getZExtValue();
    if (!isUInt<32>(ImmVal))
      return false;

    Imm = CurDAG->getTargetConstant(ImmVal, SDLoc(N), MVT::i64);
    return true;
  }

  // In static codegen with small code model, we can get the address of a label
  // into a register with 'movl'. TableGen has already made sure we're looking
  // at a label of some kind.
  assert(N->getOpcode() == X86ISD::Wrapper &&
         "Unexpected node type for MOV32ri64");
  N = N.getOperand(0);

  // At least GNU as does not accept 'movl' for TPOFF relocations.
  // FIXME: We could use 'movl' when we know we are targeting MC.
  if (N->getOpcode() == ISD::TargetGlobalTLSAddress)
    return false;

  Imm = N;
  if (N->getOpcode() != ISD::TargetGlobalAddress)
    return TM.getCodeModel() == CodeModel::Small;

  Optional<ConstantRange> CR =
      cast<GlobalAddressSDNode>(N)->getGlobal()->getAbsoluteSymbolRange();
  if (!CR)
    return TM.getCodeModel() == CodeModel::Small;

  return CR->getUnsignedMax().ult(1ull << 32);
}

bool X86DAGToDAGISel::selectLEA64_32Addr(SDValue N, SDValue &Base,
                                         SDValue &Scale, SDValue &Index,
                                         SDValue &Disp, SDValue &Segment) {
  // Save the debug loc before calling selectLEAAddr, in case it invalidates N.
  SDLoc DL(N);

  if (!selectLEAAddr(N, Base, Scale, Index, Disp, Segment))
    return false;

  RegisterSDNode *RN = dyn_cast<RegisterSDNode>(Base);
  if (RN && RN->getReg() == 0)
    Base = CurDAG->getRegister(0, MVT::i64);
  else if (Base.getValueType() == MVT::i32 && !dyn_cast<FrameIndexSDNode>(Base)) {
    // Base could already be %rip, particularly in the x32 ABI.
    Base = SDValue(CurDAG->getMachineNode(
                       TargetOpcode::SUBREG_TO_REG, DL, MVT::i64,
                       CurDAG->getTargetConstant(0, DL, MVT::i64),
                       Base,
                       CurDAG->getTargetConstant(X86::sub_32bit, DL, MVT::i32)),
                   0);
  }

  RN = dyn_cast<RegisterSDNode>(Index);
  if (RN && RN->getReg() == 0)
    Index = CurDAG->getRegister(0, MVT::i64);
  else {
    assert(Index.getValueType() == MVT::i32 &&
           "Expect to be extending 32-bit registers for use in LEA");
    Index = SDValue(CurDAG->getMachineNode(
                        TargetOpcode::SUBREG_TO_REG, DL, MVT::i64,
                        CurDAG->getTargetConstant(0, DL, MVT::i64),
                        Index,
                        CurDAG->getTargetConstant(X86::sub_32bit, DL,
                                                  MVT::i32)),
                    0);
  }

  return true;
}

/// Calls SelectAddr and determines if the maximal addressing
/// mode it matches can be cost effectively emitted as an LEA instruction.
bool X86DAGToDAGISel::selectLEAAddr(SDValue N,
                                    SDValue &Base, SDValue &Scale,
                                    SDValue &Index, SDValue &Disp,
                                    SDValue &Segment) {
  X86ISelAddressMode AM;

  // Save the DL and VT before calling matchAddress, it can invalidate N.
  SDLoc DL(N);
  MVT VT = N.getSimpleValueType();

  // Set AM.Segment to prevent MatchAddress from using one. LEA doesn't support
  // segments.
  SDValue Copy = AM.Segment;
  SDValue T = CurDAG->getRegister(0, MVT::i32);
  AM.Segment = T;
  if (matchAddress(N, AM))
    return false;
  assert (T == AM.Segment);
  AM.Segment = Copy;

  unsigned Complexity = 0;
  if (AM.BaseType == X86ISelAddressMode::RegBase)
    if (AM.Base_Reg.getNode())
      Complexity = 1;
    else
      AM.Base_Reg = CurDAG->getRegister(0, VT);
  else if (AM.BaseType == X86ISelAddressMode::FrameIndexBase)
    Complexity = 4;

  if (AM.IndexReg.getNode())
    Complexity++;
  else
    AM.IndexReg = CurDAG->getRegister(0, VT);

  // Don't match just leal(,%reg,2). It's cheaper to do addl %reg, %reg, or with
  // a simple shift.
  if (AM.Scale > 1)
    Complexity++;

  // FIXME: We are artificially lowering the criteria to turn ADD %reg, $GA
  // to a LEA. This is determined with some experimentation but is by no means
  // optimal (especially for code size consideration). LEA is nice because of
  // its three-address nature. Tweak the cost function again when we can run
  // convertToThreeAddress() at register allocation time.
  if (AM.hasSymbolicDisplacement()) {
    // For X86-64, always use LEA to materialize RIP-relative addresses.
    if (Subtarget->is64Bit())
      Complexity = 4;
    else
      Complexity += 2;
  }

  if (AM.Disp && (AM.Base_Reg.getNode() || AM.IndexReg.getNode()))
    Complexity++;

  // If it isn't worth using an LEA, reject it.
  if (Complexity <= 2)
    return false;

  getAddressOperands(AM, DL, Base, Scale, Index, Disp, Segment);
  return true;
}

/// This is only run on TargetGlobalTLSAddress nodes.
bool X86DAGToDAGISel::selectTLSADDRAddr(SDValue N, SDValue &Base,
                                        SDValue &Scale, SDValue &Index,
                                        SDValue &Disp, SDValue &Segment) {
  assert(N.getOpcode() == ISD::TargetGlobalTLSAddress);
  const GlobalAddressSDNode *GA = cast<GlobalAddressSDNode>(N);

  X86ISelAddressMode AM;
  AM.GV = GA->getGlobal();
  AM.Disp += GA->getOffset();
  AM.Base_Reg = CurDAG->getRegister(0, N.getValueType());
  AM.SymbolFlags = GA->getTargetFlags();

  if (N.getValueType() == MVT::i32) {
    AM.Scale = 1;
    AM.IndexReg = CurDAG->getRegister(X86::EBX, MVT::i32);
  } else {
    AM.IndexReg = CurDAG->getRegister(0, MVT::i64);
  }

  getAddressOperands(AM, SDLoc(N), Base, Scale, Index, Disp, Segment);
  return true;
}

bool X86DAGToDAGISel::selectRelocImm(SDValue N, SDValue &Op) {
  if (auto *CN = dyn_cast<ConstantSDNode>(N)) {
    Op = CurDAG->getTargetConstant(CN->getAPIntValue(), SDLoc(CN),
                                   N.getValueType());
    return true;
  }

  // Keep track of the original value type and whether this value was
  // truncated. If we see a truncation from pointer type to VT that truncates
  // bits that are known to be zero, we can use a narrow reference.
  EVT VT = N.getValueType();
  bool WasTruncated = false;
  if (N.getOpcode() == ISD::TRUNCATE) {
    WasTruncated = true;
    N = N.getOperand(0);
  }

  if (N.getOpcode() != X86ISD::Wrapper)
    return false;

  // We can only use non-GlobalValues as immediates if they were not truncated,
  // as we do not have any range information. If we have a GlobalValue and the
  // address was not truncated, we can select it as an operand directly.
  unsigned Opc = N.getOperand(0)->getOpcode();
  if (Opc != ISD::TargetGlobalAddress || !WasTruncated) {
    Op = N.getOperand(0);
    // We can only select the operand directly if we didn't have to look past a
    // truncate.
    return !WasTruncated;
  }

  // Check that the global's range fits into VT.
  auto *GA = cast<GlobalAddressSDNode>(N.getOperand(0));
  Optional<ConstantRange> CR = GA->getGlobal()->getAbsoluteSymbolRange();
  if (!CR || CR->getUnsignedMax().uge(1ull << VT.getSizeInBits()))
    return false;

  // Okay, we can use a narrow reference.
  Op = CurDAG->getTargetGlobalAddress(GA->getGlobal(), SDLoc(N), VT,
                                      GA->getOffset(), GA->getTargetFlags());
  return true;
}

bool X86DAGToDAGISel::tryFoldLoad(SDNode *P, SDValue N,
                                  SDValue &Base, SDValue &Scale,
                                  SDValue &Index, SDValue &Disp,
                                  SDValue &Segment) {
  if (!ISD::isNON_EXTLoad(N.getNode()) ||
      !IsProfitableToFold(N, P, P) ||
      !IsLegalToFold(N, P, P, OptLevel))
    return false;

  return selectAddr(N.getNode(),
                    N.getOperand(1), Base, Scale, Index, Disp, Segment);
}

/// Return an SDNode that returns the value of the global base register.
/// Output instructions required to initialize the global base register,
/// if necessary.
SDNode *X86DAGToDAGISel::getGlobalBaseReg() {
  unsigned GlobalBaseReg = getInstrInfo()->getGlobalBaseReg(MF);
  auto &DL = MF->getDataLayout();
  return CurDAG->getRegister(GlobalBaseReg, TLI->getPointerTy(DL)).getNode();
}

bool X86DAGToDAGISel::isSExtAbsoluteSymbolRef(unsigned Width, SDNode *N) const {
  if (N->getOpcode() == ISD::TRUNCATE)
    N = N->getOperand(0).getNode();
  if (N->getOpcode() != X86ISD::Wrapper)
    return false;

  auto *GA = dyn_cast<GlobalAddressSDNode>(N->getOperand(0));
  if (!GA)
    return false;

  Optional<ConstantRange> CR = GA->getGlobal()->getAbsoluteSymbolRange();
  return CR && CR->getSignedMin().sge(-1ull << Width) &&
         CR->getSignedMax().slt(1ull << Width);
}

/// Test whether the given X86ISD::CMP node has any uses which require the SF
/// or OF bits to be accurate.
static bool hasNoSignedComparisonUses(SDNode *N) {
  // Examine each user of the node.
  for (SDNode::use_iterator UI = N->use_begin(),
         UE = N->use_end(); UI != UE; ++UI) {
    // Only examine CopyToReg uses.
    if (UI->getOpcode() != ISD::CopyToReg)
      return false;
    // Only examine CopyToReg uses that copy to EFLAGS.
    if (cast<RegisterSDNode>(UI->getOperand(1))->getReg() !=
          X86::EFLAGS)
      return false;
    // Examine each user of the CopyToReg use.
    for (SDNode::use_iterator FlagUI = UI->use_begin(),
           FlagUE = UI->use_end(); FlagUI != FlagUE; ++FlagUI) {
      // Only examine the Flag result.
      if (FlagUI.getUse().getResNo() != 1) continue;
      // Anything unusual: assume conservatively.
      if (!FlagUI->isMachineOpcode()) return false;
      // Examine the opcode of the user.
      switch (FlagUI->getMachineOpcode()) {
      // These comparisons don't treat the most significant bit specially.
      case X86::SETAr: case X86::SETAEr: case X86::SETBr: case X86::SETBEr:
      case X86::SETEr: case X86::SETNEr: case X86::SETPr: case X86::SETNPr:
      case X86::SETAm: case X86::SETAEm: case X86::SETBm: case X86::SETBEm:
      case X86::SETEm: case X86::SETNEm: case X86::SETPm: case X86::SETNPm:
      case X86::JA_1: case X86::JAE_1: case X86::JB_1: case X86::JBE_1:
      case X86::JE_1: case X86::JNE_1: case X86::JP_1: case X86::JNP_1:
      case X86::CMOVA16rr: case X86::CMOVA16rm:
      case X86::CMOVA32rr: case X86::CMOVA32rm:
      case X86::CMOVA64rr: case X86::CMOVA64rm:
      case X86::CMOVAE16rr: case X86::CMOVAE16rm:
      case X86::CMOVAE32rr: case X86::CMOVAE32rm:
      case X86::CMOVAE64rr: case X86::CMOVAE64rm:
      case X86::CMOVB16rr: case X86::CMOVB16rm:
      case X86::CMOVB32rr: case X86::CMOVB32rm:
      case X86::CMOVB64rr: case X86::CMOVB64rm:
      case X86::CMOVBE16rr: case X86::CMOVBE16rm:
      case X86::CMOVBE32rr: case X86::CMOVBE32rm:
      case X86::CMOVBE64rr: case X86::CMOVBE64rm:
      case X86::CMOVE16rr: case X86::CMOVE16rm:
      case X86::CMOVE32rr: case X86::CMOVE32rm:
      case X86::CMOVE64rr: case X86::CMOVE64rm:
      case X86::CMOVNE16rr: case X86::CMOVNE16rm:
      case X86::CMOVNE32rr: case X86::CMOVNE32rm:
      case X86::CMOVNE64rr: case X86::CMOVNE64rm:
      case X86::CMOVNP16rr: case X86::CMOVNP16rm:
      case X86::CMOVNP32rr: case X86::CMOVNP32rm:
      case X86::CMOVNP64rr: case X86::CMOVNP64rm:
      case X86::CMOVP16rr: case X86::CMOVP16rm:
      case X86::CMOVP32rr: case X86::CMOVP32rm:
      case X86::CMOVP64rr: case X86::CMOVP64rm:
        continue;
      // Anything else: assume conservatively.
      default: return false;
      }
    }
  }
  return true;
}

/// Test whether the given node which sets flags has any uses which require the
/// CF flag to be accurate.
static bool hasNoCarryFlagUses(SDNode *N) {
  // Examine each user of the node.
  for (SDNode::use_iterator UI = N->use_begin(), UE = N->use_end(); UI != UE;
       ++UI) {
    // Only check things that use the flags.
    if (UI.getUse().getResNo() != 1)
      continue;
    // Only examine CopyToReg uses.
    if (UI->getOpcode() != ISD::CopyToReg)
      return false;
    // Only examine CopyToReg uses that copy to EFLAGS.
    if (cast<RegisterSDNode>(UI->getOperand(1))->getReg() != X86::EFLAGS)
      return false;
    // Examine each user of the CopyToReg use.
    for (SDNode::use_iterator FlagUI = UI->use_begin(), FlagUE = UI->use_end();
         FlagUI != FlagUE; ++FlagUI) {
      // Only examine the Flag result.
      if (FlagUI.getUse().getResNo() != 1)
        continue;
      // Anything unusual: assume conservatively.
      if (!FlagUI->isMachineOpcode())
        return false;
      // Examine the opcode of the user.
      switch (FlagUI->getMachineOpcode()) {
      // Comparisons which don't examine the CF flag.
      case X86::SETOr: case X86::SETNOr: case X86::SETEr: case X86::SETNEr:
      case X86::SETSr: case X86::SETNSr: case X86::SETPr: case X86::SETNPr:
      case X86::SETLr: case X86::SETGEr: case X86::SETLEr: case X86::SETGr:
      case X86::JO_1: case X86::JNO_1: case X86::JE_1: case X86::JNE_1:
      case X86::JS_1: case X86::JNS_1: case X86::JP_1: case X86::JNP_1:
      case X86::JL_1: case X86::JGE_1: case X86::JLE_1: case X86::JG_1:
      case X86::CMOVO16rr: case X86::CMOVO32rr: case X86::CMOVO64rr:
      case X86::CMOVO16rm: case X86::CMOVO32rm: case X86::CMOVO64rm:
      case X86::CMOVNO16rr: case X86::CMOVNO32rr: case X86::CMOVNO64rr:
      case X86::CMOVNO16rm: case X86::CMOVNO32rm: case X86::CMOVNO64rm:
      case X86::CMOVE16rr: case X86::CMOVE32rr: case X86::CMOVE64rr:
      case X86::CMOVE16rm: case X86::CMOVE32rm: case X86::CMOVE64rm:
      case X86::CMOVNE16rr: case X86::CMOVNE32rr: case X86::CMOVNE64rr:
      case X86::CMOVNE16rm: case X86::CMOVNE32rm: case X86::CMOVNE64rm:
      case X86::CMOVS16rr: case X86::CMOVS32rr: case X86::CMOVS64rr:
      case X86::CMOVS16rm: case X86::CMOVS32rm: case X86::CMOVS64rm:
      case X86::CMOVNS16rr: case X86::CMOVNS32rr: case X86::CMOVNS64rr:
      case X86::CMOVNS16rm: case X86::CMOVNS32rm: case X86::CMOVNS64rm:
      case X86::CMOVP16rr: case X86::CMOVP32rr: case X86::CMOVP64rr:
      case X86::CMOVP16rm: case X86::CMOVP32rm: case X86::CMOVP64rm:
      case X86::CMOVNP16rr: case X86::CMOVNP32rr: case X86::CMOVNP64rr:
      case X86::CMOVNP16rm: case X86::CMOVNP32rm: case X86::CMOVNP64rm:
      case X86::CMOVL16rr: case X86::CMOVL32rr: case X86::CMOVL64rr:
      case X86::CMOVL16rm: case X86::CMOVL32rm: case X86::CMOVL64rm:
      case X86::CMOVGE16rr: case X86::CMOVGE32rr: case X86::CMOVGE64rr:
      case X86::CMOVGE16rm: case X86::CMOVGE32rm: case X86::CMOVGE64rm:
      case X86::CMOVLE16rr: case X86::CMOVLE32rr: case X86::CMOVLE64rr:
      case X86::CMOVLE16rm: case X86::CMOVLE32rm: case X86::CMOVLE64rm:
      case X86::CMOVG16rr: case X86::CMOVG32rr: case X86::CMOVG64rr:
      case X86::CMOVG16rm: case X86::CMOVG32rm: case X86::CMOVG64rm:
        continue;
      // Anything else: assume conservatively.
      default:
        return false;
      }
    }
  }
  return true;
}

/// Check whether or not the chain ending in StoreNode is suitable for doing
/// the {load; op; store} to modify transformation.
static bool isFusableLoadOpStorePattern(StoreSDNode *StoreNode,
                                        SDValue StoredVal, SelectionDAG *CurDAG,
                                        LoadSDNode *&LoadNode,
                                        SDValue &InputChain) {
  // is the stored value result 0 of the load?
  if (StoredVal.getResNo() != 0) return false;

  // are there other uses of the loaded value than the inc or dec?
  if (!StoredVal.getNode()->hasNUsesOfValue(1, 0)) return false;

  // is the store non-extending and non-indexed?
  if (!ISD::isNormalStore(StoreNode) || StoreNode->isNonTemporal())
    return false;

  SDValue Load = StoredVal->getOperand(0);
  // Is the stored value a non-extending and non-indexed load?
  if (!ISD::isNormalLoad(Load.getNode())) return false;

  // Return LoadNode by reference.
  LoadNode = cast<LoadSDNode>(Load);

  // Is store the only read of the loaded value?
  if (!Load.hasOneUse())
    return false;

  // Is the address of the store the same as the load?
  if (LoadNode->getBasePtr() != StoreNode->getBasePtr() ||
      LoadNode->getOffset() != StoreNode->getOffset())
    return false;

  // Check if the chain is produced by the load or is a TokenFactor with
  // the load output chain as an operand. Return InputChain by reference.
  SDValue Chain = StoreNode->getChain();

  bool ChainCheck = false;
  if (Chain == Load.getValue(1)) {
    ChainCheck = true;
    InputChain = LoadNode->getChain();
  } else if (Chain.getOpcode() == ISD::TokenFactor) {
    SmallVector<SDValue, 4> ChainOps;
    for (unsigned i = 0, e = Chain.getNumOperands(); i != e; ++i) {
      SDValue Op = Chain.getOperand(i);
      if (Op == Load.getValue(1)) {
        ChainCheck = true;
        // Drop Load, but keep its chain. No cycle check necessary.
        ChainOps.push_back(Load.getOperand(0));
        continue;
      }

      // Make sure using Op as part of the chain would not cause a cycle here.
      // In theory, we could check whether the chain node is a predecessor of
      // the load. But that can be very expensive. Instead visit the uses and
      // make sure they all have smaller node id than the load.
      int LoadId = LoadNode->getNodeId();
      for (SDNode::use_iterator UI = Op.getNode()->use_begin(),
             UE = UI->use_end(); UI != UE; ++UI) {
        if (UI.getUse().getResNo() != 0)
          continue;
        if (UI->getNodeId() > LoadId)
          return false;
      }

      ChainOps.push_back(Op);
    }

    if (ChainCheck)
      // Make a new TokenFactor with all the other input chains except
      // for the load.
      InputChain = CurDAG->getNode(ISD::TokenFactor, SDLoc(Chain),
                                   MVT::Other, ChainOps);
  }
  if (!ChainCheck)
    return false;

  return true;
}

// Change a chain of {load; op; store} of the same value into a simple op
// through memory of that value, if the uses of the modified value and its
// address are suitable.
//
// The tablegen pattern memory operand pattern is currently not able to match
// the case where the EFLAGS on the original operation are used.
//
// To move this to tablegen, we'll need to improve tablegen to allow flags to
// be transferred from a node in the pattern to the result node, probably with
// a new keyword. For example, we have this
// def DEC64m : RI<0xFF, MRM1m, (outs), (ins i64mem:$dst), "dec{q}\t$dst",
//  [(store (add (loadi64 addr:$dst), -1), addr:$dst),
//   (implicit EFLAGS)]>;
// but maybe need something like this
// def DEC64m : RI<0xFF, MRM1m, (outs), (ins i64mem:$dst), "dec{q}\t$dst",
//  [(store (add (loadi64 addr:$dst), -1), addr:$dst),
//   (transferrable EFLAGS)]>;
//
// Until then, we manually fold these and instruction select the operation
// here.
bool X86DAGToDAGISel::foldLoadStoreIntoMemOperand(SDNode *Node) {
  StoreSDNode *StoreNode = cast<StoreSDNode>(Node);
  SDValue StoredVal = StoreNode->getOperand(1);
  unsigned Opc = StoredVal->getOpcode();

  // Before we try to select anything, make sure this is memory operand size
  // and opcode we can handle. Note that this must match the code below that
  // actually lowers the opcodes.
  EVT MemVT = StoreNode->getMemoryVT();
  if (MemVT != MVT::i64 && MemVT != MVT::i32 && MemVT != MVT::i16 &&
      MemVT != MVT::i8)
    return false;
  switch (Opc) {
  default:
    return false;
  case X86ISD::INC:
  case X86ISD::DEC:
  case X86ISD::ADD:
  case X86ISD::SUB:
  case X86ISD::AND:
  case X86ISD::OR:
  case X86ISD::XOR:
    break;
  }

  LoadSDNode *LoadNode = nullptr;
  SDValue InputChain;
  if (!isFusableLoadOpStorePattern(StoreNode, StoredVal, CurDAG, LoadNode,
                                   InputChain))
    return false;

  SDValue Base, Scale, Index, Disp, Segment;
  if (!selectAddr(LoadNode, LoadNode->getBasePtr(), Base, Scale, Index, Disp,
                  Segment))
    return false;

  auto SelectOpcode = [&](unsigned Opc64, unsigned Opc32, unsigned Opc16,
                          unsigned Opc8) {
    switch (MemVT.getSimpleVT().SimpleTy) {
    case MVT::i64:
      return Opc64;
    case MVT::i32:
      return Opc32;
    case MVT::i16:
      return Opc16;
    case MVT::i8:
      return Opc8;
    default:
      llvm_unreachable("Invalid size!");
    }
  };

  MachineSDNode *Result;
  switch (Opc) {
  case X86ISD::INC:
  case X86ISD::DEC: {
    unsigned NewOpc =
        Opc == X86ISD::INC
            ? SelectOpcode(X86::INC64m, X86::INC32m, X86::INC16m, X86::INC8m)
            : SelectOpcode(X86::DEC64m, X86::DEC32m, X86::DEC16m, X86::DEC8m);
    const SDValue Ops[] = {Base, Scale, Index, Disp, Segment, InputChain};
    Result =
        CurDAG->getMachineNode(NewOpc, SDLoc(Node), MVT::i32, MVT::Other, Ops);
    break;
  }
  case X86ISD::ADD:
  case X86ISD::SUB:
  case X86ISD::AND:
  case X86ISD::OR:
  case X86ISD::XOR: {
    auto SelectRegOpcode = [SelectOpcode](unsigned Opc) {
      switch (Opc) {
      case X86ISD::ADD:
        return SelectOpcode(X86::ADD64mr, X86::ADD32mr, X86::ADD16mr,
                            X86::ADD8mr);
      case X86ISD::SUB:
        return SelectOpcode(X86::SUB64mr, X86::SUB32mr, X86::SUB16mr,
                            X86::SUB8mr);
      case X86ISD::AND:
        return SelectOpcode(X86::AND64mr, X86::AND32mr, X86::AND16mr,
                            X86::AND8mr);
      case X86ISD::OR:
        return SelectOpcode(X86::OR64mr, X86::OR32mr, X86::OR16mr, X86::OR8mr);
      case X86ISD::XOR:
        return SelectOpcode(X86::XOR64mr, X86::XOR32mr, X86::XOR16mr,
                            X86::XOR8mr);
      default:
        llvm_unreachable("Invalid opcode!");
      }
    };
    auto SelectImm8Opcode = [SelectOpcode](unsigned Opc) {
      switch (Opc) {
      case X86ISD::ADD:
        return SelectOpcode(X86::ADD64mi8, X86::ADD32mi8, X86::ADD16mi8, 0);
      case X86ISD::SUB:
        return SelectOpcode(X86::SUB64mi8, X86::SUB32mi8, X86::SUB16mi8, 0);
      case X86ISD::AND:
        return SelectOpcode(X86::AND64mi8, X86::AND32mi8, X86::AND16mi8, 0);
      case X86ISD::OR:
        return SelectOpcode(X86::OR64mi8, X86::OR32mi8, X86::OR16mi8, 0);
      case X86ISD::XOR:
        return SelectOpcode(X86::XOR64mi8, X86::XOR32mi8, X86::XOR16mi8, 0);
      default:
        llvm_unreachable("Invalid opcode!");
      }
    };
    auto SelectImmOpcode = [SelectOpcode](unsigned Opc) {
      switch (Opc) {
      case X86ISD::ADD:
        return SelectOpcode(X86::ADD64mi32, X86::ADD32mi, X86::ADD16mi,
                            X86::ADD8mi);
      case X86ISD::SUB:
        return SelectOpcode(X86::SUB64mi32, X86::SUB32mi, X86::SUB16mi,
                            X86::SUB8mi);
      case X86ISD::AND:
        return SelectOpcode(X86::AND64mi32, X86::AND32mi, X86::AND16mi,
                            X86::AND8mi);
      case X86ISD::OR:
        return SelectOpcode(X86::OR64mi32, X86::OR32mi, X86::OR16mi,
                            X86::OR8mi);
      case X86ISD::XOR:
        return SelectOpcode(X86::XOR64mi32, X86::XOR32mi, X86::XOR16mi,
                            X86::XOR8mi);
      default:
        llvm_unreachable("Invalid opcode!");
      }
    };

    unsigned NewOpc = SelectRegOpcode(Opc);
    SDValue Operand = StoredVal->getOperand(1);

    // See if the operand is a constant that we can fold into an immediate
    // operand.
    if (auto *OperandC = dyn_cast<ConstantSDNode>(Operand)) {
      auto OperandV = OperandC->getAPIntValue();

      // Check if we can shrink the operand enough to fit in an immediate (or
      // fit into a smaller immediate) by negating it and switching the
      // operation.
      if ((Opc == X86ISD::ADD || Opc == X86ISD::SUB) &&
          ((MemVT != MVT::i8 && OperandV.getMinSignedBits() > 8 &&
            (-OperandV).getMinSignedBits() <= 8) ||
           (MemVT == MVT::i64 && OperandV.getMinSignedBits() > 32 &&
            (-OperandV).getMinSignedBits() <= 32)) &&
          hasNoCarryFlagUses(StoredVal.getNode())) {
        OperandV = -OperandV;
        Opc = Opc == X86ISD::ADD ? X86ISD::SUB : X86ISD::ADD;
      }

      // First try to fit this into an Imm8 operand. If it doesn't fit, then try
      // the larger immediate operand.
      if (MemVT != MVT::i8 && OperandV.getMinSignedBits() <= 8) {
        Operand = CurDAG->getTargetConstant(OperandV, SDLoc(Node), MemVT);
        NewOpc = SelectImm8Opcode(Opc);
      } else if (OperandV.getActiveBits() <= MemVT.getSizeInBits() &&
                 (MemVT != MVT::i64 || OperandV.getMinSignedBits() <= 32)) {
        Operand = CurDAG->getTargetConstant(OperandV, SDLoc(Node), MemVT);
        NewOpc = SelectImmOpcode(Opc);
      }
    }

    const SDValue Ops[] = {Base,    Scale,   Index,     Disp,
                           Segment, Operand, InputChain};
    Result =
        CurDAG->getMachineNode(NewOpc, SDLoc(Node), MVT::i32, MVT::Other, Ops);
    break;
  }
  default:
    llvm_unreachable("Invalid opcode!");
  }

  MachineSDNode::mmo_iterator MemOp = MF->allocateMemRefsArray(2);
  MemOp[0] = StoreNode->getMemOperand();
  MemOp[1] = LoadNode->getMemOperand();
  Result->setMemRefs(MemOp, MemOp + 2);

  ReplaceUses(SDValue(StoreNode, 0), SDValue(Result, 1));
  ReplaceUses(SDValue(StoredVal.getNode(), 1), SDValue(Result, 0));
  CurDAG->RemoveDeadNode(Node);
  return true;
}

// See if this is an (X >> C1) & C2 that we can match to BEXTR/BEXTRI.
bool X86DAGToDAGISel::matchBEXTRFromAnd(SDNode *Node) {
  MVT NVT = Node->getSimpleValueType(0);
  SDLoc dl(Node);

  SDValue N0 = Node->getOperand(0);
  SDValue N1 = Node->getOperand(1);

  if (!Subtarget->hasBMI() && !Subtarget->hasTBM())
    return false;

  // Must have a shift right.
  if (N0->getOpcode() != ISD::SRL && N0->getOpcode() != ISD::SRA)
    return false;

  // Shift can't have additional users.
  if (!N0->hasOneUse())
    return false;

  // Only supported for 32 and 64 bits.
  if (NVT != MVT::i32 && NVT != MVT::i64)
    return false;

  // Shift amount and RHS of and must be constant.
  ConstantSDNode *MaskCst = dyn_cast<ConstantSDNode>(N1);
  ConstantSDNode *ShiftCst = dyn_cast<ConstantSDNode>(N0->getOperand(1));
  if (!MaskCst || !ShiftCst)
    return false;

  // And RHS must be a mask.
  uint64_t Mask = MaskCst->getZExtValue();
  if (!isMask_64(Mask))
    return false;

  uint64_t Shift = ShiftCst->getZExtValue();
  uint64_t MaskSize = countPopulation(Mask);

  // Don't interfere with something that can be handled by extracting AH.
  // TODO: If we are able to fold a load, BEXTR might still be better than AH.
  if (Shift == 8 && MaskSize == 8)
    return false;

  // Make sure we are only using bits that were in the original value, not
  // shifted in.
  if (Shift + MaskSize > NVT.getSizeInBits())
    return false;

  SDValue New = CurDAG->getTargetConstant(Shift | (MaskSize << 8), dl, NVT);
  unsigned ROpc = NVT == MVT::i64 ? X86::BEXTRI64ri : X86::BEXTRI32ri;
  unsigned MOpc = NVT == MVT::i64 ? X86::BEXTRI64mi : X86::BEXTRI32mi;

  // BMI requires the immediate to placed in a register.
  if (!Subtarget->hasTBM()) {
    ROpc = NVT == MVT::i64 ? X86::BEXTR64rr : X86::BEXTR32rr;
    MOpc = NVT == MVT::i64 ? X86::BEXTR64rm : X86::BEXTR32rm;
    New = SDValue(CurDAG->getMachineNode(X86::MOV32ri, dl, NVT, New), 0);
    if (NVT == MVT::i64) {
      New =
          SDValue(CurDAG->getMachineNode(
                      TargetOpcode::SUBREG_TO_REG, dl, MVT::i64,
                      CurDAG->getTargetConstant(0, dl, MVT::i64), New,
                      CurDAG->getTargetConstant(X86::sub_32bit, dl, MVT::i32)),
                  0);
    }
  }

  MachineSDNode *NewNode;
  SDValue Input = N0->getOperand(0);
  SDValue Tmp0, Tmp1, Tmp2, Tmp3, Tmp4;
  if (tryFoldLoad(Node, Input, Tmp0, Tmp1, Tmp2, Tmp3, Tmp4)) {
    SDValue Ops[] = { Tmp0, Tmp1, Tmp2, Tmp3, Tmp4, New, Input.getOperand(0) };
    SDVTList VTs = CurDAG->getVTList(NVT, MVT::Other);
    NewNode = CurDAG->getMachineNode(MOpc, dl, VTs, Ops);
    // Update the chain.
    ReplaceUses(N1.getValue(1), SDValue(NewNode, 1));
    // Record the mem-refs
    LoadSDNode *LoadNode = cast<LoadSDNode>(Input);
    if (LoadNode) {
      MachineSDNode::mmo_iterator MemOp = MF->allocateMemRefsArray(1);
      MemOp[0] = LoadNode->getMemOperand();
      NewNode->setMemRefs(MemOp, MemOp + 1);
    }
  } else {
    NewNode = CurDAG->getMachineNode(ROpc, dl, NVT, Input, New);
  }

  ReplaceUses(SDValue(Node, 0), SDValue(NewNode, 0));
  CurDAG->RemoveDeadNode(Node);
  return true;
}

void X86DAGToDAGISel::Select(SDNode *Node) {
  MVT NVT = Node->getSimpleValueType(0);
  unsigned Opc, MOpc;
  unsigned Opcode = Node->getOpcode();
  SDLoc dl(Node);

  DEBUG(dbgs() << "Selecting: "; Node->dump(CurDAG); dbgs() << '\n');

  if (Node->isMachineOpcode()) {
    DEBUG(dbgs() << "== ";  Node->dump(CurDAG); dbgs() << '\n');
    Node->setNodeId(-1);
    return;   // Already selected.
  }

  switch (Opcode) {
  default: break;
  case ISD::BRIND: {
    if (Subtarget->isTargetNaCl())
      // NaCl has its own pass where jmp %r32 are converted to jmp %r64. We
      // leave the instruction alone.
      break;
    if (Subtarget->isTarget64BitILP32()) {
      // Converts a 32-bit register to a 64-bit, zero-extended version of
      // it. This is needed because x86-64 can do many things, but jmp %r32
      // ain't one of them.
      const SDValue &Target = Node->getOperand(1);
      assert(Target.getSimpleValueType() == llvm::MVT::i32);
      SDValue ZextTarget = CurDAG->getZExtOrTrunc(Target, dl, EVT(MVT::i64));
      SDValue Brind = CurDAG->getNode(ISD::BRIND, dl, MVT::Other,
                                      Node->getOperand(0), ZextTarget);
      ReplaceNode(Node, Brind.getNode());
      SelectCode(ZextTarget.getNode());
      SelectCode(Brind.getNode());
      return;
    }
    break;
  }
  case X86ISD::GlobalBaseReg:
    ReplaceNode(Node, getGlobalBaseReg());
    return;

  case X86ISD::SELECT:
  case X86ISD::SHRUNKBLEND: {
    // SHRUNKBLEND selects like a regular VSELECT. Same with X86ISD::SELECT.
    SDValue VSelect = CurDAG->getNode(
        ISD::VSELECT, SDLoc(Node), Node->getValueType(0), Node->getOperand(0),
        Node->getOperand(1), Node->getOperand(2));
    ReplaceNode(Node, VSelect.getNode());
    SelectCode(VSelect.getNode());
    // We already called ReplaceUses.
    return;
  }

  case ISD::AND:
    // Try to match BEXTR/BEXTRI instruction.
    if (matchBEXTRFromAnd(Node))
      return;

    LLVM_FALLTHROUGH;
  case ISD::OR:
  case ISD::XOR: {

    // For operations of the form (x << C1) op C2, check if we can use a smaller
    // encoding for C2 by transforming it into (x op (C2>>C1)) << C1.
    SDValue N0 = Node->getOperand(0);
    SDValue N1 = Node->getOperand(1);

    if (N0->getOpcode() != ISD::SHL || !N0->hasOneUse())
      break;

    // i8 is unshrinkable, i16 should be promoted to i32.
    if (NVT != MVT::i32 && NVT != MVT::i64)
      break;

    ConstantSDNode *Cst = dyn_cast<ConstantSDNode>(N1);
    ConstantSDNode *ShlCst = dyn_cast<ConstantSDNode>(N0->getOperand(1));
    if (!Cst || !ShlCst)
      break;

    int64_t Val = Cst->getSExtValue();
    uint64_t ShlVal = ShlCst->getZExtValue();

    // Make sure that we don't change the operation by removing bits.
    // This only matters for OR and XOR, AND is unaffected.
    uint64_t RemovedBitsMask = (1ULL << ShlVal) - 1;
    if (Opcode != ISD::AND && (Val & RemovedBitsMask) != 0)
      break;

    unsigned ShlOp, AddOp, Op;
    MVT CstVT = NVT;

    // Check the minimum bitwidth for the new constant.
    // TODO: AND32ri is the same as AND64ri32 with zext imm.
    // TODO: MOV32ri+OR64r is cheaper than MOV64ri64+OR64rr
    // TODO: Using 16 and 8 bit operations is also possible for or32 & xor32.
    if (!isInt<8>(Val) && isInt<8>(Val >> ShlVal))
      CstVT = MVT::i8;
    else if (!isInt<32>(Val) && isInt<32>(Val >> ShlVal))
      CstVT = MVT::i32;

    // Bail if there is no smaller encoding.
    if (NVT == CstVT)
      break;

    switch (NVT.SimpleTy) {
    default: llvm_unreachable("Unsupported VT!");
    case MVT::i32:
      assert(CstVT == MVT::i8);
      ShlOp = X86::SHL32ri;
      AddOp = X86::ADD32rr;

      switch (Opcode) {
      default: llvm_unreachable("Impossible opcode");
      case ISD::AND: Op = X86::AND32ri8; break;
      case ISD::OR:  Op =  X86::OR32ri8; break;
      case ISD::XOR: Op = X86::XOR32ri8; break;
      }
      break;
    case MVT::i64:
      assert(CstVT == MVT::i8 || CstVT == MVT::i32);
      ShlOp = X86::SHL64ri;
      AddOp = X86::ADD64rr;

      switch (Opcode) {
      default: llvm_unreachable("Impossible opcode");
      case ISD::AND: Op = CstVT==MVT::i8? X86::AND64ri8 : X86::AND64ri32; break;
      case ISD::OR:  Op = CstVT==MVT::i8?  X86::OR64ri8 :  X86::OR64ri32; break;
      case ISD::XOR: Op = CstVT==MVT::i8? X86::XOR64ri8 : X86::XOR64ri32; break;
      }
      break;
    }

    // Emit the smaller op and the shift.
    SDValue NewCst = CurDAG->getTargetConstant(Val >> ShlVal, dl, CstVT);
    SDNode *New = CurDAG->getMachineNode(Op, dl, NVT, N0->getOperand(0),NewCst);
    if (ShlVal == 1)
      CurDAG->SelectNodeTo(Node, AddOp, NVT, SDValue(New, 0),
                           SDValue(New, 0));
    else
      CurDAG->SelectNodeTo(Node, ShlOp, NVT, SDValue(New, 0),
                           getI8Imm(ShlVal, dl));
    return;
  }
  case X86ISD::UMUL8:
  case X86ISD::SMUL8: {
    SDValue N0 = Node->getOperand(0);
    SDValue N1 = Node->getOperand(1);

    Opc = (Opcode == X86ISD::SMUL8 ? X86::IMUL8r : X86::MUL8r);

    SDValue InFlag = CurDAG->getCopyToReg(CurDAG->getEntryNode(), dl, X86::AL,
                                          N0, SDValue()).getValue(1);

    SDVTList VTs = CurDAG->getVTList(NVT, MVT::i32);
    SDValue Ops[] = {N1, InFlag};
    SDNode *CNode = CurDAG->getMachineNode(Opc, dl, VTs, Ops);

    ReplaceNode(Node, CNode);
    return;
  }

  case X86ISD::UMUL: {
    SDValue N0 = Node->getOperand(0);
    SDValue N1 = Node->getOperand(1);

    unsigned LoReg;
    switch (NVT.SimpleTy) {
    default: llvm_unreachable("Unsupported VT!");
    // MVT::i8 is handled by X86ISD::UMUL8.
    case MVT::i16: LoReg = X86::AX;  Opc = X86::MUL16r; break;
    case MVT::i32: LoReg = X86::EAX; Opc = X86::MUL32r; break;
    case MVT::i64: LoReg = X86::RAX; Opc = X86::MUL64r; break;
    }

    SDValue InFlag = CurDAG->getCopyToReg(CurDAG->getEntryNode(), dl, LoReg,
                                          N0, SDValue()).getValue(1);

    SDVTList VTs = CurDAG->getVTList(NVT, NVT, MVT::i32);
    SDValue Ops[] = {N1, InFlag};
    SDNode *CNode = CurDAG->getMachineNode(Opc, dl, VTs, Ops);

    ReplaceNode(Node, CNode);
    return;
  }

  case ISD::SMUL_LOHI:
  case ISD::UMUL_LOHI: {
    SDValue N0 = Node->getOperand(0);
    SDValue N1 = Node->getOperand(1);

    bool isSigned = Opcode == ISD::SMUL_LOHI;
    bool hasBMI2 = Subtarget->hasBMI2();
    if (!isSigned) {
      switch (NVT.SimpleTy) {
      default: llvm_unreachable("Unsupported VT!");
      case MVT::i8:  Opc = X86::MUL8r;  MOpc = X86::MUL8m;  break;
      case MVT::i16: Opc = X86::MUL16r; MOpc = X86::MUL16m; break;
      case MVT::i32: Opc = hasBMI2 ? X86::MULX32rr : X86::MUL32r;
                     MOpc = hasBMI2 ? X86::MULX32rm : X86::MUL32m; break;
      case MVT::i64: Opc = hasBMI2 ? X86::MULX64rr : X86::MUL64r;
                     MOpc = hasBMI2 ? X86::MULX64rm : X86::MUL64m; break;
      }
    } else {
      switch (NVT.SimpleTy) {
      default: llvm_unreachable("Unsupported VT!");
      case MVT::i8:  Opc = X86::IMUL8r;  MOpc = X86::IMUL8m;  break;
      case MVT::i16: Opc = X86::IMUL16r; MOpc = X86::IMUL16m; break;
      case MVT::i32: Opc = X86::IMUL32r; MOpc = X86::IMUL32m; break;
      case MVT::i64: Opc = X86::IMUL64r; MOpc = X86::IMUL64m; break;
      }
    }

    unsigned SrcReg, LoReg, HiReg;
    switch (Opc) {
    default: llvm_unreachable("Unknown MUL opcode!");
    case X86::IMUL8r:
    case X86::MUL8r:
      SrcReg = LoReg = X86::AL; HiReg = X86::AH;
      break;
    case X86::IMUL16r:
    case X86::MUL16r:
      SrcReg = LoReg = X86::AX; HiReg = X86::DX;
      break;
    case X86::IMUL32r:
    case X86::MUL32r:
      SrcReg = LoReg = X86::EAX; HiReg = X86::EDX;
      break;
    case X86::IMUL64r:
    case X86::MUL64r:
      SrcReg = LoReg = X86::RAX; HiReg = X86::RDX;
      break;
    case X86::MULX32rr:
      SrcReg = X86::EDX; LoReg = HiReg = 0;
      break;
    case X86::MULX64rr:
      SrcReg = X86::RDX; LoReg = HiReg = 0;
      break;
    }

    SDValue Tmp0, Tmp1, Tmp2, Tmp3, Tmp4;
    bool foldedLoad = tryFoldLoad(Node, N1, Tmp0, Tmp1, Tmp2, Tmp3, Tmp4);
    // Multiply is commmutative.
    if (!foldedLoad) {
      foldedLoad = tryFoldLoad(Node, N0, Tmp0, Tmp1, Tmp2, Tmp3, Tmp4);
      if (foldedLoad)
        std::swap(N0, N1);
    }

    SDValue InFlag = CurDAG->getCopyToReg(CurDAG->getEntryNode(), dl, SrcReg,
                                          N0, SDValue()).getValue(1);
    SDValue ResHi, ResLo;

    if (foldedLoad) {
      SDValue Chain;
      MachineSDNode *CNode = nullptr;
      SDValue Ops[] = { Tmp0, Tmp1, Tmp2, Tmp3, Tmp4, N1.getOperand(0),
                        InFlag };
      if (MOpc == X86::MULX32rm || MOpc == X86::MULX64rm) {
        SDVTList VTs = CurDAG->getVTList(NVT, NVT, MVT::Other, MVT::Glue);
        CNode = CurDAG->getMachineNode(MOpc, dl, VTs, Ops);
        ResHi = SDValue(CNode, 0);
        ResLo = SDValue(CNode, 1);
        Chain = SDValue(CNode, 2);
        InFlag = SDValue(CNode, 3);
      } else {
        SDVTList VTs = CurDAG->getVTList(MVT::Other, MVT::Glue);
        CNode = CurDAG->getMachineNode(MOpc, dl, VTs, Ops);
        Chain = SDValue(CNode, 0);
        InFlag = SDValue(CNode, 1);
      }

      // Update the chain.
      ReplaceUses(N1.getValue(1), Chain);
      // Record the mem-refs
      LoadSDNode *LoadNode = cast<LoadSDNode>(N1);
      if (LoadNode) {
        MachineSDNode::mmo_iterator MemOp = MF->allocateMemRefsArray(1);
        MemOp[0] = LoadNode->getMemOperand();
        CNode->setMemRefs(MemOp, MemOp + 1);
      }
    } else {
      SDValue Ops[] = { N1, InFlag };
      if (Opc == X86::MULX32rr || Opc == X86::MULX64rr) {
        SDVTList VTs = CurDAG->getVTList(NVT, NVT, MVT::Glue);
        SDNode *CNode = CurDAG->getMachineNode(Opc, dl, VTs, Ops);
        ResHi = SDValue(CNode, 0);
        ResLo = SDValue(CNode, 1);
        InFlag = SDValue(CNode, 2);
      } else {
        SDVTList VTs = CurDAG->getVTList(MVT::Glue);
        SDNode *CNode = CurDAG->getMachineNode(Opc, dl, VTs, Ops);
        InFlag = SDValue(CNode, 0);
      }
    }

    // Prevent use of AH in a REX instruction by referencing AX instead.
    if (HiReg == X86::AH && Subtarget->is64Bit() &&
        !SDValue(Node, 1).use_empty()) {
      SDValue Result = CurDAG->getCopyFromReg(CurDAG->getEntryNode(), dl,
                                              X86::AX, MVT::i16, InFlag);
      InFlag = Result.getValue(2);
      // Get the low part if needed. Don't use getCopyFromReg for aliasing
      // registers.
      if (!SDValue(Node, 0).use_empty())
        ReplaceUses(SDValue(Node, 0),
          CurDAG->getTargetExtractSubreg(X86::sub_8bit, dl, MVT::i8, Result));

      // Shift AX down 8 bits.
      Result = SDValue(CurDAG->getMachineNode(X86::SHR16ri, dl, MVT::i16,
                                              Result,
                                     CurDAG->getTargetConstant(8, dl, MVT::i8)),
                       0);
      // Then truncate it down to i8.
      ReplaceUses(SDValue(Node, 1),
        CurDAG->getTargetExtractSubreg(X86::sub_8bit, dl, MVT::i8, Result));
    }
    // Copy the low half of the result, if it is needed.
    if (!SDValue(Node, 0).use_empty()) {
      if (!ResLo.getNode()) {
        assert(LoReg && "Register for low half is not defined!");
        ResLo = CurDAG->getCopyFromReg(CurDAG->getEntryNode(), dl, LoReg, NVT,
                                       InFlag);
        InFlag = ResLo.getValue(2);
      }
      ReplaceUses(SDValue(Node, 0), ResLo);
      DEBUG(dbgs() << "=> "; ResLo.getNode()->dump(CurDAG); dbgs() << '\n');
    }
    // Copy the high half of the result, if it is needed.
    if (!SDValue(Node, 1).use_empty()) {
      if (!ResHi.getNode()) {
        assert(HiReg && "Register for high half is not defined!");
        ResHi = CurDAG->getCopyFromReg(CurDAG->getEntryNode(), dl, HiReg, NVT,
                                       InFlag);
        InFlag = ResHi.getValue(2);
      }
      ReplaceUses(SDValue(Node, 1), ResHi);
      DEBUG(dbgs() << "=> "; ResHi.getNode()->dump(CurDAG); dbgs() << '\n');
    }

    CurDAG->RemoveDeadNode(Node);
    return;
  }

  case ISD::SDIVREM:
  case ISD::UDIVREM:
  case X86ISD::SDIVREM8_SEXT_HREG:
  case X86ISD::UDIVREM8_ZEXT_HREG: {
    SDValue N0 = Node->getOperand(0);
    SDValue N1 = Node->getOperand(1);

    bool isSigned = (Opcode == ISD::SDIVREM ||
                     Opcode == X86ISD::SDIVREM8_SEXT_HREG);
    if (!isSigned) {
      switch (NVT.SimpleTy) {
      default: llvm_unreachable("Unsupported VT!");
      case MVT::i8:  Opc = X86::DIV8r;  MOpc = X86::DIV8m;  break;
      case MVT::i16: Opc = X86::DIV16r; MOpc = X86::DIV16m; break;
      case MVT::i32: Opc = X86::DIV32r; MOpc = X86::DIV32m; break;
      case MVT::i64: Opc = X86::DIV64r; MOpc = X86::DIV64m; break;
      }
    } else {
      switch (NVT.SimpleTy) {
      default: llvm_unreachable("Unsupported VT!");
      case MVT::i8:  Opc = X86::IDIV8r;  MOpc = X86::IDIV8m;  break;
      case MVT::i16: Opc = X86::IDIV16r; MOpc = X86::IDIV16m; break;
      case MVT::i32: Opc = X86::IDIV32r; MOpc = X86::IDIV32m; break;
      case MVT::i64: Opc = X86::IDIV64r; MOpc = X86::IDIV64m; break;
      }
    }

    unsigned LoReg, HiReg, ClrReg;
    unsigned SExtOpcode;
    switch (NVT.SimpleTy) {
    default: llvm_unreachable("Unsupported VT!");
    case MVT::i8:
      LoReg = X86::AL;  ClrReg = HiReg = X86::AH;
      SExtOpcode = X86::CBW;
      break;
    case MVT::i16:
      LoReg = X86::AX;  HiReg = X86::DX;
      ClrReg = X86::DX;
      SExtOpcode = X86::CWD;
      break;
    case MVT::i32:
      LoReg = X86::EAX; ClrReg = HiReg = X86::EDX;
      SExtOpcode = X86::CDQ;
      break;
    case MVT::i64:
      LoReg = X86::RAX; ClrReg = HiReg = X86::RDX;
      SExtOpcode = X86::CQO;
      break;
    }

    SDValue Tmp0, Tmp1, Tmp2, Tmp3, Tmp4;
    bool foldedLoad = tryFoldLoad(Node, N1, Tmp0, Tmp1, Tmp2, Tmp3, Tmp4);
    bool signBitIsZero = CurDAG->SignBitIsZero(N0);

    SDValue InFlag;
    if (NVT == MVT::i8 && (!isSigned || signBitIsZero)) {
      // Special case for div8, just use a move with zero extension to AX to
      // clear the upper 8 bits (AH).
      SDValue Tmp0, Tmp1, Tmp2, Tmp3, Tmp4, Move, Chain;
      if (tryFoldLoad(Node, N0, Tmp0, Tmp1, Tmp2, Tmp3, Tmp4)) {
        SDValue Ops[] = { Tmp0, Tmp1, Tmp2, Tmp3, Tmp4, N0.getOperand(0) };
        Move =
          SDValue(CurDAG->getMachineNode(X86::MOVZX32rm8, dl, MVT::i32,
                                         MVT::Other, Ops), 0);
        Chain = Move.getValue(1);
        ReplaceUses(N0.getValue(1), Chain);
      } else {
        Move =
          SDValue(CurDAG->getMachineNode(X86::MOVZX32rr8, dl, MVT::i32, N0),0);
        Chain = CurDAG->getEntryNode();
      }
      Chain  = CurDAG->getCopyToReg(Chain, dl, X86::EAX, Move, SDValue());
      InFlag = Chain.getValue(1);
    } else {
      InFlag =
        CurDAG->getCopyToReg(CurDAG->getEntryNode(), dl,
                             LoReg, N0, SDValue()).getValue(1);
      if (isSigned && !signBitIsZero) {
        // Sign extend the low part into the high part.
        InFlag =
          SDValue(CurDAG->getMachineNode(SExtOpcode, dl, MVT::Glue, InFlag),0);
      } else {
        // Zero out the high part, effectively zero extending the input.
        SDValue ClrNode = SDValue(CurDAG->getMachineNode(X86::MOV32r0, dl, NVT), 0);
        switch (NVT.SimpleTy) {
        case MVT::i16:
          ClrNode =
              SDValue(CurDAG->getMachineNode(
                          TargetOpcode::EXTRACT_SUBREG, dl, MVT::i16, ClrNode,
                          CurDAG->getTargetConstant(X86::sub_16bit, dl,
                                                    MVT::i32)),
                      0);
          break;
        case MVT::i32:
          break;
        case MVT::i64:
          ClrNode =
              SDValue(CurDAG->getMachineNode(
                          TargetOpcode::SUBREG_TO_REG, dl, MVT::i64,
                          CurDAG->getTargetConstant(0, dl, MVT::i64), ClrNode,
                          CurDAG->getTargetConstant(X86::sub_32bit, dl,
                                                    MVT::i32)),
                      0);
          break;
        default:
          llvm_unreachable("Unexpected division source");
        }

        InFlag = CurDAG->getCopyToReg(CurDAG->getEntryNode(), dl, ClrReg,
                                      ClrNode, InFlag).getValue(1);
      }
    }

    if (foldedLoad) {
      SDValue Ops[] = { Tmp0, Tmp1, Tmp2, Tmp3, Tmp4, N1.getOperand(0),
                        InFlag };
      SDNode *CNode =
        CurDAG->getMachineNode(MOpc, dl, MVT::Other, MVT::Glue, Ops);
      InFlag = SDValue(CNode, 1);
      // Update the chain.
      ReplaceUses(N1.getValue(1), SDValue(CNode, 0));
    } else {
      InFlag =
        SDValue(CurDAG->getMachineNode(Opc, dl, MVT::Glue, N1, InFlag), 0);
    }

    // Prevent use of AH in a REX instruction by explicitly copying it to
    // an ABCD_L register.
    //
    // The current assumption of the register allocator is that isel
    // won't generate explicit references to the GR8_ABCD_H registers. If
    // the allocator and/or the backend get enhanced to be more robust in
    // that regard, this can be, and should be, removed.
    if (HiReg == X86::AH && !SDValue(Node, 1).use_empty()) {
      SDValue AHCopy = CurDAG->getRegister(X86::AH, MVT::i8);
      unsigned AHExtOpcode =
          isSigned ? X86::MOVSX32_NOREXrr8 : X86::MOVZX32_NOREXrr8;

      SDNode *RNode = CurDAG->getMachineNode(AHExtOpcode, dl, MVT::i32,
                                             MVT::Glue, AHCopy, InFlag);
      SDValue Result(RNode, 0);
      InFlag = SDValue(RNode, 1);

      if (Opcode == X86ISD::UDIVREM8_ZEXT_HREG ||
          Opcode == X86ISD::SDIVREM8_SEXT_HREG) {
        assert(Node->getValueType(1) == MVT::i32 && "Unexpected result type!");
      } else {
        Result =
            CurDAG->getTargetExtractSubreg(X86::sub_8bit, dl, MVT::i8, Result);
      }
      ReplaceUses(SDValue(Node, 1), Result);
      DEBUG(dbgs() << "=> "; Result.getNode()->dump(CurDAG); dbgs() << '\n');
    }
    // Copy the division (low) result, if it is needed.
    if (!SDValue(Node, 0).use_empty()) {
      SDValue Result = CurDAG->getCopyFromReg(CurDAG->getEntryNode(), dl,
                                                LoReg, NVT, InFlag);
      InFlag = Result.getValue(2);
      ReplaceUses(SDValue(Node, 0), Result);
      DEBUG(dbgs() << "=> "; Result.getNode()->dump(CurDAG); dbgs() << '\n');
    }
    // Copy the remainder (high) result, if it is needed.
    if (!SDValue(Node, 1).use_empty()) {
      SDValue Result = CurDAG->getCopyFromReg(CurDAG->getEntryNode(), dl,
                                              HiReg, NVT, InFlag);
      InFlag = Result.getValue(2);
      ReplaceUses(SDValue(Node, 1), Result);
      DEBUG(dbgs() << "=> "; Result.getNode()->dump(CurDAG); dbgs() << '\n');
    }
    CurDAG->RemoveDeadNode(Node);
    return;
  }

  case X86ISD::CMP:
  case X86ISD::SUB: {
    // Sometimes a SUB is used to perform comparison.
    if (Opcode == X86ISD::SUB && Node->hasAnyUseOfValue(0))
      // This node is not a CMP.
      break;
    SDValue N0 = Node->getOperand(0);
    SDValue N1 = Node->getOperand(1);

    if (N0.getOpcode() == ISD::TRUNCATE && N0.hasOneUse() &&
        hasNoSignedComparisonUses(Node))
      N0 = N0.getOperand(0);

    // Look for (X86cmp (and $op, $imm), 0) and see if we can convert it to
    // use a smaller encoding.
    // Look past the truncate if CMP is the only use of it.
    if ((N0.getOpcode() == ISD::AND ||
         (N0.getResNo() == 0 && N0.getOpcode() == X86ISD::AND)) &&
        N0.getNode()->hasOneUse() &&
        N0.getValueType() != MVT::i8 &&
        X86::isZeroNode(N1)) {
      ConstantSDNode *C = dyn_cast<ConstantSDNode>(N0.getOperand(1));
      if (!C) break;
      uint64_t Mask = C->getZExtValue();

      // For example, convert "testl %eax, $8" to "testb %al, $8"
      if (isUInt<8>(Mask) &&
          (!(Mask & 0x80) || hasNoSignedComparisonUses(Node))) {
        SDValue Imm = CurDAG->getTargetConstant(Mask, dl, MVT::i8);
        SDValue Reg = N0.getOperand(0);

        // Extract the l-register.
        SDValue Subreg = CurDAG->getTargetExtractSubreg(X86::sub_8bit, dl,
                                                        MVT::i8, Reg);

        // Emit a testb.
        SDNode *NewNode = CurDAG->getMachineNode(X86::TEST8ri, dl, MVT::i32,
                                                 Subreg, Imm);
        // Replace SUB|CMP with TEST, since SUB has two outputs while TEST has
        // one, do not call ReplaceAllUsesWith.
        ReplaceUses(SDValue(Node, (Opcode == X86ISD::SUB ? 1 : 0)),
                    SDValue(NewNode, 0));
        CurDAG->RemoveDeadNode(Node);
        return;
      }

      // For example, "testl %eax, $2048" to "testb %ah, $8".
      if (isShiftedUInt<8, 8>(Mask) &&
          (!(Mask & 0x8000) || hasNoSignedComparisonUses(Node))) {
        // Shift the immediate right by 8 bits.
        SDValue ShiftedImm = CurDAG->getTargetConstant(Mask >> 8, dl, MVT::i8);
        SDValue Reg = N0.getOperand(0);

        // Extract the h-register.
        SDValue Subreg = CurDAG->getTargetExtractSubreg(X86::sub_8bit_hi, dl,
                                                        MVT::i8, Reg);

        // Emit a testb.  The EXTRACT_SUBREG becomes a COPY that can only
        // target GR8_NOREX registers, so make sure the register class is
        // forced.
        SDNode *NewNode = CurDAG->getMachineNode(X86::TEST8ri_NOREX, dl,
                                                 MVT::i32, Subreg, ShiftedImm);
        // Replace SUB|CMP with TEST, since SUB has two outputs while TEST has
        // one, do not call ReplaceAllUsesWith.
        ReplaceUses(SDValue(Node, (Opcode == X86ISD::SUB ? 1 : 0)),
                    SDValue(NewNode, 0));
        CurDAG->RemoveDeadNode(Node);
        return;
      }

      // For example, "testl %eax, $32776" to "testw %ax, $32776".
      // NOTE: We only want to form TESTW instructions if optimizing for
      // min size. Otherwise we only save one byte and possibly get a length
      // changing prefix penalty in the decoders.
      if (OptForMinSize && isUInt<16>(Mask) && N0.getValueType() != MVT::i16 &&
          (!(Mask & 0x8000) || hasNoSignedComparisonUses(Node))) {
        SDValue Imm = CurDAG->getTargetConstant(Mask, dl, MVT::i16);
        SDValue Reg = N0.getOperand(0);

        // Extract the 16-bit subregister.
        SDValue Subreg = CurDAG->getTargetExtractSubreg(X86::sub_16bit, dl,
                                                        MVT::i16, Reg);

        // Emit a testw.
        SDNode *NewNode = CurDAG->getMachineNode(X86::TEST16ri, dl, MVT::i32,
                                                 Subreg, Imm);
        // Replace SUB|CMP with TEST, since SUB has two outputs while TEST has
        // one, do not call ReplaceAllUsesWith.
        ReplaceUses(SDValue(Node, (Opcode == X86ISD::SUB ? 1 : 0)),
                    SDValue(NewNode, 0));
        CurDAG->RemoveDeadNode(Node);
        return;
      }

      // For example, "testq %rax, $268468232" to "testl %eax, $268468232".
      if (isUInt<32>(Mask) && N0.getValueType() == MVT::i64 &&
          (!(Mask & 0x80000000) || hasNoSignedComparisonUses(Node))) {
        SDValue Imm = CurDAG->getTargetConstant(Mask, dl, MVT::i32);
        SDValue Reg = N0.getOperand(0);

        // Extract the 32-bit subregister.
        SDValue Subreg = CurDAG->getTargetExtractSubreg(X86::sub_32bit, dl,
                                                        MVT::i32, Reg);

        // Emit a testl.
        SDNode *NewNode = CurDAG->getMachineNode(X86::TEST32ri, dl, MVT::i32,
                                                 Subreg, Imm);
        // Replace SUB|CMP with TEST, since SUB has two outputs while TEST has
        // one, do not call ReplaceAllUsesWith.
        ReplaceUses(SDValue(Node, (Opcode == X86ISD::SUB ? 1 : 0)),
                    SDValue(NewNode, 0));
        CurDAG->RemoveDeadNode(Node);
        return;
      }
    }
    break;
  }
  case ISD::STORE:
    if (foldLoadStoreIntoMemOperand(Node))
      return;
    break;
  }

  SelectCode(Node);
}

bool X86DAGToDAGISel::
SelectInlineAsmMemoryOperand(const SDValue &Op, unsigned ConstraintID,
                             std::vector<SDValue> &OutOps) {
  SDValue Op0, Op1, Op2, Op3, Op4;
  switch (ConstraintID) {
  default:
    llvm_unreachable("Unexpected asm memory constraint");
  case InlineAsm::Constraint_i:
    // FIXME: It seems strange that 'i' is needed here since it's supposed to
    //        be an immediate and not a memory constraint.
    LLVM_FALLTHROUGH;
  case InlineAsm::Constraint_o: // offsetable        ??
  case InlineAsm::Constraint_v: // not offsetable    ??
  case InlineAsm::Constraint_m: // memory
  case InlineAsm::Constraint_X:
    if (!selectAddr(nullptr, Op, Op0, Op1, Op2, Op3, Op4))
      return true;
    break;
  }

  OutOps.push_back(Op0);
  OutOps.push_back(Op1);
  OutOps.push_back(Op2);
  OutOps.push_back(Op3);
  OutOps.push_back(Op4);
  return false;
}

/// This pass converts a legalized DAG into a X86-specific DAG,
/// ready for instruction scheduling.
FunctionPass *llvm::createX86ISelDag(X86TargetMachine &TM,
                                     CodeGenOpt::Level OptLevel) {
  return new X86DAGToDAGISel(TM, OptLevel);
}
