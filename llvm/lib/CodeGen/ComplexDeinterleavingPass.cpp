//===- ComplexDeinterleavingPass.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Identification:
// This step is responsible for finding the patterns that can be lowered to
// complex instructions, and building a graph to represent the complex
// structures. Starting from the "Converging Shuffle" (a shuffle that
// reinterleaves the complex components, with a mask of <0, 2, 1, 3>), the
// operands are evaluated and identified as "Composite Nodes" (collections of
// instructions that can potentially be lowered to a single complex
// instruction). This is performed by checking the real and imaginary components
// and tracking the data flow for each component while following the operand
// pairs. Validity of each node is expected to be done upon creation, and any
// validation errors should halt traversal and prevent further graph
// construction.
//
// Replacement:
// This step traverses the graph built up by identification, delegating to the
// target to validate and generate the correct intrinsics, and plumbs them
// together connecting each end of the new intrinsics graph to the existing
// use-def chain. This step is assumed to finish successfully, as all
// information is expected to be correct by this point.
//
//
// Internal data structure:
// ComplexDeinterleavingGraph:
// Keeps references to all the valid CompositeNodes formed as part of the
// transformation, and every Instruction contained within said nodes. It also
// holds onto a reference to the root Instruction, and the root node that should
// replace it.
//
// ComplexDeinterleavingCompositeNode:
// A CompositeNode represents a single transformation point; each node should
// transform into a single complex instruction (ignoring vector splitting, which
// would generate more instructions per node). They are identified in a
// depth-first manner, traversing and identifying the operands of each
// instruction in the order they appear in the IR.
// Each node maintains a reference  to its Real and Imaginary instructions,
// as well as any additional instructions that make up the identified operation
// (Internal instructions should only have uses within their containing node).
// A Node also contains the rotation and operation type that it represents.
// Operands contains pointers to other CompositeNodes, acting as the edges in
// the graph. ReplacementValue is the transformed Value* that has been emitted
// to the IR.
//
// Note: If the operation of a Node is Shuffle, only the Real, Imaginary, and
// ReplacementValue fields of that Node are relevant, where the ReplacementValue
// should be pre-populated.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/ComplexDeinterleavingPass.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/InitializePasses.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Utils/Local.h"
#include <algorithm>

using namespace llvm;
using namespace PatternMatch;

#define DEBUG_TYPE "complex-deinterleaving"

STATISTIC(NumComplexTransformations, "Amount of complex patterns transformed");

static cl::opt<bool> ComplexDeinterleavingEnabled(
    "enable-complex-deinterleaving",
    cl::desc("Enable generation of complex instructions"), cl::init(true),
    cl::Hidden);

/// Checks the given mask, and determines whether said mask is interleaving.
///
/// To be interleaving, a mask must alternate between `i` and `i + (Length /
/// 2)`, and must contain all numbers within the range of `[0..Length)` (e.g. a
/// 4x vector interleaving mask would be <0, 2, 1, 3>).
static bool isInterleavingMask(ArrayRef<int> Mask);

/// Checks the given mask, and determines whether said mask is deinterleaving.
///
/// To be deinterleaving, a mask must increment in steps of 2, and either start
/// with 0 or 1.
/// (e.g. an 8x vector deinterleaving mask would be either <0, 2, 4, 6> or
/// <1, 3, 5, 7>).
static bool isDeinterleavingMask(ArrayRef<int> Mask);

namespace {

class ComplexDeinterleavingLegacyPass : public FunctionPass {
public:
  static char ID;

  ComplexDeinterleavingLegacyPass(const TargetMachine *TM = nullptr)
      : FunctionPass(ID), TM(TM) {
    initializeComplexDeinterleavingLegacyPassPass(
        *PassRegistry::getPassRegistry());
  }

  StringRef getPassName() const override {
    return "Complex Deinterleaving Pass";
  }

  bool runOnFunction(Function &F) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetLibraryInfoWrapperPass>();
    AU.setPreservesCFG();
  }

private:
  const TargetMachine *TM;
};

class ComplexDeinterleavingGraph;
struct ComplexDeinterleavingCompositeNode {

  ComplexDeinterleavingCompositeNode(ComplexDeinterleavingOperation Op,
                                     Instruction *R, Instruction *I)
      : Operation(Op), Real(R), Imag(I) {}

private:
  friend class ComplexDeinterleavingGraph;
  using NodePtr = std::shared_ptr<ComplexDeinterleavingCompositeNode>;
  using RawNodePtr = ComplexDeinterleavingCompositeNode *;

public:
  ComplexDeinterleavingOperation Operation;
  Instruction *Real;
  Instruction *Imag;

  // Instructions that should only exist within this node, there should be no
  // users of these instructions outside the node. An example of these would be
  // the multiply instructions of a partial multiply operation.
  SmallVector<Instruction *> InternalInstructions;
  ComplexDeinterleavingRotation Rotation;
  SmallVector<RawNodePtr> Operands;
  Value *ReplacementNode = nullptr;

  void addInstruction(Instruction *I) { InternalInstructions.push_back(I); }
  void addOperand(NodePtr Node) { Operands.push_back(Node.get()); }

  bool hasAllInternalUses(SmallPtrSet<Instruction *, 16> &AllInstructions);

  void dump() { dump(dbgs()); }
  void dump(raw_ostream &OS) {
    auto PrintValue = [&](Value *V) {
      if (V) {
        OS << "\"";
        V->print(OS, true);
        OS << "\"\n";
      } else
        OS << "nullptr\n";
    };
    auto PrintNodeRef = [&](RawNodePtr Ptr) {
      if (Ptr)
        OS << Ptr << "\n";
      else
        OS << "nullptr\n";
    };

    OS << "- CompositeNode: " << this << "\n";
    OS << "  Real: ";
    PrintValue(Real);
    OS << "  Imag: ";
    PrintValue(Imag);
    OS << "  ReplacementNode: ";
    PrintValue(ReplacementNode);
    OS << "  Operation: " << (int)Operation << "\n";
    OS << "  Rotation: " << ((int)Rotation * 90) << "\n";
    OS << "  Operands: \n";
    for (const auto &Op : Operands) {
      OS << "    - ";
      PrintNodeRef(Op);
    }
    OS << "  InternalInstructions:\n";
    for (const auto &I : InternalInstructions) {
      OS << "    - \"";
      I->print(OS, true);
      OS << "\"\n";
    }
  }
};

class ComplexDeinterleavingGraph {
public:
  using NodePtr = ComplexDeinterleavingCompositeNode::NodePtr;
  using RawNodePtr = ComplexDeinterleavingCompositeNode::RawNodePtr;
  explicit ComplexDeinterleavingGraph(const TargetLowering *tl) : TL(tl) {}

private:
  const TargetLowering *TL;
  Instruction *RootValue;
  NodePtr RootNode;
  SmallVector<NodePtr> CompositeNodes;
  SmallPtrSet<Instruction *, 16> AllInstructions;

  NodePtr prepareCompositeNode(ComplexDeinterleavingOperation Operation,
                               Instruction *R, Instruction *I) {
    return std::make_shared<ComplexDeinterleavingCompositeNode>(Operation, R,
                                                                I);
  }

  NodePtr submitCompositeNode(NodePtr Node) {
    CompositeNodes.push_back(Node);
    AllInstructions.insert(Node->Real);
    AllInstructions.insert(Node->Imag);
    for (auto *I : Node->InternalInstructions)
      AllInstructions.insert(I);
    return Node;
  }

  NodePtr getContainingComposite(Value *R, Value *I) {
    for (const auto &CN : CompositeNodes) {
      if (CN->Real == R && CN->Imag == I)
        return CN;
    }
    return nullptr;
  }

  /// Identifies a complex partial multiply pattern and its rotation, based on
  /// the following patterns
  ///
  ///  0:  r: cr + ar * br
  ///      i: ci + ar * bi
  /// 90:  r: cr - ai * bi
  ///      i: ci + ai * br
  /// 180: r: cr - ar * br
  ///      i: ci - ar * bi
  /// 270: r: cr + ai * bi
  ///      i: ci - ai * br
  NodePtr identifyPartialMul(Instruction *Real, Instruction *Imag);

  /// Identify the other branch of a Partial Mul, taking the CommonOperandI that
  /// is partially known from identifyPartialMul, filling in the other half of
  /// the complex pair.
  NodePtr identifyNodeWithImplicitAdd(
      Instruction *I, Instruction *J,
      std::pair<Instruction *, Instruction *> &CommonOperandI);

  /// Identifies a complex add pattern and its rotation, based on the following
  /// patterns.
  ///
  /// 90:  r: ar - bi
  ///      i: ai + br
  /// 270: r: ar + bi
  ///      i: ai - br
  NodePtr identifyAdd(Instruction *Real, Instruction *Imag);

  NodePtr identifyNode(Instruction *I, Instruction *J);

  Value *replaceNode(RawNodePtr Node);

public:
  void dump() { dump(dbgs()); }
  void dump(raw_ostream &OS) {
    for (const auto &Node : CompositeNodes)
      Node->dump(OS);
  }

  /// Returns false if the deinterleaving operation should be cancelled for the
  /// current graph.
  bool identifyNodes(Instruction *RootI);

  /// Perform the actual replacement of the underlying instruction graph.
  /// Returns false if the deinterleaving operation should be cancelled for the
  /// current graph.
  void replaceNodes();
};

class ComplexDeinterleaving {
public:
  ComplexDeinterleaving(const TargetLowering *tl, const TargetLibraryInfo *tli)
      : TL(tl), TLI(tli) {}
  bool runOnFunction(Function &F);

private:
  bool evaluateBasicBlock(BasicBlock *B);

  const TargetLowering *TL = nullptr;
  const TargetLibraryInfo *TLI = nullptr;
};

} // namespace

char ComplexDeinterleavingLegacyPass::ID = 0;

INITIALIZE_PASS_BEGIN(ComplexDeinterleavingLegacyPass, DEBUG_TYPE,
                      "Complex Deinterleaving", false, false)
INITIALIZE_PASS_END(ComplexDeinterleavingLegacyPass, DEBUG_TYPE,
                    "Complex Deinterleaving", false, false)

PreservedAnalyses ComplexDeinterleavingPass::run(Function &F,
                                                 FunctionAnalysisManager &AM) {
  const TargetLowering *TL = TM->getSubtargetImpl(F)->getTargetLowering();
  auto &TLI = AM.getResult<llvm::TargetLibraryAnalysis>(F);
  if (!ComplexDeinterleaving(TL, &TLI).runOnFunction(F))
    return PreservedAnalyses::all();

  PreservedAnalyses PA;
  PA.preserve<FunctionAnalysisManagerModuleProxy>();
  return PA;
}

FunctionPass *llvm::createComplexDeinterleavingPass(const TargetMachine *TM) {
  return new ComplexDeinterleavingLegacyPass(TM);
}

bool ComplexDeinterleavingLegacyPass::runOnFunction(Function &F) {
  const auto *TL = TM->getSubtargetImpl(F)->getTargetLowering();
  auto TLI = getAnalysis<TargetLibraryInfoWrapperPass>().getTLI(F);
  return ComplexDeinterleaving(TL, &TLI).runOnFunction(F);
}

bool ComplexDeinterleaving::runOnFunction(Function &F) {
  if (!ComplexDeinterleavingEnabled) {
    LLVM_DEBUG(
        dbgs() << "Complex deinterleaving has been explicitly disabled.\n");
    return false;
  }

  if (!TL->isComplexDeinterleavingSupported()) {
    LLVM_DEBUG(
        dbgs() << "Complex deinterleaving has been disabled, target does "
                  "not support lowering of complex number operations.\n");
    return false;
  }

  bool Changed = false;
  for (auto &B : F)
    Changed |= evaluateBasicBlock(&B);

  return Changed;
}

static bool isInterleavingMask(ArrayRef<int> Mask) {
  // If the size is not even, it's not an interleaving mask
  if ((Mask.size() & 1))
    return false;

  int HalfNumElements = Mask.size() / 2;
  for (int Idx = 0; Idx < HalfNumElements; ++Idx) {
    int MaskIdx = Idx * 2;
    if (Mask[MaskIdx] != Idx || Mask[MaskIdx + 1] != (Idx + HalfNumElements))
      return false;
  }

  return true;
}

static bool isDeinterleavingMask(ArrayRef<int> Mask) {
  int Offset = Mask[0];
  int HalfNumElements = Mask.size() / 2;

  for (int Idx = 1; Idx < HalfNumElements; ++Idx) {
    if (Mask[Idx] != (Idx * 2) + Offset)
      return false;
  }

  return true;
}

bool ComplexDeinterleaving::evaluateBasicBlock(BasicBlock *B) {
  bool Changed = false;

  SmallVector<Instruction *> DeadInstrRoots;

  for (auto &I : *B) {
    auto *SVI = dyn_cast<ShuffleVectorInst>(&I);
    if (!SVI)
      continue;

    // Look for a shufflevector that takes separate vectors of the real and
    // imaginary components and recombines them into a single vector.
    if (!isInterleavingMask(SVI->getShuffleMask()))
      continue;

    ComplexDeinterleavingGraph Graph(TL);
    if (!Graph.identifyNodes(SVI))
      continue;

    Graph.replaceNodes();
    DeadInstrRoots.push_back(SVI);
    Changed = true;
  }

  for (const auto &I : DeadInstrRoots) {
    if (!I || I->getParent() == nullptr)
      continue;
    llvm::RecursivelyDeleteTriviallyDeadInstructions(I, TLI);
  }

  return Changed;
}

ComplexDeinterleavingGraph::NodePtr
ComplexDeinterleavingGraph::identifyNodeWithImplicitAdd(
    Instruction *Real, Instruction *Imag,
    std::pair<Instruction *, Instruction *> &PartialMatch) {
  LLVM_DEBUG(dbgs() << "identifyNodeWithImplicitAdd " << *Real << " / " << *Imag
                    << "\n");

  if (!Real->hasOneUse() || !Imag->hasOneUse()) {
    LLVM_DEBUG(dbgs() << "  - Mul operand has multiple uses.\n");
    return nullptr;
  }

  if (Real->getOpcode() != Instruction::FMul ||
      Imag->getOpcode() != Instruction::FMul) {
    LLVM_DEBUG(dbgs() << "  - Real or imaginary instruction is not fmul\n");
    return nullptr;
  }

  Instruction *R0 = dyn_cast<Instruction>(Real->getOperand(0));
  Instruction *R1 = dyn_cast<Instruction>(Real->getOperand(1));
  Instruction *I0 = dyn_cast<Instruction>(Imag->getOperand(0));
  Instruction *I1 = dyn_cast<Instruction>(Imag->getOperand(1));
  if (!R0 || !R1 || !I0 || !I1) {
    LLVM_DEBUG(dbgs() << "  - Mul operand not Instruction\n");
    return nullptr;
  }

  // A +/+ has a rotation of 0. If any of the operands are fneg, we flip the
  // rotations and use the operand.
  unsigned Negs = 0;
  SmallVector<Instruction *> FNegs;
  if (R0->getOpcode() == Instruction::FNeg ||
      R1->getOpcode() == Instruction::FNeg) {
    Negs |= 1;
    if (R0->getOpcode() == Instruction::FNeg) {
      FNegs.push_back(R0);
      R0 = dyn_cast<Instruction>(R0->getOperand(0));
    } else {
      FNegs.push_back(R1);
      R1 = dyn_cast<Instruction>(R1->getOperand(0));
    }
    if (!R0 || !R1)
      return nullptr;
  }
  if (I0->getOpcode() == Instruction::FNeg ||
      I1->getOpcode() == Instruction::FNeg) {
    Negs |= 2;
    Negs ^= 1;
    if (I0->getOpcode() == Instruction::FNeg) {
      FNegs.push_back(I0);
      I0 = dyn_cast<Instruction>(I0->getOperand(0));
    } else {
      FNegs.push_back(I1);
      I1 = dyn_cast<Instruction>(I1->getOperand(0));
    }
    if (!I0 || !I1)
      return nullptr;
  }

  ComplexDeinterleavingRotation Rotation = (ComplexDeinterleavingRotation)Negs;

  Instruction *CommonOperand;
  Instruction *UncommonRealOp;
  Instruction *UncommonImagOp;

  if (R0 == I0 || R0 == I1) {
    CommonOperand = R0;
    UncommonRealOp = R1;
  } else if (R1 == I0 || R1 == I1) {
    CommonOperand = R1;
    UncommonRealOp = R0;
  } else {
    LLVM_DEBUG(dbgs() << "  - No equal operand\n");
    return nullptr;
  }

  UncommonImagOp = (CommonOperand == I0) ? I1 : I0;
  if (Rotation == ComplexDeinterleavingRotation::Rotation_90 ||
      Rotation == ComplexDeinterleavingRotation::Rotation_270)
    std::swap(UncommonRealOp, UncommonImagOp);

  // Between identifyPartialMul and here we need to have found a complete valid
  // pair from the CommonOperand of each part.
  if (Rotation == ComplexDeinterleavingRotation::Rotation_0 ||
      Rotation == ComplexDeinterleavingRotation::Rotation_180)
    PartialMatch.first = CommonOperand;
  else
    PartialMatch.second = CommonOperand;

  if (!PartialMatch.first || !PartialMatch.second) {
    LLVM_DEBUG(dbgs() << "  - Incomplete partial match\n");
    return nullptr;
  }

  NodePtr CommonNode = identifyNode(PartialMatch.first, PartialMatch.second);
  if (!CommonNode) {
    LLVM_DEBUG(dbgs() << "  - No CommonNode identified\n");
    return nullptr;
  }

  NodePtr UncommonNode = identifyNode(UncommonRealOp, UncommonImagOp);
  if (!UncommonNode) {
    LLVM_DEBUG(dbgs() << "  - No UncommonNode identified\n");
    return nullptr;
  }

  NodePtr Node = prepareCompositeNode(
      ComplexDeinterleavingOperation::CMulPartial, Real, Imag);
  Node->Rotation = Rotation;
  Node->addOperand(CommonNode);
  Node->addOperand(UncommonNode);
  Node->InternalInstructions.append(FNegs);
  return submitCompositeNode(Node);
}

ComplexDeinterleavingGraph::NodePtr
ComplexDeinterleavingGraph::identifyPartialMul(Instruction *Real,
                                               Instruction *Imag) {
  LLVM_DEBUG(dbgs() << "identifyPartialMul " << *Real << " / " << *Imag
                    << "\n");
  // Determine rotation
  ComplexDeinterleavingRotation Rotation;
  if (Real->getOpcode() == Instruction::FAdd &&
      Imag->getOpcode() == Instruction::FAdd)
    Rotation = ComplexDeinterleavingRotation::Rotation_0;
  else if (Real->getOpcode() == Instruction::FSub &&
           Imag->getOpcode() == Instruction::FAdd)
    Rotation = ComplexDeinterleavingRotation::Rotation_90;
  else if (Real->getOpcode() == Instruction::FSub &&
           Imag->getOpcode() == Instruction::FSub)
    Rotation = ComplexDeinterleavingRotation::Rotation_180;
  else if (Real->getOpcode() == Instruction::FAdd &&
           Imag->getOpcode() == Instruction::FSub)
    Rotation = ComplexDeinterleavingRotation::Rotation_270;
  else {
    LLVM_DEBUG(dbgs() << "  - Unhandled rotation.\n");
    return nullptr;
  }

  if (!Real->getFastMathFlags().allowContract() ||
      !Imag->getFastMathFlags().allowContract()) {
    LLVM_DEBUG(dbgs() << "  - Contract is missing from the FastMath flags.\n");
    return nullptr;
  }

  Value *CR = Real->getOperand(0);
  Instruction *RealMulI = dyn_cast<Instruction>(Real->getOperand(1));
  if (!RealMulI)
    return nullptr;
  Value *CI = Imag->getOperand(0);
  Instruction *ImagMulI = dyn_cast<Instruction>(Imag->getOperand(1));
  if (!ImagMulI)
    return nullptr;

  if (!RealMulI->hasOneUse() || !ImagMulI->hasOneUse()) {
    LLVM_DEBUG(dbgs() << "  - Mul instruction has multiple uses\n");
    return nullptr;
  }

  Instruction *R0 = dyn_cast<Instruction>(RealMulI->getOperand(0));
  Instruction *R1 = dyn_cast<Instruction>(RealMulI->getOperand(1));
  Instruction *I0 = dyn_cast<Instruction>(ImagMulI->getOperand(0));
  Instruction *I1 = dyn_cast<Instruction>(ImagMulI->getOperand(1));
  if (!R0 || !R1 || !I0 || !I1) {
    LLVM_DEBUG(dbgs() << "  - Mul operand not Instruction\n");
    return nullptr;
  }

  Instruction *CommonOperand;
  Instruction *UncommonRealOp;
  Instruction *UncommonImagOp;

  if (R0 == I0 || R0 == I1) {
    CommonOperand = R0;
    UncommonRealOp = R1;
  } else if (R1 == I0 || R1 == I1) {
    CommonOperand = R1;
    UncommonRealOp = R0;
  } else {
    LLVM_DEBUG(dbgs() << "  - No equal operand\n");
    return nullptr;
  }

  UncommonImagOp = (CommonOperand == I0) ? I1 : I0;
  if (Rotation == ComplexDeinterleavingRotation::Rotation_90 ||
      Rotation == ComplexDeinterleavingRotation::Rotation_270)
    std::swap(UncommonRealOp, UncommonImagOp);

  std::pair<Instruction *, Instruction *> PartialMatch(
      (Rotation == ComplexDeinterleavingRotation::Rotation_0 ||
       Rotation == ComplexDeinterleavingRotation::Rotation_180)
          ? CommonOperand
          : nullptr,
      (Rotation == ComplexDeinterleavingRotation::Rotation_90 ||
       Rotation == ComplexDeinterleavingRotation::Rotation_270)
          ? CommonOperand
          : nullptr);
  NodePtr CNode = identifyNodeWithImplicitAdd(
      cast<Instruction>(CR), cast<Instruction>(CI), PartialMatch);
  if (!CNode) {
    LLVM_DEBUG(dbgs() << "  - No cnode identified\n");
    return nullptr;
  }

  NodePtr UncommonRes = identifyNode(UncommonRealOp, UncommonImagOp);
  if (!UncommonRes) {
    LLVM_DEBUG(dbgs() << "  - No UncommonRes identified\n");
    return nullptr;
  }

  assert(PartialMatch.first && PartialMatch.second);
  NodePtr CommonRes = identifyNode(PartialMatch.first, PartialMatch.second);
  if (!CommonRes) {
    LLVM_DEBUG(dbgs() << "  - No CommonRes identified\n");
    return nullptr;
  }

  NodePtr Node = prepareCompositeNode(
      ComplexDeinterleavingOperation::CMulPartial, Real, Imag);
  Node->addInstruction(RealMulI);
  Node->addInstruction(ImagMulI);
  Node->Rotation = Rotation;
  Node->addOperand(CommonRes);
  Node->addOperand(UncommonRes);
  Node->addOperand(CNode);
  return submitCompositeNode(Node);
}

ComplexDeinterleavingGraph::NodePtr
ComplexDeinterleavingGraph::identifyAdd(Instruction *Real, Instruction *Imag) {
  LLVM_DEBUG(dbgs() << "identifyAdd " << *Real << " / " << *Imag << "\n");

  // Determine rotation
  ComplexDeinterleavingRotation Rotation;
  if (Real->getOpcode() == Instruction::FSub &&
      Imag->getOpcode() == Instruction::FAdd)
    Rotation = ComplexDeinterleavingRotation::Rotation_90;
  else if (Real->getOpcode() == Instruction::FAdd &&
           Imag->getOpcode() == Instruction::FSub)
    Rotation = ComplexDeinterleavingRotation::Rotation_270;
  else {
    LLVM_DEBUG(dbgs() << " - Unhandled case, rotation is not assigned.\n");
    return nullptr;
  }

  auto *AR = cast<Instruction>(Real->getOperand(0));
  auto *BI = cast<Instruction>(Real->getOperand(1));
  auto *AI = cast<Instruction>(Imag->getOperand(0));
  auto *BR = cast<Instruction>(Imag->getOperand(1));

  NodePtr ResA = identifyNode(AR, AI);
  if (!ResA) {
    LLVM_DEBUG(dbgs() << " - AR/AI is not identified as a composite node.\n");
    return nullptr;
  }
  NodePtr ResB = identifyNode(BR, BI);
  if (!ResB) {
    LLVM_DEBUG(dbgs() << " - BR/BI is not identified as a composite node.\n");
    return nullptr;
  }

  NodePtr Node =
      prepareCompositeNode(ComplexDeinterleavingOperation::CAdd, Real, Imag);
  Node->Rotation = Rotation;
  Node->addOperand(ResA);
  Node->addOperand(ResB);
  return submitCompositeNode(Node);
}

static bool isInstructionPairAdd(Instruction *A, Instruction *B) {
  unsigned OpcA = A->getOpcode();
  unsigned OpcB = B->getOpcode();
  return (OpcA == Instruction::FSub && OpcB == Instruction::FAdd) ||
         (OpcA == Instruction::FAdd && OpcB == Instruction::FSub);
}

static bool isInstructionPairMul(Instruction *A, Instruction *B) {
  auto Pattern =
      m_BinOp(m_FMul(m_Value(), m_Value()), m_FMul(m_Value(), m_Value()));

  return match(A, Pattern) && match(B, Pattern);
}

ComplexDeinterleavingGraph::NodePtr
ComplexDeinterleavingGraph::identifyNode(Instruction *Real, Instruction *Imag) {
  LLVM_DEBUG(dbgs() << "identifyNode on " << *Real << " / " << *Imag << "\n");
  if (NodePtr CN = getContainingComposite(Real, Imag)) {
    LLVM_DEBUG(dbgs() << " - Folding to existing node\n");
    return CN;
  }

  auto *RealShuffle = dyn_cast<ShuffleVectorInst>(Real);
  auto *ImagShuffle = dyn_cast<ShuffleVectorInst>(Imag);
  if (RealShuffle && ImagShuffle) {
    Value *RealOp1 = RealShuffle->getOperand(1);
    if (!isa<UndefValue>(RealOp1) && !isa<ConstantAggregateZero>(RealOp1)) {
      LLVM_DEBUG(dbgs() << " - RealOp1 is not undef or zero.\n");
      return nullptr;
    }
    Value *ImagOp1 = ImagShuffle->getOperand(1);
    if (!isa<UndefValue>(ImagOp1) && !isa<ConstantAggregateZero>(ImagOp1)) {
      LLVM_DEBUG(dbgs() << " - ImagOp1 is not undef or zero.\n");
      return nullptr;
    }

    Value *RealOp0 = RealShuffle->getOperand(0);
    Value *ImagOp0 = ImagShuffle->getOperand(0);

    if (RealOp0 != ImagOp0) {
      LLVM_DEBUG(dbgs() << " - Shuffle operands are not equal.\n");
      return nullptr;
    }

    ArrayRef<int> RealMask = RealShuffle->getShuffleMask();
    ArrayRef<int> ImagMask = ImagShuffle->getShuffleMask();
    if (!isDeinterleavingMask(RealMask) || !isDeinterleavingMask(ImagMask)) {
      LLVM_DEBUG(dbgs() << " - Masks are not deinterleaving.\n");
      return nullptr;
    }

    if (RealMask[0] != 0 || ImagMask[0] != 1) {
      LLVM_DEBUG(dbgs() << " - Masks do not have the correct initial value.\n");
      return nullptr;
    }

    // Type checking, the shuffle type should be a vector type of the same
    // scalar type, but half the size
    auto CheckType = [&](ShuffleVectorInst *Shuffle) {
      Value *Op = Shuffle->getOperand(0);
      auto *ShuffleTy = cast<FixedVectorType>(Shuffle->getType());
      auto *OpTy = cast<FixedVectorType>(Op->getType());

      if (OpTy->getScalarType() != ShuffleTy->getScalarType())
        return false;
      if ((ShuffleTy->getNumElements() * 2) != OpTy->getNumElements())
        return false;

      return true;
    };

    auto CheckDeinterleavingShuffle = [&](ShuffleVectorInst *Shuffle) -> bool {
      if (!CheckType(Shuffle))
        return false;

      ArrayRef<int> Mask = Shuffle->getShuffleMask();
      int Last = *Mask.rbegin();

      Value *Op = Shuffle->getOperand(0);
      auto *OpTy = cast<FixedVectorType>(Op->getType());
      int NumElements = OpTy->getNumElements();

      // Ensure that the deinterleaving shuffle only pulls from the first
      // shuffle operand.
      return Last < NumElements;
    };

    if (RealShuffle->getType() != ImagShuffle->getType()) {
      LLVM_DEBUG(dbgs() << " - Shuffle types aren't equal.\n");
      return nullptr;
    }
    if (!CheckDeinterleavingShuffle(RealShuffle)) {
      LLVM_DEBUG(dbgs() << " - RealShuffle is invalid type.\n");
      return nullptr;
    }
    if (!CheckDeinterleavingShuffle(ImagShuffle)) {
      LLVM_DEBUG(dbgs() << " - ImagShuffle is invalid type.\n");
      return nullptr;
    }

    NodePtr PlaceholderNode =
        prepareCompositeNode(llvm::ComplexDeinterleavingOperation::Shuffle,
                             RealShuffle, ImagShuffle);
    PlaceholderNode->ReplacementNode = RealShuffle->getOperand(0);
    return submitCompositeNode(PlaceholderNode);
  }
  if (RealShuffle || ImagShuffle)
    return nullptr;

  auto *VTy = cast<FixedVectorType>(Real->getType());
  auto *NewVTy =
      FixedVectorType::get(VTy->getScalarType(), VTy->getNumElements() * 2);

  if (TL->isComplexDeinterleavingOperationSupported(
          ComplexDeinterleavingOperation::CMulPartial, NewVTy) &&
      isInstructionPairMul(Real, Imag)) {
    return identifyPartialMul(Real, Imag);
  }

  if (TL->isComplexDeinterleavingOperationSupported(
          ComplexDeinterleavingOperation::CAdd, NewVTy) &&
      isInstructionPairAdd(Real, Imag)) {
    return identifyAdd(Real, Imag);
  }

  return nullptr;
}

bool ComplexDeinterleavingGraph::identifyNodes(Instruction *RootI) {
  Instruction *Real;
  Instruction *Imag;
  if (!match(RootI, m_Shuffle(m_Instruction(Real), m_Instruction(Imag))))
    return false;

  RootValue = RootI;
  AllInstructions.insert(RootI);
  RootNode = identifyNode(Real, Imag);

  LLVM_DEBUG({
    Function *F = RootI->getFunction();
    BasicBlock *B = RootI->getParent();
    dbgs() << "Complex deinterleaving graph for " << F->getName()
           << "::" << B->getName() << ".\n";
    dump(dbgs());
    dbgs() << "\n";
  });

  // Check all instructions have internal uses
  for (const auto &Node : CompositeNodes) {
    if (!Node->hasAllInternalUses(AllInstructions)) {
      LLVM_DEBUG(dbgs() << "  - Invalid internal uses\n");
      return false;
    }
  }
  return RootNode != nullptr;
}

Value *ComplexDeinterleavingGraph::replaceNode(
    ComplexDeinterleavingGraph::RawNodePtr Node) {
  if (Node->ReplacementNode)
    return Node->ReplacementNode;

  Value *Input0 = replaceNode(Node->Operands[0]);
  Value *Input1 = replaceNode(Node->Operands[1]);
  Value *Accumulator =
      Node->Operands.size() > 2 ? replaceNode(Node->Operands[2]) : nullptr;

  assert(Input0->getType() == Input1->getType() &&
         "Node inputs need to be of the same type");

  Node->ReplacementNode = TL->createComplexDeinterleavingIR(
      Node->Real, Node->Operation, Node->Rotation, Input0, Input1, Accumulator);

  assert(Node->ReplacementNode && "Target failed to create Intrinsic call.");
  NumComplexTransformations += 1;
  return Node->ReplacementNode;
}

void ComplexDeinterleavingGraph::replaceNodes() {
  Value *R = replaceNode(RootNode.get());
  assert(R && "Unable to find replacement for RootValue");
  RootValue->replaceAllUsesWith(R);
}

bool ComplexDeinterleavingCompositeNode::hasAllInternalUses(
    SmallPtrSet<Instruction *, 16> &AllInstructions) {
  if (Operation == ComplexDeinterleavingOperation::Shuffle)
    return true;

  for (auto *User : Real->users()) {
    if (!AllInstructions.contains(cast<Instruction>(User)))
      return false;
  }
  for (auto *User : Imag->users()) {
    if (!AllInstructions.contains(cast<Instruction>(User)))
      return false;
  }
  for (auto *I : InternalInstructions) {
    for (auto *User : I->users()) {
      if (!AllInstructions.contains(cast<Instruction>(User)))
        return false;
    }
  }
  return true;
}