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
// Instead of relying on Shuffle operations, vector interleaving and
// deinterleaving can be represented by vector.interleave2 and
// vector.deinterleave2 intrinsics. Scalable vectors can be represented only by
// these intrinsics, whereas, fixed-width vectors are recognized for both
// shufflevector instruction and intrinsics.
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

  ComplexDeinterleavingRotation Rotation =
      ComplexDeinterleavingRotation::Rotation_0;
  SmallVector<RawNodePtr> Operands;
  Value *ReplacementNode = nullptr;

  void addOperand(NodePtr Node) { Operands.push_back(Node.get()); }

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
  }
};

class ComplexDeinterleavingGraph {
public:
  using NodePtr = ComplexDeinterleavingCompositeNode::NodePtr;
  using RawNodePtr = ComplexDeinterleavingCompositeNode::RawNodePtr;
  explicit ComplexDeinterleavingGraph(const TargetLowering *TL,
                                      const TargetLibraryInfo *TLI)
      : TL(TL), TLI(TLI) {}

private:
  const TargetLowering *TL = nullptr;
  const TargetLibraryInfo *TLI = nullptr;
  SmallVector<NodePtr> CompositeNodes;

  SmallPtrSet<Instruction *, 16> FinalInstructions;

  /// Root instructions are instructions from which complex computation starts
  std::map<Instruction *, NodePtr> RootToNode;

  /// Topologically sorted root instructions
  SmallVector<Instruction *, 1> OrderedRoots;

  NodePtr prepareCompositeNode(ComplexDeinterleavingOperation Operation,
                               Instruction *R, Instruction *I) {
    return std::make_shared<ComplexDeinterleavingCompositeNode>(Operation, R,
                                                                I);
  }

  NodePtr submitCompositeNode(NodePtr Node) {
    CompositeNodes.push_back(Node);
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
  NodePtr identifySymmetricOperation(Instruction *Real, Instruction *Imag);

  NodePtr identifyNode(Instruction *I, Instruction *J);

  NodePtr identifyRoot(Instruction *I);

  /// Identifies the Deinterleave operation applied to a vector containing
  /// complex numbers. There are two ways to represent the Deinterleave
  /// operation:
  /// * Using two shufflevectors with even indices for /pReal instruction and
  /// odd indices for /pImag instructions (only for fixed-width vectors)
  /// * Using two extractvalue instructions applied to `vector.deinterleave2`
  /// intrinsic (for both fixed and scalable vectors)
  NodePtr identifyDeinterleave(Instruction *Real, Instruction *Imag);

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

  /// Check that every instruction, from the roots to the leaves, has internal
  /// uses.
  bool checkNodes();

  /// Perform the actual replacement of the underlying instruction graph.
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
  ComplexDeinterleavingGraph Graph(TL, TLI);
  for (auto &I : *B)
    Graph.identifyNodes(&I);

  if (Graph.checkNodes()) {
    Graph.replaceNodes();
    return true;
  }

  return false;
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

  auto *CRInst = dyn_cast<Instruction>(CR);
  auto *CIInst = dyn_cast<Instruction>(CI);

  if (!CRInst || !CIInst) {
    LLVM_DEBUG(dbgs() << "  - Common operands are not instructions.\n");
    return nullptr;
  }

  NodePtr CNode = identifyNodeWithImplicitAdd(CRInst, CIInst, PartialMatch);
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
  if ((Real->getOpcode() == Instruction::FSub &&
       Imag->getOpcode() == Instruction::FAdd) ||
      (Real->getOpcode() == Instruction::Sub &&
       Imag->getOpcode() == Instruction::Add))
    Rotation = ComplexDeinterleavingRotation::Rotation_90;
  else if ((Real->getOpcode() == Instruction::FAdd &&
            Imag->getOpcode() == Instruction::FSub) ||
           (Real->getOpcode() == Instruction::Add &&
            Imag->getOpcode() == Instruction::Sub))
    Rotation = ComplexDeinterleavingRotation::Rotation_270;
  else {
    LLVM_DEBUG(dbgs() << " - Unhandled case, rotation is not assigned.\n");
    return nullptr;
  }

  auto *AR = dyn_cast<Instruction>(Real->getOperand(0));
  auto *BI = dyn_cast<Instruction>(Real->getOperand(1));
  auto *AI = dyn_cast<Instruction>(Imag->getOperand(0));
  auto *BR = dyn_cast<Instruction>(Imag->getOperand(1));

  if (!AR || !AI || !BR || !BI) {
    LLVM_DEBUG(dbgs() << " - Not all operands are instructions.\n");
    return nullptr;
  }

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
         (OpcA == Instruction::FAdd && OpcB == Instruction::FSub) ||
         (OpcA == Instruction::Sub && OpcB == Instruction::Add) ||
         (OpcA == Instruction::Add && OpcB == Instruction::Sub);
}

static bool isInstructionPairMul(Instruction *A, Instruction *B) {
  auto Pattern =
      m_BinOp(m_FMul(m_Value(), m_Value()), m_FMul(m_Value(), m_Value()));

  return match(A, Pattern) && match(B, Pattern);
}

static bool isInstructionPotentiallySymmetric(Instruction *I) {
  switch (I->getOpcode()) {
  case Instruction::FAdd:
  case Instruction::FSub:
  case Instruction::FMul:
  case Instruction::FNeg:
    return true;
  default:
    return false;
  }
}

ComplexDeinterleavingGraph::NodePtr
ComplexDeinterleavingGraph::identifySymmetricOperation(Instruction *Real,
                                                       Instruction *Imag) {
  if (Real->getOpcode() != Imag->getOpcode())
    return nullptr;

  if (!isInstructionPotentiallySymmetric(Real) ||
      !isInstructionPotentiallySymmetric(Imag))
    return nullptr;

  auto *R0 = dyn_cast<Instruction>(Real->getOperand(0));
  auto *I0 = dyn_cast<Instruction>(Imag->getOperand(0));

  if (!R0 || !I0)
    return nullptr;

  NodePtr Op0 = identifyNode(R0, I0);
  NodePtr Op1 = nullptr;
  if (Op0 == nullptr)
    return nullptr;

  if (Real->isBinaryOp()) {
    auto *R1 = dyn_cast<Instruction>(Real->getOperand(1));
    auto *I1 = dyn_cast<Instruction>(Imag->getOperand(1));
    if (!R1 || !I1)
      return nullptr;

    Op1 = identifyNode(R1, I1);
    if (Op1 == nullptr)
      return nullptr;
  }

  auto Node = prepareCompositeNode(ComplexDeinterleavingOperation::Symmetric,
                                   Real, Imag);
  Node->addOperand(Op0);
  if (Real->isBinaryOp())
    Node->addOperand(Op1);

  return submitCompositeNode(Node);
}

ComplexDeinterleavingGraph::NodePtr
ComplexDeinterleavingGraph::identifyNode(Instruction *Real, Instruction *Imag) {
  LLVM_DEBUG(dbgs() << "identifyNode on " << *Real << " / " << *Imag << "\n");
  if (NodePtr CN = getContainingComposite(Real, Imag)) {
    LLVM_DEBUG(dbgs() << " - Folding to existing node\n");
    return CN;
  }

  NodePtr Node = identifyDeinterleave(Real, Imag);
  if (Node)
    return Node;

  auto *VTy = cast<VectorType>(Real->getType());
  auto *NewVTy = VectorType::getDoubleElementsVectorType(VTy);

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

  auto Symmetric = identifySymmetricOperation(Real, Imag);
  LLVM_DEBUG(if (Symmetric == nullptr) dbgs()
             << "  - Not recognised as a valid pattern.\n");
  return Symmetric;
}

bool ComplexDeinterleavingGraph::identifyNodes(Instruction *RootI) {
  auto RootNode = identifyRoot(RootI);
  if (!RootNode)
    return false;

  LLVM_DEBUG({
    Function *F = RootI->getFunction();
    BasicBlock *B = RootI->getParent();
    dbgs() << "Complex deinterleaving graph for " << F->getName()
           << "::" << B->getName() << ".\n";
    dump(dbgs());
    dbgs() << "\n";
  });
  RootToNode[RootI] = RootNode;
  OrderedRoots.push_back(RootI);
  return true;
}

bool ComplexDeinterleavingGraph::checkNodes() {
  // Collect all instructions from roots to leaves
  SmallPtrSet<Instruction *, 16> AllInstructions;
  SmallVector<Instruction *, 8> Worklist;
  for (auto *I : OrderedRoots)
    Worklist.push_back(I);

  // Extract all instructions that are used by all XCMLA/XCADD/ADD/SUB/NEG
  // chains
  while (!Worklist.empty()) {
    auto *I = Worklist.back();
    Worklist.pop_back();

    if (!AllInstructions.insert(I).second)
      continue;

    for (Value *Op : I->operands()) {
      if (auto *OpI = dyn_cast<Instruction>(Op)) {
        if (!FinalInstructions.count(I))
          Worklist.emplace_back(OpI);
      }
    }
  }

  // Find instructions that have users outside of chain
  SmallVector<Instruction *, 2> OuterInstructions;
  for (auto *I : AllInstructions) {
    // Skip root nodes
    if (RootToNode.count(I))
      continue;

    for (User *U : I->users()) {
      if (AllInstructions.count(cast<Instruction>(U)))
        continue;

      // Found an instruction that is not used by XCMLA/XCADD chain
      Worklist.emplace_back(I);
      break;
    }
  }

  // If any instructions are found to be used outside, find and remove roots
  // that somehow connect to those instructions.
  SmallPtrSet<Instruction *, 16> Visited;
  while (!Worklist.empty()) {
    auto *I = Worklist.back();
    Worklist.pop_back();
    if (!Visited.insert(I).second)
      continue;

    // Found an impacted root node. Removing it from the nodes to be
    // deinterleaved
    if (RootToNode.count(I)) {
      LLVM_DEBUG(dbgs() << "Instruction " << *I
                        << " could be deinterleaved but its chain of complex "
                           "operations have an outside user\n");
      RootToNode.erase(I);
    }

    if (!AllInstructions.count(I) || FinalInstructions.count(I))
      continue;

    for (User *U : I->users())
      Worklist.emplace_back(cast<Instruction>(U));

    for (Value *Op : I->operands()) {
      if (auto *OpI = dyn_cast<Instruction>(Op))
        Worklist.emplace_back(OpI);
    }
  }
  return !RootToNode.empty();
}

ComplexDeinterleavingGraph::NodePtr
ComplexDeinterleavingGraph::identifyRoot(Instruction *RootI) {
  if (auto *Intrinsic = dyn_cast<IntrinsicInst>(RootI)) {
    if (Intrinsic->getIntrinsicID() !=
        Intrinsic::experimental_vector_interleave2)
      return nullptr;

    auto *Real = dyn_cast<Instruction>(Intrinsic->getOperand(0));
    auto *Imag = dyn_cast<Instruction>(Intrinsic->getOperand(1));
    if (!Real || !Imag)
      return nullptr;

    return identifyNode(Real, Imag);
  }

  auto *SVI = dyn_cast<ShuffleVectorInst>(RootI);
  if (!SVI)
    return nullptr;

  // Look for a shufflevector that takes separate vectors of the real and
  // imaginary components and recombines them into a single vector.
  if (!isInterleavingMask(SVI->getShuffleMask()))
    return nullptr;

  Instruction *Real;
  Instruction *Imag;
  if (!match(RootI, m_Shuffle(m_Instruction(Real), m_Instruction(Imag))))
    return nullptr;

  return identifyNode(Real, Imag);
}

ComplexDeinterleavingGraph::NodePtr
ComplexDeinterleavingGraph::identifyDeinterleave(Instruction *Real,
                                                 Instruction *Imag) {
  Instruction *I = nullptr;
  Value *FinalValue = nullptr;
  if (match(Real, m_ExtractValue<0>(m_Instruction(I))) &&
      match(Imag, m_ExtractValue<1>(m_Specific(I))) &&
      match(I, m_Intrinsic<Intrinsic::experimental_vector_deinterleave2>(
                   m_Value(FinalValue)))) {
    NodePtr PlaceholderNode = prepareCompositeNode(
        llvm::ComplexDeinterleavingOperation::Deinterleave, Real, Imag);
    PlaceholderNode->ReplacementNode = FinalValue;
    FinalInstructions.insert(Real);
    FinalInstructions.insert(Imag);
    return submitCompositeNode(PlaceholderNode);
  }

  auto *RealShuffle = dyn_cast<ShuffleVectorInst>(Real);
  auto *ImagShuffle = dyn_cast<ShuffleVectorInst>(Imag);
  if (!RealShuffle || !ImagShuffle) {
    if (RealShuffle || ImagShuffle)
      LLVM_DEBUG(dbgs() << " - There's a shuffle where there shouldn't be.\n");
    return nullptr;
  }

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
      prepareCompositeNode(llvm::ComplexDeinterleavingOperation::Deinterleave,
                           RealShuffle, ImagShuffle);
  PlaceholderNode->ReplacementNode = RealShuffle->getOperand(0);
  FinalInstructions.insert(RealShuffle);
  FinalInstructions.insert(ImagShuffle);
  return submitCompositeNode(PlaceholderNode);
}

static Value *replaceSymmetricNode(ComplexDeinterleavingGraph::RawNodePtr Node,
                                   Value *InputA, Value *InputB) {
  Instruction *I = Node->Real;
  if (I->isUnaryOp())
    assert(!InputB &&
           "Unary symmetric operations need one input, but two were provided.");
  else if (I->isBinaryOp())
    assert(InputB && "Binary symmetric operations need two inputs, only one "
                     "was provided.");

  IRBuilder<> B(I);

  switch (I->getOpcode()) {
  case Instruction::FNeg:
    return B.CreateFNegFMF(InputA, I);
  case Instruction::FAdd:
    return B.CreateFAddFMF(InputA, InputB, I);
  case Instruction::FSub:
    return B.CreateFSubFMF(InputA, InputB, I);
  case Instruction::FMul:
    return B.CreateFMulFMF(InputA, InputB, I);
  }

  return nullptr;
}

Value *ComplexDeinterleavingGraph::replaceNode(
    ComplexDeinterleavingGraph::RawNodePtr Node) {
  if (Node->ReplacementNode)
    return Node->ReplacementNode;

  Value *Input0 = replaceNode(Node->Operands[0]);
  Value *Input1 =
      Node->Operands.size() > 1 ? replaceNode(Node->Operands[1]) : nullptr;
  Value *Accumulator =
      Node->Operands.size() > 2 ? replaceNode(Node->Operands[2]) : nullptr;

  if (Input1)
    assert(Input0->getType() == Input1->getType() &&
           "Node inputs need to be of the same type");

  if (Node->Operation == ComplexDeinterleavingOperation::Symmetric)
    Node->ReplacementNode = replaceSymmetricNode(Node, Input0, Input1);
  else
    Node->ReplacementNode = TL->createComplexDeinterleavingIR(
        Node->Real, Node->Operation, Node->Rotation, Input0, Input1,
        Accumulator);

  assert(Node->ReplacementNode && "Target failed to create Intrinsic call.");
  NumComplexTransformations += 1;
  return Node->ReplacementNode;
}

void ComplexDeinterleavingGraph::replaceNodes() {
  SmallVector<Instruction *, 16> DeadInstrRoots;
  for (auto *RootInstruction : OrderedRoots) {
    // Check if this potential root went through check process and we can
    // deinterleave it
    if (!RootToNode.count(RootInstruction))
      continue;

    IRBuilder<> Builder(RootInstruction);
    auto RootNode = RootToNode[RootInstruction];
    Value *R = replaceNode(RootNode.get());
    assert(R && "Unable to find replacement for RootInstruction");
    DeadInstrRoots.push_back(RootInstruction);
    RootInstruction->replaceAllUsesWith(R);
  }

  for (auto *I : DeadInstrRoots)
    RecursivelyDeleteTriviallyDeadInstructions(I, TLI);
}
