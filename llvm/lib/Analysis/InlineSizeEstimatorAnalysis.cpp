//===- InlineSizeEstimatorAnalysis.cpp - IR to native size from ML model --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This implements feature and label extraction for offline supervised learning
// of a IR to native size model.
//
//===----------------------------------------------------------------------===//
#include "llvm/Analysis/InlineSizeEstimatorAnalysis.h"

#ifdef LLVM_HAVE_TF_API
#include "llvm/Analysis/Utils/TFUtils.h"
#endif
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PassManager.h"
#include "llvm/MC/MCAsmLayout.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <deque>

using namespace llvm;

AnalysisKey InlineSizeEstimatorAnalysis::Key;

#define DEBUG_TYPE "inline-size-estimator"

#ifdef LLVM_HAVE_TF_API
cl::opt<std::string> TFIR2NativeModelPath(
    "ml-inliner-ir2native-model", cl::Hidden,
    cl::desc("Path to saved model evaluating native size from IR."));

namespace {
unsigned getMaxInstructionID() {
#define LAST_OTHER_INST(NR) return NR;
#include "llvm/IR/Instruction.def"
}

class IRToNativeSizeLearning {
public:
  enum class NamedFeatureIndex : size_t {
    InitialSize,
    Blocks,
    Calls,
    IsLocal,
    IsLinkOnceODR,
    IsLinkOnce,
    Loops,
    MaxLoopDepth,
    MaxDomTreeLevel,

    NumNamedFeatures
  };
  static const size_t NumNamedFeatures =
      static_cast<size_t>(NamedFeatureIndex::NumNamedFeatures);
  struct FunctionFeatures {
    static std::vector<std::pair<size_t, size_t>>
        ImportantInstructionSuccessions;
    static const size_t FeatureCount;

    std::array<int32_t, NumNamedFeatures> NamedFeatures = {0};
    std::vector<int32_t> InstructionHistogram;
    std::vector<int32_t> InstructionPairHistogram;

    void fillTensor(int32_t *Ptr) const;
    int32_t &operator[](NamedFeatureIndex Pos) {
      return NamedFeatures[static_cast<size_t>(Pos)];
    }
  };
  IRToNativeSizeLearning() = default;

  static FunctionFeatures getFunctionFeatures(Function &F,
                                              FunctionAnalysisManager &FAM);

private:
  /// Sort once the feature tuples.
  struct SortFeatureTuples {
    bool IsSorted = false;
    SortFeatureTuples() {
      std::sort(FunctionFeatures::ImportantInstructionSuccessions.begin(),
                FunctionFeatures::ImportantInstructionSuccessions.end());
      IsSorted = true;
    }
  };

  static llvm::ManagedStatic<SortFeatureTuples> TupleSorter;

  static bool ensureSortedTuples() { return TupleSorter->IsSorted; }
};
llvm::ManagedStatic<IRToNativeSizeLearning::SortFeatureTuples>
    IRToNativeSizeLearning::TupleSorter;

// This is a point in time - we determined including these pairs of
// consecutive instructions (in the IR layout available at inline time) as
// features improves the model performance. We want to move away from manual
// feature selection.
// The vector is given in opcode pairs rather than labels because 1) labels
// weren't readily available, and 2) the successions were hand - extracted
std::vector<std::pair<size_t, size_t>>
    IRToNativeSizeLearning::FunctionFeatures::ImportantInstructionSuccessions =
        {{1, 34},  {15, 27}, {53, 53}, {53, 34}, {1, 11},  {32, 2},  {2, 48},
         {28, 48}, {1, 45},  {49, 32}, {57, 56}, {55, 53}, {1, 28},  {57, 34},
         {1, 1},   {32, 28}, {32, 15}, {49, 28}, {53, 1},  {2, 53},  {48, 34},
         {28, 53}, {2, 32},  {1, 40},  {32, 48}, {29, 56}, {56, 32}, {55, 56},
         {48, 56}, {1, 31},  {33, 34}, {2, 28},  {1, 12},  {55, 1},  {31, 31},
         {65, 1},  {33, 56}, {32, 32}, {13, 13}, {1, 26},  {13, 26}, {2, 1},
         {1, 33},  {47, 49}, {64, 1},  {2, 38},  {34, 53}, {48, 2},  {55, 34},
         {34, 32}, {1, 5},   {56, 13}, {2, 2},   {2, 49},  {33, 2},  {49, 39},
         {56, 49}, {33, 49}, {32, 39}, {39, 57}, {29, 33}, {31, 34}, {32, 29},
         {47, 15}, {13, 34}, {2, 33},  {32, 49}, {49, 34}, {56, 33}, {1, 30},
         {33, 33}, {31, 33}, {2, 29},  {56, 7},  {32, 13}, {2, 55},  {56, 56},
         {2, 34},  {1, 42},  {34, 49}, {1, 20},  {32, 33}, {1, 25},  {53, 28},
         {1, 14},  {31, 49}, {28, 2},  {2, 13},  {2, 56},  {1, 32},  {56, 53},
         {65, 65}, {33, 53}, {64, 64}, {13, 2},  {34, 33}, {1, 4},   {49, 2},
         {1, 9},   {56, 1},  {33, 1},  {53, 57}, {32, 53}, {13, 56}, {32, 56},
         {55, 55}, {1, 18},  {49, 56}, {34, 34}, {1, 7},   {56, 64}, {32, 1},
         {13, 33}, {55, 28}, {49, 33}, {57, 57}, {56, 34}, {34, 56}, {33, 32},
         {32, 40}, {1, 29},  {53, 2},  {34, 1},  {32, 34}, {49, 49}, {1, 24},
         {40, 34}, {1, 13},  {38, 34}, {29, 2},  {34, 2},  {1, 39},  {1, 22},
         {1, 27},  {49, 1},  {1, 8},   {56, 2}};

// We have: 9 calculated features (the features here); 1 feature for each
// instruction opcode; and 1 feature for each manually-identified sequence.
// For the latter 2, we build a histogram: we count the number of
// occurrences of each instruction opcode or succession of instructions,
// respectively.
// Note that instruction opcodes start from 1. For convenience, we also have an
// always 0 feature for the '0' opcode, hence the extra 1.
const size_t IRToNativeSizeLearning::FunctionFeatures::FeatureCount =
    IRToNativeSizeLearning::FunctionFeatures::ImportantInstructionSuccessions
        .size() +
    getMaxInstructionID() + 1 + IRToNativeSizeLearning::NumNamedFeatures;

size_t getSize(Function &F, TargetTransformInfo &TTI) {
  size_t Ret = 0;
  for (auto &BB : F)
    for (auto &I : BB)
      Ret += TTI.getInstructionCost(
          &I, TargetTransformInfo::TargetCostKind::TCK_CodeSize);
  return Ret;
}

size_t getSize(Function &F, FunctionAnalysisManager &FAM) {
  auto &TTI = FAM.getResult<TargetIRAnalysis>(F);
  return getSize(F, TTI);
}

unsigned getMaxDominatorTreeDepth(const Function &F,
                                  const DominatorTree &Tree) {
  unsigned Ret = 0;
  for (auto &BB : F)
    if (auto *TN = Tree.getNode(&BB))
      Ret = std::max(Ret, TN->getLevel());
  return Ret;
}
} // namespace

IRToNativeSizeLearning::FunctionFeatures
IRToNativeSizeLearning::getFunctionFeatures(Function &F,
                                            FunctionAnalysisManager &FAM) {
  assert(ensureSortedTuples() && "expected lazy initialization");

  auto &DomTree = FAM.getResult<DominatorTreeAnalysis>(F);
  FunctionFeatures FF;
  size_t InstrCount = getMaxInstructionID() + 1;
  FF.InstructionHistogram.resize(InstrCount);

  FF.InstructionPairHistogram.resize(
      FunctionFeatures::ImportantInstructionSuccessions.size());

  auto StartID = 0;
  auto LastID = StartID;
  auto getPairIndex = [](size_t a, size_t b) {
    auto I =
        std::find(FunctionFeatures::ImportantInstructionSuccessions.begin(),
                  FunctionFeatures::ImportantInstructionSuccessions.end(),
                  std::make_pair(a, b));
    if (I == FunctionFeatures::ImportantInstructionSuccessions.end())
      return -1;
    return static_cast<int>(std::distance(
        FunctionFeatures::ImportantInstructionSuccessions.begin(), I));
  };

  // We don't want debug calls, because they'd just add noise.
  for (auto &BB : F) {
    for (auto I = BB.instructionsWithoutDebug().begin(),
              E = BB.instructionsWithoutDebug().end();
         I != E; ++I) {
      auto ID = I->getOpcode();

      ++FF.InstructionHistogram[ID];
      int PairIndex = getPairIndex(LastID, ID);
      if (PairIndex >= 0)
        ++FF.InstructionPairHistogram[PairIndex];
      LastID = ID;
      if (isa<CallBase>(*I))
        ++FF[NamedFeatureIndex::Calls];
    }
  }

  FF[NamedFeatureIndex::InitialSize] = getSize(F, FAM);
  FF[NamedFeatureIndex::IsLocal] = F.hasLocalLinkage();
  FF[NamedFeatureIndex::IsLinkOnceODR] = F.hasLinkOnceODRLinkage();
  FF[NamedFeatureIndex::IsLinkOnce] = F.hasLinkOnceLinkage();
  FF[NamedFeatureIndex::Blocks] =
      std::distance(F.getBasicBlockList().begin(), F.getBasicBlockList().end());
  auto &LI = FAM.getResult<LoopAnalysis>(F);
  FF[NamedFeatureIndex::Loops] = std::distance(LI.begin(), LI.end());
  for (auto &L : LI)
    FF[NamedFeatureIndex::MaxLoopDepth] =
        std::max(FF[NamedFeatureIndex::MaxLoopDepth],
                 static_cast<int32_t>(L->getLoopDepth()));
  FF[NamedFeatureIndex::MaxDomTreeLevel] = getMaxDominatorTreeDepth(F, DomTree);
  return FF;
}

void IRToNativeSizeLearning::FunctionFeatures::fillTensor(int32_t *Ptr) const {
  std::copy(NamedFeatures.begin(), NamedFeatures.end(), Ptr);
  Ptr += NamedFeatures.size();
  std::copy(InstructionHistogram.begin(), InstructionHistogram.end(), Ptr);
  Ptr += InstructionHistogram.size();
  std::copy(InstructionPairHistogram.begin(), InstructionPairHistogram.end(),
            Ptr);
}

bool InlineSizeEstimatorAnalysis::isEvaluatorRequested() {
  return !TFIR2NativeModelPath.empty();
}

InlineSizeEstimatorAnalysis::InlineSizeEstimatorAnalysis() {
  if (!isEvaluatorRequested()) {
    return;
  }
  std::vector<std::string> InputNames{"serving_default_input_1"};
  std::vector<std::string> OutputName{"StatefulPartitionedCall"};
  Evaluator = std::make_unique<TFModelEvaluator>(
      TFIR2NativeModelPath.getValue().c_str(), InputNames, OutputName);
  if (!Evaluator || !Evaluator->isValid()) {
    Evaluator.reset();
    return;
  }
  static const std::vector<int64_t> Dim{
      1, static_cast<int64_t>(
             IRToNativeSizeLearning::FunctionFeatures::FeatureCount)};

  Evaluator->initInput<int32_t>(0, Dim);
}

InlineSizeEstimatorAnalysis::Result
InlineSizeEstimatorAnalysis::run(const Function &F,
                                 FunctionAnalysisManager &FAM) {
  if (!Evaluator)
    return None;
  auto Features = IRToNativeSizeLearning::getFunctionFeatures(
      const_cast<Function &>(F), FAM);
  int32_t *V = Evaluator->getInput<int32_t>(0);
  Features.fillTensor(V);
  auto ER = Evaluator->evaluate();
  if (!ER)
    return None;
  float Ret = *ER->getTensorValue<float>(0);
  if (Ret < 0.0)
    Ret = 0.0;
  return static_cast<size_t>(Ret);
}

InlineSizeEstimatorAnalysis::~InlineSizeEstimatorAnalysis() {}
InlineSizeEstimatorAnalysis::InlineSizeEstimatorAnalysis(
    InlineSizeEstimatorAnalysis &&Other)
    : Evaluator(std::move(Other.Evaluator)) {}

#else
namespace llvm {
class TFModelEvaluator {};
} // namespace llvm
InlineSizeEstimatorAnalysis::InlineSizeEstimatorAnalysis() {}
InlineSizeEstimatorAnalysis ::InlineSizeEstimatorAnalysis(
    InlineSizeEstimatorAnalysis &&) {}
InlineSizeEstimatorAnalysis::~InlineSizeEstimatorAnalysis() {}
InlineSizeEstimatorAnalysis::Result
InlineSizeEstimatorAnalysis::run(const Function &F,
                                 FunctionAnalysisManager &FAM) {
  return None;
}
bool InlineSizeEstimatorAnalysis::isEvaluatorRequested() { return false; }
#endif

PreservedAnalyses
InlineSizeEstimatorAnalysisPrinterPass::run(Function &F,
                                            FunctionAnalysisManager &AM) {
  OS << "[InlineSizeEstimatorAnalysis] size estimate for " << F.getName()
     << ": " << AM.getResult<InlineSizeEstimatorAnalysis>(F) << "\n";
  return PreservedAnalyses::all();
}
