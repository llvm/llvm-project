#include "llvm/CodeGen/MachineSchedSearch.h"
#include "llvm/CodeGen/CodeGenTargetMachineImpl.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/TargetFrameLowering.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Triple.h"
#include "gtest/gtest.h"
#include <array>

using namespace llvm;

namespace {

// Include helper functions to construct a target-independent MachineFunction.
#include "MFCommon.inc"

void initializeTestDAG(std::array<SUnit, 5> &Nodes) {
  for (unsigned I = 0; I != Nodes.size(); ++I)
    Nodes[I].NodeNum = I;

  // 0 --\
  //       2 -> 3    4
  // 1 --/
  Nodes[2].addPred(SDep(&Nodes[0], SDep::Artificial));
  Nodes[2].addPred(SDep(&Nodes[1], SDep::Artificial));
  Nodes[3].addPred(SDep(&Nodes[2], SDep::Artificial));
}

class TestCompleteScheduleOptimizer final
    : public MachineSchedCompleteScheduleOptimizer {
  SmallVector<unsigned> Result;
  SmallVector<unsigned> *SeenFounder;
  bool AcceptResult;
  bool AcceptValidation;

public:
  TestCompleteScheduleOptimizer(ArrayRef<unsigned> Result,
                                bool AcceptResult = true,
                                bool AcceptValidation = true,
                                SmallVector<unsigned> *SeenFounder = nullptr)
      : Result(Result), SeenFounder(SeenFounder), AcceptResult(AcceptResult),
        AcceptValidation(AcceptValidation) {}

  bool optimizeCompleteSchedule(const MachineSchedSearchRegion &,
                                ArrayRef<unsigned> Founder,
                                SmallVectorImpl<unsigned> &Order) override {
    if (SeenFounder)
      SeenFounder->assign(Founder.begin(), Founder.end());
    Order.assign(Result.begin(), Result.end());
    return AcceptResult;
  }

  bool validateCompleteSchedule(const MachineSchedSearchRegion &,
                                ArrayRef<unsigned>) override {
    return AcceptValidation;
  }
};

class TestScheduleDAGMI final : public ScheduleDAGMI {
public:
  TestScheduleDAGMI(MachineSchedContext *Context,
                    std::unique_ptr<MachineSchedStrategy> Strategy)
      : ScheduleDAGMI(Context, std::move(Strategy),
                      /*RemoveKillFlags=*/false) {}

  using ScheduleDAGMI::runPostScheduleOptimizer;
};

SmallVector<unsigned> initializeAndPick(ArrayRef<unsigned> Result,
                                        bool AcceptResult = true,
                                        bool AcceptValidation = true) {
  LLVMContext Context;
  Module M("MachineSchedSearchTest", Context);
  std::unique_ptr<MachineFunction> MF = createMachineFunction(Context, M);
  MachineSchedContext SchedContext;
  SchedContext.MF = MF.get();

  auto Optimizer = std::make_unique<TestCompleteScheduleOptimizer>(
      Result, AcceptResult, AcceptValidation);
  auto Strategy = std::make_unique<MachineSchedCompleteScheduleReplayer>(
      std::move(Optimizer));
  MachineSchedCompleteScheduleReplayer *StrategyPtr = Strategy.get();
  ScheduleDAGMI DAG(&SchedContext, std::move(Strategy),
                    /*RemoveKillFlags=*/false);
  DAG.SUnits.resize(3);
  for (unsigned I = 0; I != DAG.SUnits.size(); ++I)
    DAG.SUnits[I].NodeNum = I;

  StrategyPtr->initialize(&DAG);
  SmallVector<unsigned> Picked;
  bool IsTopNode = false;
  while (SUnit *SU = StrategyPtr->pickNode(IsTopNode)) {
    EXPECT_TRUE(IsTopNode);
    Picked.push_back(SU->NodeNum);
  }
  return Picked;
}

SmallVector<unsigned>
runPostOptimization(ArrayRef<unsigned> Result, bool AcceptResult = true,
                    bool AcceptValidation = true,
                    bool MaterializeCompleteFounder = true,
                    SmallVector<unsigned> *SeenFounder = nullptr,
                    ArrayRef<unsigned> MaterializedFounder = {}) {
  LLVMContext Context;
  Module M("MachineSchedCompleteScheduleOptimizerTest", Context);
  std::unique_ptr<MachineFunction> MF = createMachineFunction(Context, M);
  MachineBasicBlock *MBB = MF->CreateMachineBasicBlock();
  MF->push_back(MBB);

  MCInstrDesc Desc = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::array<MachineInstr *, 3> Instructions;
  for (MachineInstr *&MI : Instructions)
    MI = MF->CreateMachineInstr(Desc, DebugLoc());
  if (MaterializedFounder.empty()) {
    for (MachineInstr *MI : Instructions)
      MBB->push_back(MI);
  } else {
    for (unsigned Node : MaterializedFounder)
      MBB->push_back(Instructions[Node]);
  }

  MachineSchedContext SchedContext;
  SchedContext.MF = MF.get();
  auto ReplayOptimizer = std::make_unique<TestCompleteScheduleOptimizer>(
      SmallVector<unsigned, 3>{0, 1, 2});
  auto Strategy = std::make_unique<MachineSchedCompleteScheduleReplayer>(
      std::move(ReplayOptimizer));
  TestScheduleDAGMI DAG(&SchedContext, std::move(Strategy));
  DAG.startBlock(MBB);
  DAG.enterRegion(MBB, MBB->begin(), MBB->end(), Instructions.size());
  for (auto [Ordinal, MI] : enumerate(Instructions)) {
    DAG.SUnits.emplace_back(MI, Ordinal);
    DAG.SUnits.back().isScheduled = MaterializeCompleteFounder;
  }

  DAG.setPostScheduleOptimizer(std::make_unique<TestCompleteScheduleOptimizer>(
      Result, AcceptResult, AcceptValidation, SeenFounder));
  DAG.runPostScheduleOptimizer();

  SmallVector<unsigned> Order;
  for (MachineInstr &MI : *MBB) {
    auto I = llvm::find(Instructions, &MI);
    if (I == Instructions.end()) {
      ADD_FAILURE() << "instruction not found in original region";
      return {};
    }
    Order.push_back(std::distance(Instructions.begin(), I));
  }
  return Order;
}

} // namespace

TEST(MachineSchedSearchRegion, ValidatesCompleteOrders) {
  std::array<SUnit, 5> Nodes;
  initializeTestDAG(Nodes);
  MachineSchedSearchRegion Region(Nodes);

  EXPECT_TRUE(Region.isLegalOrder({0, 1, 2, 3, 4}));
  EXPECT_TRUE(Region.isLegalOrder({4, 1, 0, 2, 3}));

  EXPECT_FALSE(Region.isLegalOrder({0, 1, 2, 3}));
  EXPECT_FALSE(Region.isLegalOrder({0, 1, 2, 3, 5}));
  EXPECT_FALSE(Region.isLegalOrder({0, 1, 2, 3, 3}));
  EXPECT_FALSE(Region.isLegalOrder({2, 0, 1, 3, 4}));
  EXPECT_FALSE(Region.isLegalOrder({0, 1, 3, 2, 4}));
}

TEST(MachineSchedSearchRegion, WeakEdgesArePreferences) {
  std::array<SUnit, 2> Nodes;
  for (unsigned I = 0; I != Nodes.size(); ++I)
    Nodes[I].NodeNum = I;
  Nodes[1].addPred(SDep(&Nodes[0], SDep::Weak));

  MachineSchedSearchRegion Region(Nodes);
  EXPECT_TRUE(Region.isLegalOrder({0, 1}));
  EXPECT_TRUE(Region.isLegalOrder({1, 0}));
}

TEST(MachineSchedSearchRegion, UsesViewLocalOrdinals) {
  std::array<SUnit, 2> Nodes;
  Nodes[0].NodeNum = 17;
  Nodes[1].NodeNum = 9;
  Nodes[1].addPred(SDep(&Nodes[0], SDep::Artificial));

  MachineSchedSearchRegion Region(Nodes);
  EXPECT_TRUE(Region.isLegalOrder({0, 1}));
  EXPECT_FALSE(Region.isLegalOrder({1, 0}));
  EXPECT_EQ(&Region.getSUnit(0), &Nodes[0]);
  EXPECT_EQ(&Region.getSUnit(1), &Nodes[1]);
}

TEST(MachineSchedSearchRegion, BuildsStableTopologicalFallback) {
  std::array<SUnit, 3> Nodes;
  for (unsigned I = 0; I != Nodes.size(); ++I)
    Nodes[I].NodeNum = I;
  Nodes[0].addPred(SDep(&Nodes[1], SDep::Artificial));

  MachineSchedSearchRegion Region(Nodes);
  EXPECT_FALSE(Region.isLegalOrder(Region.getInitialOrder()));
  EXPECT_EQ(Region.getTopologicalOrder(), (SmallVector<unsigned, 3>{1, 0, 2}));
}

TEST(MachineSchedSearchRegion, ComputesLegalRelocationRange) {
  std::array<SUnit, 5> Nodes;
  initializeTestDAG(Nodes);
  MachineSchedSearchRegion Region(Nodes);
  const unsigned Order[] = {0, 1, 2, 3, 4};
  MachineSchedSearchRegion::MoveRange Range;

  ASSERT_TRUE(Region.getLegalMoveRange(Order, 0, Range));
  EXPECT_EQ(Range.Begin, 0u);
  EXPECT_EQ(Range.End, 1u);

  ASSERT_TRUE(Region.getLegalMoveRange(Order, 2, Range));
  EXPECT_EQ(Range.Begin, 2u);
  EXPECT_EQ(Range.End, 2u);

  ASSERT_TRUE(Region.getLegalMoveRange(Order, 4, Range));
  EXPECT_EQ(Range.Begin, 0u);
  EXPECT_EQ(Range.End, 4u);

  EXPECT_FALSE(Region.getLegalMoveRange({2, 0, 1, 3, 4}, 0, Range));
  EXPECT_FALSE(Region.getLegalMoveRange(Order, 5, Range));
}

TEST(MachineSchedCompleteScheduleReplayer, ReplaysValidatedCompleteOrder) {
  EXPECT_EQ(initializeAndPick({2, 0, 1}), (SmallVector<unsigned, 3>{2, 0, 1}));
}

TEST(MachineSchedCompleteScheduleReplayer, PreservesExistingOrderOnFailure) {
  EXPECT_EQ(initializeAndPick({2, 2, 1}), (SmallVector<unsigned, 3>{0, 1, 2}));
  EXPECT_EQ(initializeAndPick({2, 0, 1}, /*AcceptResult=*/false),
            (SmallVector<unsigned, 3>{0, 1, 2}));
  EXPECT_EQ(initializeAndPick({2, 0, 1}, /*AcceptResult=*/true,
                              /*AcceptValidation=*/false),
            (SmallVector<unsigned, 3>{0, 1, 2}));
}

TEST(MachineSchedCompleteScheduleOptimizer, OptimizesMaterializedFounder) {
  SmallVector<unsigned> SeenFounder;
  EXPECT_EQ(runPostOptimization({2, 0, 1}, /*AcceptResult=*/true,
                                /*AcceptValidation=*/true,
                                /*MaterializeCompleteFounder=*/true,
                                &SeenFounder,
                                /*MaterializedFounder=*/{1, 2, 0}),
            (SmallVector<unsigned, 3>{2, 0, 1}));
  EXPECT_EQ(SeenFounder, (SmallVector<unsigned, 3>{1, 2, 0}));
}

TEST(MachineSchedCompleteScheduleOptimizer, PreservesFounderOnFailure) {
  EXPECT_EQ(runPostOptimization({2, 2, 1}),
            (SmallVector<unsigned, 3>{0, 1, 2}));
  EXPECT_EQ(runPostOptimization({2, 0, 1}, /*AcceptResult=*/false),
            (SmallVector<unsigned, 3>{0, 1, 2}));
  EXPECT_EQ(runPostOptimization({2, 0, 1}, /*AcceptResult=*/true,
                                /*AcceptValidation=*/false),
            (SmallVector<unsigned, 3>{0, 1, 2}));
}

TEST(MachineSchedCompleteScheduleOptimizer, SkipsIncompleteFounder) {
  SmallVector<unsigned> SeenFounder;
  EXPECT_EQ(runPostOptimization({2, 0, 1}, /*AcceptResult=*/true,
                                /*AcceptValidation=*/true,
                                /*MaterializeCompleteFounder=*/false,
                                &SeenFounder),
            (SmallVector<unsigned, 3>{0, 1, 2}));
  EXPECT_TRUE(SeenFounder.empty());
}
