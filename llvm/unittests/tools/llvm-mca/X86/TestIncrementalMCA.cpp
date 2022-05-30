#include "Views/SummaryView.h"
#include "X86TestBase.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/MCA/CustomBehaviour.h"
#include "llvm/MCA/IncrementalSourceMgr.h"
#include "llvm/MCA/InstrBuilder.h"
#include "llvm/MCA/Pipeline.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include <unordered_map>

using namespace llvm;
using namespace mca;

TEST_F(X86TestBase, TestResumablePipeline) {
  mca::Context MCA(*MRI, *STI);

  mca::IncrementalSourceMgr ISM;
  // Empty CustomBehaviour.
  auto CB = std::make_unique<mca::CustomBehaviour>(*STI, ISM, *MCII);

  auto PO = getDefaultPipelineOptions();
  auto P = MCA.createDefaultPipeline(PO, ISM, *CB);
  ASSERT_TRUE(P);

  SmallVector<MCInst> MCIs;
  getSimpleInsts(MCIs, /*Repeats=*/100);

  // Add views.
  auto SV = std::make_unique<SummaryView>(STI->getSchedModel(), MCIs,
                                          PO.DispatchWidth);
  P->addEventListener(SV.get());

  mca::InstrBuilder IB(*STI, *MCII, *MRI, MCIA.get());

  // Tile size = 7
  for (unsigned i = 0U, E = MCIs.size(); i < E;) {
    for (unsigned TE = i + 7; i < TE && i < E; ++i) {
      Expected<std::unique_ptr<mca::Instruction>> InstOrErr =
          IB.createInstruction(MCIs[i]);
      ASSERT_TRUE(bool(InstOrErr));
      ISM.addInst(std::move(InstOrErr.get()));
    }

    // Run the pipeline.
    Expected<unsigned> Cycles = P->run();
    if (!Cycles) {
      // Should be a stream pause error.
      ASSERT_TRUE(Cycles.errorIsA<mca::InstStreamPause>());
      llvm::consumeError(Cycles.takeError());
    }
  }

  ISM.endOfStream();
  // Has to terminate properly.
  Expected<unsigned> Cycles = P->run();
  ASSERT_TRUE(bool(Cycles));

  json::Value Result = SV->toJSON();
  auto *ResultObj = Result.getAsObject();
  ASSERT_TRUE(ResultObj);

  // Run the baseline.
  json::Object BaselineResult;
  auto E = runBaselineMCA(BaselineResult, MCIs);
  ASSERT_FALSE(bool(E)) << "Failed to run baseline";
  auto *BaselineObj = BaselineResult.getObject(SV->getNameAsString());
  ASSERT_TRUE(BaselineObj) << "Does not contain SummaryView result";

  // Compare the results.
  constexpr const char *Fields[] = {"Instructions", "TotalCycles", "TotaluOps",
                                    "BlockRThroughput"};
  for (const auto *F : Fields) {
    auto V = ResultObj->getInteger(F);
    auto BV = BaselineObj->getInteger(F);
    ASSERT_TRUE(V && BV);
    ASSERT_EQ(*BV, *V) << "Value of '" << F << "' does not match";
  }
}
