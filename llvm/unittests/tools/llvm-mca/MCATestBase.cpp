#include "MCATestBase.h"
#include "Views/SummaryView.h"
#include "llvm/MCA/CustomBehaviour.h"
#include "llvm/MCA/InstrBuilder.h"
#include "llvm/MCA/Pipeline.h"
#include "llvm/MCA/SourceMgr.h"
#include "llvm/MCA/View.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/WithColor.h"
#include <string>

using namespace llvm;
using namespace mca;

const Target *MCATestBase::getLLVMTarget() const {
  std::string Error;
  return TargetRegistry::lookupTarget(TheTriple.getTriple(), Error);
}

mca::PipelineOptions MCATestBase::getDefaultPipelineOptions() {
  mca::PipelineOptions PO(/*MicroOpQueue=*/0, /*DecoderThroughput=*/0,
                          /*DispatchWidth=*/0,
                          /*RegisterFileSize=*/0,
                          /*LoadQueueSize=*/0, /*StoreQueueSize=*/0,
                          /*AssumeNoAlias=*/true,
                          /*EnableBottleneckAnalysis=*/false);
  return PO;
}

void MCATestBase::SetUp() {
  TheTarget = getLLVMTarget();
  ASSERT_NE(TheTarget, nullptr);

  StringRef TripleName = TheTriple.getTriple();

  STI.reset(TheTarget->createMCSubtargetInfo(TripleName, CPUName, MAttr));
  ASSERT_TRUE(STI);
  ASSERT_TRUE(STI->isCPUStringValid(CPUName));

  MRI.reset(TheTarget->createMCRegInfo(TripleName));
  ASSERT_TRUE(MRI);

  auto MCOptions = getMCTargetOptions();
  MAI.reset(TheTarget->createMCAsmInfo(*MRI, TripleName, MCOptions));
  ASSERT_TRUE(MAI);

  Ctx = std::make_unique<MCContext>(TheTriple, MAI.get(), MRI.get(), STI.get());
  MOFI.reset(TheTarget->createMCObjectFileInfo(*Ctx, /*PIC=*/false));
  Ctx->setObjectFileInfo(MOFI.get());

  MCII.reset(TheTarget->createMCInstrInfo());
  ASSERT_TRUE(MCII);

  MCIA.reset(TheTarget->createMCInstrAnalysis(MCII.get()));
  ASSERT_TRUE(MCIA);

  IP.reset(TheTarget->createMCInstPrinter(TheTriple, /*AssemblerDialect=*/0,
                                          *MAI, *MCII, *MRI));
  ASSERT_TRUE(IP);
}

Error MCATestBase::runBaselineMCA(json::Object &Result, ArrayRef<MCInst> Insts,
                                  ArrayRef<mca::View *> Views,
                                  const mca::PipelineOptions *PO) {
  mca::Context MCA(*MRI, *STI);

  mca::InstrBuilder IB(*STI, *MCII, *MRI, MCIA.get());

  SmallVector<std::unique_ptr<mca::Instruction>> LoweredInsts;
  for (const auto &MCI : Insts) {
    Expected<std::unique_ptr<mca::Instruction>> Inst =
        IB.createInstruction(MCI);
    if (!Inst) {
      if (auto NewE =
              handleErrors(Inst.takeError(),
                           [this](const mca::InstructionError<MCInst> &IE) {
                             std::string InstructionStr;
                             raw_string_ostream SS(InstructionStr);
                             WithColor::error() << IE.Message << '\n';
                             IP->printInst(&IE.Inst, 0, "", *STI, SS);
                             WithColor::note()
                                 << "instruction: " << InstructionStr << '\n';
                           })) {
        // Default case.
        return NewE;
      }
    } else {
      LoweredInsts.emplace_back(std::move(Inst.get()));
    }
  }

  mca::CircularSourceMgr SM(LoweredInsts, /*Iterations=*/1);

  // Empty CustomBehaviour.
  auto CB = std::make_unique<mca::CustomBehaviour>(*STI, SM, *MCII);

  mca::PipelineOptions ThePO = PO ? *PO : getDefaultPipelineOptions();
  auto P = MCA.createDefaultPipeline(ThePO, SM, *CB);

  SmallVector<std::unique_ptr<mca::View>, 1> DefaultViews;
  if (Views.empty()) {
    // By default, we only add SummaryView.
    auto SV = std::make_unique<SummaryView>(STI->getSchedModel(), Insts,
                                            ThePO.DispatchWidth);
    P->addEventListener(SV.get());
    DefaultViews.emplace_back(std::move(SV));
  } else {
    for (auto *V : Views)
      P->addEventListener(V);
  }

  // Run the pipeline.
  Expected<unsigned> Cycles = P->run();
  if (!Cycles)
    return Cycles.takeError();

  for (const auto *V : Views)
    Result[V->getNameAsString()] = V->toJSON();
  for (const auto &V : DefaultViews)
    Result[V->getNameAsString()] = V->toJSON();

  return Error::success();
}
