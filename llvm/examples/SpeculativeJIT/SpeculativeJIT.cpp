#include "llvm/ExecutionEngine/Orc/CompileOnDemandLayer.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/IndirectionUtils.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/Orc/SpeculateAnalyses.h"
#include "llvm/ExecutionEngine/Orc/Speculation.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ThreadPool.h"

#include <list>
#include <string>

using namespace llvm;
using namespace llvm::orc;

static cl::list<std::string> InputFiles(cl::Positional, cl::OneOrMore,
                                        cl::desc("input files"));

static cl::list<std::string> InputArgv("args", cl::Positional,
                                       cl::desc("<program arguments>..."),
                                       cl::ZeroOrMore, cl::PositionalEatsArgs);

static cl::opt<unsigned> NumThreads("num-threads", cl::Optional,
                                    cl::desc("Number of compile threads"),
                                    cl::init(4));

ExitOnError ExitOnErr;

// Add Layers
class SpeculativeJIT {
public:
  static Expected<std::unique_ptr<SpeculativeJIT>> Create() {
    auto JTMB = orc::JITTargetMachineBuilder::detectHost();
    if (!JTMB)
      return JTMB.takeError();

    auto DL = JTMB->getDefaultDataLayoutForTarget();
    if (!DL)
      return DL.takeError();

    auto ES = std::make_unique<ExecutionSession>();

    auto LCTMgr = createLocalLazyCallThroughManager(
        JTMB->getTargetTriple(), *ES,
        pointerToJITTargetAddress(explodeOnLazyCompileFailure));
    if (!LCTMgr)
      return LCTMgr.takeError();

    auto ISMBuilder =
        createLocalIndirectStubsManagerBuilder(JTMB->getTargetTriple());
    if (!ISMBuilder)
      return make_error<StringError>("No indirect stubs manager for target",
                                     inconvertibleErrorCode());

    auto ProcessSymbolsSearchGenerator =
        DynamicLibrarySearchGenerator::GetForCurrentProcess(
            DL->getGlobalPrefix());
    if (!ProcessSymbolsSearchGenerator)
      return ProcessSymbolsSearchGenerator.takeError();

    std::unique_ptr<SpeculativeJIT> SJ(new SpeculativeJIT(
        std::move(ES), std::move(*DL), std::move(*JTMB), std::move(*LCTMgr),
        std::move(ISMBuilder), std::move(*ProcessSymbolsSearchGenerator)));
    return std::move(SJ);
  }

  ExecutionSession &getES() { return *ES; }

  Error addModule(JITDylib &JD, ThreadSafeModule TSM) {
    return CODLayer.add(JD, std::move(TSM));
  }

  Expected<JITEvaluatedSymbol> lookup(StringRef UnmangledName) {
    return ES->lookup({&ES->getMainJITDylib()}, Mangle(UnmangledName));
  }

  ~SpeculativeJIT() { CompileThreads.wait(); }

private:
  using IndirectStubsManagerBuilderFunction =
      std::function<std::unique_ptr<IndirectStubsManager>()>;

  static void explodeOnLazyCompileFailure() {
    errs() << "Lazy compilation failed, Symbol Implmentation not found!\n";
    exit(1);
  }

  SpeculativeJIT(
      std::unique_ptr<ExecutionSession> ES, DataLayout DL,
      orc::JITTargetMachineBuilder JTMB,
      std::unique_ptr<LazyCallThroughManager> LCTMgr,
      IndirectStubsManagerBuilderFunction ISMBuilder,
      std::unique_ptr<DynamicLibrarySearchGenerator> ProcessSymbolsGenerator)
      : ES(std::move(ES)), DL(std::move(DL)), LCTMgr(std::move(LCTMgr)),
        CompileLayer(*this->ES, ObjLayer,
                     ConcurrentIRCompiler(std::move(JTMB))),
        S(Imps, *this->ES),
        SpeculateLayer(*this->ES, CompileLayer, S, Mangle, BlockFreqQuery()),
        CODLayer(*this->ES, SpeculateLayer, *this->LCTMgr,
                 std::move(ISMBuilder)) {
    this->ES->getMainJITDylib().addGenerator(
        std::move(ProcessSymbolsGenerator));
    this->CODLayer.setImplMap(&Imps);
    this->ES->setDispatchMaterialization(

        [this](JITDylib &JD, std::unique_ptr<MaterializationUnit> MU) {
          // FIXME: Switch to move capture once we have C++14.
          auto SharedMU = std::shared_ptr<MaterializationUnit>(std::move(MU));
          auto Work = [SharedMU, &JD]() { SharedMU->doMaterialize(JD); };
          CompileThreads.async(std::move(Work));
        });
    ExitOnErr(S.addSpeculationRuntime(this->ES->getMainJITDylib(), Mangle));
    LocalCXXRuntimeOverrides CXXRuntimeoverrides;
    ExitOnErr(CXXRuntimeoverrides.enable(this->ES->getMainJITDylib(), Mangle));
  }

  static std::unique_ptr<SectionMemoryManager> createMemMgr() {
    return std::make_unique<SectionMemoryManager>();
  }

  std::unique_ptr<ExecutionSession> ES;
  DataLayout DL;
  MangleAndInterner Mangle{*ES, DL};
  ThreadPool CompileThreads{NumThreads};

  Triple TT;
  std::unique_ptr<LazyCallThroughManager> LCTMgr;
  IRCompileLayer CompileLayer;
  ImplSymbolMap Imps;
  Speculator S;
  RTDyldObjectLinkingLayer ObjLayer{*ES, createMemMgr};
  IRSpeculationLayer SpeculateLayer;
  CompileOnDemandLayer CODLayer;
};

int main(int argc, char *argv[]) {
  // Initialize LLVM.
  InitLLVM X(argc, argv);

  InitializeNativeTarget();
  InitializeNativeTargetAsmPrinter();

  cl::ParseCommandLineOptions(argc, argv, "SpeculativeJIT");
  ExitOnErr.setBanner(std::string(argv[0]) + ": ");

  if (NumThreads < 1) {
    errs() << "Speculative compilation requires one or more dedicated compile "
              "threads\n";
    return 1;
  }

  // Create a JIT instance.
  auto SJ = ExitOnErr(SpeculativeJIT::Create());

  // Load the IR inputs.
  for (const auto &InputFile : InputFiles) {
    SMDiagnostic Err;
    auto Ctx = std::make_unique<LLVMContext>();
    auto M = parseIRFile(InputFile, Err, *Ctx);
    if (!M) {
      Err.print(argv[0], errs());
      return 1;
    }

    ExitOnErr(SJ->addModule(SJ->getES().getMainJITDylib(),
                            ThreadSafeModule(std::move(M), std::move(Ctx))));
  }

  // Build an argv array for the JIT'd main.
  std::vector<const char *> ArgV;
  ArgV.push_back(argv[0]);
  for (const auto &InputArg : InputArgv)
    ArgV.push_back(InputArg.data());
  ArgV.push_back(nullptr);

  // Look up the JIT'd main, cast it to a function pointer, then call it.

  auto MainSym = ExitOnErr(SJ->lookup("main"));
  int (*Main)(int, const char *[]) =
      (int (*)(int, const char *[]))MainSym.getAddress();

  Main(ArgV.size() - 1, ArgV.data());

  return 0;
}
