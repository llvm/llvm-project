//===-- llc.cpp - Implement the LLVM Native Code Generator ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the llc code generator driver. It provides a convenient
// command-line interface for generating native assembly-language code
// or C code, given LLVM bitcode.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/CodeGen/LinkAllAsmWriterComponents.h"
#include "llvm/CodeGen/LinkAllCodegenComponents.h"
#include "llvm/CodeGen/MIRParser/MIRParser.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/AutoUpgrade.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/IRPrintingPasses.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LLVMRemarkStreamer.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/InitializePasses.h"
#include "llvm/MC/SubtargetFeature.h"
#include "llvm/Pass.h"
#include "llvm/Remarks/HotnessThresholdParser.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/PluginLoader.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include <memory>
using namespace llvm;

static codegen::RegisterCodeGenFlags CGF;

// General options for llc.  Other pass-specific options are specified
// within the corresponding llc passes, and target-specific options
// and back-end code generation options are specified with the target machine.
//
static cl::opt<std::string>
InputFilename(cl::Positional, cl::desc("<input bitcode>"), cl::init("-"));

static cl::opt<std::string>
InputLanguage("x", cl::desc("Input language ('ir' or 'mir')"));

static cl::opt<std::string>
OutputFilename("o", cl::desc("Output filename"), cl::value_desc("filename"));

static cl::opt<std::string>
    SplitDwarfOutputFile("split-dwarf-output",
                         cl::desc(".dwo output filename"),
                         cl::value_desc("filename"));

static cl::opt<unsigned>
TimeCompilations("time-compilations", cl::Hidden, cl::init(1u),
                 cl::value_desc("N"),
                 cl::desc("Repeat compilation N times for timing"));

static cl::opt<std::string>
    BinutilsVersion("binutils-version", cl::Hidden,
                    cl::desc("Produced object files can use all ELF features "
                             "supported by this binutils version and newer."
                             "If -no-integrated-as is specified, the generated "
                             "assembly will consider GNU as support."
                             "'none' means that all ELF features can be used, "
                             "regardless of binutils support"));

static cl::opt<bool>
NoIntegratedAssembler("no-integrated-as", cl::Hidden,
                      cl::desc("Disable integrated assembler"));

static cl::opt<bool>
    PreserveComments("preserve-as-comments", cl::Hidden,
                     cl::desc("Preserve Comments in outputted assembly"),
                     cl::init(true));

// Determine optimization level.
static cl::opt<char>
OptLevel("O",
         cl::desc("Optimization level. [-O0, -O1, -O2, or -O3] "
                  "(default = '-O2')"),
         cl::Prefix,
         cl::ZeroOrMore,
         cl::init(' '));

static cl::opt<std::string>
TargetTriple("mtriple", cl::desc("Override target triple for module"));

static cl::opt<std::string> SplitDwarfFile(
    "split-dwarf-file",
    cl::desc(
        "Specify the name of the .dwo file to encode in the DWARF output"));

static cl::opt<bool> NoVerify("disable-verify", cl::Hidden,
                              cl::desc("Do not verify input module"));

static cl::opt<bool> DisableSimplifyLibCalls("disable-simplify-libcalls",
                                             cl::desc("Disable simplify-libcalls"));

static cl::opt<bool> ShowMCEncoding("show-mc-encoding", cl::Hidden,
                                    cl::desc("Show encoding in .s output"));

static cl::opt<bool> EnableDwarfDirectory(
    "enable-dwarf-directory", cl::Hidden,
    cl::desc("Use .file directives with an explicit directory."));

static cl::opt<bool> AsmVerbose("asm-verbose",
                                cl::desc("Add comments to directives."),
                                cl::init(true));

static cl::opt<bool>
    CompileTwice("compile-twice", cl::Hidden,
                 cl::desc("Run everything twice, re-using the same pass "
                          "manager and verify the result is the same."),
                 cl::init(false));

static cl::opt<bool> DiscardValueNames(
    "discard-value-names",
    cl::desc("Discard names from Value (other than GlobalValue)."),
    cl::init(false), cl::Hidden);

static cl::list<std::string> IncludeDirs("I", cl::desc("include search path"));

static cl::opt<bool> RemarksWithHotness(
    "pass-remarks-with-hotness",
    cl::desc("With PGO, include profile count in optimization remarks"),
    cl::Hidden);

static cl::opt<Optional<uint64_t>, false, remarks::HotnessThresholdParser>
    RemarksHotnessThreshold(
        "pass-remarks-hotness-threshold",
        cl::desc("Minimum profile count required for "
                 "an optimization remark to be output. "
                 "Use 'auto' to apply the threshold from profile summary."),
        cl::value_desc("N or 'auto'"), cl::init(0), cl::Hidden);

static cl::opt<std::string>
    RemarksFilename("pass-remarks-output",
                    cl::desc("Output filename for pass remarks"),
                    cl::value_desc("filename"));

static cl::opt<std::string>
    RemarksPasses("pass-remarks-filter",
                  cl::desc("Only record optimization remarks from passes whose "
                           "names match the given regular expression"),
                  cl::value_desc("regex"));

static cl::opt<std::string> RemarksFormat(
    "pass-remarks-format",
    cl::desc("The format used for serializing remarks (default: YAML)"),
    cl::value_desc("format"), cl::init("yaml"));

namespace {
static ManagedStatic<std::vector<std::string>> RunPassNames;

struct RunPassOption {
  void operator=(const std::string &Val) const {
    if (Val.empty())
      return;
    SmallVector<StringRef, 8> PassNames;
    StringRef(Val).split(PassNames, ',', -1, false);
    for (auto PassName : PassNames)
      RunPassNames->push_back(std::string(PassName));
  }
};
}

static RunPassOption RunPassOpt;

static cl::opt<RunPassOption, true, cl::parser<std::string>> RunPass(
    "run-pass",
    cl::desc("Run compiler only for specified passes (comma separated list)"),
    cl::value_desc("pass-name"), cl::ZeroOrMore, cl::location(RunPassOpt));

static int compileModule(char **, LLVMContext &);

LLVM_ATTRIBUTE_NORETURN static void reportError(Twine Msg,
                                                StringRef Filename = "") {
  SmallString<256> Prefix;
  if (!Filename.empty()) {
    if (Filename == "-")
      Filename = "<stdin>";
    ("'" + Twine(Filename) + "': ").toStringRef(Prefix);
  }
  WithColor::error(errs(), "llc") << Prefix << Msg << "\n";
  exit(1);
}

LLVM_ATTRIBUTE_NORETURN static void reportError(Error Err, StringRef Filename) {
  assert(Err);
  handleAllErrors(createFileError(Filename, std::move(Err)),
                  [&](const ErrorInfoBase &EI) { reportError(EI.message()); });
  llvm_unreachable("reportError() should not return");
}

static std::unique_ptr<ToolOutputFile> GetOutputStream(const char *TargetName,
                                                       Triple::OSType OS,
                                                       const char *ProgName) {
  // If we don't yet have an output filename, make one.
  if (OutputFilename.empty()) {
    if (InputFilename == "-")
      OutputFilename = "-";
    else {
      // If InputFilename ends in .bc or .ll, remove it.
      StringRef IFN = InputFilename;
      if (IFN.endswith(".bc") || IFN.endswith(".ll"))
        OutputFilename = std::string(IFN.drop_back(3));
      else if (IFN.endswith(".mir"))
        OutputFilename = std::string(IFN.drop_back(4));
      else
        OutputFilename = std::string(IFN);

      switch (codegen::getFileType()) {
      case CGFT_AssemblyFile:
        if (TargetName[0] == 'c') {
          if (TargetName[1] == 0)
            OutputFilename += ".cbe.c";
          else if (TargetName[1] == 'p' && TargetName[2] == 'p')
            OutputFilename += ".cpp";
          else
            OutputFilename += ".s";
        } else
          OutputFilename += ".s";
        break;
      case CGFT_ObjectFile:
        if (OS == Triple::Win32)
          OutputFilename += ".obj";
        else
          OutputFilename += ".o";
        break;
      case CGFT_Null:
        OutputFilename = "-";
        break;
      }
    }
  }

  // Decide if we need "binary" output.
  bool Binary = false;
  switch (codegen::getFileType()) {
  case CGFT_AssemblyFile:
    break;
  case CGFT_ObjectFile:
  case CGFT_Null:
    Binary = true;
    break;
  }

  // Open the file.
  std::error_code EC;
  sys::fs::OpenFlags OpenFlags = sys::fs::OF_None;
  if (!Binary)
    OpenFlags |= sys::fs::OF_Text;
  auto FDOut = std::make_unique<ToolOutputFile>(OutputFilename, EC, OpenFlags);
  if (EC) {
    reportError(EC.message());
    return nullptr;
  }

  return FDOut;
}

struct LLCDiagnosticHandler : public DiagnosticHandler {
  bool *HasError;
  LLCDiagnosticHandler(bool *HasErrorPtr) : HasError(HasErrorPtr) {}
  bool handleDiagnostics(const DiagnosticInfo &DI) override {
    if (DI.getSeverity() == DS_Error)
      *HasError = true;

    if (auto *Remark = dyn_cast<DiagnosticInfoOptimizationBase>(&DI))
      if (!Remark->isEnabled())
        return true;

    DiagnosticPrinterRawOStream DP(errs());
    errs() << LLVMContext::getDiagnosticMessagePrefix(DI.getSeverity()) << ": ";
    DI.print(DP);
    errs() << "\n";
    return true;
  }
};

static void InlineAsmDiagHandler(const SMDiagnostic &SMD, void *Context,
                                 unsigned LocCookie) {
  bool *HasError = static_cast<bool *>(Context);
  if (SMD.getKind() == SourceMgr::DK_Error)
    *HasError = true;

  SMD.print(nullptr, errs());

  // For testing purposes, we print the LocCookie here.
  if (LocCookie)
    WithColor::note() << "!srcloc = " << LocCookie << "\n";
}

// main - Entry point for the llc compiler.
//
int main(int argc, char **argv) {
  InitLLVM X(argc, argv);

  // Enable debug stream buffering.
  EnableDebugBuffering = true;

  LLVMContext Context;

  // Initialize targets first, so that --version shows registered targets.
  InitializeAllTargets();
  InitializeAllTargetMCs();
  InitializeAllAsmPrinters();
  InitializeAllAsmParsers();

  // Initialize codegen and IR passes used by llc so that the -print-after,
  // -print-before, and -stop-after options work.
  PassRegistry *Registry = PassRegistry::getPassRegistry();
  initializeCore(*Registry);
  initializeCodeGen(*Registry);
  initializeLoopStrengthReducePass(*Registry);
  initializeLowerIntrinsicsPass(*Registry);
  initializeEntryExitInstrumenterPass(*Registry);
  initializePostInlineEntryExitInstrumenterPass(*Registry);
  initializeUnreachableBlockElimLegacyPassPass(*Registry);
  initializeConstantHoistingLegacyPassPass(*Registry);
  initializeScalarOpts(*Registry);
  initializeVectorization(*Registry);
  initializeScalarizeMaskedMemIntrinLegacyPassPass(*Registry);
  initializeExpandReductionsPass(*Registry);
  initializeHardwareLoopsPass(*Registry);
  initializeTransformUtils(*Registry);

  // Initialize debugging passes.
  initializeScavengerTestPass(*Registry);

  // Register the target printer for --version.
  cl::AddExtraVersionPrinter(TargetRegistry::printRegisteredTargetsForVersion);

  cl::ParseCommandLineOptions(argc, argv, "llvm system compiler\n");

  Context.setDiscardValueNames(DiscardValueNames);

  // Set a diagnostic handler that doesn't exit on the first error
  bool HasError = false;
  Context.setDiagnosticHandler(
      std::make_unique<LLCDiagnosticHandler>(&HasError));
  Context.setInlineAsmDiagnosticHandler(InlineAsmDiagHandler, &HasError);

  Expected<std::unique_ptr<ToolOutputFile>> RemarksFileOrErr =
      setupLLVMOptimizationRemarks(Context, RemarksFilename, RemarksPasses,
                                   RemarksFormat, RemarksWithHotness,
                                   RemarksHotnessThreshold);
  if (Error E = RemarksFileOrErr.takeError())
    reportError(std::move(E), RemarksFilename);
  std::unique_ptr<ToolOutputFile> RemarksFile = std::move(*RemarksFileOrErr);

  if (InputLanguage != "" && InputLanguage != "ir" && InputLanguage != "mir")
    reportError("input language must be '', 'IR' or 'MIR'");

  // Compile the module TimeCompilations times to give better compile time
  // metrics.
  for (unsigned I = TimeCompilations; I; --I)
    if (int RetVal = compileModule(argv, Context))
      return RetVal;

  if (RemarksFile)
    RemarksFile->keep();
  return 0;
}

static bool addPass(PassManagerBase &PM, const char *argv0,
                    StringRef PassName, TargetPassConfig &TPC) {
  if (PassName == "none")
    return false;

  const PassRegistry *PR = PassRegistry::getPassRegistry();
  const PassInfo *PI = PR->getPassInfo(PassName);
  if (!PI) {
    WithColor::error(errs(), argv0)
        << "run-pass " << PassName << " is not registered.\n";
    return true;
  }

  Pass *P;
  if (PI->getNormalCtor())
    P = PI->getNormalCtor()();
  else {
    WithColor::error(errs(), argv0)
        << "cannot create pass: " << PI->getPassName() << "\n";
    return true;
  }
  std::string Banner = std::string("After ") + std::string(P->getPassName());
  TPC.addMachinePrePasses();
  PM.add(P);
  TPC.addMachinePostPasses(Banner);

  return false;
}

static int compileModule(char **argv, LLVMContext &Context) {
  // Load the module to be compiled...
  SMDiagnostic Err;
  std::unique_ptr<Module> M;
  std::unique_ptr<MIRParser> MIR;
  Triple TheTriple;
  std::string CPUStr = codegen::getCPUStr(),
              FeaturesStr = codegen::getFeaturesStr();

  // Set attributes on functions as loaded from MIR from command line arguments.
  auto setMIRFunctionAttributes = [&CPUStr, &FeaturesStr](Function &F) {
    codegen::setFunctionAttributes(CPUStr, FeaturesStr, F);
  };

  auto MAttrs = codegen::getMAttrs();
  bool SkipModule = codegen::getMCPU() == "help" ||
                    (!MAttrs.empty() && MAttrs.front() == "help");

  CodeGenOpt::Level OLvl = CodeGenOpt::Default;
  switch (OptLevel) {
  default:
    WithColor::error(errs(), argv[0]) << "invalid optimization level.\n";
    return 1;
  case ' ': break;
  case '0': OLvl = CodeGenOpt::None; break;
  case '1': OLvl = CodeGenOpt::Less; break;
  case '2': OLvl = CodeGenOpt::Default; break;
  case '3': OLvl = CodeGenOpt::Aggressive; break;
  }

  // Parse 'none' or '$major.$minor'. Disallow -binutils-version=0 because we
  // use that to indicate the MC default.
  if (!BinutilsVersion.empty() && BinutilsVersion != "none") {
    StringRef V = BinutilsVersion.getValue();
    unsigned Num;
    if (V.consumeInteger(10, Num) || Num == 0 ||
        !(V.empty() ||
          (V.consume_front(".") && !V.consumeInteger(10, Num) && V.empty()))) {
      WithColor::error(errs(), argv[0])
          << "invalid -binutils-version, accepting 'none' or major.minor\n";
      return 1;
    }
  }
  TargetOptions Options;
  auto InitializeOptions = [&](const Triple &TheTriple) {
    Options = codegen::InitTargetOptionsFromCodeGenFlags(TheTriple);
    Options.BinutilsVersion =
        TargetMachine::parseBinutilsVersion(BinutilsVersion);
    Options.DisableIntegratedAS = NoIntegratedAssembler;
    Options.MCOptions.ShowMCEncoding = ShowMCEncoding;
    Options.MCOptions.MCUseDwarfDirectory = EnableDwarfDirectory;
    Options.MCOptions.AsmVerbose = AsmVerbose;
    Options.MCOptions.PreserveAsmComments = PreserveComments;
    Options.MCOptions.IASSearchPaths = IncludeDirs;
    Options.MCOptions.SplitDwarfFile = SplitDwarfFile;
  };

  Optional<Reloc::Model> RM = codegen::getExplicitRelocModel();

  const Target *TheTarget = nullptr;
  std::unique_ptr<TargetMachine> Target;

  // If user just wants to list available options, skip module loading
  if (!SkipModule) {
    auto SetDataLayout =
        [&](StringRef DataLayoutTargetTriple) -> Optional<std::string> {
      // If we are supposed to override the target triple, do so now.
      std::string IRTargetTriple = DataLayoutTargetTriple.str();
      if (!TargetTriple.empty())
        IRTargetTriple = Triple::normalize(TargetTriple);
      TheTriple = Triple(IRTargetTriple);
      if (TheTriple.getTriple().empty())
        TheTriple.setTriple(sys::getDefaultTargetTriple());

      std::string Error;
      TheTarget =
          TargetRegistry::lookupTarget(codegen::getMArch(), TheTriple, Error);
      if (!TheTarget) {
        WithColor::error(errs(), argv[0]) << Error;
        exit(1);
      }

      // On AIX, setting the relocation model to anything other than PIC is
      // considered a user error.
      if (TheTriple.isOSAIX() && RM.hasValue() && *RM != Reloc::PIC_)
        reportError("invalid relocation model, AIX only supports PIC",
                    InputFilename);

      InitializeOptions(TheTriple);
      Target = std::unique_ptr<TargetMachine>(TheTarget->createTargetMachine(
          TheTriple.getTriple(), CPUStr, FeaturesStr, Options, RM,
          codegen::getExplicitCodeModel(), OLvl));
      assert(Target && "Could not allocate target machine!");

      return Target->createDataLayout().getStringRepresentation();
    };
    if (InputLanguage == "mir" ||
        (InputLanguage == "" && StringRef(InputFilename).endswith(".mir"))) {
      MIR = createMIRParserFromFile(InputFilename, Err, Context,
                                    setMIRFunctionAttributes);
      if (MIR)
        M = MIR->parseIRModule(SetDataLayout);
    } else {
      M = parseIRFile(InputFilename, Err, Context, SetDataLayout);
    }
    if (!M) {
      Err.print(argv[0], WithColor::error(errs(), argv[0]));
      return 1;
    }
    if (!TargetTriple.empty())
      M->setTargetTriple(Triple::normalize(TargetTriple));
  } else {
    TheTriple = Triple(Triple::normalize(TargetTriple));
    if (TheTriple.getTriple().empty())
      TheTriple.setTriple(sys::getDefaultTargetTriple());

    // Get the target specific parser.
    std::string Error;
    TheTarget =
        TargetRegistry::lookupTarget(codegen::getMArch(), TheTriple, Error);
    if (!TheTarget) {
      WithColor::error(errs(), argv[0]) << Error;
      return 1;
    }

    // On AIX, setting the relocation model to anything other than PIC is
    // considered a user error.
    if (TheTriple.isOSAIX() && RM.hasValue() && *RM != Reloc::PIC_) {
      WithColor::error(errs(), argv[0])
          << "invalid relocation model, AIX only supports PIC.\n";
      return 1;
    }

    InitializeOptions(TheTriple);
    Target = std::unique_ptr<TargetMachine>(TheTarget->createTargetMachine(
        TheTriple.getTriple(), CPUStr, FeaturesStr, Options, RM,
        codegen::getExplicitCodeModel(), OLvl));
    assert(Target && "Could not allocate target machine!");

    // If we don't have a module then just exit now. We do this down
    // here since the CPU/Feature help is underneath the target machine
    // creation.
    return 0;
  }

  assert(M && "Should have exited if we didn't have a module!");
  if (codegen::getFloatABIForCalls() != FloatABI::Default)
    Options.FloatABIType = codegen::getFloatABIForCalls();

  // Figure out where we are going to send the output.
  std::unique_ptr<ToolOutputFile> Out =
      GetOutputStream(TheTarget->getName(), TheTriple.getOS(), argv[0]);
  if (!Out) return 1;

  std::unique_ptr<ToolOutputFile> DwoOut;
  if (!SplitDwarfOutputFile.empty()) {
    std::error_code EC;
    DwoOut = std::make_unique<ToolOutputFile>(SplitDwarfOutputFile, EC,
                                               sys::fs::OF_None);
    if (EC)
      reportError(EC.message(), SplitDwarfOutputFile);
  }

  // Build up all of the passes that we want to do to the module.
  legacy::PassManager PM;

  // Add an appropriate TargetLibraryInfo pass for the module's triple.
  TargetLibraryInfoImpl TLII(Triple(M->getTargetTriple()));

  // The -disable-simplify-libcalls flag actually disables all builtin optzns.
  if (DisableSimplifyLibCalls)
    TLII.disableAllFunctions();
  PM.add(new TargetLibraryInfoWrapperPass(TLII));

  // Verify module immediately to catch problems before doInitialization() is
  // called on any passes.
  if (!NoVerify && verifyModule(*M, &errs()))
    reportError("input module cannot be verified", InputFilename);

  // Override function attributes based on CPUStr, FeaturesStr, and command line
  // flags.
  codegen::setFunctionAttributes(CPUStr, FeaturesStr, *M);

  if (mc::getExplicitRelaxAll() && codegen::getFileType() != CGFT_ObjectFile)
    WithColor::warning(errs(), argv[0])
        << ": warning: ignoring -mc-relax-all because filetype != obj";

  {
    raw_pwrite_stream *OS = &Out->os();

    // Manually do the buffering rather than using buffer_ostream,
    // so we can memcmp the contents in CompileTwice mode
    SmallVector<char, 0> Buffer;
    std::unique_ptr<raw_svector_ostream> BOS;
    if ((codegen::getFileType() != CGFT_AssemblyFile &&
         !Out->os().supportsSeeking()) ||
        CompileTwice) {
      BOS = std::make_unique<raw_svector_ostream>(Buffer);
      OS = BOS.get();
    }

    const char *argv0 = argv[0];
    LLVMTargetMachine &LLVMTM = static_cast<LLVMTargetMachine &>(*Target);
    MachineModuleInfoWrapperPass *MMIWP =
        new MachineModuleInfoWrapperPass(&LLVMTM);

    // Construct a custom pass pipeline that starts after instruction
    // selection.
    if (!RunPassNames->empty()) {
      if (!MIR) {
        WithColor::warning(errs(), argv[0])
            << "run-pass is for .mir file only.\n";
        return 1;
      }
      TargetPassConfig &TPC = *LLVMTM.createPassConfig(PM);
      if (TPC.hasLimitedCodeGenPipeline()) {
        WithColor::warning(errs(), argv[0])
            << "run-pass cannot be used with "
            << TPC.getLimitedCodeGenPipelineReason(" and ") << ".\n";
        return 1;
      }

      TPC.setDisableVerify(NoVerify);
      PM.add(&TPC);
      PM.add(MMIWP);
      TPC.printAndVerify("");
      for (const std::string &RunPassName : *RunPassNames) {
        if (addPass(PM, argv0, RunPassName, TPC))
          return 1;
      }
      TPC.setInitialized();
      PM.add(createPrintMIRPass(*OS));
      PM.add(createFreeMachineFunctionPass());
    } else if (Target->addPassesToEmitFile(
                   PM, *OS, DwoOut ? &DwoOut->os() : nullptr,
                   codegen::getFileType(), NoVerify, MMIWP)) {
      reportError("target does not support generation of this file type");
    }

    const_cast<TargetLoweringObjectFile *>(LLVMTM.getObjFileLowering())
        ->Initialize(MMIWP->getMMI().getContext(), *Target);
    if (MIR) {
      assert(MMIWP && "Forgot to create MMIWP?");
      if (MIR->parseMachineFunctions(*M, MMIWP->getMMI()))
        return 1;
    }

    // Before executing passes, print the final values of the LLVM options.
    cl::PrintOptionValues();

    // If requested, run the pass manager over the same module again,
    // to catch any bugs due to persistent state in the passes. Note that
    // opt has the same functionality, so it may be worth abstracting this out
    // in the future.
    SmallVector<char, 0> CompileTwiceBuffer;
    if (CompileTwice) {
      std::unique_ptr<Module> M2(llvm::CloneModule(*M));
      PM.run(*M2);
      CompileTwiceBuffer = Buffer;
      Buffer.clear();
    }

    PM.run(*M);

    auto HasError =
        ((const LLCDiagnosticHandler *)(Context.getDiagHandlerPtr()))->HasError;
    if (*HasError)
      return 1;

    // Compare the two outputs and make sure they're the same
    if (CompileTwice) {
      if (Buffer.size() != CompileTwiceBuffer.size() ||
          (memcmp(Buffer.data(), CompileTwiceBuffer.data(), Buffer.size()) !=
           0)) {
        errs()
            << "Running the pass manager twice changed the output.\n"
               "Writing the result of the second run to the specified output\n"
               "To generate the one-run comparison binary, just run without\n"
               "the compile-twice option\n";
        Out->os() << Buffer;
        Out->keep();
        return 1;
      }
    }

    if (BOS) {
      Out->os() << Buffer;
    }
  }

  // Declare success.
  Out->keep();
  if (DwoOut)
    DwoOut->keep();

  return 0;
}
