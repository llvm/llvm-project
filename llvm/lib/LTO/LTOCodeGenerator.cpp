//===-LTOCodeGenerator.cpp - LLVM Link Time Optimizer ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Link Time Optimization library. This library is
// intended to be used by linker to optimize code at link time.
//
//===----------------------------------------------------------------------===//

#include "llvm/LTO/LTOCodeGenerator.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/CodeGen/ParallelCG.h"
#include "llvm/CodeGen/RuntimeLibcalls.h"
#include "llvm/Config/config.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Mangler.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/InitializePasses.h"
#include "llvm/LTO/LTOModule.h"
#include "llvm/Linker/Linker.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/SubtargetFeature.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetSubtargetInfo.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/ObjCARC.h"
#include <system_error>
using namespace llvm;

const char* LTOCodeGenerator::getVersionString() {
#ifdef LLVM_VERSION_INFO
  return PACKAGE_NAME " version " PACKAGE_VERSION ", " LLVM_VERSION_INFO;
#else
  return PACKAGE_NAME " version " PACKAGE_VERSION;
#endif
}

LTOCodeGenerator::LTOCodeGenerator(LLVMContext &Context)
    : Context(Context), MergedModule(new Module("ld-temp.o", Context)),
      IRLinker(new Linker(*MergedModule, [this](const DiagnosticInfo &DI) {
        MergedModule->getContext().diagnose(DI);
      })) {
  initializeLTOPasses();
}

LTOCodeGenerator::~LTOCodeGenerator() {}

// Initialize LTO passes. Please keep this function in sync with
// PassManagerBuilder::populateLTOPassManager(), and make sure all LTO
// passes are initialized.
void LTOCodeGenerator::initializeLTOPasses() {
  PassRegistry &R = *PassRegistry::getPassRegistry();

  initializeInternalizePassPass(R);
  initializeIPSCCPPass(R);
  initializeGlobalOptPass(R);
  initializeConstantMergePass(R);
  initializeDAHPass(R);
  initializeInstructionCombiningPassPass(R);
  initializeSimpleInlinerPass(R);
  initializePruneEHPass(R);
  initializeGlobalDCEPass(R);
  initializeArgPromotionPass(R);
  initializeJumpThreadingPass(R);
  initializeSROALegacyPassPass(R);
  initializeSROA_DTPass(R);
  initializeSROA_SSAUpPass(R);
  initializeFunctionAttrsPass(R);
  initializeGlobalsAAWrapperPassPass(R);
  initializeLICMPass(R);
  initializeMergedLoadStoreMotionPass(R);
  initializeGVNPass(R);
  initializeMemCpyOptPass(R);
  initializeDCEPass(R);
  initializeCFGSimplifyPassPass(R);
}

bool LTOCodeGenerator::addModule(LTOModule *Mod) {
  assert(&Mod->getModule().getContext() == &Context &&
         "Expected module in same context");

  bool ret = IRLinker->linkInModule(Mod->getModule());

  const std::vector<const char *> &undefs = Mod->getAsmUndefinedRefs();
  for (int i = 0, e = undefs.size(); i != e; ++i)
    AsmUndefinedRefs[undefs[i]] = 1;

  return !ret;
}

void LTOCodeGenerator::setModule(std::unique_ptr<LTOModule> Mod) {
  assert(&Mod->getModule().getContext() == &Context &&
         "Expected module in same context");

  AsmUndefinedRefs.clear();

  MergedModule = Mod->takeModule();
  IRLinker = llvm::make_unique<Linker>(*MergedModule,
                                       IRLinker->getDiagnosticHandler());

  const std::vector<const char*> &Undefs = Mod->getAsmUndefinedRefs();
  for (int I = 0, E = Undefs.size(); I != E; ++I)
    AsmUndefinedRefs[Undefs[I]] = 1;
}

void LTOCodeGenerator::setTargetOptions(TargetOptions Options) {
  this->Options = Options;
}

void LTOCodeGenerator::setDebugInfo(lto_debug_model Debug) {
  switch (Debug) {
  case LTO_DEBUG_MODEL_NONE:
    EmitDwarfDebugInfo = false;
    return;

  case LTO_DEBUG_MODEL_DWARF:
    EmitDwarfDebugInfo = true;
    return;
  }
  llvm_unreachable("Unknown debug format!");
}

void LTOCodeGenerator::setOptLevel(unsigned Level) {
  OptLevel = Level;
  switch (OptLevel) {
  case 0:
    CGOptLevel = CodeGenOpt::None;
    break;
  case 1:
    CGOptLevel = CodeGenOpt::Less;
    break;
  case 2:
    CGOptLevel = CodeGenOpt::Default;
    break;
  case 3:
    CGOptLevel = CodeGenOpt::Aggressive;
    break;
  }
}

bool LTOCodeGenerator::writeMergedModules(const char *Path) {
  if (!determineTarget())
    return false;

  // mark which symbols can not be internalized
  applyScopeRestrictions();

  // create output file
  std::error_code EC;
  tool_output_file Out(Path, EC, sys::fs::F_None);
  if (EC) {
    std::string ErrMsg = "could not open bitcode file for writing: ";
    ErrMsg += Path;
    emitError(ErrMsg);
    return false;
  }

  // write bitcode to it
  WriteBitcodeToFile(MergedModule.get(), Out.os(), ShouldEmbedUselists);
  Out.os().close();

  if (Out.os().has_error()) {
    std::string ErrMsg = "could not write bitcode file: ";
    ErrMsg += Path;
    emitError(ErrMsg);
    Out.os().clear_error();
    return false;
  }

  Out.keep();
  return true;
}

bool LTOCodeGenerator::compileOptimizedToFile(const char **Name) {
  // make unique temp output file to put generated code
  SmallString<128> Filename;
  int FD;

  const char *Extension =
      (FileType == TargetMachine::CGFT_AssemblyFile ? "s" : "o");

  std::error_code EC =
      sys::fs::createTemporaryFile("lto-llvm", Extension, FD, Filename);
  if (EC) {
    emitError(EC.message());
    return false;
  }

  // generate object file
  tool_output_file objFile(Filename.c_str(), FD);

  bool genResult = compileOptimized(&objFile.os());
  objFile.os().close();
  if (objFile.os().has_error()) {
    objFile.os().clear_error();
    sys::fs::remove(Twine(Filename));
    return false;
  }

  objFile.keep();
  if (!genResult) {
    sys::fs::remove(Twine(Filename));
    return false;
  }

  NativeObjectPath = Filename.c_str();
  *Name = NativeObjectPath.c_str();
  return true;
}

std::unique_ptr<MemoryBuffer>
LTOCodeGenerator::compileOptimized() {
  const char *name;
  if (!compileOptimizedToFile(&name))
    return nullptr;

  // read .o file into memory buffer
  ErrorOr<std::unique_ptr<MemoryBuffer>> BufferOrErr =
      MemoryBuffer::getFile(name, -1, false);
  if (std::error_code EC = BufferOrErr.getError()) {
    emitError(EC.message());
    sys::fs::remove(NativeObjectPath);
    return nullptr;
  }

  // remove temp files
  sys::fs::remove(NativeObjectPath);

  return std::move(*BufferOrErr);
}

bool LTOCodeGenerator::compile_to_file(const char **Name, bool DisableVerify,
                                       bool DisableInline,
                                       bool DisableGVNLoadPRE,
                                       bool DisableVectorization) {
  if (!optimize(DisableVerify, DisableInline, DisableGVNLoadPRE,
                DisableVectorization))
    return false;

  return compileOptimizedToFile(Name);
}

std::unique_ptr<MemoryBuffer>
LTOCodeGenerator::compile(bool DisableVerify, bool DisableInline,
                          bool DisableGVNLoadPRE, bool DisableVectorization) {
  if (!optimize(DisableVerify, DisableInline, DisableGVNLoadPRE,
                DisableVectorization))
    return nullptr;

  return compileOptimized();
}

bool LTOCodeGenerator::determineTarget() {
  if (TargetMach)
    return true;

  std::string TripleStr = MergedModule->getTargetTriple();
  if (TripleStr.empty()) {
    TripleStr = sys::getDefaultTargetTriple();
    MergedModule->setTargetTriple(TripleStr);
  }
  llvm::Triple Triple(TripleStr);

  // create target machine from info for merged modules
  std::string ErrMsg;
  const Target *march = TargetRegistry::lookupTarget(TripleStr, ErrMsg);
  if (!march) {
    emitError(ErrMsg);
    return false;
  }

  // Construct LTOModule, hand over ownership of module and target. Use MAttr as
  // the default set of features.
  SubtargetFeatures Features(MAttr);
  Features.getDefaultSubtargetFeatures(Triple);
  FeatureStr = Features.getString();
  // Set a default CPU for Darwin triples.
  if (MCpu.empty() && Triple.isOSDarwin()) {
    if (Triple.getArch() == llvm::Triple::x86_64)
      MCpu = "core2";
    else if (Triple.getArch() == llvm::Triple::x86)
      MCpu = "yonah";
    else if (Triple.getArch() == llvm::Triple::aarch64)
      MCpu = "cyclone";
  }

  TargetMach.reset(march->createTargetMachine(TripleStr, MCpu, FeatureStr,
                                              Options, RelocModel,
                                              CodeModel::Default, CGOptLevel));
  return true;
}

void LTOCodeGenerator::
applyRestriction(GlobalValue &GV,
                 ArrayRef<StringRef> Libcalls,
                 std::vector<const char*> &MustPreserveList,
                 SmallPtrSetImpl<GlobalValue*> &AsmUsed,
                 Mangler &Mangler) {
  // There are no restrictions to apply to declarations.
  if (GV.isDeclaration())
    return;

  // There is nothing more restrictive than private linkage.
  if (GV.hasPrivateLinkage())
    return;

  SmallString<64> Buffer;
  TargetMach->getNameWithPrefix(Buffer, &GV, Mangler);

  if (MustPreserveSymbols.count(Buffer))
    MustPreserveList.push_back(GV.getName().data());
  if (AsmUndefinedRefs.count(Buffer))
    AsmUsed.insert(&GV);

  // Conservatively append user-supplied runtime library functions to
  // llvm.compiler.used.  These could be internalized and deleted by
  // optimizations like -globalopt, causing problems when later optimizations
  // add new library calls (e.g., llvm.memset => memset and printf => puts).
  // Leave it to the linker to remove any dead code (e.g. with -dead_strip).
  if (isa<Function>(GV) &&
      std::binary_search(Libcalls.begin(), Libcalls.end(), GV.getName()))
    AsmUsed.insert(&GV);
}

static void findUsedValues(GlobalVariable *LLVMUsed,
                           SmallPtrSetImpl<GlobalValue*> &UsedValues) {
  if (!LLVMUsed) return;

  ConstantArray *Inits = cast<ConstantArray>(LLVMUsed->getInitializer());
  for (unsigned i = 0, e = Inits->getNumOperands(); i != e; ++i)
    if (GlobalValue *GV =
        dyn_cast<GlobalValue>(Inits->getOperand(i)->stripPointerCasts()))
      UsedValues.insert(GV);
}

// Collect names of runtime library functions. User-defined functions with the
// same names are added to llvm.compiler.used to prevent them from being
// deleted by optimizations.
static void accumulateAndSortLibcalls(std::vector<StringRef> &Libcalls,
                                      const TargetLibraryInfo& TLI,
                                      const Module &Mod,
                                      const TargetMachine &TM) {
  // TargetLibraryInfo has info on C runtime library calls on the current
  // target.
  for (unsigned I = 0, E = static_cast<unsigned>(LibFunc::NumLibFuncs);
       I != E; ++I) {
    LibFunc::Func F = static_cast<LibFunc::Func>(I);
    if (TLI.has(F))
      Libcalls.push_back(TLI.getName(F));
  }

  SmallPtrSet<const TargetLowering *, 1> TLSet;

  for (const Function &F : Mod) {
    const TargetLowering *Lowering =
        TM.getSubtargetImpl(F)->getTargetLowering();

    if (Lowering && TLSet.insert(Lowering).second)
      // TargetLowering has info on library calls that CodeGen expects to be
      // available, both from the C runtime and compiler-rt.
      for (unsigned I = 0, E = static_cast<unsigned>(RTLIB::UNKNOWN_LIBCALL);
           I != E; ++I)
        if (const char *Name =
                Lowering->getLibcallName(static_cast<RTLIB::Libcall>(I)))
          Libcalls.push_back(Name);
  }

  array_pod_sort(Libcalls.begin(), Libcalls.end());
  Libcalls.erase(std::unique(Libcalls.begin(), Libcalls.end()),
                 Libcalls.end());
}

void LTOCodeGenerator::applyScopeRestrictions() {
  if (ScopeRestrictionsDone || !ShouldInternalize)
    return;

  // Start off with a verification pass.
  legacy::PassManager passes;
  passes.add(createVerifierPass());

  // mark which symbols can not be internalized
  Mangler Mangler;
  std::vector<const char*> MustPreserveList;
  SmallPtrSet<GlobalValue*, 8> AsmUsed;
  std::vector<StringRef> Libcalls;
  TargetLibraryInfoImpl TLII(Triple(TargetMach->getTargetTriple()));
  TargetLibraryInfo TLI(TLII);

  accumulateAndSortLibcalls(Libcalls, TLI, *MergedModule, *TargetMach);

  for (Function &f : *MergedModule)
    applyRestriction(f, Libcalls, MustPreserveList, AsmUsed, Mangler);
  for (GlobalVariable &v : MergedModule->globals())
    applyRestriction(v, Libcalls, MustPreserveList, AsmUsed, Mangler);
  for (GlobalAlias &a : MergedModule->aliases())
    applyRestriction(a, Libcalls, MustPreserveList, AsmUsed, Mangler);

  GlobalVariable *LLVMCompilerUsed =
    MergedModule->getGlobalVariable("llvm.compiler.used");
  findUsedValues(LLVMCompilerUsed, AsmUsed);
  if (LLVMCompilerUsed)
    LLVMCompilerUsed->eraseFromParent();

  if (!AsmUsed.empty()) {
    llvm::Type *i8PTy = llvm::Type::getInt8PtrTy(Context);
    std::vector<Constant*> asmUsed2;
    for (auto *GV : AsmUsed) {
      Constant *c = ConstantExpr::getBitCast(GV, i8PTy);
      asmUsed2.push_back(c);
    }

    llvm::ArrayType *ATy = llvm::ArrayType::get(i8PTy, asmUsed2.size());
    LLVMCompilerUsed =
      new llvm::GlobalVariable(*MergedModule, ATy, false,
                               llvm::GlobalValue::AppendingLinkage,
                               llvm::ConstantArray::get(ATy, asmUsed2),
                               "llvm.compiler.used");

    LLVMCompilerUsed->setSection("llvm.metadata");
  }

  passes.add(createInternalizePass(MustPreserveList));

  // apply scope restrictions
  passes.run(*MergedModule);

  ScopeRestrictionsDone = true;
}

/// Optimize merged modules using various IPO passes
bool LTOCodeGenerator::optimize(bool DisableVerify, bool DisableInline,
                                bool DisableGVNLoadPRE,
                                bool DisableVectorization) {
  if (!this->determineTarget())
    return false;

  // Mark which symbols can not be internalized
  this->applyScopeRestrictions();

  // Instantiate the pass manager to organize the passes.
  legacy::PassManager passes;

  // Add an appropriate DataLayout instance for this module...
  MergedModule->setDataLayout(TargetMach->createDataLayout());

  passes.add(
      createTargetTransformInfoWrapperPass(TargetMach->getTargetIRAnalysis()));

  Triple TargetTriple(TargetMach->getTargetTriple());
  PassManagerBuilder PMB;
  PMB.DisableGVNLoadPRE = DisableGVNLoadPRE;
  PMB.LoopVectorize = !DisableVectorization;
  PMB.SLPVectorize = !DisableVectorization;
  if (!DisableInline)
    PMB.Inliner = createFunctionInliningPass();
  PMB.LibraryInfo = new TargetLibraryInfoImpl(TargetTriple);
  PMB.OptLevel = OptLevel;
  PMB.VerifyInput = !DisableVerify;
  PMB.VerifyOutput = !DisableVerify;

  PMB.populateLTOPassManager(passes);

  // Run our queue of passes all at once now, efficiently.
  passes.run(*MergedModule);

  return true;
}

bool LTOCodeGenerator::compileOptimized(ArrayRef<raw_pwrite_stream *> Out) {
  if (!this->determineTarget())
    return false;

  legacy::PassManager preCodeGenPasses;

  // If the bitcode files contain ARC code and were compiled with optimization,
  // the ObjCARCContractPass must be run, so do it unconditionally here.
  preCodeGenPasses.add(createObjCARCContractPass());
  preCodeGenPasses.run(*MergedModule);

  // Do code generation. We need to preserve the module in case the client calls
  // writeMergedModules() after compilation, but we only need to allow this at
  // parallelism level 1. This is achieved by having splitCodeGen return the
  // original module at parallelism level 1 which we then assign back to
  // MergedModule.
  MergedModule =
      splitCodeGen(std::move(MergedModule), Out, MCpu, FeatureStr, Options,
                   RelocModel, CodeModel::Default, CGOptLevel, FileType);

  return true;
}

/// setCodeGenDebugOptions - Set codegen debugging options to aid in debugging
/// LTO problems.
void LTOCodeGenerator::setCodeGenDebugOptions(const char *Options) {
  for (std::pair<StringRef, StringRef> o = getToken(Options); !o.first.empty();
       o = getToken(o.second))
    CodegenOptions.push_back(o.first);
}

void LTOCodeGenerator::parseCodeGenDebugOptions() {
  // if options were requested, set them
  if (!CodegenOptions.empty()) {
    // ParseCommandLineOptions() expects argv[0] to be program name.
    std::vector<const char *> CodegenArgv(1, "libLLVMLTO");
    for (std::string &Arg : CodegenOptions)
      CodegenArgv.push_back(Arg.c_str());
    cl::ParseCommandLineOptions(CodegenArgv.size(), CodegenArgv.data());
  }
}

void LTOCodeGenerator::DiagnosticHandler(const DiagnosticInfo &DI,
                                         void *Context) {
  ((LTOCodeGenerator *)Context)->DiagnosticHandler2(DI);
}

void LTOCodeGenerator::DiagnosticHandler2(const DiagnosticInfo &DI) {
  // Map the LLVM internal diagnostic severity to the LTO diagnostic severity.
  lto_codegen_diagnostic_severity_t Severity;
  switch (DI.getSeverity()) {
  case DS_Error:
    Severity = LTO_DS_ERROR;
    break;
  case DS_Warning:
    Severity = LTO_DS_WARNING;
    break;
  case DS_Remark:
    Severity = LTO_DS_REMARK;
    break;
  case DS_Note:
    Severity = LTO_DS_NOTE;
    break;
  }
  // Create the string that will be reported to the external diagnostic handler.
  std::string MsgStorage;
  raw_string_ostream Stream(MsgStorage);
  DiagnosticPrinterRawOStream DP(Stream);
  DI.print(DP);
  Stream.flush();

  // If this method has been called it means someone has set up an external
  // diagnostic handler. Assert on that.
  assert(DiagHandler && "Invalid diagnostic handler");
  (*DiagHandler)(Severity, MsgStorage.c_str(), DiagContext);
}

void
LTOCodeGenerator::setDiagnosticHandler(lto_diagnostic_handler_t DiagHandler,
                                       void *Ctxt) {
  this->DiagHandler = DiagHandler;
  this->DiagContext = Ctxt;
  if (!DiagHandler)
    return Context.setDiagnosticHandler(nullptr, nullptr);
  // Register the LTOCodeGenerator stub in the LLVMContext to forward the
  // diagnostic to the external DiagHandler.
  Context.setDiagnosticHandler(LTOCodeGenerator::DiagnosticHandler, this,
                               /* RespectFilters */ true);
}

namespace {
class LTODiagnosticInfo : public DiagnosticInfo {
  const Twine &Msg;
public:
  LTODiagnosticInfo(const Twine &DiagMsg, DiagnosticSeverity Severity=DS_Error)
      : DiagnosticInfo(DK_Linker, Severity), Msg(DiagMsg) {}
  void print(DiagnosticPrinter &DP) const override { DP << Msg; }
};
}

void LTOCodeGenerator::emitError(const std::string &ErrMsg) {
  if (DiagHandler)
    (*DiagHandler)(LTO_DS_ERROR, ErrMsg.c_str(), DiagContext);
  else
    Context.diagnose(LTODiagnosticInfo(ErrMsg));
}
