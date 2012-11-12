//===- lli.cpp - LLVM Interpreter / Dynamic compiler ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This utility provides a simple wrapper around the LLVM Execution Engines,
// which allow the direct execution of LLVM programs through a Just-In-Time
// compiler, or through an interpreter if no JIT is available for this platform.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "lli"
#include "RecordingMemoryManager.h"
#include "RemoteTarget.h"
#include "llvm/LLVMContext.h"
#include "llvm/Module.h"
#include "llvm/Type.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/CodeGen/LinkAllCodegenComponents.h"
#include "llvm/ExecutionEngine/GenericValue.h"
#include "llvm/ExecutionEngine/Interpreter.h"
#include "llvm/ExecutionEngine/JIT.h"
#include "llvm/ExecutionEngine/JITEventListener.h"
#include "llvm/ExecutionEngine/JITMemoryManager.h"
#include "llvm/ExecutionEngine/MCJIT.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/IRReader.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/PluginLoader.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/Memory.h"
#include "llvm/Support/MathExtras.h"
#include <cerrno>

#ifdef __linux__
// These includes used by LLIMCJITMemoryManager::getPointerToNamedFunction()
// for Glibc trickery. Look comments in this function for more information.
#ifdef HAVE_SYS_STAT_H
#include <sys/stat.h>
#endif
#include <fcntl.h>
#include <unistd.h>
#endif

#ifdef __CYGWIN__
#include <cygwin/version.h>
#if defined(CYGWIN_VERSION_DLL_MAJOR) && CYGWIN_VERSION_DLL_MAJOR<1007
#define DO_NOTHING_ATEXIT 1
#endif
#endif

using namespace llvm;

namespace {
  cl::opt<std::string>
  InputFile(cl::desc("<input bitcode>"), cl::Positional, cl::init("-"));

  cl::list<std::string>
  InputArgv(cl::ConsumeAfter, cl::desc("<program arguments>..."));

  cl::opt<bool> ForceInterpreter("force-interpreter",
                                 cl::desc("Force interpretation: disable JIT"),
                                 cl::init(false));

  cl::opt<bool> UseMCJIT(
    "use-mcjit", cl::desc("Enable use of the MC-based JIT (if available)"),
    cl::init(false));

  // The MCJIT supports building for a target address space separate from
  // the JIT compilation process. Use a forked process and a copying
  // memory manager with IPC to execute using this functionality.
  cl::opt<bool> RemoteMCJIT("remote-mcjit",
    cl::desc("Execute MCJIT'ed code in a separate process."),
    cl::init(false));

  // Determine optimization level.
  cl::opt<char>
  OptLevel("O",
           cl::desc("Optimization level. [-O0, -O1, -O2, or -O3] "
                    "(default = '-O2')"),
           cl::Prefix,
           cl::ZeroOrMore,
           cl::init(' '));

  cl::opt<std::string>
  TargetTriple("mtriple", cl::desc("Override target triple for module"));

  cl::opt<std::string>
  MArch("march",
        cl::desc("Architecture to generate assembly for (see --version)"));

  cl::opt<std::string>
  MCPU("mcpu",
       cl::desc("Target a specific cpu type (-mcpu=help for details)"),
       cl::value_desc("cpu-name"),
       cl::init(""));

  cl::list<std::string>
  MAttrs("mattr",
         cl::CommaSeparated,
         cl::desc("Target specific attributes (-mattr=help for details)"),
         cl::value_desc("a1,+a2,-a3,..."));

  cl::opt<std::string>
  EntryFunc("entry-function",
            cl::desc("Specify the entry function (default = 'main') "
                     "of the executable"),
            cl::value_desc("function"),
            cl::init("main"));

  cl::opt<std::string>
  FakeArgv0("fake-argv0",
            cl::desc("Override the 'argv[0]' value passed into the executing"
                     " program"), cl::value_desc("executable"));

  cl::opt<bool>
  DisableCoreFiles("disable-core-files", cl::Hidden,
                   cl::desc("Disable emission of core files if possible"));

  cl::opt<bool>
  NoLazyCompilation("disable-lazy-compilation",
                  cl::desc("Disable JIT lazy compilation"),
                  cl::init(false));

  cl::opt<Reloc::Model>
  RelocModel("relocation-model",
             cl::desc("Choose relocation model"),
             cl::init(Reloc::Default),
             cl::values(
            clEnumValN(Reloc::Default, "default",
                       "Target default relocation model"),
            clEnumValN(Reloc::Static, "static",
                       "Non-relocatable code"),
            clEnumValN(Reloc::PIC_, "pic",
                       "Fully relocatable, position independent code"),
            clEnumValN(Reloc::DynamicNoPIC, "dynamic-no-pic",
                       "Relocatable external references, non-relocatable code"),
            clEnumValEnd));

  cl::opt<llvm::CodeModel::Model>
  CMModel("code-model",
          cl::desc("Choose code model"),
          cl::init(CodeModel::JITDefault),
          cl::values(clEnumValN(CodeModel::JITDefault, "default",
                                "Target default JIT code model"),
                     clEnumValN(CodeModel::Small, "small",
                                "Small code model"),
                     clEnumValN(CodeModel::Kernel, "kernel",
                                "Kernel code model"),
                     clEnumValN(CodeModel::Medium, "medium",
                                "Medium code model"),
                     clEnumValN(CodeModel::Large, "large",
                                "Large code model"),
                     clEnumValEnd));

  cl::opt<bool>
  EnableJITExceptionHandling("jit-enable-eh",
    cl::desc("Emit exception handling information"),
    cl::init(false));

  cl::opt<bool>
  GenerateSoftFloatCalls("soft-float",
    cl::desc("Generate software floating point library calls"),
    cl::init(false));

  cl::opt<llvm::FloatABI::ABIType>
  FloatABIForCalls("float-abi",
                   cl::desc("Choose float ABI type"),
                   cl::init(FloatABI::Default),
                   cl::values(
                     clEnumValN(FloatABI::Default, "default",
                                "Target default float ABI type"),
                     clEnumValN(FloatABI::Soft, "soft",
                                "Soft float ABI (implied by -soft-float)"),
                     clEnumValN(FloatABI::Hard, "hard",
                                "Hard float ABI (uses FP registers)"),
                     clEnumValEnd));
  cl::opt<bool>
// In debug builds, make this default to true.
#ifdef NDEBUG
#define EMIT_DEBUG false
#else
#define EMIT_DEBUG true
#endif
  EmitJitDebugInfo("jit-emit-debug",
    cl::desc("Emit debug information to debugger"),
    cl::init(EMIT_DEBUG));
#undef EMIT_DEBUG

  static cl::opt<bool>
  EmitJitDebugInfoToDisk("jit-emit-debug-to-disk",
    cl::Hidden,
    cl::desc("Emit debug info objfiles to disk"),
    cl::init(false));
}

static ExecutionEngine *EE = 0;

static void do_shutdown() {
  // Cygwin-1.5 invokes DLL's dtors before atexit handler.
#ifndef DO_NOTHING_ATEXIT
  delete EE;
  llvm_shutdown();
#endif
}

// Memory manager for MCJIT
class LLIMCJITMemoryManager : public JITMemoryManager {
public:
  SmallVector<sys::MemoryBlock, 16> AllocatedDataMem;
  SmallVector<sys::MemoryBlock, 16> AllocatedCodeMem;
  SmallVector<sys::MemoryBlock, 16> FreeCodeMem;

  LLIMCJITMemoryManager() { }
  ~LLIMCJITMemoryManager();

  virtual uint8_t *allocateCodeSection(uintptr_t Size, unsigned Alignment,
                                       unsigned SectionID);

  virtual uint8_t *allocateDataSection(uintptr_t Size, unsigned Alignment,
                                       unsigned SectionID);

  virtual void *getPointerToNamedFunction(const std::string &Name,
                                          bool AbortOnFailure = true);

  // Invalidate instruction cache for code sections. Some platforms with
  // separate data cache and instruction cache require explicit cache flush,
  // otherwise JIT code manipulations (like resolved relocations) will get to
  // the data cache but not to the instruction cache.
  virtual void invalidateInstructionCache();

  // The RTDyldMemoryManager doesn't use the following functions, so we don't
  // need implement them.
  virtual void setMemoryWritable() {
    llvm_unreachable("Unexpected call!");
  }
  virtual void setMemoryExecutable() {
    llvm_unreachable("Unexpected call!");
  }
  virtual void setPoisonMemory(bool poison) {
    llvm_unreachable("Unexpected call!");
  }
  virtual void AllocateGOT() {
    llvm_unreachable("Unexpected call!");
  }
  virtual uint8_t *getGOTBase() const {
    llvm_unreachable("Unexpected call!");
    return 0;
  }
  virtual uint8_t *startFunctionBody(const Function *F,
                                     uintptr_t &ActualSize){
    llvm_unreachable("Unexpected call!");
    return 0;
  }
  virtual uint8_t *allocateStub(const GlobalValue* F, unsigned StubSize,
                                unsigned Alignment) {
    llvm_unreachable("Unexpected call!");
    return 0;
  }
  virtual void endFunctionBody(const Function *F, uint8_t *FunctionStart,
                               uint8_t *FunctionEnd) {
    llvm_unreachable("Unexpected call!");
  }
  virtual uint8_t *allocateSpace(intptr_t Size, unsigned Alignment) {
    llvm_unreachable("Unexpected call!");
    return 0;
  }
  virtual uint8_t *allocateGlobal(uintptr_t Size, unsigned Alignment) {
    llvm_unreachable("Unexpected call!");
    return 0;
  }
  virtual void deallocateFunctionBody(void *Body) {
    llvm_unreachable("Unexpected call!");
  }
  virtual uint8_t* startExceptionTable(const Function* F,
                                       uintptr_t &ActualSize) {
    llvm_unreachable("Unexpected call!");
    return 0;
  }
  virtual void endExceptionTable(const Function *F, uint8_t *TableStart,
                                 uint8_t *TableEnd, uint8_t* FrameRegister) {
    llvm_unreachable("Unexpected call!");
  }
  virtual void deallocateExceptionTable(void *ET) {
    llvm_unreachable("Unexpected call!");
  }
};

uint8_t *LLIMCJITMemoryManager::allocateDataSection(uintptr_t Size,
                                                    unsigned Alignment,
                                                    unsigned SectionID) {
  if (!Alignment)
    Alignment = 16;
  // Ensure that enough memory is requested to allow aligning.
  size_t NumElementsAligned = 1 + (Size + Alignment - 1)/Alignment;
  uint8_t *Addr = (uint8_t*)calloc(NumElementsAligned, Alignment);

  // Honour the alignment requirement.
  uint8_t *AlignedAddr = (uint8_t*)RoundUpToAlignment((uint64_t)Addr, Alignment);

  // Store the original address from calloc so we can free it later.
  AllocatedDataMem.push_back(sys::MemoryBlock(Addr, NumElementsAligned*Alignment));
  return AlignedAddr;
}

uint8_t *LLIMCJITMemoryManager::allocateCodeSection(uintptr_t Size,
                                                    unsigned Alignment,
                                                    unsigned SectionID) {
  if (!Alignment)
    Alignment = 16;
  unsigned NeedAllocate = Alignment * ((Size + Alignment - 1)/Alignment + 1);
  uintptr_t Addr = 0;
  // Look in the list of free code memory regions and use a block there if one
  // is available.
  for (int i = 0, e = FreeCodeMem.size(); i != e; ++i) {
    sys::MemoryBlock &MB = FreeCodeMem[i];
    if (MB.size() >= NeedAllocate) {
      Addr = (uintptr_t)MB.base();
      uintptr_t EndOfBlock = Addr + MB.size();
      // Align the address.
      Addr = (Addr + Alignment - 1) & ~(uintptr_t)(Alignment - 1);
      // Store cutted free memory block.
      FreeCodeMem[i] = sys::MemoryBlock((void*)(Addr + Size),
                                        EndOfBlock - Addr - Size);
      return (uint8_t*)Addr;
    }
  }

  // No pre-allocated free block was large enough. Allocate a new memory region.
  sys::MemoryBlock MB = sys::Memory::AllocateRWX(NeedAllocate, 0, 0);

  AllocatedCodeMem.push_back(MB);
  Addr = (uintptr_t)MB.base();
  uintptr_t EndOfBlock = Addr + MB.size();
  // Align the address.
  Addr = (Addr + Alignment - 1) & ~(uintptr_t)(Alignment - 1);
  // The AllocateRWX may allocate much more memory than we need. In this case,
  // we store the unused memory as a free memory block.
  unsigned FreeSize = EndOfBlock-Addr-Size;
  if (FreeSize > 16)
    FreeCodeMem.push_back(sys::MemoryBlock((void*)(Addr + Size), FreeSize));

  // Return aligned address
  return (uint8_t*)Addr;
}

void LLIMCJITMemoryManager::invalidateInstructionCache() {
  for (int i = 0, e = AllocatedCodeMem.size(); i != e; ++i)
    sys::Memory::InvalidateInstructionCache(AllocatedCodeMem[i].base(),
                                            AllocatedCodeMem[i].size());
}

static int jit_noop() {
  return 0;
}

void *LLIMCJITMemoryManager::getPointerToNamedFunction(const std::string &Name,
                                                       bool AbortOnFailure) {
#if defined(__linux__)
  //===--------------------------------------------------------------------===//
  // Function stubs that are invoked instead of certain library calls
  //
  // Force the following functions to be linked in to anything that uses the
  // JIT. This is a hack designed to work around the all-too-clever Glibc
  // strategy of making these functions work differently when inlined vs. when
  // not inlined, and hiding their real definitions in a separate archive file
  // that the dynamic linker can't see. For more info, search for
  // 'libc_nonshared.a' on Google, or read http://llvm.org/PR274.
  if (Name == "stat") return (void*)(intptr_t)&stat;
  if (Name == "fstat") return (void*)(intptr_t)&fstat;
  if (Name == "lstat") return (void*)(intptr_t)&lstat;
  if (Name == "stat64") return (void*)(intptr_t)&stat64;
  if (Name == "fstat64") return (void*)(intptr_t)&fstat64;
  if (Name == "lstat64") return (void*)(intptr_t)&lstat64;
  if (Name == "atexit") return (void*)(intptr_t)&atexit;
  if (Name == "mknod") return (void*)(intptr_t)&mknod;
#endif // __linux__

  // We should not invoke parent's ctors/dtors from generated main()!
  // On Mingw and Cygwin, the symbol __main is resolved to
  // callee's(eg. tools/lli) one, to invoke wrong duplicated ctors
  // (and register wrong callee's dtors with atexit(3)).
  // We expect ExecutionEngine::runStaticConstructorsDestructors()
  // is called before ExecutionEngine::runFunctionAsMain() is called.
  if (Name == "__main") return (void*)(intptr_t)&jit_noop;

  const char *NameStr = Name.c_str();
  void *Ptr = sys::DynamicLibrary::SearchForAddressOfSymbol(NameStr);
  if (Ptr) return Ptr;

  // If it wasn't found and if it starts with an underscore ('_') character,
  // try again without the underscore.
  if (NameStr[0] == '_') {
    Ptr = sys::DynamicLibrary::SearchForAddressOfSymbol(NameStr+1);
    if (Ptr) return Ptr;
  }

  if (AbortOnFailure)
    report_fatal_error("Program used external function '" + Name +
                      "' which could not be resolved!");
  return 0;
}

LLIMCJITMemoryManager::~LLIMCJITMemoryManager() {
  for (unsigned i = 0, e = AllocatedCodeMem.size(); i != e; ++i)
    sys::Memory::ReleaseRWX(AllocatedCodeMem[i]);
  for (unsigned i = 0, e = AllocatedDataMem.size(); i != e; ++i)
    free(AllocatedDataMem[i].base());
}


void layoutRemoteTargetMemory(RemoteTarget *T, RecordingMemoryManager *JMM) {
  // Lay out our sections in order, with all the code sections first, then
  // all the data sections.
  uint64_t CurOffset = 0;
  unsigned MaxAlign = T->getPageAlignment();
  SmallVector<std::pair<const void*, uint64_t>, 16> Offsets;
  SmallVector<unsigned, 16> Sizes;
  for (RecordingMemoryManager::const_code_iterator I = JMM->code_begin(),
                                                   E = JMM->code_end();
       I != E; ++I) {
    DEBUG(dbgs() << "code region: size " << I->first.size()
                 << ", alignment " << I->second << "\n");
    // Align the current offset up to whatever is needed for the next
    // section.
    unsigned Align = I->second;
    CurOffset = (CurOffset + Align - 1) / Align * Align;
    // Save off the address of the new section and allocate its space.
    Offsets.push_back(std::pair<const void*,uint64_t>(I->first.base(), CurOffset));
    Sizes.push_back(I->first.size());
    CurOffset += I->first.size();
  }
  // Adjust to keep code and data aligned on seperate pages.
  CurOffset = (CurOffset + MaxAlign - 1) / MaxAlign * MaxAlign;
  unsigned FirstDataIndex = Offsets.size();
  for (RecordingMemoryManager::const_data_iterator I = JMM->data_begin(),
                                                   E = JMM->data_end();
       I != E; ++I) {
    DEBUG(dbgs() << "data region: size " << I->first.size()
                 << ", alignment " << I->second << "\n");
    // Align the current offset up to whatever is needed for the next
    // section.
    unsigned Align = I->second;
    CurOffset = (CurOffset + Align - 1) / Align * Align;
    // Save off the address of the new section and allocate its space.
    Offsets.push_back(std::pair<const void*,uint64_t>(I->first.base(), CurOffset));
    Sizes.push_back(I->first.size());
    CurOffset += I->first.size();
  }

  // Allocate space in the remote target.
  uint64_t RemoteAddr;
  if (T->allocateSpace(CurOffset, MaxAlign, RemoteAddr))
    report_fatal_error(T->getErrorMsg());
  // Map the section addresses so relocations will get updated in the local
  // copies of the sections.
  for (unsigned i = 0, e = Offsets.size(); i != e; ++i) {
    uint64_t Addr = RemoteAddr + Offsets[i].second;
    EE->mapSectionAddress(const_cast<void*>(Offsets[i].first), Addr);

    DEBUG(dbgs() << "  Mapping local: " << Offsets[i].first
                 << " to remote: " << format("%p", Addr) << "\n");

  }

  // Trigger application of relocations
  EE->finalizeObject();

  // Now load it all to the target.
  for (unsigned i = 0, e = Offsets.size(); i != e; ++i) {
    uint64_t Addr = RemoteAddr + Offsets[i].second;

    if (i < FirstDataIndex) {
      T->loadCode(Addr, Offsets[i].first, Sizes[i]);

      DEBUG(dbgs() << "  loading code: " << Offsets[i].first
            << " to remote: " << format("%p", Addr) << "\n");
    } else {
      T->loadData(Addr, Offsets[i].first, Sizes[i]);

      DEBUG(dbgs() << "  loading data: " << Offsets[i].first
            << " to remote: " << format("%p", Addr) << "\n");
    }

  }
}

//===----------------------------------------------------------------------===//
// main Driver function
//
int main(int argc, char **argv, char * const *envp) {
  sys::PrintStackTraceOnErrorSignal();
  PrettyStackTraceProgram X(argc, argv);

  LLVMContext &Context = getGlobalContext();
  atexit(do_shutdown);  // Call llvm_shutdown() on exit.

  // If we have a native target, initialize it to ensure it is linked in and
  // usable by the JIT.
  InitializeNativeTarget();
  InitializeNativeTargetAsmPrinter();
  InitializeNativeTargetAsmParser();

  cl::ParseCommandLineOptions(argc, argv,
                              "llvm interpreter & dynamic compiler\n");

  // If the user doesn't want core files, disable them.
  if (DisableCoreFiles)
    sys::Process::PreventCoreFiles();

  // Load the bitcode...
  SMDiagnostic Err;
  Module *Mod = ParseIRFile(InputFile, Err, Context);
  if (!Mod) {
    Err.print(argv[0], errs());
    return 1;
  }

  // If not jitting lazily, load the whole bitcode file eagerly too.
  std::string ErrorMsg;
  if (NoLazyCompilation) {
    if (Mod->MaterializeAllPermanently(&ErrorMsg)) {
      errs() << argv[0] << ": bitcode didn't read correctly.\n";
      errs() << "Reason: " << ErrorMsg << "\n";
      exit(1);
    }
  }

  EngineBuilder builder(Mod);
  builder.setMArch(MArch);
  builder.setMCPU(MCPU);
  builder.setMAttrs(MAttrs);
  builder.setRelocationModel(RelocModel);
  builder.setCodeModel(CMModel);
  builder.setErrorStr(&ErrorMsg);
  builder.setEngineKind(ForceInterpreter
                        ? EngineKind::Interpreter
                        : EngineKind::JIT);

  // If we are supposed to override the target triple, do so now.
  if (!TargetTriple.empty())
    Mod->setTargetTriple(Triple::normalize(TargetTriple));

  // Enable MCJIT if desired.
  JITMemoryManager *JMM = 0;
  if (UseMCJIT && !ForceInterpreter) {
    builder.setUseMCJIT(true);
    if (RemoteMCJIT)
      JMM = new RecordingMemoryManager();
    else
      JMM = new LLIMCJITMemoryManager();
    builder.setJITMemoryManager(JMM);
  } else {
    if (RemoteMCJIT) {
      errs() << "error: Remote process execution requires -use-mcjit\n";
      exit(1);
    }
    builder.setJITMemoryManager(ForceInterpreter ? 0 :
                                JITMemoryManager::CreateDefaultMemManager());
  }

  CodeGenOpt::Level OLvl = CodeGenOpt::Default;
  switch (OptLevel) {
  default:
    errs() << argv[0] << ": invalid optimization level.\n";
    return 1;
  case ' ': break;
  case '0': OLvl = CodeGenOpt::None; break;
  case '1': OLvl = CodeGenOpt::Less; break;
  case '2': OLvl = CodeGenOpt::Default; break;
  case '3': OLvl = CodeGenOpt::Aggressive; break;
  }
  builder.setOptLevel(OLvl);

  TargetOptions Options;
  Options.UseSoftFloat = GenerateSoftFloatCalls;
  if (FloatABIForCalls != FloatABI::Default)
    Options.FloatABIType = FloatABIForCalls;
  if (GenerateSoftFloatCalls)
    FloatABIForCalls = FloatABI::Soft;

  // Remote target execution doesn't handle EH or debug registration.
  if (!RemoteMCJIT) {
    Options.JITExceptionHandling = EnableJITExceptionHandling;
    Options.JITEmitDebugInfo = EmitJitDebugInfo;
    Options.JITEmitDebugInfoToDisk = EmitJitDebugInfoToDisk;
  }

  builder.setTargetOptions(Options);

  EE = builder.create();
  if (!EE) {
    if (!ErrorMsg.empty())
      errs() << argv[0] << ": error creating EE: " << ErrorMsg << "\n";
    else
      errs() << argv[0] << ": unknown error creating EE!\n";
    exit(1);
  }

  // The following functions have no effect if their respective profiling
  // support wasn't enabled in the build configuration.
  EE->RegisterJITEventListener(
                JITEventListener::createOProfileJITEventListener());
  EE->RegisterJITEventListener(
                JITEventListener::createIntelJITEventListener());

  if (!NoLazyCompilation && RemoteMCJIT) {
    errs() << "warning: remote mcjit does not support lazy compilation\n";
    NoLazyCompilation = true;
  }
  EE->DisableLazyCompilation(NoLazyCompilation);

  // If the user specifically requested an argv[0] to pass into the program,
  // do it now.
  if (!FakeArgv0.empty()) {
    InputFile = FakeArgv0;
  } else {
    // Otherwise, if there is a .bc suffix on the executable strip it off, it
    // might confuse the program.
    if (StringRef(InputFile).endswith(".bc"))
      InputFile.erase(InputFile.length() - 3);
  }

  // Add the module's name to the start of the vector of arguments to main().
  InputArgv.insert(InputArgv.begin(), InputFile);

  // Call the main function from M as if its signature were:
  //   int main (int argc, char **argv, const char **envp)
  // using the contents of Args to determine argc & argv, and the contents of
  // EnvVars to determine envp.
  //
  Function *EntryFn = Mod->getFunction(EntryFunc);
  if (!EntryFn) {
    errs() << '\'' << EntryFunc << "\' function not found in module.\n";
    return -1;
  }

  // If the program doesn't explicitly call exit, we will need the Exit
  // function later on to make an explicit call, so get the function now.
  Constant *Exit = Mod->getOrInsertFunction("exit", Type::getVoidTy(Context),
                                                    Type::getInt32Ty(Context),
                                                    NULL);

  // Reset errno to zero on entry to main.
  errno = 0;

  // Remote target MCJIT doesn't (yet) support static constructors. No reason
  // it couldn't. This is a limitation of the LLI implemantation, not the
  // MCJIT itself. FIXME.
  //
  // Run static constructors.
  if (!RemoteMCJIT)
    EE->runStaticConstructorsDestructors(false);

  if (NoLazyCompilation) {
    for (Module::iterator I = Mod->begin(), E = Mod->end(); I != E; ++I) {
      Function *Fn = &*I;
      if (Fn != EntryFn && !Fn->isDeclaration())
        EE->getPointerToFunction(Fn);
    }
  }

  int Result;
  if (RemoteMCJIT) {
    RecordingMemoryManager *MM = static_cast<RecordingMemoryManager*>(JMM);
    // Everything is prepared now, so lay out our program for the target
    // address space, assign the section addresses to resolve any relocations,
    // and send it to the target.
    RemoteTarget Target;
    Target.create();

    // Ask for a pointer to the entry function. This triggers the actual
    // compilation.
    (void)EE->getPointerToFunction(EntryFn);

    // Enough has been compiled to execute the entry function now, so
    // layout the target memory.
    layoutRemoteTargetMemory(&Target, MM);

    // Since we're executing in a (at least simulated) remote address space,
    // we can't use the ExecutionEngine::runFunctionAsMain(). We have to
    // grab the function address directly here and tell the remote target
    // to execute the function.
    // FIXME: argv and envp handling.
    uint64_t Entry = (uint64_t)EE->getPointerToFunction(EntryFn);

    DEBUG(dbgs() << "Executing '" << EntryFn->getName() << "' at "
                 << format("%p", Entry) << "\n");

    if (Target.executeCode(Entry, Result))
      errs() << "ERROR: " << Target.getErrorMsg() << "\n";

    Target.stop();
  } else {
    // Trigger compilation separately so code regions that need to be 
    // invalidated will be known.
    (void)EE->getPointerToFunction(EntryFn);
    // Clear instruction cache before code will be executed.
    if (JMM)
      static_cast<LLIMCJITMemoryManager*>(JMM)->invalidateInstructionCache();

    // Run main.
    Result = EE->runFunctionAsMain(EntryFn, InputArgv, envp);
  }

  // Like static constructors, the remote target MCJIT support doesn't handle
  // this yet. It could. FIXME.
  if (!RemoteMCJIT) {
    // Run static destructors.
    EE->runStaticConstructorsDestructors(true);

    // If the program didn't call exit explicitly, we should call it now.
    // This ensures that any atexit handlers get called correctly.
    if (Function *ExitF = dyn_cast<Function>(Exit)) {
      std::vector<GenericValue> Args;
      GenericValue ResultGV;
      ResultGV.IntVal = APInt(32, Result);
      Args.push_back(ResultGV);
      EE->runFunction(ExitF, Args);
      errs() << "ERROR: exit(" << Result << ") returned!\n";
      abort();
    } else {
      errs() << "ERROR: exit defined with wrong prototype!\n";
      abort();
    }
  }
  return Result;
}
