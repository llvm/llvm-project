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

#include "LTOCodeGenerator.h"
#include "LTOModule.h"
#include "llvm/Constants.h"
#include "llvm/DataLayout.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Linker.h"
#include "llvm/LLVMContext.h"
#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/Config/config.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/SubtargetFeature.h"
#include "llvm/Target/Mangler.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/system_error.h"
#include "llvm/ADT/StringExtras.h"
using namespace llvm;

static cl::opt<bool>
DisableInline("disable-inlining", cl::init(false),
  cl::desc("Do not run the inliner pass"));

static cl::opt<bool>
DisableGVNLoadPRE("disable-gvn-loadpre", cl::init(false),
  cl::desc("Do not run the GVN load PRE pass"));

const char* LTOCodeGenerator::getVersionString() {
#ifdef LLVM_VERSION_INFO
  return PACKAGE_NAME " version " PACKAGE_VERSION ", " LLVM_VERSION_INFO;
#else
  return PACKAGE_NAME " version " PACKAGE_VERSION;
#endif
}

LTOCodeGenerator::LTOCodeGenerator()
  : _context(getGlobalContext()),
    _linker("LinkTimeOptimizer", "ld-temp.o", _context), _target(NULL),
    _emitDwarfDebugInfo(false), _scopeRestrictionsDone(false),
    _codeModel(LTO_CODEGEN_PIC_MODEL_DYNAMIC),
    _nativeObjectFile(NULL) {
  InitializeAllTargets();
  InitializeAllTargetMCs();
  InitializeAllAsmPrinters();
}

LTOCodeGenerator::~LTOCodeGenerator() {
  delete _target;
  delete _nativeObjectFile;

  for (std::vector<char*>::iterator I = _codegenOptions.begin(),
         E = _codegenOptions.end(); I != E; ++I)
    free(*I);
}

bool LTOCodeGenerator::addModule(LTOModule* mod, std::string& errMsg) {
  bool ret = _linker.LinkInModule(mod->getLLVVMModule(), &errMsg);

  const std::vector<const char*> &undefs = mod->getAsmUndefinedRefs();
  for (int i = 0, e = undefs.size(); i != e; ++i)
    _asmUndefinedRefs[undefs[i]] = 1;

  return ret;
}

bool LTOCodeGenerator::setDebugInfo(lto_debug_model debug,
                                    std::string& errMsg) {
  switch (debug) {
  case LTO_DEBUG_MODEL_NONE:
    _emitDwarfDebugInfo = false;
    return false;

  case LTO_DEBUG_MODEL_DWARF:
    _emitDwarfDebugInfo = true;
    return false;
  }
  llvm_unreachable("Unknown debug format!");
}

bool LTOCodeGenerator::setCodePICModel(lto_codegen_model model,
                                       std::string& errMsg) {
  switch (model) {
  case LTO_CODEGEN_PIC_MODEL_STATIC:
  case LTO_CODEGEN_PIC_MODEL_DYNAMIC:
  case LTO_CODEGEN_PIC_MODEL_DYNAMIC_NO_PIC:
    _codeModel = model;
    return false;
  }
  llvm_unreachable("Unknown PIC model!");
}

bool LTOCodeGenerator::writeMergedModules(const char *path,
                                          std::string &errMsg) {
  if (determineTarget(errMsg))
    return true;

  // mark which symbols can not be internalized
  applyScopeRestrictions();

  // create output file
  std::string ErrInfo;
  tool_output_file Out(path, ErrInfo,
                       raw_fd_ostream::F_Binary);
  if (!ErrInfo.empty()) {
    errMsg = "could not open bitcode file for writing: ";
    errMsg += path;
    return true;
  }

  // write bitcode to it
  WriteBitcodeToFile(_linker.getModule(), Out.os());
  Out.os().close();

  if (Out.os().has_error()) {
    errMsg = "could not write bitcode file: ";
    errMsg += path;
    Out.os().clear_error();
    return true;
  }

  Out.keep();
  return false;
}

bool LTOCodeGenerator::compile_to_file(const char** name, std::string& errMsg) {
  // make unique temp .o file to put generated object file
  sys::PathWithStatus uniqueObjPath("lto-llvm.o");
  if (uniqueObjPath.createTemporaryFileOnDisk(false, &errMsg)) {
    uniqueObjPath.eraseFromDisk();
    return true;
  }
  sys::RemoveFileOnSignal(uniqueObjPath);

  // generate object file
  bool genResult = false;
  tool_output_file objFile(uniqueObjPath.c_str(), errMsg);
  if (!errMsg.empty()) {
    uniqueObjPath.eraseFromDisk();
    return true;
  }

  genResult = this->generateObjectFile(objFile.os(), errMsg);
  objFile.os().close();
  if (objFile.os().has_error()) {
    objFile.os().clear_error();
    uniqueObjPath.eraseFromDisk();
    return true;
  }

  objFile.keep();
  if (genResult) {
    uniqueObjPath.eraseFromDisk();
    return true;
  }

  _nativeObjectPath = uniqueObjPath.str();
  *name = _nativeObjectPath.c_str();
  return false;
}

const void* LTOCodeGenerator::compile(size_t* length, std::string& errMsg) {
  const char *name;
  if (compile_to_file(&name, errMsg))
    return NULL;

  // remove old buffer if compile() called twice
  delete _nativeObjectFile;

  // read .o file into memory buffer
  OwningPtr<MemoryBuffer> BuffPtr;
  if (error_code ec = MemoryBuffer::getFile(name, BuffPtr, -1, false)) {
    errMsg = ec.message();
    sys::Path(_nativeObjectPath).eraseFromDisk();
    return NULL;
  }
  _nativeObjectFile = BuffPtr.take();

  // remove temp files
  sys::Path(_nativeObjectPath).eraseFromDisk();

  // return buffer, unless error
  if (_nativeObjectFile == NULL)
    return NULL;
  *length = _nativeObjectFile->getBufferSize();
  return _nativeObjectFile->getBufferStart();
}

bool LTOCodeGenerator::determineTarget(std::string& errMsg) {
  if (_target != NULL)
    return false;

  std::string TripleStr = _linker.getModule()->getTargetTriple();
  if (TripleStr.empty())
    TripleStr = sys::getDefaultTargetTriple();
  llvm::Triple Triple(TripleStr);

  // create target machine from info for merged modules
  const Target *march = TargetRegistry::lookupTarget(TripleStr, errMsg);
  if (march == NULL)
    return true;

  // The relocation model is actually a static member of TargetMachine and
  // needs to be set before the TargetMachine is instantiated.
  Reloc::Model RelocModel = Reloc::Default;
  switch (_codeModel) {
  case LTO_CODEGEN_PIC_MODEL_STATIC:
    RelocModel = Reloc::Static;
    break;
  case LTO_CODEGEN_PIC_MODEL_DYNAMIC:
    RelocModel = Reloc::PIC_;
    break;
  case LTO_CODEGEN_PIC_MODEL_DYNAMIC_NO_PIC:
    RelocModel = Reloc::DynamicNoPIC;
    break;
  }

  // construct LTOModule, hand over ownership of module and target
  SubtargetFeatures Features;
  Features.getDefaultSubtargetFeatures(Triple);
  std::string FeatureStr = Features.getString();
  // Set a default CPU for Darwin triples.
  if (_mCpu.empty() && Triple.isOSDarwin()) {
    if (Triple.getArch() == llvm::Triple::x86_64)
      _mCpu = "core2";
    else if (Triple.getArch() == llvm::Triple::x86)
      _mCpu = "yonah";
  }
  TargetOptions Options;
  LTOModule::getTargetOptions(Options);
  _target = march->createTargetMachine(TripleStr, _mCpu, FeatureStr, Options,
                                       RelocModel, CodeModel::Default,
                                       CodeGenOpt::Aggressive);
  return false;
}

void LTOCodeGenerator::
applyRestriction(GlobalValue &GV,
                 std::vector<const char*> &mustPreserveList,
                 SmallPtrSet<GlobalValue*, 8> &asmUsed,
                 Mangler &mangler) {
  SmallString<64> Buffer;
  mangler.getNameWithPrefix(Buffer, &GV, false);

  if (GV.isDeclaration())
    return;
  if (_mustPreserveSymbols.count(Buffer))
    mustPreserveList.push_back(GV.getName().data());
  if (_asmUndefinedRefs.count(Buffer))
    asmUsed.insert(&GV);
}

static void findUsedValues(GlobalVariable *LLVMUsed,
                           SmallPtrSet<GlobalValue*, 8> &UsedValues) {
  if (LLVMUsed == 0) return;

  ConstantArray *Inits = dyn_cast<ConstantArray>(LLVMUsed->getInitializer());
  if (Inits == 0) return;

  for (unsigned i = 0, e = Inits->getNumOperands(); i != e; ++i)
    if (GlobalValue *GV =
        dyn_cast<GlobalValue>(Inits->getOperand(i)->stripPointerCasts()))
      UsedValues.insert(GV);
}

void LTOCodeGenerator::applyScopeRestrictions() {
  if (_scopeRestrictionsDone) return;
  Module *mergedModule = _linker.getModule();

  // Start off with a verification pass.
  PassManager passes;
  passes.add(createVerifierPass());

  // mark which symbols can not be internalized
  MCContext Context(*_target->getMCAsmInfo(), *_target->getRegisterInfo(),NULL);
  Mangler mangler(Context, *_target->getDataLayout());
  std::vector<const char*> mustPreserveList;
  SmallPtrSet<GlobalValue*, 8> asmUsed;

  for (Module::iterator f = mergedModule->begin(),
         e = mergedModule->end(); f != e; ++f)
    applyRestriction(*f, mustPreserveList, asmUsed, mangler);
  for (Module::global_iterator v = mergedModule->global_begin(),
         e = mergedModule->global_end(); v !=  e; ++v)
    applyRestriction(*v, mustPreserveList, asmUsed, mangler);
  for (Module::alias_iterator a = mergedModule->alias_begin(),
         e = mergedModule->alias_end(); a != e; ++a)
    applyRestriction(*a, mustPreserveList, asmUsed, mangler);

  GlobalVariable *LLVMCompilerUsed =
    mergedModule->getGlobalVariable("llvm.compiler.used");
  findUsedValues(LLVMCompilerUsed, asmUsed);
  if (LLVMCompilerUsed)
    LLVMCompilerUsed->eraseFromParent();

  llvm::Type *i8PTy = llvm::Type::getInt8PtrTy(_context);
  std::vector<Constant*> asmUsed2;
  for (SmallPtrSet<GlobalValue*, 16>::const_iterator i = asmUsed.begin(),
         e = asmUsed.end(); i !=e; ++i) {
    GlobalValue *GV = *i;
    Constant *c = ConstantExpr::getBitCast(GV, i8PTy);
    asmUsed2.push_back(c);
  }

  llvm::ArrayType *ATy = llvm::ArrayType::get(i8PTy, asmUsed2.size());
  LLVMCompilerUsed =
    new llvm::GlobalVariable(*mergedModule, ATy, false,
                             llvm::GlobalValue::AppendingLinkage,
                             llvm::ConstantArray::get(ATy, asmUsed2),
                             "llvm.compiler.used");

  LLVMCompilerUsed->setSection("llvm.metadata");

  passes.add(createInternalizePass(mustPreserveList));

  // apply scope restrictions
  passes.run(*mergedModule);

  _scopeRestrictionsDone = true;
}

/// Optimize merged modules using various IPO passes
bool LTOCodeGenerator::generateObjectFile(raw_ostream &out,
                                          std::string &errMsg) {
  if (this->determineTarget(errMsg))
    return true;

  Module* mergedModule = _linker.getModule();

  // if options were requested, set them
  if (!_codegenOptions.empty())
    cl::ParseCommandLineOptions(_codegenOptions.size(),
                                const_cast<char **>(&_codegenOptions[0]));

  // mark which symbols can not be internalized
  this->applyScopeRestrictions();

  // Instantiate the pass manager to organize the passes.
  PassManager passes;

  // Start off with a verification pass.
  passes.add(createVerifierPass());

  // Add an appropriate DataLayout instance for this module...
  passes.add(new DataLayout(*_target->getDataLayout()));
  passes.add(new TargetTransformInfo(_target->getScalarTargetTransformInfo(),
                                     _target->getVectorTargetTransformInfo()));

  // Enabling internalize here would use its AllButMain variant. It
  // keeps only main if it exists and does nothing for libraries. Instead
  // we create the pass ourselves with the symbol list provided by the linker.
  PassManagerBuilder().populateLTOPassManager(passes, /*Internalize=*/false,
                                              !DisableInline,
                                              DisableGVNLoadPRE);

  // Make sure everything is still good.
  passes.add(createVerifierPass());

  FunctionPassManager *codeGenPasses = new FunctionPassManager(mergedModule);

  codeGenPasses->add(new DataLayout(*_target->getDataLayout()));

  formatted_raw_ostream Out(out);

  if (_target->addPassesToEmitFile(*codeGenPasses, Out,
                                   TargetMachine::CGFT_ObjectFile)) {
    errMsg = "target file type not supported";
    return true;
  }

  // Run our queue of passes all at once now, efficiently.
  passes.run(*mergedModule);

  // Run the code generator, and write assembly file
  codeGenPasses->doInitialization();

  for (Module::iterator
         it = mergedModule->begin(), e = mergedModule->end(); it != e; ++it)
    if (!it->isDeclaration())
      codeGenPasses->run(*it);

  codeGenPasses->doFinalization();
  delete codeGenPasses;

  return false; // success
}

/// setCodeGenDebugOptions - Set codegen debugging options to aid in debugging
/// LTO problems.
void LTOCodeGenerator::setCodeGenDebugOptions(const char *options) {
  for (std::pair<StringRef, StringRef> o = getToken(options);
       !o.first.empty(); o = getToken(o.second)) {
    // ParseCommandLineOptions() expects argv[0] to be program name. Lazily add
    // that.
    if (_codegenOptions.empty())
      _codegenOptions.push_back(strdup("libLTO"));
    _codegenOptions.push_back(strdup(o.first.str().c_str()));
  }
}
