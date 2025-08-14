//===- ExecutionEngine.cpp - MLIR Execution engine and utils --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the execution engine for MLIR modules based on LLVM Orc
// JIT engine.
//
//===----------------------------------------------------------------------===//
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "llvm/ExecutionEngine/JITEventListener.h"
#include "llvm/ExecutionEngine/ObjectCache.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/IRTransformLayer.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/SubtargetFeature.h"

#define DEBUG_TYPE "execution-engine"

using namespace mlir;
using llvm::dbgs;
using llvm::Error;
using llvm::errs;
using llvm::Expected;
using llvm::LLVMContext;
using llvm::MemoryBuffer;
using llvm::MemoryBufferRef;
using llvm::Module;
using llvm::SectionMemoryManager;
using llvm::StringError;
using llvm::Triple;
using llvm::orc::DynamicLibrarySearchGenerator;
using llvm::orc::ExecutionSession;
using llvm::orc::IRCompileLayer;
using llvm::orc::JITTargetMachineBuilder;
using llvm::orc::MangleAndInterner;
using llvm::orc::RTDyldObjectLinkingLayer;
using llvm::orc::SymbolMap;
using llvm::orc::ThreadSafeModule;
using llvm::orc::TMOwningSimpleCompiler;

/// Wrap a string into an llvm::StringError.
static Error makeStringError(const Twine &message) {
  return llvm::make_error<StringError>(message.str(),
                                       llvm::inconvertibleErrorCode());
}

void SimpleObjectCache::notifyObjectCompiled(const Module *m,
                                             MemoryBufferRef objBuffer) {
  cachedObjects[m->getModuleIdentifier()] = MemoryBuffer::getMemBufferCopy(
      objBuffer.getBuffer(), objBuffer.getBufferIdentifier());
}

std::unique_ptr<MemoryBuffer> SimpleObjectCache::getObject(const Module *m) {
  auto i = cachedObjects.find(m->getModuleIdentifier());
  if (i == cachedObjects.end()) {
    LLVM_DEBUG(dbgs() << "No object for " << m->getModuleIdentifier()
                      << " in cache. Compiling.\n");
    return nullptr;
  }
  LLVM_DEBUG(dbgs() << "Object for " << m->getModuleIdentifier()
                    << " loaded from cache.\n");
  return MemoryBuffer::getMemBuffer(i->second->getMemBufferRef());
}

void SimpleObjectCache::dumpToObjectFile(StringRef outputFilename) {
  // Set up the output file.
  std::string errorMessage;
  auto file = openOutputFile(outputFilename, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    return;
  }

  // Dump the object generated for a single module to the output file.
  assert(cachedObjects.size() == 1 && "Expected only one object entry.");
  auto &cachedObject = cachedObjects.begin()->second;
  file->os() << cachedObject->getBuffer();
  file->keep();
}

bool SimpleObjectCache::isEmpty() { return cachedObjects.empty(); }

void ExecutionEngine::dumpToObjectFile(StringRef filename) {
  if (cache == nullptr) {
    llvm::errs() << "cannot dump ExecutionEngine object code to file: "
                    "object cache is disabled\n";
    return;
  }
  // Compilation is lazy and it doesn't populate object cache unless requested.
  // In case object dump is requested before cache is populated, we need to
  // force compilation manually. 
  if (cache->isEmpty()) {
    for (std::string &functionName : functionNames) {
      auto result = lookupPacked(functionName);
      if (!result) {
        llvm::errs() << "Could not compile " << functionName << ":\n  "
                     << result.takeError() << "\n";
        return;
      }
    }
  }
  cache->dumpToObjectFile(filename);
}

void ExecutionEngine::registerSymbols(
    llvm::function_ref<SymbolMap(MangleAndInterner)> symbolMap) {
  auto &mainJitDylib = jit->getMainJITDylib();
  cantFail(mainJitDylib.define(
      absoluteSymbols(symbolMap(llvm::orc::MangleAndInterner(
          mainJitDylib.getExecutionSession(), jit->getDataLayout())))));
}

void ExecutionEngine::setupTargetTripleAndDataLayout(Module *llvmModule,
                                                     llvm::TargetMachine *tm) {
  llvmModule->setDataLayout(tm->createDataLayout());
  llvmModule->setTargetTriple(tm->getTargetTriple());
}

static std::string makePackedFunctionName(StringRef name) {
  return "_mlir_" + name.str();
}

// For each function in the LLVM module, define an interface function that wraps
// all the arguments of the original function and all its results into an i8**
// pointer to provide a unified invocation interface.
static void packFunctionArguments(Module *module) {
  auto &ctx = module->getContext();
  llvm::IRBuilder<> builder(ctx);
  DenseSet<llvm::Function *> interfaceFunctions;
  for (auto &func : module->getFunctionList()) {
    if (func.isDeclaration()) {
      continue;
    }
    if (interfaceFunctions.count(&func)) {
      continue;
    }

    // Given a function `foo(<...>)`, define the interface function
    // `mlir_foo(i8**)`.
    auto *newType =
        llvm::FunctionType::get(builder.getVoidTy(), builder.getPtrTy(),
                                /*isVarArg=*/false);
    auto newName = makePackedFunctionName(func.getName());
    auto funcCst = module->getOrInsertFunction(newName, newType);
    llvm::Function *interfaceFunc = cast<llvm::Function>(funcCst.getCallee());
    interfaceFunctions.insert(interfaceFunc);

    // Extract the arguments from the type-erased argument list and cast them to
    // the proper types.
    auto *bb = llvm::BasicBlock::Create(ctx);
    bb->insertInto(interfaceFunc);
    builder.SetInsertPoint(bb);
    llvm::Value *argList = interfaceFunc->arg_begin();
    SmallVector<llvm::Value *, 8> args;
    args.reserve(llvm::size(func.args()));
    for (auto [index, arg] : llvm::enumerate(func.args())) {
      llvm::Value *argIndex = llvm::Constant::getIntegerValue(
          builder.getInt64Ty(), APInt(64, index));
      llvm::Value *argPtrPtr =
          builder.CreateGEP(builder.getPtrTy(), argList, argIndex);
      llvm::Value *argPtr = builder.CreateLoad(builder.getPtrTy(), argPtrPtr);
      llvm::Type *argTy = arg.getType();
      llvm::Value *load = builder.CreateLoad(argTy, argPtr);
      args.push_back(load);
    }

    // Call the implementation function with the extracted arguments.
    llvm::Value *result = builder.CreateCall(&func, args);

    // Assuming the result is one value, potentially of type `void`.
    if (!result->getType()->isVoidTy()) {
      llvm::Value *retIndex = llvm::Constant::getIntegerValue(
          builder.getInt64Ty(), APInt(64, llvm::size(func.args())));
      llvm::Value *retPtrPtr =
          builder.CreateGEP(builder.getPtrTy(), argList, retIndex);
      llvm::Value *retPtr = builder.CreateLoad(builder.getPtrTy(), retPtrPtr);
      builder.CreateStore(result, retPtr);
    }

    // The interface function returns void.
    builder.CreateRetVoid();
  }
}

ExecutionEngine::ExecutionEngine(bool enableObjectDump,
                                 bool enableGDBNotificationListener,
                                 bool enablePerfNotificationListener)
    : cache(enableObjectDump ? new SimpleObjectCache() : nullptr),
      functionNames(),
      gdbListener(enableGDBNotificationListener
                      ? llvm::JITEventListener::createGDBRegistrationListener()
                      : nullptr),
      perfListener(nullptr) {
  if (enablePerfNotificationListener) {
    if (auto *listener = llvm::JITEventListener::createPerfJITEventListener())
      perfListener = listener;
    else if (auto *listener =
                 llvm::JITEventListener::createIntelJITEventListener())
      perfListener = listener;
  }
}

ExecutionEngine::~ExecutionEngine() {
  // Execute the global destructors from the module being processed.
  // TODO: Allow JIT deinitialize for AArch64. Currently there's a bug causing a
  // crash for AArch64 see related issue #71963.
  if (jit && !jit->getTargetTriple().isAArch64())
    llvm::consumeError(jit->deinitialize(jit->getMainJITDylib()));
  // Run all dynamic library destroy callbacks to prepare for the shutdown.
  for (LibraryDestroyFn destroy : destroyFns)
    destroy();
}

Expected<std::unique_ptr<ExecutionEngine>>
ExecutionEngine::create(Operation *m, const ExecutionEngineOptions &options,
                        std::unique_ptr<llvm::TargetMachine> tm) {
  auto engine = std::make_unique<ExecutionEngine>(
      options.enableObjectDump, options.enableGDBNotificationListener,
      options.enablePerfNotificationListener);

  // Remember all entry-points if object dumping is enabled.
  if (options.enableObjectDump) {
    for (auto funcOp : m->getRegion(0).getOps<LLVM::LLVMFuncOp>()) {
      StringRef funcName = funcOp.getSymName();
      engine->functionNames.push_back(funcName.str());
    }
  }

  std::unique_ptr<llvm::LLVMContext> ctx(new llvm::LLVMContext);
  auto llvmModule = options.llvmModuleBuilder
                        ? options.llvmModuleBuilder(m, *ctx)
                        : translateModuleToLLVMIR(m, *ctx);
  if (!llvmModule)
    return makeStringError("could not convert to LLVM IR");

  // If no valid TargetMachine was passed, create a default TM ignoring any
  // input arguments from the user.
  if (!tm) {
    auto tmBuilderOrError = llvm::orc::JITTargetMachineBuilder::detectHost();
    if (!tmBuilderOrError)
      return tmBuilderOrError.takeError();

    auto tmOrError = tmBuilderOrError->createTargetMachine();
    if (!tmOrError)
      return tmOrError.takeError();
    tm = std::move(tmOrError.get());
  }

  // TODO: Currently, the LLVM module created above has no triple associated
  // with it. Instead, the triple is extracted from the TargetMachine, which is
  // either based on the host defaults or command line arguments when specified
  // (set-up by callers of this method). It could also be passed to the
  // translation or dialect conversion instead of this.
  setupTargetTripleAndDataLayout(llvmModule.get(), tm.get());
  packFunctionArguments(llvmModule.get());

  auto dataLayout = llvmModule->getDataLayout();

  // Use absolute library path so that gdb can find the symbol table.
  SmallVector<SmallString<256>, 4> sharedLibPaths;
  transform(
      options.sharedLibPaths, std::back_inserter(sharedLibPaths),
      [](StringRef libPath) {
        SmallString<256> absPath(libPath.begin(), libPath.end());
        cantFail(llvm::errorCodeToError(llvm::sys::fs::make_absolute(absPath)));
        return absPath;
      });

  // If shared library implements custom execution layer library init and
  // destroy functions, we'll use them to register the library. Otherwise, load
  // the library as JITDyLib below.
  llvm::StringMap<void *> exportSymbols;
  SmallVector<LibraryDestroyFn> destroyFns;
  SmallVector<StringRef> jitDyLibPaths;

  for (auto &libPath : sharedLibPaths) {
    auto lib = llvm::sys::DynamicLibrary::getPermanentLibrary(
        libPath.str().str().c_str());
    void *initSym = lib.getAddressOfSymbol(kLibraryInitFnName);
    void *destroySim = lib.getAddressOfSymbol(kLibraryDestroyFnName);

    // Library does not provide call backs, rely on symbol visiblity.
    if (!initSym || !destroySim) {
      jitDyLibPaths.push_back(libPath);
      continue;
    }

    auto initFn = reinterpret_cast<LibraryInitFn>(initSym);
    initFn(exportSymbols);

    auto destroyFn = reinterpret_cast<LibraryDestroyFn>(destroySim);
    destroyFns.push_back(destroyFn);
  }
  engine->destroyFns = std::move(destroyFns);

  // Callback to create the object layer with symbol resolution to current
  // process and dynamically linked libraries.
  auto objectLinkingLayerCreator = [&](ExecutionSession &session) {
    auto objectLayer = std::make_unique<RTDyldObjectLinkingLayer>(
        session, [sectionMemoryMapper =
                      options.sectionMemoryMapper](const MemoryBuffer &) {
          return std::make_unique<SectionMemoryManager>(sectionMemoryMapper);
        });

    // Register JIT event listeners if they are enabled.
    if (engine->gdbListener)
      objectLayer->registerJITEventListener(*engine->gdbListener);
    if (engine->perfListener)
      objectLayer->registerJITEventListener(*engine->perfListener);

    // COFF format binaries (Windows) need special handling to deal with
    // exported symbol visibility.
    // cf llvm/lib/ExecutionEngine/Orc/LLJIT.cpp LLJIT::createObjectLinkingLayer
    const llvm::Triple &targetTriple = llvmModule->getTargetTriple();
    if (targetTriple.isOSBinFormatCOFF()) {
      objectLayer->setOverrideObjectFlagsWithResponsibilityFlags(true);
      objectLayer->setAutoClaimResponsibilityForObjectSymbols(true);
    }

    // Resolve symbols from shared libraries.
    for (auto &libPath : jitDyLibPaths) {
      auto mb = llvm::MemoryBuffer::getFile(libPath);
      if (!mb) {
        errs() << "Failed to create MemoryBuffer for: " << libPath
               << "\nError: " << mb.getError().message() << "\n";
        continue;
      }
      auto &jd = session.createBareJITDylib(std::string(libPath));
      auto loaded = DynamicLibrarySearchGenerator::Load(
          libPath.str().c_str(), dataLayout.getGlobalPrefix());
      if (!loaded) {
        errs() << "Could not load " << libPath << ":\n  " << loaded.takeError()
               << "\n";
        continue;
      }
      jd.addGenerator(std::move(*loaded));
      cantFail(objectLayer->add(jd, std::move(mb.get())));
    }

    return objectLayer;
  };

  // Callback to inspect the cache and recompile on demand. This follows Lang's
  // LLJITWithObjectCache example.
  auto compileFunctionCreator = [&](JITTargetMachineBuilder jtmb)
      -> Expected<std::unique_ptr<IRCompileLayer::IRCompiler>> {
    if (options.jitCodeGenOptLevel)
      jtmb.setCodeGenOptLevel(*options.jitCodeGenOptLevel);
    return std::make_unique<TMOwningSimpleCompiler>(std::move(tm),
                                                    engine->cache.get());
  };

  // Create the LLJIT by calling the LLJITBuilder with 2 callbacks.
  auto jit =
      cantFail(llvm::orc::LLJITBuilder()
                   .setCompileFunctionCreator(compileFunctionCreator)
                   .setObjectLinkingLayerCreator(objectLinkingLayerCreator)
                   .setDataLayout(dataLayout)
                   .create());

  // Add a ThreadSafemodule to the engine and return.
  ThreadSafeModule tsm(std::move(llvmModule), std::move(ctx));
  if (options.transformer)
    cantFail(tsm.withModuleDo(
        [&](llvm::Module &module) { return options.transformer(&module); }));
  cantFail(jit->addIRModule(std::move(tsm)));
  engine->jit = std::move(jit);

  // Resolve symbols that are statically linked in the current process.
  llvm::orc::JITDylib &mainJD = engine->jit->getMainJITDylib();
  mainJD.addGenerator(
      cantFail(DynamicLibrarySearchGenerator::GetForCurrentProcess(
          dataLayout.getGlobalPrefix())));

  // Build a runtime symbol map from the exported symbols and register them.
  auto runtimeSymbolMap = [&](llvm::orc::MangleAndInterner interner) {
    auto symbolMap = llvm::orc::SymbolMap();
    for (auto &exportSymbol : exportSymbols)
      symbolMap[interner(exportSymbol.getKey())] = {
          llvm::orc::ExecutorAddr::fromPtr(exportSymbol.getValue()),
          llvm::JITSymbolFlags::Exported};
    return symbolMap;
  };
  engine->registerSymbols(runtimeSymbolMap);

  // Execute the global constructors from the module being processed.
  // TODO: Allow JIT initialize for AArch64. Currently there's a bug causing a
  // crash for AArch64 see related issue #71963.
  if (!engine->jit->getTargetTriple().isAArch64())
    cantFail(engine->jit->initialize(engine->jit->getMainJITDylib()));

  return std::move(engine);
}

Expected<void (*)(void **)>
ExecutionEngine::lookupPacked(StringRef name) const {
  auto result = lookup(makePackedFunctionName(name));
  if (!result)
    return result.takeError();
  return reinterpret_cast<void (*)(void **)>(result.get());
}

Expected<void *> ExecutionEngine::lookup(StringRef name) const {
  auto expectedSymbol = jit->lookup(name);

  // JIT lookup may return an Error referring to strings stored internally by
  // the JIT. If the Error outlives the ExecutionEngine, it would want have a
  // dangling reference, which is currently caught by an assertion inside JIT
  // thanks to hand-rolled reference counting. Rewrap the error message into a
  // string before returning. Alternatively, ORC JIT should consider copying
  // the string into the error message.
  if (!expectedSymbol) {
    std::string errorMessage;
    llvm::raw_string_ostream os(errorMessage);
    llvm::handleAllErrors(expectedSymbol.takeError(),
                          [&os](llvm::ErrorInfoBase &ei) { ei.log(os); });
    return makeStringError(errorMessage);
  }

  if (void *fptr = expectedSymbol->toPtr<void *>())
    return fptr;
  return makeStringError("looked up function is null");
}

Error ExecutionEngine::invokePacked(StringRef name,
                                    MutableArrayRef<void *> args) {
  auto expectedFPtr = lookupPacked(name);
  if (!expectedFPtr)
    return expectedFPtr.takeError();
  auto fptr = *expectedFPtr;

  (*fptr)(args.data());

  return Error::success();
}
