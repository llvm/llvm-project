//===--------- LLJIT.cpp - An ORC-based JIT for compiling LLVM IR ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/JITLink/EHFrameSupport.h"
#include "llvm/ExecutionEngine/JITLink/JITLinkMemoryManager.h"
#include "llvm/ExecutionEngine/Orc/MachOPlatform.h"
#include "llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/Orc/OrcError.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/Orc/TargetProcessControl.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Mangler.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/DynamicLibrary.h"

#include <map>

#define DEBUG_TYPE "orc"

using namespace llvm;
using namespace llvm::orc;

namespace {

/// Adds helper function decls and wrapper functions that call the helper with
/// some additional prefix arguments.
///
/// E.g. For wrapper "foo" with type i8(i8, i64), helper "bar", and prefix
/// args i32 4 and i16 12345, this function will add:
///
/// declare i8 @bar(i32, i16, i8, i64)
///
/// define i8 @foo(i8, i64) {
/// entry:
///   %2 = call i8 @bar(i32 4, i16 12345, i8 %0, i64 %1)
///   ret i8 %2
/// }
///
Function *addHelperAndWrapper(Module &M, StringRef WrapperName,
                              FunctionType *WrapperFnType,
                              GlobalValue::VisibilityTypes WrapperVisibility,
                              StringRef HelperName,
                              ArrayRef<Value *> HelperPrefixArgs) {
  std::vector<Type *> HelperArgTypes;
  for (auto *Arg : HelperPrefixArgs)
    HelperArgTypes.push_back(Arg->getType());
  for (auto *T : WrapperFnType->params())
    HelperArgTypes.push_back(T);
  auto *HelperFnType =
      FunctionType::get(WrapperFnType->getReturnType(), HelperArgTypes, false);
  auto *HelperFn = Function::Create(HelperFnType, GlobalValue::ExternalLinkage,
                                    HelperName, M);

  auto *WrapperFn = Function::Create(
      WrapperFnType, GlobalValue::ExternalLinkage, WrapperName, M);
  WrapperFn->setVisibility(WrapperVisibility);

  auto *EntryBlock = BasicBlock::Create(M.getContext(), "entry", WrapperFn);
  IRBuilder<> IB(EntryBlock);

  std::vector<Value *> HelperArgs;
  for (auto *Arg : HelperPrefixArgs)
    HelperArgs.push_back(Arg);
  for (auto &Arg : WrapperFn->args())
    HelperArgs.push_back(&Arg);
  auto *HelperResult = IB.CreateCall(HelperFn, HelperArgs);
  if (HelperFn->getReturnType()->isVoidTy())
    IB.CreateRetVoid();
  else
    IB.CreateRet(HelperResult);

  return WrapperFn;
}

class GenericLLVMIRPlatformSupport;

/// orc::Platform component of Generic LLVM IR Platform support.
/// Just forwards calls to the GenericLLVMIRPlatformSupport class below.
class GenericLLVMIRPlatform : public Platform {
public:
  GenericLLVMIRPlatform(GenericLLVMIRPlatformSupport &S) : S(S) {}
  Error setupJITDylib(JITDylib &JD) override;
  Error notifyAdding(ResourceTracker &RT,
                     const MaterializationUnit &MU) override;
  Error notifyRemoving(ResourceTracker &RT) override {
    // Noop -- Nothing to do (yet).
    return Error::success();
  }

private:
  GenericLLVMIRPlatformSupport &S;
};

/// This transform parses llvm.global_ctors to produce a single initialization
/// function for the module, records the function, then deletes
/// llvm.global_ctors.
class GlobalCtorDtorScraper {
public:

  GlobalCtorDtorScraper(GenericLLVMIRPlatformSupport &PS,
                        StringRef InitFunctionPrefix)
    : PS(PS), InitFunctionPrefix(InitFunctionPrefix) {}
  Expected<ThreadSafeModule> operator()(ThreadSafeModule TSM,
                                        MaterializationResponsibility &R);

private:
  GenericLLVMIRPlatformSupport &PS;
  StringRef InitFunctionPrefix;
};

/// Generic IR Platform Support
///
/// Scrapes llvm.global_ctors and llvm.global_dtors and replaces them with
/// specially named 'init' and 'deinit'. Injects definitions / interposes for
/// some runtime API, including __cxa_atexit, dlopen, and dlclose.
class GenericLLVMIRPlatformSupport : public LLJIT::PlatformSupport {
public:
  // GenericLLVMIRPlatform &P) : P(P) {
  GenericLLVMIRPlatformSupport(LLJIT &J)
      : J(J), InitFunctionPrefix(J.mangle("__orc_init_func.")) {

    getExecutionSession().setPlatform(
        std::make_unique<GenericLLVMIRPlatform>(*this));

    setInitTransform(J, GlobalCtorDtorScraper(*this, InitFunctionPrefix));

    SymbolMap StdInterposes;

    StdInterposes[J.mangleAndIntern("__lljit.platform_support_instance")] =
        JITEvaluatedSymbol(pointerToJITTargetAddress(this),
                           JITSymbolFlags::Exported);
    StdInterposes[J.mangleAndIntern("__lljit.cxa_atexit_helper")] =
        JITEvaluatedSymbol(pointerToJITTargetAddress(registerAtExitHelper),
                           JITSymbolFlags());

    cantFail(
        J.getMainJITDylib().define(absoluteSymbols(std::move(StdInterposes))));
    cantFail(setupJITDylib(J.getMainJITDylib()));
    cantFail(J.addIRModule(J.getMainJITDylib(), createPlatformRuntimeModule()));
  }

  ExecutionSession &getExecutionSession() { return J.getExecutionSession(); }

  /// Adds a module that defines the __dso_handle global.
  Error setupJITDylib(JITDylib &JD) {

    // Add per-jitdylib standard interposes.
    SymbolMap PerJDInterposes;
    PerJDInterposes[J.mangleAndIntern("__lljit.run_atexits_helper")] =
        JITEvaluatedSymbol(pointerToJITTargetAddress(runAtExitsHelper),
                           JITSymbolFlags());
    cantFail(JD.define(absoluteSymbols(std::move(PerJDInterposes))));

    auto Ctx = std::make_unique<LLVMContext>();
    auto M = std::make_unique<Module>("__standard_lib", *Ctx);
    M->setDataLayout(J.getDataLayout());

    auto *Int64Ty = Type::getInt64Ty(*Ctx);
    auto *DSOHandle = new GlobalVariable(
        *M, Int64Ty, true, GlobalValue::ExternalLinkage,
        ConstantInt::get(Int64Ty, reinterpret_cast<uintptr_t>(&JD)),
        "__dso_handle");
    DSOHandle->setVisibility(GlobalValue::DefaultVisibility);
    DSOHandle->setInitializer(
        ConstantInt::get(Int64Ty, pointerToJITTargetAddress(&JD)));

    auto *GenericIRPlatformSupportTy =
        StructType::create(*Ctx, "lljit.GenericLLJITIRPlatformSupport");

    auto *PlatformInstanceDecl = new GlobalVariable(
        *M, GenericIRPlatformSupportTy, true, GlobalValue::ExternalLinkage,
        nullptr, "__lljit.platform_support_instance");

    auto *VoidTy = Type::getVoidTy(*Ctx);
    addHelperAndWrapper(
        *M, "__lljit_run_atexits", FunctionType::get(VoidTy, {}, false),
        GlobalValue::HiddenVisibility, "__lljit.run_atexits_helper",
        {PlatformInstanceDecl, DSOHandle});

    return J.addIRModule(JD, ThreadSafeModule(std::move(M), std::move(Ctx)));
  }

  Error notifyAdding(ResourceTracker &RT, const MaterializationUnit &MU) {
    auto &JD = RT.getJITDylib();
    if (auto &InitSym = MU.getInitializerSymbol())
      InitSymbols[&JD].add(InitSym, SymbolLookupFlags::WeaklyReferencedSymbol);
    else {
      // If there's no identified init symbol attached, but there is a symbol
      // with the GenericIRPlatform::InitFunctionPrefix, then treat that as
      // an init function. Add the symbol to both the InitSymbols map (which
      // will trigger a lookup to materialize the module) and the InitFunctions
      // map (which holds the names of the symbols to execute).
      for (auto &KV : MU.getSymbols())
        if ((*KV.first).startswith(InitFunctionPrefix)) {
          InitSymbols[&JD].add(KV.first,
                               SymbolLookupFlags::WeaklyReferencedSymbol);
          InitFunctions[&JD].add(KV.first);
        }
    }
    return Error::success();
  }

  Error initialize(JITDylib &JD) override {
    LLVM_DEBUG({
      dbgs() << "GenericLLVMIRPlatformSupport getting initializers to run\n";
    });
    if (auto Initializers = getInitializers(JD)) {
      LLVM_DEBUG(
          { dbgs() << "GenericLLVMIRPlatformSupport running initializers\n"; });
      for (auto InitFnAddr : *Initializers) {
        LLVM_DEBUG({
          dbgs() << "  Running init " << formatv("{0:x16}", InitFnAddr)
                 << "...\n";
        });
        auto *InitFn = jitTargetAddressToFunction<void (*)()>(InitFnAddr);
        InitFn();
      }
    } else
      return Initializers.takeError();
    return Error::success();
  }

  Error deinitialize(JITDylib &JD) override {
    LLVM_DEBUG({
      dbgs() << "GenericLLVMIRPlatformSupport getting deinitializers to run\n";
    });
    if (auto Deinitializers = getDeinitializers(JD)) {
      LLVM_DEBUG({
        dbgs() << "GenericLLVMIRPlatformSupport running deinitializers\n";
      });
      for (auto DeinitFnAddr : *Deinitializers) {
        LLVM_DEBUG({
          dbgs() << "  Running init " << formatv("{0:x16}", DeinitFnAddr)
                 << "...\n";
        });
        auto *DeinitFn = jitTargetAddressToFunction<void (*)()>(DeinitFnAddr);
        DeinitFn();
      }
    } else
      return Deinitializers.takeError();

    return Error::success();
  }

  void registerInitFunc(JITDylib &JD, SymbolStringPtr InitName) {
    getExecutionSession().runSessionLocked([&]() {
        InitFunctions[&JD].add(InitName);
      });
  }

private:

  Expected<std::vector<JITTargetAddress>> getInitializers(JITDylib &JD) {
    if (auto Err = issueInitLookups(JD))
      return std::move(Err);

    DenseMap<JITDylib *, SymbolLookupSet> LookupSymbols;
    std::vector<JITDylibSP> DFSLinkOrder;

    getExecutionSession().runSessionLocked([&]() {
      DFSLinkOrder = JD.getDFSLinkOrder();

      for (auto &NextJD : DFSLinkOrder) {
        auto IFItr = InitFunctions.find(NextJD.get());
        if (IFItr != InitFunctions.end()) {
          LookupSymbols[NextJD.get()] = std::move(IFItr->second);
          InitFunctions.erase(IFItr);
        }
      }
    });

    LLVM_DEBUG({
      dbgs() << "JITDylib init order is [ ";
      for (auto &JD : llvm::reverse(DFSLinkOrder))
        dbgs() << "\"" << JD->getName() << "\" ";
      dbgs() << "]\n";
      dbgs() << "Looking up init functions:\n";
      for (auto &KV : LookupSymbols)
        dbgs() << "  \"" << KV.first->getName() << "\": " << KV.second << "\n";
    });

    auto &ES = getExecutionSession();
    auto LookupResult = Platform::lookupInitSymbols(ES, LookupSymbols);

    if (!LookupResult)
      return LookupResult.takeError();

    std::vector<JITTargetAddress> Initializers;
    while (!DFSLinkOrder.empty()) {
      auto &NextJD = *DFSLinkOrder.back();
      DFSLinkOrder.pop_back();
      auto InitsItr = LookupResult->find(&NextJD);
      if (InitsItr == LookupResult->end())
        continue;
      for (auto &KV : InitsItr->second)
        Initializers.push_back(KV.second.getAddress());
    }

    return Initializers;
  }

  Expected<std::vector<JITTargetAddress>> getDeinitializers(JITDylib &JD) {
    auto &ES = getExecutionSession();

    auto LLJITRunAtExits = J.mangleAndIntern("__lljit_run_atexits");

    DenseMap<JITDylib *, SymbolLookupSet> LookupSymbols;
    std::vector<JITDylibSP> DFSLinkOrder;

    ES.runSessionLocked([&]() {
      DFSLinkOrder = JD.getDFSLinkOrder();

      for (auto &NextJD : DFSLinkOrder) {
        auto &JDLookupSymbols = LookupSymbols[NextJD.get()];
        auto DIFItr = DeInitFunctions.find(NextJD.get());
        if (DIFItr != DeInitFunctions.end()) {
          LookupSymbols[NextJD.get()] = std::move(DIFItr->second);
          DeInitFunctions.erase(DIFItr);
        }
        JDLookupSymbols.add(LLJITRunAtExits,
                            SymbolLookupFlags::WeaklyReferencedSymbol);
      }
    });

    LLVM_DEBUG({
      dbgs() << "JITDylib deinit order is [ ";
      for (auto &JD : DFSLinkOrder)
        dbgs() << "\"" << JD->getName() << "\" ";
      dbgs() << "]\n";
      dbgs() << "Looking up deinit functions:\n";
      for (auto &KV : LookupSymbols)
        dbgs() << "  \"" << KV.first->getName() << "\": " << KV.second << "\n";
    });

    auto LookupResult = Platform::lookupInitSymbols(ES, LookupSymbols);

    if (!LookupResult)
      return LookupResult.takeError();

    std::vector<JITTargetAddress> DeInitializers;
    for (auto &NextJD : DFSLinkOrder) {
      auto DeInitsItr = LookupResult->find(NextJD.get());
      assert(DeInitsItr != LookupResult->end() &&
             "Every JD should have at least __lljit_run_atexits");

      auto RunAtExitsItr = DeInitsItr->second.find(LLJITRunAtExits);
      if (RunAtExitsItr != DeInitsItr->second.end())
        DeInitializers.push_back(RunAtExitsItr->second.getAddress());

      for (auto &KV : DeInitsItr->second)
        if (KV.first != LLJITRunAtExits)
          DeInitializers.push_back(KV.second.getAddress());
    }

    return DeInitializers;
  }

  /// Issue lookups for all init symbols required to initialize JD (and any
  /// JITDylibs that it depends on).
  Error issueInitLookups(JITDylib &JD) {
    DenseMap<JITDylib *, SymbolLookupSet> RequiredInitSymbols;
    std::vector<JITDylibSP> DFSLinkOrder;

    getExecutionSession().runSessionLocked([&]() {
      DFSLinkOrder = JD.getDFSLinkOrder();

      for (auto &NextJD : DFSLinkOrder) {
        auto ISItr = InitSymbols.find(NextJD.get());
        if (ISItr != InitSymbols.end()) {
          RequiredInitSymbols[NextJD.get()] = std::move(ISItr->second);
          InitSymbols.erase(ISItr);
        }
      }
    });

    return Platform::lookupInitSymbols(getExecutionSession(),
                                       RequiredInitSymbols)
        .takeError();
  }

  static void registerAtExitHelper(void *Self, void (*F)(void *), void *Ctx,
                                   void *DSOHandle) {
    LLVM_DEBUG({
      dbgs() << "Registering atexit function " << (void *)F << " for JD "
             << (*static_cast<JITDylib **>(DSOHandle))->getName() << "\n";
    });
    static_cast<GenericLLVMIRPlatformSupport *>(Self)->AtExitMgr.registerAtExit(
        F, Ctx, DSOHandle);
  }

  static void runAtExitsHelper(void *Self, void *DSOHandle) {
    LLVM_DEBUG({
      dbgs() << "Running atexit functions for JD "
             << (*static_cast<JITDylib **>(DSOHandle))->getName() << "\n";
    });
    static_cast<GenericLLVMIRPlatformSupport *>(Self)->AtExitMgr.runAtExits(
        DSOHandle);
  }

  // Constructs an LLVM IR module containing platform runtime globals,
  // functions, and interposes.
  ThreadSafeModule createPlatformRuntimeModule() {
    auto Ctx = std::make_unique<LLVMContext>();
    auto M = std::make_unique<Module>("__standard_lib", *Ctx);
    M->setDataLayout(J.getDataLayout());

    auto *GenericIRPlatformSupportTy =
        StructType::create(*Ctx, "lljit.GenericLLJITIRPlatformSupport");

    auto *PlatformInstanceDecl = new GlobalVariable(
        *M, GenericIRPlatformSupportTy, true, GlobalValue::ExternalLinkage,
        nullptr, "__lljit.platform_support_instance");

    auto *Int8Ty = Type::getInt8Ty(*Ctx);
    auto *IntTy = Type::getIntNTy(*Ctx, sizeof(int) * CHAR_BIT);
    auto *VoidTy = Type::getVoidTy(*Ctx);
    auto *BytePtrTy = PointerType::getUnqual(Int8Ty);
    auto *AtExitCallbackTy = FunctionType::get(VoidTy, {BytePtrTy}, false);
    auto *AtExitCallbackPtrTy = PointerType::getUnqual(AtExitCallbackTy);

    addHelperAndWrapper(
        *M, "__cxa_atexit",
        FunctionType::get(IntTy, {AtExitCallbackPtrTy, BytePtrTy, BytePtrTy},
                          false),
        GlobalValue::DefaultVisibility, "__lljit.cxa_atexit_helper",
        {PlatformInstanceDecl});

    return ThreadSafeModule(std::move(M), std::move(Ctx));
  }

  LLJIT &J;
  std::string InitFunctionPrefix;
  DenseMap<JITDylib *, SymbolLookupSet> InitSymbols;
  DenseMap<JITDylib *, SymbolLookupSet> InitFunctions;
  DenseMap<JITDylib *, SymbolLookupSet> DeInitFunctions;
  ItaniumCXAAtExitSupport AtExitMgr;
};

Error GenericLLVMIRPlatform::setupJITDylib(JITDylib &JD) {
  return S.setupJITDylib(JD);
}

Error GenericLLVMIRPlatform::notifyAdding(ResourceTracker &RT,
                                          const MaterializationUnit &MU) {
  return S.notifyAdding(RT, MU);
}

Expected<ThreadSafeModule>
GlobalCtorDtorScraper::operator()(ThreadSafeModule TSM,
                                  MaterializationResponsibility &R) {
  auto Err = TSM.withModuleDo([&](Module &M) -> Error {
    auto &Ctx = M.getContext();
    auto *GlobalCtors = M.getNamedGlobal("llvm.global_ctors");

    // If there's no llvm.global_ctors or it's just a decl then skip.
    if (!GlobalCtors || GlobalCtors->isDeclaration())
      return Error::success();

    std::string InitFunctionName;
    raw_string_ostream(InitFunctionName)
        << InitFunctionPrefix << M.getModuleIdentifier();

    MangleAndInterner Mangle(PS.getExecutionSession(), M.getDataLayout());
    auto InternedName = Mangle(InitFunctionName);
    if (auto Err =
            R.defineMaterializing({{InternedName, JITSymbolFlags::Callable}}))
      return Err;

    auto *InitFunc =
        Function::Create(FunctionType::get(Type::getVoidTy(Ctx), {}, false),
                         GlobalValue::ExternalLinkage, InitFunctionName, &M);
    InitFunc->setVisibility(GlobalValue::HiddenVisibility);
    std::vector<std::pair<Function *, unsigned>> Inits;
    for (auto E : getConstructors(M))
      Inits.push_back(std::make_pair(E.Func, E.Priority));
    llvm::sort(Inits, [](const std::pair<Function *, unsigned> &LHS,
                         const std::pair<Function *, unsigned> &RHS) {
      return LHS.first < RHS.first;
    });
    auto *EntryBlock = BasicBlock::Create(Ctx, "entry", InitFunc);
    IRBuilder<> IB(EntryBlock);
    for (auto &KV : Inits)
      IB.CreateCall(KV.first);
    IB.CreateRetVoid();

    PS.registerInitFunc(R.getTargetJITDylib(), InternedName);
    GlobalCtors->eraseFromParent();
    return Error::success();
  });

  if (Err)
    return std::move(Err);

  return std::move(TSM);
}

class MachOPlatformSupport : public LLJIT::PlatformSupport {
public:
  using DLOpenType = void *(*)(const char *Name, int Mode);
  using DLCloseType = int (*)(void *Handle);
  using DLSymType = void *(*)(void *Handle, const char *Name);
  using DLErrorType = const char *(*)();

  struct DlFcnValues {
    Optional<void *> RTLDDefault;
    DLOpenType dlopen = nullptr;
    DLCloseType dlclose = nullptr;
    DLSymType dlsym = nullptr;
    DLErrorType dlerror = nullptr;
  };

  static Expected<std::unique_ptr<MachOPlatformSupport>>
  Create(LLJIT &J, JITDylib &PlatformJITDylib) {

    // Make process symbols visible.
    {
      std::string ErrMsg;
      auto Lib = sys::DynamicLibrary::getPermanentLibrary(nullptr, &ErrMsg);
      if (!Lib.isValid())
        return make_error<StringError>(std::move(ErrMsg),
                                       inconvertibleErrorCode());
    }

    DlFcnValues DlFcn;

    // Add support for RTLDDefault on known platforms.
#ifdef __APPLE__
    DlFcn.RTLDDefault = reinterpret_cast<void *>(-2);
#endif // __APPLE__

    if (auto Err = hookUpFunction(DlFcn.dlopen, "dlopen"))
      return std::move(Err);
    if (auto Err = hookUpFunction(DlFcn.dlclose, "dlclose"))
      return std::move(Err);
    if (auto Err = hookUpFunction(DlFcn.dlsym, "dlsym"))
      return std::move(Err);
    if (auto Err = hookUpFunction(DlFcn.dlerror, "dlerror"))
      return std::move(Err);

    std::unique_ptr<MachOPlatformSupport> MP(
        new MachOPlatformSupport(J, PlatformJITDylib, DlFcn));
    return std::move(MP);
  }

  Error initialize(JITDylib &JD) override {
    LLVM_DEBUG({
      dbgs() << "MachOPlatformSupport initializing \"" << JD.getName()
             << "\"\n";
    });

    auto InitSeq = MP.getInitializerSequence(JD);
    if (!InitSeq)
      return InitSeq.takeError();

    // If ObjC is not enabled but there are JIT'd ObjC inits then return
    // an error.
    if (!objCRegistrationEnabled())
      for (auto &KV : *InitSeq) {
        if (!KV.second.getObjCSelRefsSections().empty() ||
            !KV.second.getObjCClassListSections().empty())
          return make_error<StringError>("JITDylib " + KV.first->getName() +
                                             " contains objc metadata but objc"
                                             " is not enabled",
                                         inconvertibleErrorCode());
      }

    // Run the initializers.
    for (auto &KV : *InitSeq) {
      if (objCRegistrationEnabled()) {
        KV.second.registerObjCSelectors();
        if (auto Err = KV.second.registerObjCClasses()) {
          // FIXME: Roll back registrations on error?
          return Err;
        }
      }
      KV.second.runModInits();
    }

    return Error::success();
  }

  Error deinitialize(JITDylib &JD) override {
    auto &ES = J.getExecutionSession();
    if (auto DeinitSeq = MP.getDeinitializerSequence(JD)) {
      for (auto &KV : *DeinitSeq) {
        auto DSOHandleName = ES.intern("___dso_handle");

        // FIXME: Run DeInits here.
        auto Result = ES.lookup(
            {{KV.first, JITDylibLookupFlags::MatchAllSymbols}},
            SymbolLookupSet(DSOHandleName,
                            SymbolLookupFlags::WeaklyReferencedSymbol));
        if (!Result)
          return Result.takeError();
        if (Result->empty())
          continue;
        assert(Result->count(DSOHandleName) &&
               "Result does not contain __dso_handle");
        auto *DSOHandle = jitTargetAddressToPointer<void *>(
            Result->begin()->second.getAddress());
        AtExitMgr.runAtExits(DSOHandle);
      }
    } else
      return DeinitSeq.takeError();
    return Error::success();
  }

private:
  template <typename FunctionPtrTy>
  static Error hookUpFunction(FunctionPtrTy &Fn, const char *Name) {
    if (auto *FnAddr = sys::DynamicLibrary::SearchForAddressOfSymbol(Name)) {
      Fn = reinterpret_cast<FunctionPtrTy>(Fn);
      return Error::success();
    }

    return make_error<StringError>((Twine("Can not enable MachO JIT Platform: "
                                          "missing function: ") +
                                    Name)
                                       .str(),
                                   inconvertibleErrorCode());
  }

  MachOPlatformSupport(LLJIT &J, JITDylib &PlatformJITDylib, DlFcnValues DlFcn)
      : J(J), MP(setupPlatform(J)), DlFcn(std::move(DlFcn)) {

    SymbolMap HelperSymbols;

    // platform and atexit helpers.
    HelperSymbols[J.mangleAndIntern("__lljit.platform_support_instance")] =
        JITEvaluatedSymbol(pointerToJITTargetAddress(this), JITSymbolFlags());
    HelperSymbols[J.mangleAndIntern("__lljit.cxa_atexit_helper")] =
        JITEvaluatedSymbol(pointerToJITTargetAddress(registerAtExitHelper),
                           JITSymbolFlags());
    HelperSymbols[J.mangleAndIntern("__lljit.run_atexits_helper")] =
        JITEvaluatedSymbol(pointerToJITTargetAddress(runAtExitsHelper),
                           JITSymbolFlags());

    // dlfcn helpers.
    HelperSymbols[J.mangleAndIntern("__lljit.dlopen_helper")] =
        JITEvaluatedSymbol(pointerToJITTargetAddress(dlopenHelper),
                           JITSymbolFlags());
    HelperSymbols[J.mangleAndIntern("__lljit.dlclose_helper")] =
        JITEvaluatedSymbol(pointerToJITTargetAddress(dlcloseHelper),
                           JITSymbolFlags());
    HelperSymbols[J.mangleAndIntern("__lljit.dlsym_helper")] =
        JITEvaluatedSymbol(pointerToJITTargetAddress(dlsymHelper),
                           JITSymbolFlags());
    HelperSymbols[J.mangleAndIntern("__lljit.dlerror_helper")] =
        JITEvaluatedSymbol(pointerToJITTargetAddress(dlerrorHelper),
                           JITSymbolFlags());

    cantFail(
        PlatformJITDylib.define(absoluteSymbols(std::move(HelperSymbols))));
    cantFail(MP.setupJITDylib(J.getMainJITDylib()));
    cantFail(J.addIRModule(PlatformJITDylib, createPlatformRuntimeModule()));
  }

  static MachOPlatform &setupPlatform(LLJIT &J) {
    auto Tmp = std::make_unique<MachOPlatform>(
        J.getExecutionSession(),
        static_cast<ObjectLinkingLayer &>(J.getObjLinkingLayer()),
        createStandardSymbolsObject(J));
    auto &MP = *Tmp;
    J.getExecutionSession().setPlatform(std::move(Tmp));
    return MP;
  }

  static std::unique_ptr<MemoryBuffer> createStandardSymbolsObject(LLJIT &J) {
    LLVMContext Ctx;
    Module M("__standard_symbols", Ctx);
    M.setDataLayout(J.getDataLayout());

    auto *Int64Ty = Type::getInt64Ty(Ctx);

    auto *DSOHandle =
        new GlobalVariable(M, Int64Ty, true, GlobalValue::ExternalLinkage,
                           ConstantInt::get(Int64Ty, 0), "__dso_handle");
    DSOHandle->setVisibility(GlobalValue::DefaultVisibility);

    return cantFail(J.getIRCompileLayer().getCompiler()(M));
  }

  ThreadSafeModule createPlatformRuntimeModule() {
    auto Ctx = std::make_unique<LLVMContext>();
    auto M = std::make_unique<Module>("__standard_lib", *Ctx);
    M->setDataLayout(J.getDataLayout());

    auto *MachOPlatformSupportTy =
        StructType::create(*Ctx, "lljit.MachOPlatformSupport");

    auto *PlatformInstanceDecl = new GlobalVariable(
        *M, MachOPlatformSupportTy, true, GlobalValue::ExternalLinkage, nullptr,
        "__lljit.platform_support_instance");

    auto *Int8Ty = Type::getInt8Ty(*Ctx);
    auto *IntTy = Type::getIntNTy(*Ctx, sizeof(int) * CHAR_BIT);
    auto *VoidTy = Type::getVoidTy(*Ctx);
    auto *BytePtrTy = PointerType::getUnqual(Int8Ty);
    auto *AtExitCallbackTy = FunctionType::get(VoidTy, {BytePtrTy}, false);
    auto *AtExitCallbackPtrTy = PointerType::getUnqual(AtExitCallbackTy);

    addHelperAndWrapper(
        *M, "__cxa_atexit",
        FunctionType::get(IntTy, {AtExitCallbackPtrTy, BytePtrTy, BytePtrTy},
                          false),
        GlobalValue::DefaultVisibility, "__lljit.cxa_atexit_helper",
        {PlatformInstanceDecl});

    addHelperAndWrapper(*M, "dlopen",
                        FunctionType::get(BytePtrTy, {BytePtrTy, IntTy}, false),
                        GlobalValue::DefaultVisibility, "__lljit.dlopen_helper",
                        {PlatformInstanceDecl});

    addHelperAndWrapper(*M, "dlclose",
                        FunctionType::get(IntTy, {BytePtrTy}, false),
                        GlobalValue::DefaultVisibility,
                        "__lljit.dlclose_helper", {PlatformInstanceDecl});

    addHelperAndWrapper(
        *M, "dlsym",
        FunctionType::get(BytePtrTy, {BytePtrTy, BytePtrTy}, false),
        GlobalValue::DefaultVisibility, "__lljit.dlsym_helper",
        {PlatformInstanceDecl});

    addHelperAndWrapper(*M, "dlerror", FunctionType::get(BytePtrTy, {}, false),
                        GlobalValue::DefaultVisibility,
                        "__lljit.dlerror_helper", {PlatformInstanceDecl});

    return ThreadSafeModule(std::move(M), std::move(Ctx));
  }

  static void registerAtExitHelper(void *Self, void (*F)(void *), void *Ctx,
                                   void *DSOHandle) {
    static_cast<MachOPlatformSupport *>(Self)->AtExitMgr.registerAtExit(
        F, Ctx, DSOHandle);
  }

  static void runAtExitsHelper(void *Self, void *DSOHandle) {
    static_cast<MachOPlatformSupport *>(Self)->AtExitMgr.runAtExits(DSOHandle);
  }

  void *jit_dlopen(const char *Path, int Mode) {
    JITDylib *JDToOpen = nullptr;
    // FIXME: Do the right thing with Mode flags.
    {
      std::lock_guard<std::mutex> Lock(PlatformSupportMutex);

      // Clear any existing error messages.
      dlErrorMsgs.erase(std::this_thread::get_id());

      if (auto *JD = J.getExecutionSession().getJITDylibByName(Path)) {
        auto I = JDRefCounts.find(JD);
        if (I != JDRefCounts.end()) {
          ++I->second;
          return JD;
        }

        JDRefCounts[JD] = 1;
        JDToOpen = JD;
      }
    }

    if (JDToOpen) {
      if (auto Err = initialize(*JDToOpen)) {
        recordError(std::move(Err));
        return 0;
      }
    }

    // Fall through to dlopen if no JITDylib found for Path.
    return DlFcn.dlopen(Path, Mode);
  }

  static void *dlopenHelper(void *Self, const char *Path, int Mode) {
    return static_cast<MachOPlatformSupport *>(Self)->jit_dlopen(Path, Mode);
  }

  int jit_dlclose(void *Handle) {
    JITDylib *JDToClose = nullptr;

    {
      std::lock_guard<std::mutex> Lock(PlatformSupportMutex);

      // Clear any existing error messages.
      dlErrorMsgs.erase(std::this_thread::get_id());

      auto I = JDRefCounts.find(Handle);
      if (I != JDRefCounts.end()) {
        --I->second;
        if (I->second == 0) {
          JDRefCounts.erase(I);
          JDToClose = static_cast<JITDylib *>(Handle);
        } else
          return 0;
      }
    }

    if (JDToClose) {
      if (auto Err = deinitialize(*JDToClose)) {
        recordError(std::move(Err));
        return -1;
      }
      return 0;
    }

    // Fall through to dlclose if no JITDylib found for Path.
    return DlFcn.dlclose(Handle);
  }

  static int dlcloseHelper(void *Self, void *Handle) {
    return static_cast<MachOPlatformSupport *>(Self)->jit_dlclose(Handle);
  }

  void *jit_dlsym(void *Handle, const char *Name) {
    JITDylibSearchOrder JITSymSearchOrder;

    // FIXME: RTLD_NEXT, RTLD_SELF not supported.
    {
      std::lock_guard<std::mutex> Lock(PlatformSupportMutex);

      // Clear any existing error messages.
      dlErrorMsgs.erase(std::this_thread::get_id());

      if (JDRefCounts.count(Handle)) {
        JITSymSearchOrder.push_back(
            {static_cast<JITDylib *>(Handle),
             JITDylibLookupFlags::MatchExportedSymbolsOnly});
      } else if (Handle == DlFcn.RTLDDefault) {
        for (auto &KV : JDRefCounts)
          JITSymSearchOrder.push_back(
              {static_cast<JITDylib *>(KV.first),
               JITDylibLookupFlags::MatchExportedSymbolsOnly});
      }
    }

    if (!JITSymSearchOrder.empty()) {
      auto MangledName = J.mangleAndIntern(Name);
      SymbolLookupSet Syms(MangledName,
                           SymbolLookupFlags::WeaklyReferencedSymbol);
      if (auto Result = J.getExecutionSession().lookup(JITSymSearchOrder, Syms,
                                                       LookupKind::DLSym)) {
        auto I = Result->find(MangledName);
        if (I != Result->end())
          return jitTargetAddressToPointer<void *>(I->second.getAddress());
      } else {
        recordError(Result.takeError());
        return 0;
      }
    }

    // Fall through to dlsym.
    return DlFcn.dlsym(Handle, Name);
  }

  static void *dlsymHelper(void *Self, void *Handle, const char *Name) {
    return static_cast<MachOPlatformSupport *>(Self)->jit_dlsym(Handle, Name);
  }

  const char *jit_dlerror() {
    {
      std::lock_guard<std::mutex> Lock(PlatformSupportMutex);
      auto I = dlErrorMsgs.find(std::this_thread::get_id());
      if (I != dlErrorMsgs.end())
        return I->second->c_str();
    }
    return DlFcn.dlerror();
  }

  static const char *dlerrorHelper(void *Self) {
    return static_cast<MachOPlatformSupport *>(Self)->jit_dlerror();
  }

  void recordError(Error Err) {
    std::lock_guard<std::mutex> Lock(PlatformSupportMutex);
    dlErrorMsgs[std::this_thread::get_id()] =
        std::make_unique<std::string>(toString(std::move(Err)));
  }

  std::mutex PlatformSupportMutex;
  LLJIT &J;
  MachOPlatform &MP;
  DlFcnValues DlFcn;
  ItaniumCXAAtExitSupport AtExitMgr;
  DenseMap<void *, unsigned> JDRefCounts;
  std::map<std::thread::id, std::unique_ptr<std::string>> dlErrorMsgs;
};

} // end anonymous namespace

namespace llvm {
namespace orc {

void LLJIT::PlatformSupport::setInitTransform(
    LLJIT &J, IRTransformLayer::TransformFunction T) {
  J.InitHelperTransformLayer->setTransform(std::move(T));
}

LLJIT::PlatformSupport::~PlatformSupport() {}

Error LLJITBuilderState::prepareForConstruction() {

  LLVM_DEBUG(dbgs() << "Preparing to create LLJIT instance...\n");

  if (!JTMB) {
    LLVM_DEBUG({
      dbgs() << "  No explicitly set JITTargetMachineBuilder. "
                "Detecting host...\n";
    });
    if (auto JTMBOrErr = JITTargetMachineBuilder::detectHost())
      JTMB = std::move(*JTMBOrErr);
    else
      return JTMBOrErr.takeError();
  }

  LLVM_DEBUG({
    dbgs() << "  JITTargetMachineBuilder is " << JTMB << "\n"
           << "  Pre-constructed ExecutionSession: " << (ES ? "Yes" : "No")
           << "\n"
           << "  DataLayout: ";
    if (DL)
      dbgs() << DL->getStringRepresentation() << "\n";
    else
      dbgs() << "None (will be created by JITTargetMachineBuilder)\n";

    dbgs() << "  Custom object-linking-layer creator: "
           << (CreateObjectLinkingLayer ? "Yes" : "No") << "\n"
           << "  Custom compile-function creator: "
           << (CreateCompileFunction ? "Yes" : "No") << "\n"
           << "  Custom platform-setup function: "
           << (SetUpPlatform ? "Yes" : "No") << "\n"
           << "  Number of compile threads: " << NumCompileThreads;
    if (!NumCompileThreads)
      dbgs() << " (code will be compiled on the execution thread)\n";
    else
      dbgs() << "\n";
  });

  // If the client didn't configure any linker options then auto-configure the
  // JIT linker.
  if (!CreateObjectLinkingLayer) {
    auto &TT = JTMB->getTargetTriple();
    if (TT.isOSBinFormatMachO() &&
        (TT.getArch() == Triple::aarch64 || TT.getArch() == Triple::x86_64)) {

      JTMB->setRelocationModel(Reloc::PIC_);
      JTMB->setCodeModel(CodeModel::Small);
      CreateObjectLinkingLayer =
          [TPC = this->TPC](ExecutionSession &ES,
                            const Triple &) -> std::unique_ptr<ObjectLayer> {
        std::unique_ptr<ObjectLinkingLayer> ObjLinkingLayer;
        if (TPC)
          ObjLinkingLayer =
              std::make_unique<ObjectLinkingLayer>(ES, TPC->getMemMgr());
        else
          ObjLinkingLayer = std::make_unique<ObjectLinkingLayer>(
              ES, std::make_unique<jitlink::InProcessMemoryManager>());
        ObjLinkingLayer->addPlugin(std::make_unique<EHFrameRegistrationPlugin>(
            ES, std::make_unique<jitlink::InProcessEHFrameRegistrar>()));
        return std::move(ObjLinkingLayer);
      };
    }
  }

  return Error::success();
}

LLJIT::~LLJIT() {
  if (CompileThreads)
    CompileThreads->wait();
  if (auto Err = ES->endSession())
    ES->reportError(std::move(Err));
}

Error LLJIT::addIRModule(ResourceTrackerSP RT, ThreadSafeModule TSM) {
  assert(TSM && "Can not add null module");

  if (auto Err =
          TSM.withModuleDo([&](Module &M) { return applyDataLayout(M); }))
    return Err;

  return InitHelperTransformLayer->add(std::move(RT), std::move(TSM));
}

Error LLJIT::addIRModule(JITDylib &JD, ThreadSafeModule TSM) {
  return addIRModule(JD.getDefaultResourceTracker(), std::move(TSM));
}

Error LLJIT::addObjectFile(ResourceTrackerSP RT,
                           std::unique_ptr<MemoryBuffer> Obj) {
  assert(Obj && "Can not add null object");

  return ObjTransformLayer.add(std::move(RT), std::move(Obj));
}

Error LLJIT::addObjectFile(JITDylib &JD, std::unique_ptr<MemoryBuffer> Obj) {
  return addObjectFile(JD.getDefaultResourceTracker(), std::move(Obj));
}

Expected<JITEvaluatedSymbol> LLJIT::lookupLinkerMangled(JITDylib &JD,
                                                        SymbolStringPtr Name) {
  return ES->lookup(
      makeJITDylibSearchOrder(&JD, JITDylibLookupFlags::MatchAllSymbols), Name);
}

std::unique_ptr<ObjectLayer>
LLJIT::createObjectLinkingLayer(LLJITBuilderState &S, ExecutionSession &ES) {

  // If the config state provided an ObjectLinkingLayer factory then use it.
  if (S.CreateObjectLinkingLayer)
    return S.CreateObjectLinkingLayer(ES, S.JTMB->getTargetTriple());

  // Otherwise default to creating an RTDyldObjectLinkingLayer that constructs
  // a new SectionMemoryManager for each object.
  auto GetMemMgr = []() { return std::make_unique<SectionMemoryManager>(); };
  auto ObjLinkingLayer =
      std::make_unique<RTDyldObjectLinkingLayer>(ES, std::move(GetMemMgr));

  if (S.JTMB->getTargetTriple().isOSBinFormatCOFF()) {
    ObjLinkingLayer->setOverrideObjectFlagsWithResponsibilityFlags(true);
    ObjLinkingLayer->setAutoClaimResponsibilityForObjectSymbols(true);
  }

  // FIXME: Explicit conversion to std::unique_ptr<ObjectLayer> added to silence
  //        errors from some GCC / libstdc++ bots. Remove this conversion (i.e.
  //        just return ObjLinkingLayer) once those bots are upgraded.
  return std::unique_ptr<ObjectLayer>(std::move(ObjLinkingLayer));
}

Expected<std::unique_ptr<IRCompileLayer::IRCompiler>>
LLJIT::createCompileFunction(LLJITBuilderState &S,
                             JITTargetMachineBuilder JTMB) {

  /// If there is a custom compile function creator set then use it.
  if (S.CreateCompileFunction)
    return S.CreateCompileFunction(std::move(JTMB));

  // Otherwise default to creating a SimpleCompiler, or ConcurrentIRCompiler,
  // depending on the number of threads requested.
  if (S.NumCompileThreads > 0)
    return std::make_unique<ConcurrentIRCompiler>(std::move(JTMB));

  auto TM = JTMB.createTargetMachine();
  if (!TM)
    return TM.takeError();

  return std::make_unique<TMOwningSimpleCompiler>(std::move(*TM));
}

LLJIT::LLJIT(LLJITBuilderState &S, Error &Err)
    : ES(S.ES ? std::move(S.ES) : std::make_unique<ExecutionSession>()), Main(),
      DL(""), TT(S.JTMB->getTargetTriple()),
      ObjLinkingLayer(createObjectLinkingLayer(S, *ES)),
      ObjTransformLayer(*this->ES, *ObjLinkingLayer) {

  ErrorAsOutParameter _(&Err);

  if (auto MainOrErr = this->ES->createJITDylib("main"))
    Main = &*MainOrErr;
  else {
    Err = MainOrErr.takeError();
    return;
  }

  if (S.DL)
    DL = std::move(*S.DL);
  else if (auto DLOrErr = S.JTMB->getDefaultDataLayoutForTarget())
    DL = std::move(*DLOrErr);
  else {
    Err = DLOrErr.takeError();
    return;
  }

  {
    auto CompileFunction = createCompileFunction(S, std::move(*S.JTMB));
    if (!CompileFunction) {
      Err = CompileFunction.takeError();
      return;
    }
    CompileLayer = std::make_unique<IRCompileLayer>(
        *ES, ObjTransformLayer, std::move(*CompileFunction));
    TransformLayer = std::make_unique<IRTransformLayer>(*ES, *CompileLayer);
    InitHelperTransformLayer =
        std::make_unique<IRTransformLayer>(*ES, *TransformLayer);
  }

  if (S.NumCompileThreads > 0) {
    InitHelperTransformLayer->setCloneToNewContextOnEmit(true);
    CompileThreads =
        std::make_unique<ThreadPool>(hardware_concurrency(S.NumCompileThreads));
    ES->setDispatchMaterialization(
        [this](std::unique_ptr<MaterializationUnit> MU,
               std::unique_ptr<MaterializationResponsibility> MR) {
          // FIXME: We should be able to use move-capture here, but ThreadPool's
          // AsyncTaskTys are std::functions rather than unique_functions
          // (because MSVC's std::packaged_tasks don't support move-only types).
          // Fix this when all the above gets sorted out.
          CompileThreads->async(
              [UnownedMU = MU.release(), UnownedMR = MR.release()]() mutable {
                std::unique_ptr<MaterializationUnit> MU(UnownedMU);
                std::unique_ptr<MaterializationResponsibility> MR(UnownedMR);
                MU->materialize(std::move(MR));
              });
        });
  }

  if (S.SetUpPlatform)
    Err = S.SetUpPlatform(*this);
  else
    setUpGenericLLVMIRPlatform(*this);
}

std::string LLJIT::mangle(StringRef UnmangledName) const {
  std::string MangledName;
  {
    raw_string_ostream MangledNameStream(MangledName);
    Mangler::getNameWithPrefix(MangledNameStream, UnmangledName, DL);
  }
  return MangledName;
}

Error LLJIT::applyDataLayout(Module &M) {
  if (M.getDataLayout().isDefault())
    M.setDataLayout(DL);

  if (M.getDataLayout() != DL)
    return make_error<StringError>(
        "Added modules have incompatible data layouts: " +
            M.getDataLayout().getStringRepresentation() + " (module) vs " +
            DL.getStringRepresentation() + " (jit)",
        inconvertibleErrorCode());

  return Error::success();
}

void setUpGenericLLVMIRPlatform(LLJIT &J) {
  LLVM_DEBUG(
      { dbgs() << "Setting up GenericLLVMIRPlatform support for LLJIT\n"; });
  J.setPlatformSupport(std::make_unique<GenericLLVMIRPlatformSupport>(J));
}

Error setUpMachOPlatform(LLJIT &J) {
  LLVM_DEBUG({ dbgs() << "Setting up MachOPlatform support for LLJIT\n"; });
  auto MP = MachOPlatformSupport::Create(J, J.getMainJITDylib());
  if (!MP)
    return MP.takeError();
  J.setPlatformSupport(std::move(*MP));
  return Error::success();
}

Error LLLazyJITBuilderState::prepareForConstruction() {
  if (auto Err = LLJITBuilderState::prepareForConstruction())
    return Err;
  TT = JTMB->getTargetTriple();
  return Error::success();
}

Error LLLazyJIT::addLazyIRModule(JITDylib &JD, ThreadSafeModule TSM) {
  assert(TSM && "Can not add null module");

  if (auto Err = TSM.withModuleDo(
          [&](Module &M) -> Error { return applyDataLayout(M); }))
    return Err;

  return CODLayer->add(JD, std::move(TSM));
}

LLLazyJIT::LLLazyJIT(LLLazyJITBuilderState &S, Error &Err) : LLJIT(S, Err) {

  // If LLJIT construction failed then bail out.
  if (Err)
    return;

  ErrorAsOutParameter _(&Err);

  /// Take/Create the lazy-compile callthrough manager.
  if (S.LCTMgr)
    LCTMgr = std::move(S.LCTMgr);
  else {
    if (auto LCTMgrOrErr = createLocalLazyCallThroughManager(
            S.TT, *ES, S.LazyCompileFailureAddr))
      LCTMgr = std::move(*LCTMgrOrErr);
    else {
      Err = LCTMgrOrErr.takeError();
      return;
    }
  }

  // Take/Create the indirect stubs manager builder.
  auto ISMBuilder = std::move(S.ISMBuilder);

  // If none was provided, try to build one.
  if (!ISMBuilder)
    ISMBuilder = createLocalIndirectStubsManagerBuilder(S.TT);

  // No luck. Bail out.
  if (!ISMBuilder) {
    Err = make_error<StringError>("Could not construct "
                                  "IndirectStubsManagerBuilder for target " +
                                      S.TT.str(),
                                  inconvertibleErrorCode());
    return;
  }

  // Create the COD layer.
  CODLayer = std::make_unique<CompileOnDemandLayer>(
      *ES, *InitHelperTransformLayer, *LCTMgr, std::move(ISMBuilder));

  if (S.NumCompileThreads > 0)
    CODLayer->setCloneToNewContextOnEmit(true);
}

} // End namespace orc.
} // End namespace llvm.
