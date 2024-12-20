//===------ ELFNixPlatform.cpp - Utilities for executing ELFNix in Orc
//-----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/ELFNixPlatform.h"

#include "llvm/ExecutionEngine/JITLink/aarch64.h"
#include "llvm/ExecutionEngine/JITLink/ppc64.h"
#include "llvm/ExecutionEngine/JITLink/x86_64.h"
#include "llvm/ExecutionEngine/Orc/AbsoluteSymbols.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/Shared/ObjectFormats.h"
#include "llvm/Support/Debug.h"
#include <optional>

#define DEBUG_TYPE "orc"

using namespace llvm;
using namespace llvm::orc;
using namespace llvm::orc::shared;

namespace {

template <typename SPSSerializer, typename... ArgTs>
shared::WrapperFunctionCall::ArgDataBufferType
getArgDataBufferType(const ArgTs &...Args) {
  shared::WrapperFunctionCall::ArgDataBufferType ArgData;
  ArgData.resize(SPSSerializer::size(Args...));
  SPSOutputBuffer OB(ArgData.empty() ? nullptr : ArgData.data(),
                     ArgData.size());
  if (SPSSerializer::serialize(OB, Args...))
    return ArgData;
  return {};
}

std::unique_ptr<jitlink::LinkGraph> createPlatformGraph(ELFNixPlatform &MOP,
                                                        std::string Name) {
  unsigned PointerSize;
  llvm::endianness Endianness;
  const auto &TT = MOP.getExecutionSession().getTargetTriple();

  switch (TT.getArch()) {
  case Triple::x86_64:
    PointerSize = 8;
    Endianness = llvm::endianness::little;
    break;
  case Triple::aarch64:
    PointerSize = 8;
    Endianness = llvm::endianness::little;
    break;
  case Triple::ppc64:
    PointerSize = 8;
    Endianness = llvm::endianness::big;
    break;
  case Triple::ppc64le:
    PointerSize = 8;
    Endianness = llvm::endianness::little;
    break;
  default:
    llvm_unreachable("Unrecognized architecture");
  }

  return std::make_unique<jitlink::LinkGraph>(
      std::move(Name), MOP.getExecutionSession().getSymbolStringPool(), TT,
      PointerSize, Endianness, jitlink::getGenericEdgeKindName);
}

// Creates a Bootstrap-Complete LinkGraph to run deferred actions.
class ELFNixPlatformCompleteBootstrapMaterializationUnit
    : public MaterializationUnit {
public:
  ELFNixPlatformCompleteBootstrapMaterializationUnit(
      ELFNixPlatform &MOP, StringRef PlatformJDName,
      SymbolStringPtr CompleteBootstrapSymbol, DeferredRuntimeFnMap DeferredAAs,
      ExecutorAddr ELFNixHeaderAddr, ExecutorAddr PlatformBootstrap,
      ExecutorAddr PlatformShutdown, ExecutorAddr RegisterJITDylib,
      ExecutorAddr DeregisterJITDylib)
      : MaterializationUnit(
            {{{CompleteBootstrapSymbol, JITSymbolFlags::None}}, nullptr}),
        MOP(MOP), PlatformJDName(PlatformJDName),
        CompleteBootstrapSymbol(std::move(CompleteBootstrapSymbol)),
        DeferredAAsMap(std::move(DeferredAAs)),
        ELFNixHeaderAddr(ELFNixHeaderAddr),
        PlatformBootstrap(PlatformBootstrap),
        PlatformShutdown(PlatformShutdown), RegisterJITDylib(RegisterJITDylib),
        DeregisterJITDylib(DeregisterJITDylib) {}

  StringRef getName() const override {
    return "ELFNixPlatformCompleteBootstrap";
  }

  void materialize(std::unique_ptr<MaterializationResponsibility> R) override {
    using namespace jitlink;
    auto G = createPlatformGraph(MOP, "<OrcRTCompleteBootstrap>");
    auto &PlaceholderSection =
        G->createSection("__orc_rt_cplt_bs", MemProt::Read);
    auto &PlaceholderBlock =
        G->createZeroFillBlock(PlaceholderSection, 1, ExecutorAddr(), 1, 0);
    G->addDefinedSymbol(PlaceholderBlock, 0, *CompleteBootstrapSymbol, 1,
                        Linkage::Strong, Scope::Hidden, false, true);

    // 1. Bootstrap the platform support code.
    G->allocActions().push_back(
        {cantFail(WrapperFunctionCall::Create<SPSArgList<SPSExecutorAddr>>(
             PlatformBootstrap, ELFNixHeaderAddr)),
         cantFail(
             WrapperFunctionCall::Create<SPSArgList<>>(PlatformShutdown))});

    // 2. Register the platform JITDylib.
    G->allocActions().push_back(
        {cantFail(WrapperFunctionCall::Create<
                  SPSArgList<SPSString, SPSExecutorAddr>>(
             RegisterJITDylib, PlatformJDName, ELFNixHeaderAddr)),
         cantFail(WrapperFunctionCall::Create<SPSArgList<SPSExecutorAddr>>(
             DeregisterJITDylib, ELFNixHeaderAddr))});

    // 4. Add the deferred actions to the graph.
    for (auto &[Fn, CallDatas] : DeferredAAsMap) {
      for (auto &CallData : CallDatas) {
        G->allocActions().push_back(
            {WrapperFunctionCall(Fn.first->Addr, std::move(CallData.first)),
             WrapperFunctionCall(Fn.second->Addr, std::move(CallData.second))});
      }
    }

    MOP.getObjectLinkingLayer().emit(std::move(R), std::move(G));
  }

  void discard(const JITDylib &JD, const SymbolStringPtr &Sym) override {}

private:
  ELFNixPlatform &MOP;
  StringRef PlatformJDName;
  SymbolStringPtr CompleteBootstrapSymbol;
  DeferredRuntimeFnMap DeferredAAsMap;
  ExecutorAddr ELFNixHeaderAddr;
  ExecutorAddr PlatformBootstrap;
  ExecutorAddr PlatformShutdown;
  ExecutorAddr RegisterJITDylib;
  ExecutorAddr DeregisterJITDylib;
};

class DSOHandleMaterializationUnit : public MaterializationUnit {
public:
  DSOHandleMaterializationUnit(ELFNixPlatform &ENP,
                               const SymbolStringPtr &DSOHandleSymbol)
      : MaterializationUnit(
            createDSOHandleSectionInterface(ENP, DSOHandleSymbol)),
        ENP(ENP) {}

  StringRef getName() const override { return "DSOHandleMU"; }

  void materialize(std::unique_ptr<MaterializationResponsibility> R) override {
    unsigned PointerSize;
    llvm::endianness Endianness;
    jitlink::Edge::Kind EdgeKind;
    const auto &TT = ENP.getExecutionSession().getTargetTriple();

    switch (TT.getArch()) {
    case Triple::x86_64:
      PointerSize = 8;
      Endianness = llvm::endianness::little;
      EdgeKind = jitlink::x86_64::Pointer64;
      break;
    case Triple::aarch64:
      PointerSize = 8;
      Endianness = llvm::endianness::little;
      EdgeKind = jitlink::aarch64::Pointer64;
      break;
    case Triple::ppc64:
      PointerSize = 8;
      Endianness = llvm::endianness::big;
      EdgeKind = jitlink::ppc64::Pointer64;
      break;
    case Triple::ppc64le:
      PointerSize = 8;
      Endianness = llvm::endianness::little;
      EdgeKind = jitlink::ppc64::Pointer64;
      break;
    default:
      llvm_unreachable("Unrecognized architecture");
    }

    // void *__dso_handle = &__dso_handle;
    auto G = std::make_unique<jitlink::LinkGraph>(
        "<DSOHandleMU>", ENP.getExecutionSession().getSymbolStringPool(), TT,
        PointerSize, Endianness, jitlink::getGenericEdgeKindName);
    auto &DSOHandleSection =
        G->createSection(".data.__dso_handle", MemProt::Read);
    auto &DSOHandleBlock = G->createContentBlock(
        DSOHandleSection, getDSOHandleContent(PointerSize), orc::ExecutorAddr(),
        8, 0);
    auto &DSOHandleSymbol = G->addDefinedSymbol(
        DSOHandleBlock, 0, *R->getInitializerSymbol(), DSOHandleBlock.getSize(),
        jitlink::Linkage::Strong, jitlink::Scope::Default, false, true);
    DSOHandleBlock.addEdge(EdgeKind, 0, DSOHandleSymbol, 0);

    ENP.getObjectLinkingLayer().emit(std::move(R), std::move(G));
  }

  void discard(const JITDylib &JD, const SymbolStringPtr &Sym) override {}

private:
  static MaterializationUnit::Interface
  createDSOHandleSectionInterface(ELFNixPlatform &ENP,
                                  const SymbolStringPtr &DSOHandleSymbol) {
    SymbolFlagsMap SymbolFlags;
    SymbolFlags[DSOHandleSymbol] = JITSymbolFlags::Exported;
    return MaterializationUnit::Interface(std::move(SymbolFlags),
                                          DSOHandleSymbol);
  }

  ArrayRef<char> getDSOHandleContent(size_t PointerSize) {
    static const char Content[8] = {0};
    assert(PointerSize <= sizeof Content);
    return {Content, PointerSize};
  }

  ELFNixPlatform &ENP;
};

} // end anonymous namespace

namespace llvm {
namespace orc {

Expected<std::unique_ptr<ELFNixPlatform>>
ELFNixPlatform::Create(ObjectLinkingLayer &ObjLinkingLayer,
                       JITDylib &PlatformJD,
                       std::unique_ptr<DefinitionGenerator> OrcRuntime,
                       std::optional<SymbolAliasMap> RuntimeAliases) {

  auto &ES = ObjLinkingLayer.getExecutionSession();

  // If the target is not supported then bail out immediately.
  if (!supportedTarget(ES.getTargetTriple()))
    return make_error<StringError>("Unsupported ELFNixPlatform triple: " +
                                       ES.getTargetTriple().str(),
                                   inconvertibleErrorCode());

  auto &EPC = ES.getExecutorProcessControl();

  // Create default aliases if the caller didn't supply any.
  if (!RuntimeAliases) {
    auto StandardRuntimeAliases = standardPlatformAliases(ES, PlatformJD);
    if (!StandardRuntimeAliases)
      return StandardRuntimeAliases.takeError();
    RuntimeAliases = std::move(*StandardRuntimeAliases);
  }

  // Define the aliases.
  if (auto Err = PlatformJD.define(symbolAliases(std::move(*RuntimeAliases))))
    return std::move(Err);

  // Add JIT-dispatch function support symbols.
  if (auto Err = PlatformJD.define(
          absoluteSymbols({{ES.intern("__orc_rt_jit_dispatch"),
                            {EPC.getJITDispatchInfo().JITDispatchFunction,
                             JITSymbolFlags::Exported}},
                           {ES.intern("__orc_rt_jit_dispatch_ctx"),
                            {EPC.getJITDispatchInfo().JITDispatchContext,
                             JITSymbolFlags::Exported}}})))
    return std::move(Err);

  // Create the instance.
  Error Err = Error::success();
  auto P = std::unique_ptr<ELFNixPlatform>(new ELFNixPlatform(
      ObjLinkingLayer, PlatformJD, std::move(OrcRuntime), Err));
  if (Err)
    return std::move(Err);
  return std::move(P);
}

Expected<std::unique_ptr<ELFNixPlatform>>
ELFNixPlatform::Create(ObjectLinkingLayer &ObjLinkingLayer,
                       JITDylib &PlatformJD, const char *OrcRuntimePath,
                       std::optional<SymbolAliasMap> RuntimeAliases) {

  // Create a generator for the ORC runtime archive.
  auto OrcRuntimeArchiveGenerator =
      StaticLibraryDefinitionGenerator::Load(ObjLinkingLayer, OrcRuntimePath);
  if (!OrcRuntimeArchiveGenerator)
    return OrcRuntimeArchiveGenerator.takeError();

  return Create(ObjLinkingLayer, PlatformJD,
                std::move(*OrcRuntimeArchiveGenerator),
                std::move(RuntimeAliases));
}

Error ELFNixPlatform::setupJITDylib(JITDylib &JD) {
  if (auto Err = JD.define(std::make_unique<DSOHandleMaterializationUnit>(
          *this, DSOHandleSymbol)))
    return Err;

  return ES.lookup({&JD}, DSOHandleSymbol).takeError();
}

Error ELFNixPlatform::teardownJITDylib(JITDylib &JD) {
  std::lock_guard<std::mutex> Lock(PlatformMutex);
  auto I = JITDylibToHandleAddr.find(&JD);
  if (I != JITDylibToHandleAddr.end()) {
    assert(HandleAddrToJITDylib.count(I->second) &&
           "HandleAddrToJITDylib missing entry");
    HandleAddrToJITDylib.erase(I->second);
    JITDylibToHandleAddr.erase(I);
  }
  return Error::success();
}

Error ELFNixPlatform::notifyAdding(ResourceTracker &RT,
                                   const MaterializationUnit &MU) {

  auto &JD = RT.getJITDylib();
  const auto &InitSym = MU.getInitializerSymbol();
  if (!InitSym)
    return Error::success();

  RegisteredInitSymbols[&JD].add(InitSym,
                                 SymbolLookupFlags::WeaklyReferencedSymbol);
  LLVM_DEBUG({
    dbgs() << "ELFNixPlatform: Registered init symbol " << *InitSym
           << " for MU " << MU.getName() << "\n";
  });
  return Error::success();
}

Error ELFNixPlatform::notifyRemoving(ResourceTracker &RT) {
  llvm_unreachable("Not supported yet");
}

static void addAliases(ExecutionSession &ES, SymbolAliasMap &Aliases,
                       ArrayRef<std::pair<const char *, const char *>> AL) {
  for (auto &KV : AL) {
    auto AliasName = ES.intern(KV.first);
    assert(!Aliases.count(AliasName) && "Duplicate symbol name in alias map");
    Aliases[std::move(AliasName)] = {ES.intern(KV.second),
                                     JITSymbolFlags::Exported};
  }
}

Expected<SymbolAliasMap>
ELFNixPlatform::standardPlatformAliases(ExecutionSession &ES,
                                        JITDylib &PlatformJD) {
  SymbolAliasMap Aliases;
  addAliases(ES, Aliases, requiredCXXAliases());
  addAliases(ES, Aliases, standardRuntimeUtilityAliases());
  addAliases(ES, Aliases, standardLazyCompilationAliases());
  return Aliases;
}

ArrayRef<std::pair<const char *, const char *>>
ELFNixPlatform::requiredCXXAliases() {
  static const std::pair<const char *, const char *> RequiredCXXAliases[] = {
      {"__cxa_atexit", "__orc_rt_elfnix_cxa_atexit"},
      {"atexit", "__orc_rt_elfnix_atexit"}};

  return ArrayRef<std::pair<const char *, const char *>>(RequiredCXXAliases);
}

ArrayRef<std::pair<const char *, const char *>>
ELFNixPlatform::standardRuntimeUtilityAliases() {
  static const std::pair<const char *, const char *>
      StandardRuntimeUtilityAliases[] = {
          {"__orc_rt_run_program", "__orc_rt_elfnix_run_program"},
          {"__orc_rt_jit_dlerror", "__orc_rt_elfnix_jit_dlerror"},
          {"__orc_rt_jit_dlopen", "__orc_rt_elfnix_jit_dlopen"},
          {"__orc_rt_jit_dlupdate", "__orc_rt_elfnix_jit_dlupdate"},
          {"__orc_rt_jit_dlclose", "__orc_rt_elfnix_jit_dlclose"},
          {"__orc_rt_jit_dlsym", "__orc_rt_elfnix_jit_dlsym"},
          {"__orc_rt_log_error", "__orc_rt_log_error_to_stderr"}};

  return ArrayRef<std::pair<const char *, const char *>>(
      StandardRuntimeUtilityAliases);
}

ArrayRef<std::pair<const char *, const char *>>
ELFNixPlatform::standardLazyCompilationAliases() {
  static const std::pair<const char *, const char *>
      StandardLazyCompilationAliases[] = {
          {"__orc_rt_reenter", "__orc_rt_sysv_reenter"}};

  return ArrayRef<std::pair<const char *, const char *>>(
      StandardLazyCompilationAliases);
}

bool ELFNixPlatform::supportedTarget(const Triple &TT) {
  switch (TT.getArch()) {
  case Triple::x86_64:
  case Triple::aarch64:
  // FIXME: jitlink for ppc64 hasn't been well tested, leave it unsupported
  // right now.
  case Triple::ppc64le:
    return true;
  default:
    return false;
  }
}

ELFNixPlatform::ELFNixPlatform(
    ObjectLinkingLayer &ObjLinkingLayer, JITDylib &PlatformJD,
    std::unique_ptr<DefinitionGenerator> OrcRuntimeGenerator, Error &Err)
    : ES(ObjLinkingLayer.getExecutionSession()), PlatformJD(PlatformJD),
      ObjLinkingLayer(ObjLinkingLayer),
      DSOHandleSymbol(ES.intern("__dso_handle")) {
  ErrorAsOutParameter _(Err);
  ObjLinkingLayer.addPlugin(std::make_unique<ELFNixPlatformPlugin>(*this));

  PlatformJD.addGenerator(std::move(OrcRuntimeGenerator));

  BootstrapInfo BI;
  Bootstrap = &BI;

  // PlatformJD hasn't been 'set-up' by the platform yet (since we're creating
  // the platform now), so set it up.
  if (auto E2 = setupJITDylib(PlatformJD)) {
    Err = std::move(E2);
    return;
  }

  // Step (2) Request runtime registration functions to trigger
  // materialization..
  if ((Err = ES.lookup(
                   makeJITDylibSearchOrder(&PlatformJD),
                   SymbolLookupSet(
                       {PlatformBootstrap.Name, PlatformShutdown.Name,
                        RegisterJITDylib.Name, DeregisterJITDylib.Name,
                        RegisterInitSections.Name, DeregisterInitSections.Name,
                        RegisterObjectSections.Name,
                        DeregisterObjectSections.Name, CreatePThreadKey.Name}))
                 .takeError()))
    return;

  // Step (3) Wait for any incidental linker work to complete.
  {
    std::unique_lock<std::mutex> Lock(BI.Mutex);
    BI.CV.wait(Lock, [&]() { return BI.ActiveGraphs == 0; });
    Bootstrap = nullptr;
  }

  // Step (4) Add complete-bootstrap materialization unit and request.
  auto BootstrapCompleteSymbol =
      ES.intern("__orc_rt_elfnix_complete_bootstrap");
  if ((Err = PlatformJD.define(
           std::make_unique<ELFNixPlatformCompleteBootstrapMaterializationUnit>(
               *this, PlatformJD.getName(), BootstrapCompleteSymbol,
               std::move(BI.DeferredRTFnMap), BI.ELFNixHeaderAddr,
               PlatformBootstrap.Addr, PlatformShutdown.Addr,
               RegisterJITDylib.Addr, DeregisterJITDylib.Addr))))
    return;
  if ((Err = ES.lookup(makeJITDylibSearchOrder(
                           &PlatformJD, JITDylibLookupFlags::MatchAllSymbols),
                       std::move(BootstrapCompleteSymbol))
                 .takeError()))
    return;

  // Associate wrapper function tags with JIT-side function implementations.
  if (auto E2 = associateRuntimeSupportFunctions(PlatformJD)) {
    Err = std::move(E2);
    return;
  }
}

Error ELFNixPlatform::associateRuntimeSupportFunctions(JITDylib &PlatformJD) {
  ExecutionSession::JITDispatchHandlerAssociationMap WFs;

  using RecordInitializersSPSSig =
      SPSExpected<SPSELFNixJITDylibDepInfoMap>(SPSExecutorAddr);
  WFs[ES.intern("__orc_rt_elfnix_push_initializers_tag")] =
      ES.wrapAsyncWithSPS<RecordInitializersSPSSig>(
          this, &ELFNixPlatform::rt_recordInitializers);

  using LookupSymbolSPSSig =
      SPSExpected<SPSExecutorAddr>(SPSExecutorAddr, SPSString);
  WFs[ES.intern("__orc_rt_elfnix_symbol_lookup_tag")] =
      ES.wrapAsyncWithSPS<LookupSymbolSPSSig>(this,
                                              &ELFNixPlatform::rt_lookupSymbol);

  return ES.registerJITDispatchHandlers(PlatformJD, std::move(WFs));
}

void ELFNixPlatform::pushInitializersLoop(
    PushInitializersSendResultFn SendResult, JITDylibSP JD) {
  DenseMap<JITDylib *, SymbolLookupSet> NewInitSymbols;
  DenseMap<JITDylib *, SmallVector<JITDylib *>> JDDepMap;
  SmallVector<JITDylib *, 16> Worklist({JD.get()});

  ES.runSessionLocked([&]() {
    while (!Worklist.empty()) {
      // FIXME: Check for defunct dylibs.

      auto DepJD = Worklist.back();
      Worklist.pop_back();

      // If we've already visited this JITDylib on this iteration then continue.
      if (JDDepMap.count(DepJD))
        continue;

      // Add dep info.
      auto &DM = JDDepMap[DepJD];
      DepJD->withLinkOrderDo([&](const JITDylibSearchOrder &O) {
        for (auto &KV : O) {
          if (KV.first == DepJD)
            continue;
          DM.push_back(KV.first);
          Worklist.push_back(KV.first);
        }
      });

      // Add any registered init symbols.
      auto RISItr = RegisteredInitSymbols.find(DepJD);
      if (RISItr != RegisteredInitSymbols.end()) {
        NewInitSymbols[DepJD] = std::move(RISItr->second);
        RegisteredInitSymbols.erase(RISItr);
      }
    }
  });

  // If there are no further init symbols to look up then send the link order
  // (as a list of header addresses) to the caller.
  if (NewInitSymbols.empty()) {

    // To make the list intelligible to the runtime we need to convert all
    // JITDylib pointers to their header addresses. Only include JITDylibs
    // that appear in the JITDylibToHandleAddr map (i.e. those that have been
    // through setupJITDylib) -- bare JITDylibs aren't managed by the platform.
    DenseMap<JITDylib *, ExecutorAddr> HeaderAddrs;
    HeaderAddrs.reserve(JDDepMap.size());
    {
      std::lock_guard<std::mutex> Lock(PlatformMutex);
      for (auto &KV : JDDepMap) {
        auto I = JITDylibToHandleAddr.find(KV.first);
        if (I != JITDylibToHandleAddr.end())
          HeaderAddrs[KV.first] = I->second;
      }
    }

    // Build the dep info map to return.
    ELFNixJITDylibDepInfoMap DIM;
    DIM.reserve(JDDepMap.size());
    for (auto &KV : JDDepMap) {
      auto HI = HeaderAddrs.find(KV.first);
      // Skip unmanaged JITDylibs.
      if (HI == HeaderAddrs.end())
        continue;
      auto H = HI->second;
      ELFNixJITDylibDepInfo DepInfo;
      for (auto &Dep : KV.second) {
        auto HJ = HeaderAddrs.find(Dep);
        if (HJ != HeaderAddrs.end())
          DepInfo.push_back(HJ->second);
      }
      DIM.push_back(std::make_pair(H, std::move(DepInfo)));
    }
    SendResult(DIM);
    return;
  }

  // Otherwise issue a lookup and re-run this phase when it completes.
  lookupInitSymbolsAsync(
      [this, SendResult = std::move(SendResult), JD](Error Err) mutable {
        if (Err)
          SendResult(std::move(Err));
        else
          pushInitializersLoop(std::move(SendResult), JD);
      },
      ES, std::move(NewInitSymbols));
}

void ELFNixPlatform::rt_recordInitializers(
    PushInitializersSendResultFn SendResult, ExecutorAddr JDHeaderAddr) {
  JITDylibSP JD;
  {
    std::lock_guard<std::mutex> Lock(PlatformMutex);
    auto I = HandleAddrToJITDylib.find(JDHeaderAddr);
    if (I != HandleAddrToJITDylib.end())
      JD = I->second;
  }

  LLVM_DEBUG({
    dbgs() << "ELFNixPlatform::rt_recordInitializers(" << JDHeaderAddr << ") ";
    if (JD)
      dbgs() << "pushing initializers for " << JD->getName() << "\n";
    else
      dbgs() << "No JITDylib for header address.\n";
  });

  if (!JD) {
    SendResult(make_error<StringError>("No JITDylib with header addr " +
                                           formatv("{0:x}", JDHeaderAddr),
                                       inconvertibleErrorCode()));
    return;
  }

  pushInitializersLoop(std::move(SendResult), JD);
}

void ELFNixPlatform::rt_lookupSymbol(SendSymbolAddressFn SendResult,
                                     ExecutorAddr Handle,
                                     StringRef SymbolName) {
  LLVM_DEBUG({
    dbgs() << "ELFNixPlatform::rt_lookupSymbol(\"" << Handle << "\")\n";
  });

  JITDylib *JD = nullptr;

  {
    std::lock_guard<std::mutex> Lock(PlatformMutex);
    auto I = HandleAddrToJITDylib.find(Handle);
    if (I != HandleAddrToJITDylib.end())
      JD = I->second;
  }

  if (!JD) {
    LLVM_DEBUG(dbgs() << "  No JITDylib for handle " << Handle << "\n");
    SendResult(make_error<StringError>("No JITDylib associated with handle " +
                                           formatv("{0:x}", Handle),
                                       inconvertibleErrorCode()));
    return;
  }

  // Use functor class to work around XL build compiler issue on AIX.
  class RtLookupNotifyComplete {
  public:
    RtLookupNotifyComplete(SendSymbolAddressFn &&SendResult)
        : SendResult(std::move(SendResult)) {}
    void operator()(Expected<SymbolMap> Result) {
      if (Result) {
        assert(Result->size() == 1 && "Unexpected result map count");
        SendResult(Result->begin()->second.getAddress());
      } else {
        SendResult(Result.takeError());
      }
    }

  private:
    SendSymbolAddressFn SendResult;
  };

  ES.lookup(
      LookupKind::DLSym, {{JD, JITDylibLookupFlags::MatchExportedSymbolsOnly}},
      SymbolLookupSet(ES.intern(SymbolName)), SymbolState::Ready,
      RtLookupNotifyComplete(std::move(SendResult)), NoDependenciesToRegister);
}

Error ELFNixPlatform::ELFNixPlatformPlugin::bootstrapPipelineStart(
    jitlink::LinkGraph &G) {
  // Increment the active graphs count in BootstrapInfo.
  std::lock_guard<std::mutex> Lock(MP.Bootstrap.load()->Mutex);
  ++MP.Bootstrap.load()->ActiveGraphs;
  return Error::success();
}

Error ELFNixPlatform::ELFNixPlatformPlugin::
    bootstrapPipelineRecordRuntimeFunctions(jitlink::LinkGraph &G) {
  // Record bootstrap function names.
  std::pair<StringRef, ExecutorAddr *> RuntimeSymbols[] = {
      {*MP.DSOHandleSymbol, &MP.Bootstrap.load()->ELFNixHeaderAddr},
      {*MP.PlatformBootstrap.Name, &MP.PlatformBootstrap.Addr},
      {*MP.PlatformShutdown.Name, &MP.PlatformShutdown.Addr},
      {*MP.RegisterJITDylib.Name, &MP.RegisterJITDylib.Addr},
      {*MP.DeregisterJITDylib.Name, &MP.DeregisterJITDylib.Addr},
      {*MP.RegisterObjectSections.Name, &MP.RegisterObjectSections.Addr},
      {*MP.DeregisterObjectSections.Name, &MP.DeregisterObjectSections.Addr},
      {*MP.RegisterInitSections.Name, &MP.RegisterInitSections.Addr},
      {*MP.DeregisterInitSections.Name, &MP.DeregisterInitSections.Addr},
      {*MP.CreatePThreadKey.Name, &MP.CreatePThreadKey.Addr}};

  bool RegisterELFNixHeader = false;

  for (auto *Sym : G.defined_symbols()) {
    for (auto &RTSym : RuntimeSymbols) {
      if (Sym->hasName() && *Sym->getName() == RTSym.first) {
        if (*RTSym.second)
          return make_error<StringError>(
              "Duplicate " + RTSym.first +
                  " detected during ELFNixPlatform bootstrap",
              inconvertibleErrorCode());

        if (*Sym->getName() == *MP.DSOHandleSymbol)
          RegisterELFNixHeader = true;

        *RTSym.second = Sym->getAddress();
      }
    }
  }

  if (RegisterELFNixHeader) {
    // If this graph defines the elfnix header symbol then create the internal
    // mapping between it and PlatformJD.
    std::lock_guard<std::mutex> Lock(MP.PlatformMutex);
    MP.JITDylibToHandleAddr[&MP.PlatformJD] =
        MP.Bootstrap.load()->ELFNixHeaderAddr;
    MP.HandleAddrToJITDylib[MP.Bootstrap.load()->ELFNixHeaderAddr] =
        &MP.PlatformJD;
  }

  return Error::success();
}

Error ELFNixPlatform::ELFNixPlatformPlugin::bootstrapPipelineEnd(
    jitlink::LinkGraph &G) {
  std::lock_guard<std::mutex> Lock(MP.Bootstrap.load()->Mutex);
  assert(MP.Bootstrap && "DeferredAAs reset before bootstrap completed");
  --MP.Bootstrap.load()->ActiveGraphs;
  // Notify Bootstrap->CV while holding the mutex because the mutex is
  // also keeping Bootstrap->CV alive.
  if (MP.Bootstrap.load()->ActiveGraphs == 0)
    MP.Bootstrap.load()->CV.notify_all();
  return Error::success();
}

Error ELFNixPlatform::registerPerObjectSections(
    jitlink::LinkGraph &G, const ELFPerObjectSectionsToRegister &POSR,
    bool IsBootstrapping) {
  using SPSRegisterPerObjSectionsArgs =
      SPSArgList<SPSELFPerObjectSectionsToRegister>;

  if (LLVM_UNLIKELY(IsBootstrapping)) {
    Bootstrap.load()->addArgumentsToRTFnMap(
        &RegisterObjectSections, &DeregisterObjectSections,
        getArgDataBufferType<SPSRegisterPerObjSectionsArgs>(POSR),
        getArgDataBufferType<SPSRegisterPerObjSectionsArgs>(POSR));
    return Error::success();
  }

  G.allocActions().push_back(
      {cantFail(WrapperFunctionCall::Create<SPSRegisterPerObjSectionsArgs>(
           RegisterObjectSections.Addr, POSR)),
       cantFail(WrapperFunctionCall::Create<SPSRegisterPerObjSectionsArgs>(
           DeregisterObjectSections.Addr, POSR))});

  return Error::success();
}

Expected<uint64_t> ELFNixPlatform::createPThreadKey() {
  if (!CreatePThreadKey.Addr)
    return make_error<StringError>(
        "Attempting to create pthread key in target, but runtime support has "
        "not been loaded yet",
        inconvertibleErrorCode());

  Expected<uint64_t> Result(0);
  if (auto Err = ES.callSPSWrapper<SPSExpected<uint64_t>(void)>(
          CreatePThreadKey.Addr, Result))
    return std::move(Err);
  return Result;
}

void ELFNixPlatform::ELFNixPlatformPlugin::modifyPassConfig(
    MaterializationResponsibility &MR, jitlink::LinkGraph &LG,
    jitlink::PassConfiguration &Config) {
  using namespace jitlink;

  bool InBootstrapPhase =
      &MR.getTargetJITDylib() == &MP.PlatformJD && MP.Bootstrap;

  // If we're in the bootstrap phase then increment the active graphs.
  if (InBootstrapPhase) {
    Config.PrePrunePasses.push_back(
        [this](LinkGraph &G) { return bootstrapPipelineStart(G); });
    Config.PostAllocationPasses.push_back([this](LinkGraph &G) {
      return bootstrapPipelineRecordRuntimeFunctions(G);
    });
  }

  // If the initializer symbol is the __dso_handle symbol then just add
  // the DSO handle support passes.
  if (auto InitSymbol = MR.getInitializerSymbol()) {
    if (InitSymbol == MP.DSOHandleSymbol && !InBootstrapPhase) {
      addDSOHandleSupportPasses(MR, Config);
      // The DSOHandle materialization unit doesn't require any other
      // support, so we can bail out early.
      return;
    }

    /// Preserve init sections.
    Config.PrePrunePasses.push_back(
        [this, &MR](jitlink::LinkGraph &G) -> Error {
          if (auto Err = preserveInitSections(G, MR))
            return Err;
          return Error::success();
        });
  }

  // Add passes for eh-frame and TLV support.
  addEHAndTLVSupportPasses(MR, Config, InBootstrapPhase);

  // If the object contains initializers then add passes to record them.
  Config.PostFixupPasses.push_back([this, &JD = MR.getTargetJITDylib(),
                                    InBootstrapPhase](jitlink::LinkGraph &G) {
    return registerInitSections(G, JD, InBootstrapPhase);
  });

  // If we're in the bootstrap phase then steal allocation actions and then
  // decrement the active graphs.
  if (InBootstrapPhase)
    Config.PostFixupPasses.push_back(
        [this](LinkGraph &G) { return bootstrapPipelineEnd(G); });
}

void ELFNixPlatform::ELFNixPlatformPlugin::addDSOHandleSupportPasses(
    MaterializationResponsibility &MR, jitlink::PassConfiguration &Config) {

  Config.PostAllocationPasses.push_back([this, &JD = MR.getTargetJITDylib()](
                                            jitlink::LinkGraph &G) -> Error {
    auto I = llvm::find_if(G.defined_symbols(), [this](jitlink::Symbol *Sym) {
      return Sym->getName() == MP.DSOHandleSymbol;
    });
    assert(I != G.defined_symbols().end() && "Missing DSO handle symbol");
    {
      std::lock_guard<std::mutex> Lock(MP.PlatformMutex);
      auto HandleAddr = (*I)->getAddress();
      MP.HandleAddrToJITDylib[HandleAddr] = &JD;
      MP.JITDylibToHandleAddr[&JD] = HandleAddr;

      G.allocActions().push_back(
          {cantFail(WrapperFunctionCall::Create<
                    SPSArgList<SPSString, SPSExecutorAddr>>(
               MP.RegisterJITDylib.Addr, JD.getName(), HandleAddr)),
           cantFail(WrapperFunctionCall::Create<SPSArgList<SPSExecutorAddr>>(
               MP.DeregisterJITDylib.Addr, HandleAddr))});
    }
    return Error::success();
  });
}

void ELFNixPlatform::ELFNixPlatformPlugin::addEHAndTLVSupportPasses(
    MaterializationResponsibility &MR, jitlink::PassConfiguration &Config,
    bool IsBootstrapping) {

  // Insert TLV lowering at the start of the PostPrunePasses, since we want
  // it to run before GOT/PLT lowering.

  // TODO: Check that before the fixTLVSectionsAndEdges pass, the GOT/PLT build
  // pass has done. Because the TLS descriptor need to be allocate in GOT.
  Config.PostPrunePasses.push_back(
      [this, &JD = MR.getTargetJITDylib()](jitlink::LinkGraph &G) {
        return fixTLVSectionsAndEdges(G, JD);
      });

  // Add a pass to register the final addresses of the eh-frame and TLV sections
  // with the runtime.
  Config.PostFixupPasses.push_back([this, IsBootstrapping](
                                       jitlink::LinkGraph &G) -> Error {
    ELFPerObjectSectionsToRegister POSR;

    if (auto *EHFrameSection = G.findSectionByName(ELFEHFrameSectionName)) {
      jitlink::SectionRange R(*EHFrameSection);
      if (!R.empty())
        POSR.EHFrameSection = R.getRange();
    }

    // Get a pointer to the thread data section if there is one. It will be used
    // below.
    jitlink::Section *ThreadDataSection =
        G.findSectionByName(ELFThreadDataSectionName);

    // Handle thread BSS section if there is one.
    if (auto *ThreadBSSSection = G.findSectionByName(ELFThreadBSSSectionName)) {
      // If there's already a thread data section in this graph then merge the
      // thread BSS section content into it, otherwise just treat the thread
      // BSS section as the thread data section.
      if (ThreadDataSection)
        G.mergeSections(*ThreadDataSection, *ThreadBSSSection);
      else
        ThreadDataSection = ThreadBSSSection;
    }

    // Having merged thread BSS (if present) and thread data (if present),
    // record the resulting section range.
    if (ThreadDataSection) {
      jitlink::SectionRange R(*ThreadDataSection);
      if (!R.empty())
        POSR.ThreadDataSection = R.getRange();
    }

    if (POSR.EHFrameSection.Start || POSR.ThreadDataSection.Start) {
      if (auto Err = MP.registerPerObjectSections(G, POSR, IsBootstrapping))
        return Err;
    }

    return Error::success();
  });
}

Error ELFNixPlatform::ELFNixPlatformPlugin::preserveInitSections(
    jitlink::LinkGraph &G, MaterializationResponsibility &MR) {

  if (const auto &InitSymName = MR.getInitializerSymbol()) {

    jitlink::Symbol *InitSym = nullptr;

    for (auto &InitSection : G.sections()) {
      // Skip non-init sections.
      if (!isELFInitializerSection(InitSection.getName()) ||
          InitSection.empty())
        continue;

      // Create the init symbol if it has not been created already and attach it
      // to the first block.
      if (!InitSym) {
        auto &B = **InitSection.blocks().begin();
        InitSym = &G.addDefinedSymbol(
            B, 0, *InitSymName, B.getSize(), jitlink::Linkage::Strong,
            jitlink::Scope::SideEffectsOnly, false, true);
      }

      // Add keep-alive edges to anonymous symbols in all other init blocks.
      for (auto *B : InitSection.blocks()) {
        if (B == &InitSym->getBlock())
          continue;

        auto &S = G.addAnonymousSymbol(*B, 0, B->getSize(), false, true);
        InitSym->getBlock().addEdge(jitlink::Edge::KeepAlive, 0, S, 0);
      }
    }
  }

  return Error::success();
}

Error ELFNixPlatform::ELFNixPlatformPlugin::registerInitSections(
    jitlink::LinkGraph &G, JITDylib &JD, bool IsBootstrapping) {
  SmallVector<ExecutorAddrRange> ELFNixPlatformSecs;
  LLVM_DEBUG(dbgs() << "ELFNixPlatform::registerInitSections\n");

  SmallVector<jitlink::Section *> OrderedInitSections;
  for (auto &Sec : G.sections())
    if (isELFInitializerSection(Sec.getName()))
      OrderedInitSections.push_back(&Sec);

  // FIXME: This handles priority order within the current graph, but we'll need
  //        to include priority information in the initializer allocation
  //        actions in order to respect the ordering across multiple graphs.
  llvm::sort(OrderedInitSections, [](const jitlink::Section *LHS,
                                     const jitlink::Section *RHS) {
    if (LHS->getName().starts_with(".init_array")) {
      if (RHS->getName().starts_with(".init_array")) {
        StringRef LHSPrioStr(LHS->getName());
        StringRef RHSPrioStr(RHS->getName());
        uint64_t LHSPriority;
        bool LHSHasPriority = LHSPrioStr.consume_front(".init_array.") &&
                              !LHSPrioStr.getAsInteger(10, LHSPriority);
        uint64_t RHSPriority;
        bool RHSHasPriority = RHSPrioStr.consume_front(".init_array.") &&
                              !RHSPrioStr.getAsInteger(10, RHSPriority);
        if (LHSHasPriority)
          return RHSHasPriority ? LHSPriority < RHSPriority : true;
        else if (RHSHasPriority)
          return false;
        // If we get here we'll fall through to the
        // LHS->getName() < RHS->getName() test below.
      } else {
        // .init_array[.N] comes before any non-.init_array[.N] section.
        return true;
      }
    }
    return LHS->getName() < RHS->getName();
  });

  for (auto &Sec : OrderedInitSections)
    ELFNixPlatformSecs.push_back(jitlink::SectionRange(*Sec).getRange());

  // Dump the scraped inits.
  LLVM_DEBUG({
    dbgs() << "ELFNixPlatform: Scraped " << G.getName() << " init sections:\n";
    for (auto &Sec : G.sections()) {
      jitlink::SectionRange R(Sec);
      dbgs() << "  " << Sec.getName() << ": " << R.getRange() << "\n";
    }
  });

  ExecutorAddr HeaderAddr;
  {
    std::lock_guard<std::mutex> Lock(MP.PlatformMutex);
    auto I = MP.JITDylibToHandleAddr.find(&JD);
    assert(I != MP.JITDylibToHandleAddr.end() && "No header registered for JD");
    assert(I->second && "Null header registered for JD");
    HeaderAddr = I->second;
  }

  using SPSRegisterInitSectionsArgs =
      SPSArgList<SPSExecutorAddr, SPSSequence<SPSExecutorAddrRange>>;

  if (LLVM_UNLIKELY(IsBootstrapping)) {
    MP.Bootstrap.load()->addArgumentsToRTFnMap(
        &MP.RegisterInitSections, &MP.DeregisterInitSections,
        getArgDataBufferType<SPSRegisterInitSectionsArgs>(HeaderAddr,
                                                          ELFNixPlatformSecs),
        getArgDataBufferType<SPSRegisterInitSectionsArgs>(HeaderAddr,
                                                          ELFNixPlatformSecs));
    return Error::success();
  }

  G.allocActions().push_back(
      {cantFail(WrapperFunctionCall::Create<SPSRegisterInitSectionsArgs>(
           MP.RegisterInitSections.Addr, HeaderAddr, ELFNixPlatformSecs)),
       cantFail(WrapperFunctionCall::Create<SPSRegisterInitSectionsArgs>(
           MP.DeregisterInitSections.Addr, HeaderAddr, ELFNixPlatformSecs))});

  return Error::success();
}

Error ELFNixPlatform::ELFNixPlatformPlugin::fixTLVSectionsAndEdges(
    jitlink::LinkGraph &G, JITDylib &JD) {
  auto TLSGetAddrSymbolName = G.intern("__tls_get_addr");
  auto TLSDescResolveSymbolName = G.intern("__tlsdesc_resolver");
  for (auto *Sym : G.external_symbols()) {
    if (Sym->getName() == TLSGetAddrSymbolName) {
      auto TLSGetAddr =
          MP.getExecutionSession().intern("___orc_rt_elfnix_tls_get_addr");
      Sym->setName(std::move(TLSGetAddr));
    } else if (Sym->getName() == TLSDescResolveSymbolName) {
      auto TLSGetAddr =
          MP.getExecutionSession().intern("___orc_rt_elfnix_tlsdesc_resolver");
      Sym->setName(std::move(TLSGetAddr));
    }
  }

  auto *TLSInfoEntrySection = G.findSectionByName("$__TLSINFO");

  if (TLSInfoEntrySection) {
    std::optional<uint64_t> Key;
    {
      std::lock_guard<std::mutex> Lock(MP.PlatformMutex);
      auto I = MP.JITDylibToPThreadKey.find(&JD);
      if (I != MP.JITDylibToPThreadKey.end())
        Key = I->second;
    }
    if (!Key) {
      if (auto KeyOrErr = MP.createPThreadKey())
        Key = *KeyOrErr;
      else
        return KeyOrErr.takeError();
    }

    uint64_t PlatformKeyBits =
        support::endian::byte_swap(*Key, G.getEndianness());

    for (auto *B : TLSInfoEntrySection->blocks()) {
      // FIXME: The TLS descriptor byte length may different with different
      // ISA
      assert(B->getSize() == (G.getPointerSize() * 2) &&
             "TLS descriptor must be 2 words length");
      auto TLSInfoEntryContent = B->getMutableContent(G);
      memcpy(TLSInfoEntryContent.data(), &PlatformKeyBits, G.getPointerSize());
    }
  }

  return Error::success();
}

} // End namespace orc.
} // End namespace llvm.
