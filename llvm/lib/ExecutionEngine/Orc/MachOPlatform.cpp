//===------ MachOPlatform.cpp - Utilities for executing MachO in Orc ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/MachOPlatform.h"

#include "llvm/BinaryFormat/MachO.h"
#include "llvm/ExecutionEngine/JITLink/x86_64.h"
#include "llvm/ExecutionEngine/Orc/DebugUtils.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/LookupAndRecordAddrs.h"
#include "llvm/Support/BinaryByteStream.h"
#include "llvm/Support/Debug.h"
#include <optional>

#define DEBUG_TYPE "orc"

using namespace llvm;
using namespace llvm::orc;
using namespace llvm::orc::shared;

namespace llvm {
namespace orc {
namespace shared {

using SPSMachOJITDylibDepInfo = SPSTuple<bool, SPSSequence<SPSExecutorAddr>>;
using SPSMachOJITDylibDepInfoMap =
    SPSSequence<SPSTuple<SPSExecutorAddr, SPSMachOJITDylibDepInfo>>;

template <>
class SPSSerializationTraits<SPSMachOJITDylibDepInfo,
                             MachOPlatform::MachOJITDylibDepInfo> {
public:
  static size_t size(const MachOPlatform::MachOJITDylibDepInfo &DDI) {
    return SPSMachOJITDylibDepInfo::AsArgList::size(DDI.Sealed, DDI.DepHeaders);
  }

  static bool serialize(SPSOutputBuffer &OB,
                        const MachOPlatform::MachOJITDylibDepInfo &DDI) {
    return SPSMachOJITDylibDepInfo::AsArgList::serialize(OB, DDI.Sealed,
                                                         DDI.DepHeaders);
  }

  static bool deserialize(SPSInputBuffer &IB,
                          MachOPlatform::MachOJITDylibDepInfo &DDI) {
    return SPSMachOJITDylibDepInfo::AsArgList::deserialize(IB, DDI.Sealed,
                                                           DDI.DepHeaders);
  }
};

} // namespace shared
} // namespace orc
} // namespace llvm

namespace {

std::unique_ptr<jitlink::LinkGraph> createPlatformGraph(MachOPlatform &MOP,
                                                        std::string Name) {
  unsigned PointerSize;
  support::endianness Endianness;
  const auto &TT =
      MOP.getExecutionSession().getExecutorProcessControl().getTargetTriple();

  switch (TT.getArch()) {
  case Triple::aarch64:
  case Triple::x86_64:
    PointerSize = 8;
    Endianness = support::endianness::little;
    break;
  default:
    llvm_unreachable("Unrecognized architecture");
  }

  return std::make_unique<jitlink::LinkGraph>(std::move(Name), TT, PointerSize,
                                              Endianness,
                                              jitlink::getGenericEdgeKindName);
}

// Generates a MachO header.
class MachOHeaderMaterializationUnit : public MaterializationUnit {
public:
  MachOHeaderMaterializationUnit(MachOPlatform &MOP,
                                 const SymbolStringPtr &HeaderStartSymbol)
      : MaterializationUnit(createHeaderInterface(MOP, HeaderStartSymbol)),
        MOP(MOP) {}

  StringRef getName() const override { return "MachOHeaderMU"; }

  void materialize(std::unique_ptr<MaterializationResponsibility> R) override {
    auto G = createPlatformGraph(MOP, "<MachOHeaderMU>");
    addMachOHeader(*G, MOP, R->getInitializerSymbol());
    MOP.getObjectLinkingLayer().emit(std::move(R), std::move(G));
  }

  void discard(const JITDylib &JD, const SymbolStringPtr &Sym) override {}

  static void addMachOHeader(jitlink::LinkGraph &G, MachOPlatform &MOP,
                             const SymbolStringPtr &InitializerSymbol) {
    auto &HeaderSection = G.createSection("__header", MemProt::Read);
    auto &HeaderBlock = createHeaderBlock(G, HeaderSection);

    // Init symbol is header-start symbol.
    G.addDefinedSymbol(HeaderBlock, 0, *InitializerSymbol,
                       HeaderBlock.getSize(), jitlink::Linkage::Strong,
                       jitlink::Scope::Default, false, true);
    for (auto &HS : AdditionalHeaderSymbols)
      G.addDefinedSymbol(HeaderBlock, HS.Offset, HS.Name, HeaderBlock.getSize(),
                         jitlink::Linkage::Strong, jitlink::Scope::Default,
                         false, true);
  }

private:
  struct HeaderSymbol {
    const char *Name;
    uint64_t Offset;
  };

  static constexpr HeaderSymbol AdditionalHeaderSymbols[] = {
      {"___mh_executable_header", 0}};

  static jitlink::Block &createHeaderBlock(jitlink::LinkGraph &G,
                                           jitlink::Section &HeaderSection) {
    MachO::mach_header_64 Hdr;
    Hdr.magic = MachO::MH_MAGIC_64;
    switch (G.getTargetTriple().getArch()) {
    case Triple::aarch64:
      Hdr.cputype = MachO::CPU_TYPE_ARM64;
      Hdr.cpusubtype = MachO::CPU_SUBTYPE_ARM64_ALL;
      break;
    case Triple::x86_64:
      Hdr.cputype = MachO::CPU_TYPE_X86_64;
      Hdr.cpusubtype = MachO::CPU_SUBTYPE_X86_64_ALL;
      break;
    default:
      llvm_unreachable("Unrecognized architecture");
    }
    Hdr.filetype = MachO::MH_DYLIB; // Custom file type?
    Hdr.ncmds = 0;
    Hdr.sizeofcmds = 0;
    Hdr.flags = 0;
    Hdr.reserved = 0;

    if (G.getEndianness() != support::endian::system_endianness())
      MachO::swapStruct(Hdr);

    auto HeaderContent = G.allocateString(
        StringRef(reinterpret_cast<const char *>(&Hdr), sizeof(Hdr)));

    return G.createContentBlock(HeaderSection, HeaderContent, ExecutorAddr(), 8,
                                0);
  }

  static MaterializationUnit::Interface
  createHeaderInterface(MachOPlatform &MOP,
                        const SymbolStringPtr &HeaderStartSymbol) {
    SymbolFlagsMap HeaderSymbolFlags;

    HeaderSymbolFlags[HeaderStartSymbol] = JITSymbolFlags::Exported;
    for (auto &HS : AdditionalHeaderSymbols)
      HeaderSymbolFlags[MOP.getExecutionSession().intern(HS.Name)] =
          JITSymbolFlags::Exported;

    return MaterializationUnit::Interface(std::move(HeaderSymbolFlags),
                                          HeaderStartSymbol);
  }

  MachOPlatform &MOP;
};

constexpr MachOHeaderMaterializationUnit::HeaderSymbol
    MachOHeaderMaterializationUnit::AdditionalHeaderSymbols[];

// Creates a Bootstrap-Complete LinkGraph to run deferred actions.
class MachOPlatformCompleteBootstrapMaterializationUnit
    : public MaterializationUnit {
public:
  MachOPlatformCompleteBootstrapMaterializationUnit(
      MachOPlatform &MOP, StringRef PlatformJDName,
      SymbolStringPtr CompleteBootstrapSymbol, shared::AllocActions DeferredAAs,
      ExecutorAddr PlatformBootstrap, ExecutorAddr PlatformShutdown,
      ExecutorAddr RegisterJITDylib, ExecutorAddr DeregisterJITDylib,
      ExecutorAddr MachOHeaderAddr)
      : MaterializationUnit(
            {{{CompleteBootstrapSymbol, JITSymbolFlags::None}}, nullptr}),
        MOP(MOP), PlatformJDName(PlatformJDName),
        CompleteBootstrapSymbol(std::move(CompleteBootstrapSymbol)),
        DeferredAAs(std::move(DeferredAAs)),
        PlatformBootstrap(PlatformBootstrap),
        PlatformShutdown(PlatformShutdown), RegisterJITDylib(RegisterJITDylib),
        DeregisterJITDylib(DeregisterJITDylib),
        MachOHeaderAddr(MachOHeaderAddr) {}

  StringRef getName() const override {
    return "MachOPlatformCompleteBootstrap";
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

    // Reserve space for the stolen actions, plus two extras.
    G->allocActions().reserve(DeferredAAs.size() + 2);

    // 1. Bootstrap the platform support code.
    G->allocActions().push_back(
        {cantFail(WrapperFunctionCall::Create<SPSArgList<>>(PlatformBootstrap)),
         cantFail(
             WrapperFunctionCall::Create<SPSArgList<>>(PlatformShutdown))});

    // 2. Register the platform JITDylib.
    G->allocActions().push_back(
        {cantFail(WrapperFunctionCall::Create<
                  SPSArgList<SPSString, SPSExecutorAddr>>(
             RegisterJITDylib, PlatformJDName, MachOHeaderAddr)),
         cantFail(WrapperFunctionCall::Create<SPSArgList<SPSExecutorAddr>>(
             DeregisterJITDylib, MachOHeaderAddr))});

    // 3. Add the deferred actions to the graph.
    std::move(DeferredAAs.begin(), DeferredAAs.end(),
              std::back_inserter(G->allocActions()));

    MOP.getObjectLinkingLayer().emit(std::move(R), std::move(G));
  }

  void discard(const JITDylib &JD, const SymbolStringPtr &Sym) override {}

private:
  MachOPlatform &MOP;
  StringRef PlatformJDName;
  SymbolStringPtr CompleteBootstrapSymbol;
  shared::AllocActions DeferredAAs;
  ExecutorAddr PlatformBootstrap;
  ExecutorAddr PlatformShutdown;
  ExecutorAddr RegisterJITDylib;
  ExecutorAddr DeregisterJITDylib;
  ExecutorAddr MachOHeaderAddr;
};

StringRef DataCommonSectionName = "__DATA,__common";
StringRef DataDataSectionName = "__DATA,__data";
StringRef EHFrameSectionName = "__TEXT,__eh_frame";
StringRef CompactUnwindInfoSectionName = "__TEXT,__unwind_info";
StringRef ModInitFuncSectionName = "__DATA,__mod_init_func";
StringRef ObjCClassListSectionName = "__DATA,__objc_classlist";
StringRef ObjCImageInfoSectionName = "__DATA,__objc_image_info";
StringRef ObjCSelRefsSectionName = "__DATA,__objc_selrefs";
StringRef Swift5ProtoSectionName = "__TEXT,__swift5_proto";
StringRef Swift5ProtosSectionName = "__TEXT,__swift5_protos";
StringRef Swift5TypesSectionName = "__TEXT,__swift5_types";
StringRef ThreadBSSSectionName = "__DATA,__thread_bss";
StringRef ThreadDataSectionName = "__DATA,__thread_data";
StringRef ThreadVarsSectionName = "__DATA,__thread_vars";

StringRef InitSectionNames[] = {
    ModInitFuncSectionName, ObjCSelRefsSectionName, ObjCClassListSectionName,
    Swift5ProtosSectionName, Swift5ProtoSectionName, Swift5TypesSectionName};

} // end anonymous namespace

namespace llvm {
namespace orc {

Expected<std::unique_ptr<MachOPlatform>>
MachOPlatform::Create(ExecutionSession &ES, ObjectLinkingLayer &ObjLinkingLayer,
                      JITDylib &PlatformJD, const char *OrcRuntimePath,
                      std::optional<SymbolAliasMap> RuntimeAliases) {

  auto &EPC = ES.getExecutorProcessControl();

  // If the target is not supported then bail out immediately.
  if (!supportedTarget(EPC.getTargetTriple()))
    return make_error<StringError>("Unsupported MachOPlatform triple: " +
                                       EPC.getTargetTriple().str(),
                                   inconvertibleErrorCode());

  // Create default aliases if the caller didn't supply any.
  if (!RuntimeAliases)
    RuntimeAliases = standardPlatformAliases(ES);

  // Define the aliases.
  if (auto Err = PlatformJD.define(symbolAliases(std::move(*RuntimeAliases))))
    return std::move(Err);

  // Add JIT-dispatch function support symbols.
  if (auto Err = PlatformJD.define(absoluteSymbols(
          {{ES.intern("___orc_rt_jit_dispatch"),
            {EPC.getJITDispatchInfo().JITDispatchFunction.getValue(),
             JITSymbolFlags::Exported}},
           {ES.intern("___orc_rt_jit_dispatch_ctx"),
            {EPC.getJITDispatchInfo().JITDispatchContext.getValue(),
             JITSymbolFlags::Exported}}})))
    return std::move(Err);

  // Create a generator for the ORC runtime archive.
  auto OrcRuntimeArchiveGenerator = StaticLibraryDefinitionGenerator::Load(
      ObjLinkingLayer, OrcRuntimePath, EPC.getTargetTriple());
  if (!OrcRuntimeArchiveGenerator)
    return OrcRuntimeArchiveGenerator.takeError();

  // Create the instance.
  Error Err = Error::success();
  auto P = std::unique_ptr<MachOPlatform>(
      new MachOPlatform(ES, ObjLinkingLayer, PlatformJD,
                        std::move(*OrcRuntimeArchiveGenerator), Err));
  if (Err)
    return std::move(Err);
  return std::move(P);
}

Error MachOPlatform::setupJITDylib(JITDylib &JD) {
  if (auto Err = JD.define(std::make_unique<MachOHeaderMaterializationUnit>(
          *this, MachOHeaderStartSymbol)))
    return Err;

  return ES.lookup({&JD}, MachOHeaderStartSymbol).takeError();
}

Error MachOPlatform::teardownJITDylib(JITDylib &JD) {
  std::lock_guard<std::mutex> Lock(PlatformMutex);
  auto I = JITDylibToHeaderAddr.find(&JD);
  if (I != JITDylibToHeaderAddr.end()) {
    assert(HeaderAddrToJITDylib.count(I->second) &&
           "HeaderAddrToJITDylib missing entry");
    HeaderAddrToJITDylib.erase(I->second);
    JITDylibToHeaderAddr.erase(I);
  }
  JITDylibToPThreadKey.erase(&JD);
  return Error::success();
}

Error MachOPlatform::notifyAdding(ResourceTracker &RT,
                                  const MaterializationUnit &MU) {
  auto &JD = RT.getJITDylib();
  const auto &InitSym = MU.getInitializerSymbol();
  if (!InitSym)
    return Error::success();

  RegisteredInitSymbols[&JD].add(InitSym,
                                 SymbolLookupFlags::WeaklyReferencedSymbol);
  LLVM_DEBUG({
    dbgs() << "MachOPlatform: Registered init symbol " << *InitSym << " for MU "
           << MU.getName() << "\n";
  });
  return Error::success();
}

Error MachOPlatform::notifyRemoving(ResourceTracker &RT) {
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

SymbolAliasMap MachOPlatform::standardPlatformAliases(ExecutionSession &ES) {
  SymbolAliasMap Aliases;
  addAliases(ES, Aliases, requiredCXXAliases());
  addAliases(ES, Aliases, standardRuntimeUtilityAliases());
  return Aliases;
}

ArrayRef<std::pair<const char *, const char *>>
MachOPlatform::requiredCXXAliases() {
  static const std::pair<const char *, const char *> RequiredCXXAliases[] = {
      {"___cxa_atexit", "___orc_rt_macho_cxa_atexit"}};

  return ArrayRef<std::pair<const char *, const char *>>(RequiredCXXAliases);
}

ArrayRef<std::pair<const char *, const char *>>
MachOPlatform::standardRuntimeUtilityAliases() {
  static const std::pair<const char *, const char *>
      StandardRuntimeUtilityAliases[] = {
          {"___orc_rt_run_program", "___orc_rt_macho_run_program"},
          {"___orc_rt_jit_dlerror", "___orc_rt_macho_jit_dlerror"},
          {"___orc_rt_jit_dlopen", "___orc_rt_macho_jit_dlopen"},
          {"___orc_rt_jit_dlclose", "___orc_rt_macho_jit_dlclose"},
          {"___orc_rt_jit_dlsym", "___orc_rt_macho_jit_dlsym"},
          {"___orc_rt_log_error", "___orc_rt_log_error_to_stderr"}};

  return ArrayRef<std::pair<const char *, const char *>>(
      StandardRuntimeUtilityAliases);
}

bool MachOPlatform::isInitializerSection(StringRef SegName,
                                         StringRef SectName) {
  for (auto &Name : InitSectionNames) {
    if (Name.startswith(SegName) && Name.substr(7) == SectName)
      return true;
  }
  return false;
}

bool MachOPlatform::supportedTarget(const Triple &TT) {
  switch (TT.getArch()) {
  case Triple::aarch64:
  case Triple::x86_64:
    return true;
  default:
    return false;
  }
}

MachOPlatform::MachOPlatform(
    ExecutionSession &ES, ObjectLinkingLayer &ObjLinkingLayer,
    JITDylib &PlatformJD,
    std::unique_ptr<DefinitionGenerator> OrcRuntimeGenerator, Error &Err)
    : ES(ES), PlatformJD(PlatformJD), ObjLinkingLayer(ObjLinkingLayer) {
  ErrorAsOutParameter _(&Err);
  ObjLinkingLayer.addPlugin(std::make_unique<MachOPlatformPlugin>(*this));
  PlatformJD.addGenerator(std::move(OrcRuntimeGenerator));

  BootstrapInfo BI;
  Bootstrap = &BI;

  // Bootstrap process -- here be phase-ordering dragons.
  //
  // The MachOPlatform class uses allocation actions to register metadata
  // sections with the ORC runtime, however the runtime contains metadata
  // registration functions that have their own metadata that they need to
  // register (e.g. the frame-info registration functions have frame-info).
  // We can't use an ordinary lookup to find these registration functions
  // because their address is needed during the link of the containing graph
  // itself (to build the allocation actions that will call the registration
  // functions). Further complicating the situation (a) the graph containing
  // the registration functions is allowed to depend on other graphs (e.g. the
  // graph containing the ORC runtime RTTI support) so we need to handle with
  // an unknown set of dependencies during bootstrap, and (b) these graphs may
  // be linked concurrently if the user has installed a concurrent dispatcher.
  //
  // We satisfy these constraint by implementing a bootstrap phase during which
  // allocation actions generated by MachOPlatform are appended to a list of
  // deferred allocation actions, rather than to the graphs themselves. At the
  // end of the bootstrap process the deferred actions are attached to a final
  // "complete-bootstrap" graph that causes them to be run.
  //
  // The bootstrap steps are as follows:
  //
  // 1. Request the graph containing the mach header. This graph is guaranteed
  //    not to have any metadata so the fact that the registration functions
  //    are not available yet is not a problem.
  //
  // 2. Look up the registration functions and discard the results. This will
  //    trigger linking of the graph containing these functions, and
  //    consequently any graphs that it depends on. We do not use the lookup
  //    result to find the addresses of the functions requested (as described
  //    above the lookup will return too late for that), instead we capture the
  //    addresses in a post-allocation pass injected by the platform runtime
  //    during bootstrap only.
  //
  // 3. During bootstrap the MachOPlatformPlugin keeps a count of the number of
  //    graphs being linked (potentially concurrently), and we block until all
  //    of these graphs have completed linking. This is to avoid a race on the
  //    deferred-actions vector: the lookup for the runtime registration
  //    functions may return while some functions (those that are being
  //    incidentally linked in, but aren't reachable via the runtime functions)
  //    are still being linked, and we need to capture any allocation actions
  //    for this incidental code before we proceed.
  //
  // 4. Once all active links are complete we transfer the deferred actions to
  //    a newly added CompleteBootstrap graph and then request a symbol from
  //    the CompleteBootstrap graph to trigger materialization. This will cause
  //    all deferred actions to be run, and once this lookup returns we can
  //    proceed.
  //
  // 5. Finally, we associate runtime support methods in MachOPlatform with
  //    the corresponding jit-dispatch tag variables in the ORC runtime to make
  //    the support methods callable. The bootstrap is now complete.

  // Step (1) Add header materialization unit and request.
  if ((Err = PlatformJD.define(std::make_unique<MachOHeaderMaterializationUnit>(
           *this, MachOHeaderStartSymbol))))
    return;
  if ((Err = ES.lookup(&PlatformJD, MachOHeaderStartSymbol).takeError()))
    return;

  // Step (2) Request runtime registration functions to trigger
  // materialization..
  if ((Err = ES.lookup(makeJITDylibSearchOrder(&PlatformJD),
                       SymbolLookupSet(
                           {PlatformBootstrap.Name, PlatformShutdown.Name,
                            RegisterJITDylib.Name, DeregisterJITDylib.Name,
                            RegisterObjectPlatformSections.Name,
                            DeregisterObjectPlatformSections.Name,
                            CreatePThreadKey.Name}))
                 .takeError()))
    return;

  // Step (3) Wait for any incidental linker work to complete.
  {
    std::unique_lock<std::mutex> Lock(BI.Mutex);
    BI.CV.wait(Lock, [&]() { return BI.ActiveGraphs == 0; });
    Bootstrap = nullptr;
  }

  // Step (4) Add complete-bootstrap materialization unit and request.
  auto BootstrapCompleteSymbol = ES.intern("__orc_rt_macho_complete_bootstrap");
  if ((Err = PlatformJD.define(
           std::make_unique<MachOPlatformCompleteBootstrapMaterializationUnit>(
               *this, PlatformJD.getName(), BootstrapCompleteSymbol,
               std::move(BI.DeferredAAs), PlatformBootstrap.Addr,
               PlatformShutdown.Addr, RegisterJITDylib.Addr,
               DeregisterJITDylib.Addr, BI.MachOHeaderAddr))))
    return;
  if ((Err = ES.lookup(makeJITDylibSearchOrder(
                           &PlatformJD, JITDylibLookupFlags::MatchAllSymbols),
                       std::move(BootstrapCompleteSymbol))
                 .takeError()))
    return;

  // (5) Associate runtime support functions.
  if ((Err = associateRuntimeSupportFunctions()))
    return;
}

Error MachOPlatform::associateRuntimeSupportFunctions() {
  ExecutionSession::JITDispatchHandlerAssociationMap WFs;

  using PushInitializersSPSSig =
      SPSExpected<SPSMachOJITDylibDepInfoMap>(SPSExecutorAddr);
  WFs[ES.intern("___orc_rt_macho_push_initializers_tag")] =
      ES.wrapAsyncWithSPS<PushInitializersSPSSig>(
          this, &MachOPlatform::rt_pushInitializers);

  using LookupSymbolSPSSig =
      SPSExpected<SPSExecutorAddr>(SPSExecutorAddr, SPSString);
  WFs[ES.intern("___orc_rt_macho_symbol_lookup_tag")] =
      ES.wrapAsyncWithSPS<LookupSymbolSPSSig>(this,
                                              &MachOPlatform::rt_lookupSymbol);

  return ES.registerJITDispatchHandlers(PlatformJD, std::move(WFs));
}

void MachOPlatform::pushInitializersLoop(
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
    // that appear in the JITDylibToHeaderAddr map (i.e. those that have been
    // through setupJITDylib) -- bare JITDylibs aren't managed by the platform.
    DenseMap<JITDylib *, ExecutorAddr> HeaderAddrs;
    HeaderAddrs.reserve(JDDepMap.size());
    {
      std::lock_guard<std::mutex> Lock(PlatformMutex);
      for (auto &KV : JDDepMap) {
        auto I = JITDylibToHeaderAddr.find(KV.first);
        if (I != JITDylibToHeaderAddr.end())
          HeaderAddrs[KV.first] = I->second;
      }
    }

    // Build the dep info map to return.
    MachOJITDylibDepInfoMap DIM;
    DIM.reserve(JDDepMap.size());
    for (auto &KV : JDDepMap) {
      auto HI = HeaderAddrs.find(KV.first);
      // Skip unmanaged JITDylibs.
      if (HI == HeaderAddrs.end())
        continue;
      auto H = HI->second;
      MachOJITDylibDepInfo DepInfo;
      for (auto &Dep : KV.second) {
        auto HJ = HeaderAddrs.find(Dep);
        if (HJ != HeaderAddrs.end())
          DepInfo.DepHeaders.push_back(HJ->second);
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

void MachOPlatform::rt_pushInitializers(PushInitializersSendResultFn SendResult,
                                        ExecutorAddr JDHeaderAddr) {
  JITDylibSP JD;
  {
    std::lock_guard<std::mutex> Lock(PlatformMutex);
    auto I = HeaderAddrToJITDylib.find(JDHeaderAddr);
    if (I != HeaderAddrToJITDylib.end())
      JD = I->second;
  }

  LLVM_DEBUG({
    dbgs() << "MachOPlatform::rt_pushInitializers(" << JDHeaderAddr << ") ";
    if (JD)
      dbgs() << "pushing initializers for " << JD->getName() << "\n";
    else
      dbgs() << "No JITDylib for header address.\n";
  });

  if (!JD) {
    SendResult(
        make_error<StringError>("No JITDylib with header addr " +
                                    formatv("{0:x}", JDHeaderAddr.getValue()),
                                inconvertibleErrorCode()));
    return;
  }

  pushInitializersLoop(std::move(SendResult), JD);
}

void MachOPlatform::rt_lookupSymbol(SendSymbolAddressFn SendResult,
                                    ExecutorAddr Handle, StringRef SymbolName) {
  LLVM_DEBUG({
    dbgs() << "MachOPlatform::rt_lookupSymbol(\""
           << formatv("{0:x}", Handle.getValue()) << "\")\n";
  });

  JITDylib *JD = nullptr;

  {
    std::lock_guard<std::mutex> Lock(PlatformMutex);
    auto I = HeaderAddrToJITDylib.find(Handle);
    if (I != HeaderAddrToJITDylib.end())
      JD = I->second;
  }

  if (!JD) {
    LLVM_DEBUG({
      dbgs() << "  No JITDylib for handle "
             << formatv("{0:x}", Handle.getValue()) << "\n";
    });
    SendResult(make_error<StringError>("No JITDylib associated with handle " +
                                           formatv("{0:x}", Handle.getValue()),
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
        SendResult(ExecutorAddr(Result->begin()->second.getAddress()));
      } else {
        SendResult(Result.takeError());
      }
    }

  private:
    SendSymbolAddressFn SendResult;
  };

  // FIXME: Proper mangling.
  auto MangledName = ("_" + SymbolName).str();
  ES.lookup(
      LookupKind::DLSym, {{JD, JITDylibLookupFlags::MatchExportedSymbolsOnly}},
      SymbolLookupSet(ES.intern(MangledName)), SymbolState::Ready,
      RtLookupNotifyComplete(std::move(SendResult)), NoDependenciesToRegister);
}

Expected<uint64_t> MachOPlatform::createPThreadKey() {
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

void MachOPlatform::MachOPlatformPlugin::modifyPassConfig(
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

  // --- Handle Initializers ---
  if (auto InitSymbol = MR.getInitializerSymbol()) {

    // If the initializer symbol is the MachOHeader start symbol then just
    // register it and then bail out -- the header materialization unit
    // definitely doesn't need any other passes.
    if (InitSymbol == MP.MachOHeaderStartSymbol && !InBootstrapPhase) {
      Config.PostAllocationPasses.push_back([this, &MR](LinkGraph &G) {
        return associateJITDylibHeaderSymbol(G, MR);
      });
      return;
    }

    // If the object contains an init symbol other than the header start symbol
    // then add passes to preserve, process and register the init
    // sections/symbols.
    Config.PrePrunePasses.push_back([this, &MR](LinkGraph &G) {
      if (auto Err = preserveInitSections(G, MR))
        return Err;
      return processObjCImageInfo(G, MR);
    });
  }

  // Insert TLV lowering at the start of the PostPrunePasses, since we want
  // it to run before GOT/PLT lowering.
  Config.PostPrunePasses.insert(
      Config.PostPrunePasses.begin(),
      [this, &JD = MR.getTargetJITDylib()](LinkGraph &G) {
        return fixTLVSectionsAndEdges(G, JD);
      });

  // Add a pass to register the final addresses of any special sections in the
  // object with the runtime.
  Config.PostAllocationPasses.push_back(
      [this, &JD = MR.getTargetJITDylib(), InBootstrapPhase](LinkGraph &G) {
        return registerObjectPlatformSections(G, JD, InBootstrapPhase);
      });

  // If we're in the bootstrap phase then steal allocation actions and then
  // decrement the active graphs.
  if (InBootstrapPhase)
    Config.PostFixupPasses.push_back(
        [this](LinkGraph &G) { return bootstrapPipelineEnd(G); });
}

ObjectLinkingLayer::Plugin::SyntheticSymbolDependenciesMap
MachOPlatform::MachOPlatformPlugin::getSyntheticSymbolDependencies(
    MaterializationResponsibility &MR) {
  std::lock_guard<std::mutex> Lock(PluginMutex);
  auto I = InitSymbolDeps.find(&MR);
  if (I != InitSymbolDeps.end()) {
    SyntheticSymbolDependenciesMap Result;
    Result[MR.getInitializerSymbol()] = std::move(I->second);
    InitSymbolDeps.erase(&MR);
    return Result;
  }
  return SyntheticSymbolDependenciesMap();
}

Error MachOPlatform::MachOPlatformPlugin::bootstrapPipelineStart(
    jitlink::LinkGraph &G) {
  // Increment the active graphs count in BootstrapInfo.
  std::lock_guard<std::mutex> Lock(MP.Bootstrap.load()->Mutex);
  ++MP.Bootstrap.load()->ActiveGraphs;
  return Error::success();
}

Error MachOPlatform::MachOPlatformPlugin::
    bootstrapPipelineRecordRuntimeFunctions(jitlink::LinkGraph &G) {
  // Record bootstrap function names.
  std::pair<StringRef, ExecutorAddr *> RuntimeSymbols[] = {
      {*MP.MachOHeaderStartSymbol, &MP.Bootstrap.load()->MachOHeaderAddr},
      {*MP.PlatformBootstrap.Name, &MP.PlatformBootstrap.Addr},
      {*MP.PlatformShutdown.Name, &MP.PlatformShutdown.Addr},
      {*MP.RegisterJITDylib.Name, &MP.RegisterJITDylib.Addr},
      {*MP.DeregisterJITDylib.Name, &MP.DeregisterJITDylib.Addr},
      {*MP.RegisterObjectPlatformSections.Name,
       &MP.RegisterObjectPlatformSections.Addr},
      {*MP.DeregisterObjectPlatformSections.Name,
       &MP.DeregisterObjectPlatformSections.Addr},
      {*MP.CreatePThreadKey.Name, &MP.CreatePThreadKey.Addr}};

  bool RegisterMachOHeader = false;

  for (auto *Sym : G.defined_symbols()) {
    for (auto &RTSym : RuntimeSymbols) {
      if (Sym->hasName() && Sym->getName() == RTSym.first) {
        if (*RTSym.second)
          return make_error<StringError>(
              "Duplicate " + RTSym.first +
                  " detected during MachOPlatform bootstrap",
              inconvertibleErrorCode());

        if (Sym->getName() == *MP.MachOHeaderStartSymbol)
          RegisterMachOHeader = true;

        *RTSym.second = Sym->getAddress();
      }
    }
  }

  if (RegisterMachOHeader) {
    // If this graph defines the macho header symbol then create the internal
    // mapping between it and PlatformJD.
    std::lock_guard<std::mutex> Lock(MP.PlatformMutex);
    MP.JITDylibToHeaderAddr[&MP.PlatformJD] =
        MP.Bootstrap.load()->MachOHeaderAddr;
    MP.HeaderAddrToJITDylib[MP.Bootstrap.load()->MachOHeaderAddr] =
        &MP.PlatformJD;
  }

  return Error::success();
}

Error MachOPlatform::MachOPlatformPlugin::bootstrapPipelineEnd(
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

Error MachOPlatform::MachOPlatformPlugin::associateJITDylibHeaderSymbol(
    jitlink::LinkGraph &G, MaterializationResponsibility &MR) {
  auto I = llvm::find_if(G.defined_symbols(), [this](jitlink::Symbol *Sym) {
    return Sym->getName() == *MP.MachOHeaderStartSymbol;
  });
  assert(I != G.defined_symbols().end() && "Missing MachO header start symbol");

  auto &JD = MR.getTargetJITDylib();
  std::lock_guard<std::mutex> Lock(MP.PlatformMutex);
  auto HeaderAddr = (*I)->getAddress();
  MP.JITDylibToHeaderAddr[&JD] = HeaderAddr;
  MP.HeaderAddrToJITDylib[HeaderAddr] = &JD;
  // We can unconditionally add these actions to the Graph because this pass
  // isn't used during bootstrap.
  G.allocActions().push_back(
      {cantFail(
           WrapperFunctionCall::Create<SPSArgList<SPSString, SPSExecutorAddr>>(
               MP.RegisterJITDylib.Addr, JD.getName(), HeaderAddr)),
       cantFail(WrapperFunctionCall::Create<SPSArgList<SPSExecutorAddr>>(
           MP.DeregisterJITDylib.Addr, HeaderAddr))});
  return Error::success();
}

Error MachOPlatform::MachOPlatformPlugin::preserveInitSections(
    jitlink::LinkGraph &G, MaterializationResponsibility &MR) {

  JITLinkSymbolSet InitSectionSymbols;
  for (auto &InitSectionName : InitSectionNames) {
    // Skip non-init sections.
    auto *InitSection = G.findSectionByName(InitSectionName);
    if (!InitSection)
      continue;

    // Make a pass over live symbols in the section: those blocks are already
    // preserved.
    DenseSet<jitlink::Block *> AlreadyLiveBlocks;
    for (auto &Sym : InitSection->symbols()) {
      auto &B = Sym->getBlock();
      if (Sym->isLive() && Sym->getOffset() == 0 &&
          Sym->getSize() == B.getSize() && !AlreadyLiveBlocks.count(&B)) {
        InitSectionSymbols.insert(Sym);
        AlreadyLiveBlocks.insert(&B);
      }
    }

    // Add anonymous symbols to preserve any not-already-preserved blocks.
    for (auto *B : InitSection->blocks())
      if (!AlreadyLiveBlocks.count(B))
        InitSectionSymbols.insert(
            &G.addAnonymousSymbol(*B, 0, B->getSize(), false, true));
  }

  if (!InitSectionSymbols.empty()) {
    std::lock_guard<std::mutex> Lock(PluginMutex);
    InitSymbolDeps[&MR] = std::move(InitSectionSymbols);
  }

  return Error::success();
}

Error MachOPlatform::MachOPlatformPlugin::processObjCImageInfo(
    jitlink::LinkGraph &G, MaterializationResponsibility &MR) {

  // If there's an ObjC imagine info then either
  //   (1) It's the first __objc_imageinfo we've seen in this JITDylib. In
  //       this case we name and record it.
  // OR
  //   (2) We already have a recorded __objc_imageinfo for this JITDylib,
  //       in which case we just verify it.
  auto *ObjCImageInfo = G.findSectionByName(ObjCImageInfoSectionName);
  if (!ObjCImageInfo)
    return Error::success();

  auto ObjCImageInfoBlocks = ObjCImageInfo->blocks();

  // Check that the section is not empty if present.
  if (ObjCImageInfoBlocks.empty())
    return make_error<StringError>("Empty " + ObjCImageInfoSectionName +
                                       " section in " + G.getName(),
                                   inconvertibleErrorCode());

  // Check that there's only one block in the section.
  if (std::next(ObjCImageInfoBlocks.begin()) != ObjCImageInfoBlocks.end())
    return make_error<StringError>("Multiple blocks in " +
                                       ObjCImageInfoSectionName +
                                       " section in " + G.getName(),
                                   inconvertibleErrorCode());

  // Check that the __objc_imageinfo section is unreferenced.
  // FIXME: We could optimize this check if Symbols had a ref-count.
  for (auto &Sec : G.sections()) {
    if (&Sec != ObjCImageInfo)
      for (auto *B : Sec.blocks())
        for (auto &E : B->edges())
          if (E.getTarget().isDefined() &&
              &E.getTarget().getBlock().getSection() == ObjCImageInfo)
            return make_error<StringError>(ObjCImageInfoSectionName +
                                               " is referenced within file " +
                                               G.getName(),
                                           inconvertibleErrorCode());
  }

  auto &ObjCImageInfoBlock = **ObjCImageInfoBlocks.begin();
  auto *ObjCImageInfoData = ObjCImageInfoBlock.getContent().data();
  auto Version = support::endian::read32(ObjCImageInfoData, G.getEndianness());
  auto Flags =
      support::endian::read32(ObjCImageInfoData + 4, G.getEndianness());

  // Lock the mutex while we verify / update the ObjCImageInfos map.
  std::lock_guard<std::mutex> Lock(PluginMutex);

  auto ObjCImageInfoItr = ObjCImageInfos.find(&MR.getTargetJITDylib());
  if (ObjCImageInfoItr != ObjCImageInfos.end()) {
    // We've already registered an __objc_imageinfo section. Verify the
    // content of this new section matches, then delete it.
    if (ObjCImageInfoItr->second.first != Version)
      return make_error<StringError>(
          "ObjC version in " + G.getName() +
              " does not match first registered version",
          inconvertibleErrorCode());
    if (ObjCImageInfoItr->second.second != Flags)
      return make_error<StringError>("ObjC flags in " + G.getName() +
                                         " do not match first registered flags",
                                     inconvertibleErrorCode());

    // __objc_imageinfo is valid. Delete the block.
    for (auto *S : ObjCImageInfo->symbols())
      G.removeDefinedSymbol(*S);
    G.removeBlock(ObjCImageInfoBlock);
  } else {
    // We haven't registered an __objc_imageinfo section yet. Register and
    // move on. The section should already be marked no-dead-strip.
    ObjCImageInfos[&MR.getTargetJITDylib()] = std::make_pair(Version, Flags);
  }

  return Error::success();
}

Error MachOPlatform::MachOPlatformPlugin::fixTLVSectionsAndEdges(
    jitlink::LinkGraph &G, JITDylib &JD) {

  // Rename external references to __tlv_bootstrap to ___orc_rt_tlv_get_addr.
  for (auto *Sym : G.external_symbols())
    if (Sym->getName() == "__tlv_bootstrap") {
      Sym->setName("___orc_rt_macho_tlv_get_addr");
      break;
    }

  // Store key in __thread_vars struct fields.
  if (auto *ThreadDataSec = G.findSectionByName(ThreadVarsSectionName)) {
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

    for (auto *B : ThreadDataSec->blocks()) {
      if (B->getSize() != 3 * G.getPointerSize())
        return make_error<StringError>("__thread_vars block at " +
                                           formatv("{0:x}", B->getAddress()) +
                                           " has unexpected size",
                                       inconvertibleErrorCode());

      auto NewBlockContent = G.allocateBuffer(B->getSize());
      llvm::copy(B->getContent(), NewBlockContent.data());
      memcpy(NewBlockContent.data() + G.getPointerSize(), &PlatformKeyBits,
             G.getPointerSize());
      B->setContent(NewBlockContent);
    }
  }

  // Transform any TLV edges into GOT edges.
  for (auto *B : G.blocks())
    for (auto &E : B->edges())
      if (E.getKind() ==
          jitlink::x86_64::RequestTLVPAndTransformToPCRel32TLVPLoadREXRelaxable)
        E.setKind(jitlink::x86_64::
                      RequestGOTAndTransformToPCRel32GOTLoadREXRelaxable);

  return Error::success();
}

std::optional<MachOPlatform::MachOPlatformPlugin::UnwindSections>
MachOPlatform::MachOPlatformPlugin::findUnwindSectionInfo(
    jitlink::LinkGraph &G) {
  using namespace jitlink;

  UnwindSections US;

  // ScanSection records a section range and adds any executable blocks that
  // that section points to to the CodeBlocks vector.
  SmallVector<Block *> CodeBlocks;
  auto ScanUnwindInfoSection = [&](Section &Sec, ExecutorAddrRange &SecRange) {
    if (Sec.blocks().empty())
      return;
    SecRange = (*Sec.blocks().begin())->getRange();
    for (auto *B : Sec.blocks()) {
      auto R = B->getRange();
      SecRange.Start = std::min(SecRange.Start, R.Start);
      SecRange.End = std::max(SecRange.End, R.End);
      for (auto &E : B->edges()) {
        if (!E.getTarget().isDefined())
          continue;
        auto &TargetBlock = E.getTarget().getBlock();
        auto &TargetSection = TargetBlock.getSection();
        if ((TargetSection.getMemProt() & MemProt::Exec) == MemProt::Exec)
          CodeBlocks.push_back(&TargetBlock);
      }
    }
  };

  if (Section *EHFrameSec = G.findSectionByName(EHFrameSectionName))
    ScanUnwindInfoSection(*EHFrameSec, US.DwarfSection);

  if (Section *CUInfoSec = G.findSectionByName(CompactUnwindInfoSectionName))
    ScanUnwindInfoSection(*CUInfoSec, US.CompactUnwindSection);

  // If we didn't find any pointed-to code-blocks then there's no need to
  // register any info.
  if (CodeBlocks.empty())
    return std::nullopt;

  // We have info to register. Sort the code blocks into address order and
  // build a list of contiguous address ranges covering them all.
  llvm::sort(CodeBlocks, [](const Block *LHS, const Block *RHS) {
    return LHS->getAddress() < RHS->getAddress();
  });
  for (auto *B : CodeBlocks) {
    if (US.CodeRanges.empty() || US.CodeRanges.back().End != B->getAddress())
      US.CodeRanges.push_back(B->getRange());
    else
      US.CodeRanges.back().End = B->getRange().End;
  }

  LLVM_DEBUG({
    dbgs() << "MachOPlatform identified unwind info in " << G.getName() << ":\n"
           << "  DWARF: ";
    if (US.DwarfSection.Start)
      dbgs() << US.DwarfSection << "\n";
    else
      dbgs() << "none\n";
    dbgs() << "  Compact-unwind: ";
    if (US.CompactUnwindSection.Start)
      dbgs() << US.CompactUnwindSection << "\n";
    else
      dbgs() << "none\n"
             << "for code ranges:\n";
    for (auto &CR : US.CodeRanges)
      dbgs() << "  " << CR << "\n";
    if (US.CodeRanges.size() >= G.sections_size())
      dbgs() << "WARNING: High number of discontiguous code ranges! "
                "Padding may be interfering with coalescing.\n";
  });

  return US;
}

Error MachOPlatform::MachOPlatformPlugin::registerObjectPlatformSections(
    jitlink::LinkGraph &G, JITDylib &JD, bool InBootstrapPhase) {

  // Get a pointer to the thread data section if there is one. It will be used
  // below.
  jitlink::Section *ThreadDataSection =
      G.findSectionByName(ThreadDataSectionName);

  // Handle thread BSS section if there is one.
  if (auto *ThreadBSSSection = G.findSectionByName(ThreadBSSSectionName)) {
    // If there's already a thread data section in this graph then merge the
    // thread BSS section content into it, otherwise just treat the thread
    // BSS section as the thread data section.
    if (ThreadDataSection)
      G.mergeSections(*ThreadDataSection, *ThreadBSSSection);
    else
      ThreadDataSection = ThreadBSSSection;
  }

  SmallVector<std::pair<StringRef, ExecutorAddrRange>, 8> MachOPlatformSecs;

  // Collect data sections to register.
  StringRef DataSections[] = {DataDataSectionName, DataCommonSectionName,
                              EHFrameSectionName};
  for (auto &SecName : DataSections) {
    if (auto *Sec = G.findSectionByName(SecName)) {
      jitlink::SectionRange R(*Sec);
      if (!R.empty())
        MachOPlatformSecs.push_back({SecName, R.getRange()});
    }
  }

  // Having merged thread BSS (if present) and thread data (if present),
  // record the resulting section range.
  if (ThreadDataSection) {
    jitlink::SectionRange R(*ThreadDataSection);
    if (!R.empty())
      MachOPlatformSecs.push_back({ThreadDataSectionName, R.getRange()});
  }

  // If any platform sections were found then add an allocation action to call
  // the registration function.
  StringRef PlatformSections[] = {
      ModInitFuncSectionName,   ObjCClassListSectionName,
      ObjCImageInfoSectionName, ObjCSelRefsSectionName,
      Swift5ProtoSectionName,   Swift5ProtosSectionName,
      Swift5TypesSectionName,
  };

  for (auto &SecName : PlatformSections) {
    auto *Sec = G.findSectionByName(SecName);
    if (!Sec)
      continue;
    jitlink::SectionRange R(*Sec);
    if (R.empty())
      continue;

    MachOPlatformSecs.push_back({SecName, R.getRange()});
  }

  std::optional<std::tuple<SmallVector<ExecutorAddrRange>, ExecutorAddrRange,
                           ExecutorAddrRange>>
      UnwindInfo;
  if (auto UI = findUnwindSectionInfo(G))
    UnwindInfo = std::make_tuple(std::move(UI->CodeRanges), UI->DwarfSection,
                                 UI->CompactUnwindSection);

  if (!MachOPlatformSecs.empty() || UnwindInfo) {
    ExecutorAddr HeaderAddr;
    {
      std::lock_guard<std::mutex> Lock(MP.PlatformMutex);
      auto I = MP.JITDylibToHeaderAddr.find(&JD);
      assert(I != MP.JITDylibToHeaderAddr.end() &&
             "Missing header for JITDylib");
      HeaderAddr = I->second;
    }

    // Dump the scraped inits.
    LLVM_DEBUG({
      dbgs() << "MachOPlatform: Scraped " << G.getName() << " init sections:\n";
      for (auto &KV : MachOPlatformSecs)
        dbgs() << "  " << KV.first << ": " << KV.second << "\n";
    });

    using SPSRegisterObjectPlatformSectionsArgs = SPSArgList<
        SPSExecutorAddr,
        SPSOptional<SPSTuple<SPSSequence<SPSExecutorAddrRange>,
                             SPSExecutorAddrRange, SPSExecutorAddrRange>>,
        SPSSequence<SPSTuple<SPSString, SPSExecutorAddrRange>>>;

    shared::AllocActions &allocActions = LLVM_LIKELY(!InBootstrapPhase)
                                             ? G.allocActions()
                                             : MP.Bootstrap.load()->DeferredAAs;

    allocActions.push_back(
        {cantFail(
             WrapperFunctionCall::Create<SPSRegisterObjectPlatformSectionsArgs>(
                 MP.RegisterObjectPlatformSections.Addr, HeaderAddr, UnwindInfo,
                 MachOPlatformSecs)),
         cantFail(
             WrapperFunctionCall::Create<SPSRegisterObjectPlatformSectionsArgs>(
                 MP.DeregisterObjectPlatformSections.Addr, HeaderAddr,
                 UnwindInfo, MachOPlatformSecs))});
  }

  return Error::success();
}

} // End namespace orc.
} // End namespace llvm.
