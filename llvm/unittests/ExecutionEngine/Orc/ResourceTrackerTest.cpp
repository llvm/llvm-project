//===------ ResourceTrackerTest.cpp - Unit tests ResourceTracker API
//-------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "OrcTestCommon.h"
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/OrcError.h"
#include "llvm/Testing/Support/Error.h"

using namespace llvm;
using namespace llvm::orc;

class ResourceTrackerStandardTest : public CoreAPIsBasedStandardTest {};

namespace {

template <typename ResourceT = unsigned>
class SimpleResourceManager : public ResourceManager {
public:
  using HandleRemoveFunction = unique_function<Error(ResourceKey)>;

  using HandleTransferFunction =
      unique_function<void(ResourceKey, ResourceKey)>;

  using RecordedResourcesMap = DenseMap<ResourceKey, ResourceT>;

  SimpleResourceManager(ExecutionSession &ES) : ES(ES) {
    HandleRemove = [&](ResourceKey K) -> Error {
      ES.runSessionLocked([&] { removeResource(K); });
      return Error::success();
    };

    HandleTransfer = [this](ResourceKey DstKey, ResourceKey SrcKey) {
      transferResources(DstKey, SrcKey);
    };

    ES.registerResourceManager(*this);
  }

  SimpleResourceManager(const SimpleResourceManager &) = delete;
  SimpleResourceManager &operator=(const SimpleResourceManager &) = delete;
  SimpleResourceManager(SimpleResourceManager &&) = delete;
  SimpleResourceManager &operator=(SimpleResourceManager &&) = delete;

  ~SimpleResourceManager() { ES.deregisterResourceManager(*this); }

  /// Set the HandleRemove function object.
  void setHandleRemove(HandleRemoveFunction HandleRemove) {
    this->HandleRemove = std::move(HandleRemove);
  }

  /// Set the HandleTransfer function object.
  void setHandleTransfer(HandleTransferFunction HandleTransfer) {
    this->HandleTransfer = std::move(HandleTransfer);
  }

  /// Create an association between the given key and resource.
  template <typename MergeOp = std::plus<ResourceT>>
  void recordResource(ResourceKey K, ResourceT Val = ResourceT(),
                      MergeOp Merge = MergeOp()) {
    auto Tmp = std::move(Resources[K]);
    Resources[K] = Merge(std::move(Tmp), std::move(Val));
  }

  /// Remove the resource associated with K from the map if present.
  void removeResource(ResourceKey K) { Resources.erase(K); }

  /// Transfer resources from DstKey to SrcKey.
  template <typename MergeOp = std::plus<ResourceT>>
  void transferResources(ResourceKey DstKey, ResourceKey SrcKey,
                         MergeOp Merge = MergeOp()) {
    auto &DstResourceRef = Resources[DstKey];
    ResourceT DstResources;
    std::swap(DstResourceRef, DstResources);

    auto SI = Resources.find(SrcKey);
    assert(SI != Resources.end() && "No resource associated with SrcKey");

    DstResourceRef = Merge(std::move(DstResources), std::move(SI->second));
    Resources.erase(SI);
  }

  /// Return a reference to the Resources map.
  RecordedResourcesMap &getRecordedResources() { return Resources; }
  const RecordedResourcesMap &getRecordedResources() const { return Resources; }

  Error handleRemoveResources(ResourceKey K) override {
    return HandleRemove(K);
  }

  void handleTransferResources(ResourceKey DstKey,
                               ResourceKey SrcKey) override {
    HandleTransfer(DstKey, SrcKey);
  }

  static void transferNotAllowed(ResourceKey DstKey, ResourceKey SrcKey) {
    llvm_unreachable("Resource transfer not allowed");
  }

private:
  ExecutionSession &ES;
  HandleRemoveFunction HandleRemove;
  HandleTransferFunction HandleTransfer;
  RecordedResourcesMap Resources;
};

TEST_F(ResourceTrackerStandardTest,
       BasicDefineAndRemoveAllBeforeMaterializing) {

  bool ResourceManagerGotRemove = false;
  SimpleResourceManager<> SRM(ES);
  SRM.setHandleRemove([&](ResourceKey K) -> Error {
    ResourceManagerGotRemove = true;
    EXPECT_EQ(SRM.getRecordedResources().size(), 0U)
        << "Unexpected resources recorded";
    SRM.removeResource(K);
    return Error::success();
  });

  bool MaterializationUnitDestroyed = false;
  auto MU = std::make_unique<SimpleMaterializationUnit>(
      SymbolFlagsMap({{Foo, FooSym.getFlags()}}),
      [&](std::unique_ptr<MaterializationResponsibility> R) {
        llvm_unreachable("Never called");
      },
      nullptr, SimpleMaterializationUnit::DiscardFunction(),
      [&]() { MaterializationUnitDestroyed = true; });

  auto RT = JD.createResourceTracker();
  cantFail(JD.define(std::move(MU), RT));
  cantFail(RT->remove());
  auto SymFlags = cantFail(ES.lookupFlags(
      LookupKind::Static,
      {{&JD, JITDylibLookupFlags::MatchExportedSymbolsOnly}},
      SymbolLookupSet(Foo, SymbolLookupFlags::WeaklyReferencedSymbol)));

  EXPECT_EQ(SymFlags.size(), 0U)
      << "Symbols should have been removed from the symbol table";
  EXPECT_TRUE(ResourceManagerGotRemove)
      << "ResourceManager did not receive handleRemoveResources";
  EXPECT_TRUE(MaterializationUnitDestroyed)
      << "MaterializationUnit not destroyed in response to removal";
}

TEST_F(ResourceTrackerStandardTest, BasicDefineAndRemoveAllAfterMaterializing) {

  bool ResourceManagerGotRemove = false;
  SimpleResourceManager<> SRM(ES);
  SRM.setHandleRemove([&](ResourceKey K) -> Error {
    ResourceManagerGotRemove = true;
    EXPECT_EQ(SRM.getRecordedResources().size(), 1U)
        << "Unexpected number of resources recorded";
    EXPECT_EQ(SRM.getRecordedResources().count(K), 1U)
        << "Unexpected recorded resource";
    SRM.removeResource(K);
    return Error::success();
  });

  auto MU = std::make_unique<SimpleMaterializationUnit>(
      SymbolFlagsMap({{Foo, FooSym.getFlags()}}),
      [&](std::unique_ptr<MaterializationResponsibility> R) {
        cantFail(R->withResourceKeyDo(
            [&](ResourceKey K) { SRM.recordResource(K); }));
        cantFail(R->notifyResolved({{Foo, FooSym}}));
        cantFail(R->notifyEmitted());
      });

  auto RT = JD.createResourceTracker();
  cantFail(JD.define(std::move(MU), RT));
  cantFail(ES.lookup({&JD}, Foo));
  cantFail(RT->remove());
  auto SymFlags = cantFail(ES.lookupFlags(
      LookupKind::Static,
      {{&JD, JITDylibLookupFlags::MatchExportedSymbolsOnly}},
      SymbolLookupSet(Foo, SymbolLookupFlags::WeaklyReferencedSymbol)));

  EXPECT_EQ(SymFlags.size(), 0U)
      << "Symbols should have been removed from the symbol table";
  EXPECT_TRUE(ResourceManagerGotRemove)
      << "ResourceManager did not receive handleRemoveResources";
}

TEST_F(ResourceTrackerStandardTest, BasicDefineAndRemoveAllWhileMaterializing) {

  bool ResourceManagerGotRemove = false;
  SimpleResourceManager<> SRM(ES);
  SRM.setHandleRemove([&](ResourceKey K) -> Error {
    ResourceManagerGotRemove = true;
    EXPECT_EQ(SRM.getRecordedResources().size(), 0U)
        << "Unexpected resources recorded";
    SRM.removeResource(K);
    return Error::success();
  });

  std::unique_ptr<MaterializationResponsibility> MR;
  auto MU = std::make_unique<SimpleMaterializationUnit>(
      SymbolFlagsMap({{Foo, FooSym.getFlags()}}),
      [&](std::unique_ptr<MaterializationResponsibility> R) {
        MR = std::move(R);
      });

  auto RT = JD.createResourceTracker();
  cantFail(JD.define(std::move(MU), RT));

  ES.lookup(
      LookupKind::Static, makeJITDylibSearchOrder(&JD), SymbolLookupSet(Foo),
      SymbolState::Ready,
      [](Expected<SymbolMap> Result) {
        EXPECT_THAT_EXPECTED(Result, Failed<FailedToMaterialize>())
            << "Lookup failed unexpectedly";
      },
      NoDependenciesToRegister);

  cantFail(RT->remove());
  auto SymFlags = cantFail(ES.lookupFlags(
      LookupKind::Static,
      {{&JD, JITDylibLookupFlags::MatchExportedSymbolsOnly}},
      SymbolLookupSet(Foo, SymbolLookupFlags::WeaklyReferencedSymbol)));

  EXPECT_EQ(SymFlags.size(), 0U)
      << "Symbols should have been removed from the symbol table";
  EXPECT_TRUE(ResourceManagerGotRemove)
      << "ResourceManager did not receive handleRemoveResources";

  EXPECT_THAT_ERROR(MR->withResourceKeyDo([](ResourceKey K) {
    ADD_FAILURE() << "Should not reach withResourceKeyDo body for removed key";
  }),
                    Failed<ResourceTrackerDefunct>())
      << "withResourceKeyDo on MR with removed tracker should have failed";
  EXPECT_THAT_ERROR(MR->notifyResolved({{Foo, FooSym}}),
                    Failed<ResourceTrackerDefunct>())
      << "notifyResolved on MR with removed tracker should have failed";

  MR->failMaterialization();
}

TEST_F(ResourceTrackerStandardTest, JITDylibClear) {
  SimpleResourceManager<> SRM(ES);

  // Add materializer for Foo.
  cantFail(JD.define(std::make_unique<SimpleMaterializationUnit>(
      SymbolFlagsMap({{Foo, FooSym.getFlags()}}),
      [&](std::unique_ptr<MaterializationResponsibility> R) {
        cantFail(R->withResourceKeyDo(
            [&](ResourceKey K) { ++SRM.getRecordedResources()[K]; }));
        cantFail(R->notifyResolved({{Foo, FooSym}}));
        cantFail(R->notifyEmitted());
      })));

  // Add materializer for Bar.
  cantFail(JD.define(std::make_unique<SimpleMaterializationUnit>(
      SymbolFlagsMap({{Bar, BarSym.getFlags()}}),
      [&](std::unique_ptr<MaterializationResponsibility> R) {
        cantFail(R->withResourceKeyDo(
            [&](ResourceKey K) { ++SRM.getRecordedResources()[K]; }));
        cantFail(R->notifyResolved({{Bar, BarSym}}));
        cantFail(R->notifyEmitted());
      })));

  EXPECT_TRUE(SRM.getRecordedResources().empty())
      << "Expected no resources recorded yet.";

  cantFail(
      ES.lookup(makeJITDylibSearchOrder(&JD), SymbolLookupSet({Foo, Bar})));

  auto JDResourceKey = JD.getDefaultResourceTracker()->getKeyUnsafe();
  EXPECT_EQ(SRM.getRecordedResources().size(), 1U)
      << "Expected exactly one entry (for JD's ResourceKey)";
  EXPECT_EQ(SRM.getRecordedResources().count(JDResourceKey), 1U)
      << "Expected an entry for JD's ResourceKey";
  EXPECT_EQ(SRM.getRecordedResources()[JDResourceKey], 2U)
      << "Expected value of 2 for JD's ResourceKey "
         "(+1 for each of Foo and Bar)";

  cantFail(JD.clear());

  EXPECT_TRUE(SRM.getRecordedResources().empty())
      << "Expected no resources recorded after clear";
}

TEST_F(ResourceTrackerStandardTest,
       BasicDefineAndExplicitTransferBeforeMaterializing) {

  bool ResourceManagerGotTransfer = false;
  SimpleResourceManager<> SRM(ES);
  SRM.setHandleTransfer([&](ResourceKey DstKey, ResourceKey SrcKey) {
    ResourceManagerGotTransfer = true;
    auto &RR = SRM.getRecordedResources();
    EXPECT_EQ(RR.size(), 0U) << "Expected no resources recorded yet";
  });

  auto MakeMU = [&](SymbolStringPtr Name, JITEvaluatedSymbol Sym) {
    return std::make_unique<SimpleMaterializationUnit>(
        SymbolFlagsMap({{Name, Sym.getFlags()}}),
        [=, &SRM](std::unique_ptr<MaterializationResponsibility> R) {
          cantFail(R->withResourceKeyDo(
              [&](ResourceKey K) { SRM.recordResource(K); }));
          cantFail(R->notifyResolved({{Name, Sym}}));
          cantFail(R->notifyEmitted());
        });
  };

  auto FooRT = JD.createResourceTracker();
  cantFail(JD.define(MakeMU(Foo, FooSym), FooRT));

  auto BarRT = JD.createResourceTracker();
  cantFail(JD.define(MakeMU(Bar, BarSym), BarRT));

  BarRT->transferTo(*FooRT);

  EXPECT_TRUE(ResourceManagerGotTransfer)
      << "ResourceManager did not receive transfer";
  EXPECT_TRUE(BarRT->isDefunct()) << "BarRT should now be defunct";

  cantFail(
      ES.lookup(makeJITDylibSearchOrder({&JD}), SymbolLookupSet({Foo, Bar})));

  EXPECT_EQ(SRM.getRecordedResources().size(), 1U)
      << "Expected exactly one entry (for FooRT's Key)";
  EXPECT_EQ(SRM.getRecordedResources().count(FooRT->getKeyUnsafe()), 1U)
      << "Expected an entry for FooRT's ResourceKey";
  EXPECT_EQ(SRM.getRecordedResources().count(BarRT->getKeyUnsafe()), 0U)
      << "Expected no entry for BarRT's ResourceKey";

  // We need to explicitly destroy FooRT or its resources will be implicitly
  // transferred to the default tracker triggering a second call to our
  // transfer function above (which expects only one call).
  cantFail(FooRT->remove());
}

TEST_F(ResourceTrackerStandardTest,
       BasicDefineAndExplicitTransferAfterMaterializing) {

  bool ResourceManagerGotTransfer = false;
  SimpleResourceManager<> SRM(ES);
  SRM.setHandleTransfer([&](ResourceKey DstKey, ResourceKey SrcKey) {
    ResourceManagerGotTransfer = true;
    SRM.transferResources(DstKey, SrcKey);
  });

  auto MakeMU = [&](SymbolStringPtr Name, JITEvaluatedSymbol Sym) {
    return std::make_unique<SimpleMaterializationUnit>(
        SymbolFlagsMap({{Name, Sym.getFlags()}}),
        [=, &SRM](std::unique_ptr<MaterializationResponsibility> R) {
          cantFail(R->withResourceKeyDo(
              [&](ResourceKey K) { SRM.recordResource(K, 1); }));
          cantFail(R->notifyResolved({{Name, Sym}}));
          cantFail(R->notifyEmitted());
        });
  };

  auto FooRT = JD.createResourceTracker();
  cantFail(JD.define(MakeMU(Foo, FooSym), FooRT));

  auto BarRT = JD.createResourceTracker();
  cantFail(JD.define(MakeMU(Bar, BarSym), BarRT));

  EXPECT_EQ(SRM.getRecordedResources().size(), 0U)
      << "Expected no recorded resources yet";

  cantFail(
      ES.lookup(makeJITDylibSearchOrder({&JD}), SymbolLookupSet({Foo, Bar})));

  EXPECT_EQ(SRM.getRecordedResources().size(), 2U)
      << "Expected recorded resources for both Foo and Bar";

  BarRT->transferTo(*FooRT);

  EXPECT_TRUE(ResourceManagerGotTransfer)
      << "ResourceManager did not receive transfer";
  EXPECT_TRUE(BarRT->isDefunct()) << "BarRT should now be defunct";

  EXPECT_EQ(SRM.getRecordedResources().size(), 1U)
      << "Expected recorded resources for Foo only";
  EXPECT_EQ(SRM.getRecordedResources().count(FooRT->getKeyUnsafe()), 1U)
      << "Expected recorded resources for Foo";
  EXPECT_EQ(SRM.getRecordedResources()[FooRT->getKeyUnsafe()], 2U)
      << "Expected resources value for for Foo to be '2'";
}

TEST_F(ResourceTrackerStandardTest,
       BasicDefineAndExplicitTransferWhileMaterializing) {

  bool ResourceManagerGotTransfer = false;
  SimpleResourceManager<> SRM(ES);
  SRM.setHandleTransfer([&](ResourceKey DstKey, ResourceKey SrcKey) {
    ResourceManagerGotTransfer = true;
    SRM.transferResources(DstKey, SrcKey);
  });

  auto FooRT = JD.createResourceTracker();
  std::unique_ptr<MaterializationResponsibility> FooMR;
  cantFail(JD.define(std::make_unique<SimpleMaterializationUnit>(
                         SymbolFlagsMap({{Foo, FooSym.getFlags()}}),
                         [&](std::unique_ptr<MaterializationResponsibility> R) {
                           FooMR = std::move(R);
                         }),
                     FooRT));

  auto BarRT = JD.createResourceTracker();

  ES.lookup(
      LookupKind::Static, makeJITDylibSearchOrder(&JD), SymbolLookupSet(Foo),
      SymbolState::Ready,
      [](Expected<SymbolMap> Result) { cantFail(Result.takeError()); },
      NoDependenciesToRegister);

  cantFail(FooMR->withResourceKeyDo([&](ResourceKey K) {
    EXPECT_EQ(FooRT->getKeyUnsafe(), K)
        << "Expected FooRT's ResourceKey for Foo here";
    SRM.recordResource(K, 1);
  }));

  EXPECT_EQ(SRM.getRecordedResources().size(), 1U)
      << "Expected one recorded resource here";
  EXPECT_EQ(SRM.getRecordedResources()[FooRT->getKeyUnsafe()], 1U)
      << "Expected Resource value for FooRT to be '1' here";

  FooRT->transferTo(*BarRT);

  EXPECT_TRUE(ResourceManagerGotTransfer)
      << "Expected resource manager to receive handleTransferResources call";

  cantFail(FooMR->withResourceKeyDo([&](ResourceKey K) {
    EXPECT_EQ(BarRT->getKeyUnsafe(), K)
        << "Expected BarRT's ResourceKey for Foo here";
    SRM.recordResource(K, 1);
  }));

  EXPECT_EQ(SRM.getRecordedResources().size(), 1U)
      << "Expected one recorded resource here";
  EXPECT_EQ(SRM.getRecordedResources().count(BarRT->getKeyUnsafe()), 1U)
      << "Expected RecordedResources to contain an entry for BarRT";
  EXPECT_EQ(SRM.getRecordedResources()[BarRT->getKeyUnsafe()], 2U)
      << "Expected Resource value for BarRT to be '2' here";

  cantFail(FooMR->notifyResolved({{Foo, FooSym}}));
  cantFail(FooMR->notifyEmitted());
}

} // namespace
