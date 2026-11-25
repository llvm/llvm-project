//===-------------- EPCGenericDylibManagerTest.cpp ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "OrcTestCommon.h"

#include "llvm/ExecutionEngine/Orc/EPCGenericDylibManagerSPS.h"
#include "llvm/ExecutionEngine/Orc/Shared/Mangler.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/ExecutionEngine/Orc/AbsoluteSymbols.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/Mangling.h"
#include "llvm/ExecutionEngine/Orc/SelfExecutorProcessControl.h"
#include "llvm/ExecutionEngine/Orc/Shared/SPSCI/NativeDylibManagerSPSCI.h"
#include "llvm/ExecutionEngine/Orc/Shared/WrapperFunctionUtils.h"
#include "llvm/Testing/Support/Error.h"

using namespace llvm;
using namespace llvm::orc;
using namespace llvm::orc::shared;

namespace {

// The instance address returned by testCreate, and the number of calls to it.
const ExecutorAddr CreatedInstance(0x1234);
unsigned NumCreateCalls = 0;

// The instance address passed to the most recent testOpen call.
ExecutorAddr OpenedWithInstance;

CWrapperFunctionBuffer testCreate(const char *ArgData, size_t ArgSize) {
  return WrapperFunction<rt::sps_ci::DylibMgrCreate::SPSSig>::handle(
             ArgData, ArgSize,
             []() -> Expected<ExecutorAddr> {
               ++NumCreateCalls;
               return CreatedInstance;
             })
      .release();
}

CWrapperFunctionBuffer testOpen(const char *ArgData, size_t ArgSize) {
  return WrapperFunction<rt::sps_ci::DylibMgrOpen::SPSSig>::handle(
             ArgData, ArgSize,
             [](ExecutorAddr Instance, std::string Path,
                uint64_t Mode) -> Expected<ExecutorAddr> {
               OpenedWithInstance = Instance;
               return ExecutorAddr(0x5678);
             })
      .release();
}

/// Define testOpen in JD under the default NativeDylibManager name. The
/// resolve operation is given a placeholder address: these tests never call
/// it.
void defineTestOps(JITDylib &JD) {
  MangleAndInterner Mangle(JD.getExecutionSession());
  auto Exported = JITSymbolFlags::Exported;
  cantFail(JD.define(absoluteSymbols({
      {Mangle(rt::sps_ci::DylibMgrOpen::Name),
       {ExecutorAddr::fromPtr(&testOpen), Exported}},
      {Mangle(rt::sps_ci::DylibMgrResolve::Name), {ExecutorAddr(3), Exported}},
  })));
}

TEST(EPCGenericDylibManagerTest, CreateFromExecutionSession) {
  // Verify that Create(ExecutionSession&) looks up symbols in the bootstrap
  // JITDylib using the default NativeDylibManager symbol names.
  class EPCWithBootstrapSymbols : public UnsupportedExecutorProcessControl {
  public:
    EPCWithBootstrapSymbols(std::shared_ptr<SymbolStringPool> SSP,
                            StringMap<ExecutorAddr> BS)
        : UnsupportedExecutorProcessControl(std::move(SSP), nullptr,
                                            sys::getProcessTriple()) {
      this->BootstrapSymbols = std::move(BS);
    }
  };

  ExecutorAddr InstanceAddr(1), OpenAddr(2), ResolveAddr(3);

  StringMap<ExecutorAddr> BootstrapSyms;
  Mangler Mangle{Triple(sys::getProcessTriple())};
  BootstrapSyms[Mangle.mangledCopy(
      rt::sps_ci::NativeDylibManagerInstanceName)] = InstanceAddr;
  BootstrapSyms[Mangle.mangledCopy(rt::sps_ci::DylibMgrOpen::Name)] = OpenAddr;
  BootstrapSyms[Mangle.mangledCopy(rt::sps_ci::DylibMgrResolve::Name)] =
      ResolveAddr;

  auto SSP = std::make_shared<SymbolStringPool>();
  auto EPC =
      std::make_unique<EPCWithBootstrapSymbols>(SSP, std::move(BootstrapSyms));
  ExecutionSession ES(std::move(EPC));

  auto Result = sps::createEPCGenericDylibManager(ES);
  EXPECT_THAT_EXPECTED(Result, Succeeded());

  cantFail(ES.endSession());
}

TEST(EPCGenericDylibManagerTest, CreateViaCreateOperation) {
  // Verify that if the executor provides the create operation, it is called to
  // create the instance that the manager's operations are applied to.
  ExecutionSession ES(cantFail(SelfExecutorProcessControl::Create()));
  MangleAndInterner Mangle(ES);
  NumCreateCalls = 0;
  OpenedWithInstance = ExecutorAddr();

  defineTestOps(ES.getBootstrapJITDylib());
  cantFail(ES.getBootstrapJITDylib().define(absoluteSymbols(
      {{Mangle(rt::sps_ci::DylibMgrCreate::Name),
        {ExecutorAddr::fromPtr(&testCreate), JITSymbolFlags::Exported}}})));

  auto DM = sps::createEPCGenericDylibManager(ES);
  ASSERT_THAT_EXPECTED(DM, Succeeded());
  EXPECT_EQ(NumCreateCalls, 1U);

  EXPECT_THAT_EXPECTED((*DM)->open("foo", 0), Succeeded());
  EXPECT_EQ(OpenedWithInstance, CreatedInstance);

  cantFail(ES.endSession());
}

TEST(EPCGenericDylibManagerTest, CreatePreferredOverInstance) {
  // If the executor provides both, the create operation should be used, and
  // the published instance ignored.
  ExecutionSession ES(cantFail(SelfExecutorProcessControl::Create()));
  MangleAndInterner Mangle(ES);
  NumCreateCalls = 0;
  OpenedWithInstance = ExecutorAddr();

  auto Exported = JITSymbolFlags::Exported;
  defineTestOps(ES.getBootstrapJITDylib());
  cantFail(ES.getBootstrapJITDylib().define(
      absoluteSymbols({{Mangle(rt::sps_ci::DylibMgrCreate::Name),
                        {ExecutorAddr::fromPtr(&testCreate), Exported}},
                       {Mangle(rt::sps_ci::NativeDylibManagerInstanceName),
                        {ExecutorAddr(1), Exported}}})));

  auto DM = sps::createEPCGenericDylibManager(ES);
  ASSERT_THAT_EXPECTED(DM, Succeeded());
  EXPECT_EQ(NumCreateCalls, 1U);

  EXPECT_THAT_EXPECTED((*DM)->open("foo", 0), Succeeded());
  EXPECT_EQ(OpenedWithInstance, CreatedInstance);

  cantFail(ES.endSession());
}

TEST(EPCGenericDylibManagerTest, CreateFailsWithoutInstanceOrCreate) {
  // Verify that creation fails if the executor provides the operations, but no
  // way to obtain an instance.
  auto SSP = std::make_shared<SymbolStringPool>();
  auto EPC =
      std::make_unique<UnsupportedExecutorProcessControl>(std::move(SSP));
  ExecutionSession ES(std::move(EPC));
  auto &JD = ES.createBareJITDylib("JD");

  defineTestOps(JD);

  EXPECT_THAT_EXPECTED(sps::createEPCGenericDylibManager(JD), Failed());

  cantFail(ES.endSession());
}

} // namespace
